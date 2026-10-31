/**
 * Default Agent driver over queued turns and step-boundary input. Every request
 * is derived from the session log.
 * @module dsh-agent-loop/agent
 */

import type {
  Agent,
  AgentCancelCause,
  AgentEventDispatch,
  AgentOptions,
  AgentStatus,
  CancelOptions,
  InboxTarget,
  PreStepDecision,
  RequestErrorAction,
} from '@deepseek-ai/dsh-agent'
import { Inbox, agentEvents, assembleContextFor } from '@deepseek-ai/dsh-agent'
import type { GenerateOptions, LlmCallConfig, Message, PreparedLlmCall } from '@deepseek-ai/dsh-llm'
import {
  BlockAssembler,
  LlmError,
  createAssistantMessage,
  deepFreeze,
  errorChain,
  markAgentLoopRequest,
} from '@deepseek-ai/dsh-llm'
import type { Scope } from '@deepseek-ai/dsh-scope'
import { createScope } from '@deepseek-ai/dsh-scope'
import type { EpochHeader, RequestContext, Session, SessionId, TurnEndReason, UserMessage } from '@deepseek-ai/dsh-session'
import { canonicalHeader, headerEquals } from '@deepseek-ai/dsh-session'
import { joinContextSections, renderContextSections, renderPrompt } from '@deepseek-ai/dsh-system-prompt'
import type { PromptAssembly } from '@deepseek-ai/dsh-system-prompt'
import type { Context } from '@deepseek-ai/cordis'
import { RuntimeContextProjection } from './runtime-context.ts'
import { executeToolCalls } from './tool-calls.ts'

type Phase =
  | { kind: 'idle'; lastTurn: number }
  | {
    kind: 'maintenance'
    abort: AbortController
    lastTurn: number
    wakeRequested: boolean
  }
  | { kind: 'running'; abort: AbortController; turn: number; step: number; wakeRequested: boolean }

type StepEndReason = Extract<TurnEndReason, { kind: 'completed' | 'max-tokens' }>

type PreparedStep =
  | { kind: 'reject' }
  | { kind: 'enter'; messages: UserMessage[]; assembly: PromptAssembly }

/** Remove adapter-derived values before plugins propose the next request config. */
function requestProposal(header: EpochHeader): LlmCallConfig {
  if (header.adapterDefaults === undefined) return header.config
  const proposal = { ...header.config }
  if (header.adapterDefaults.reasoningEffort === true) delete proposal.reasoningEffort
  if (header.adapterDefaults.maxTokens === true) delete proposal.maxTokens
  return proposal
}

/**
 * Drives one session through turn and step boundaries.
 *
 * ============================================================================
 * 【学习笔记：ReactLoopAgent 架构剖析 —— DeepSeek 默认的 Agent 执行驱动器】
 *
 * Q1: 这个是 DeepSeek 默认的 AgentLoopDriver 么？
 * A1: 是的！`ReactLoopAgent` 是 DeepSeek Harness 官方默认、唯一的执行循环引擎（Loop Driver）。
 *     它实现了 `Agent` 接口，整个 harness 的智能体执行完全围绕 ReAct（Reasoning 思考 + Acting 行动）循环构建。
 *
 * Q2: 它的层级抽象（Session / Turn / Step）是什么？
 *     - Session（会话）：持久化事件日志（Event Sourcing），是系统唯一的不可变真理源（Source of Truth）。
 *     - Turn（轮次）：由外部（如用户输入）触发的一次宏观交互周期（一问一答周期）。
 *       从 `turn/start` 开始，直到大模型最终输出文本答复或发生中止/错误（`turn/end`）。
 *       一个 Turn 内部包含 1 到 N 个 Step。
 *     - Step（单步）：一次微观的 ReAct 自循环。
 *       【思考 Reasoning】：组装上下文与 System Prompt -> 流式请求大模型。
 *       【行动 Acting】：
 *         * 若大模型返回包含 Tool Calls：并行/串行调度执行工具 -> 结果追加到会话上下文 -> 循环进入下一个 Step。
 *         * 若大模型无 Tool Calls（纯文本回答）：说明任务已得到解答，当前 Step 结束，整个 Turn 随之完成。
 *
 * Q3: 核心方法调用链路图：
 *     外部投递输入 (send / followup / steer)
 *       └──> wakeDriver() 【唤醒驱动器，防并发与排队锁存】
 *             └──> kick() 【最外层驱动循环，while (await turn())】
 *                   └──> turn() 【单个轮次，负责 turn/start -> Step 循环 -> turn/end】
 *                         ├──> preStep() 【认领 Inbox 消息，组装 SystemPrompt 与上下文】
 *                         └──> step() 【调用 LLM，流式收集，执行 Tool Calls，驱动下一步】
 * ============================================================================
 */
export class ReactLoopAgent implements Agent {
  /** Agent 拥有的持久化工作输入投影（收件箱，管理 next-turn 排队与 next-step 插队消息） */
  readonly inbox: Inbox
  /** 当前执行阶段状态机（idle 空闲 / running 运行中 / maintenance 维护中） */
  private phase: Phase
  /** 当前正在执行中的异步活动结算 Promise，用于 whenIdle() 追踪 */
  private activityDone: Promise<void> = Promise.resolve()

  /** Agent 专属的 Cordis 作用域上下文（注册在其上的插件在 agent 销毁时自动销毁） */
  readonly scope: Scope
  readonly ctx: Context

  /** 预先编译的事件派发器，避免热路径中重复分配内存 */
  private readonly dispatch: AgentEventDispatch

  /**
   * 标记当前 Agent 实例是否已向 Session Log 中写入过起始的 `request/header` 事件。
   * - 首次调用（false 时）：向会话写入首个请求头事件（标记 reason 为 'initial' 或 'resume'），
   *   并将构造时传入的 provider/model/reasoningEffort 等配置作为 seedConfig；
   * - 后续步骤（true 后）：直接复用会话中已记录的请求头，仅当模型/系统提示词/工具等发生实质改变时才写入 'change' 事件。
   */
  private requestHeaderLogged = false
  /** 运行时请求上下文投影 */
  private readonly runtimeContext: RuntimeContextProjection

  constructor(
    private loopCtx: Context,
    public readonly id: SessionId,
    public readonly options: AgentOptions,
    public readonly session: Session,
  ) {
    // 1. 初始化事件分发器，绑定此 Agent 实例
    this.dispatch = agentEvents(loopCtx, this)

    // 2. 绑定收件箱投影，当 inbox 发生插入、丢弃、认领时向系统派发相应事件
    this.inbox = new Inbox(session, {
      inserted: (message) => { this.dispatch.emit('agent/inbox/inserted', { message }) },
      discarded: (message) => { this.dispatch.emit('agent/inbox/discarded', { message }) },
      claimed: (message, turn) => { this.dispatch.emit('agent/inbox/claimed', { message, turn }) },
    })

    // 3. 从 Session 历史日志中找到最后的 turn 序号恢复状态机，默认处于 idle 阶段
    const lastTurn = session.events.findLast(event => event.type === 'turn/start')?.data.turn ?? 0
    this.phase = { kind: 'idle', lastTurn }

    // 4. 创建 Agent 专属的独立 Scope，并在其 Context 上覆盖绑定 own property `agent: this`
    this.scope = createScope(loopCtx, this)
    this.ctx = this.scope.ctx.extend({ agent: this })
    this.runtimeContext = new RuntimeContextProjection(this.ctx, session)
  }

  get status(): AgentStatus {
    return this.phase.kind === 'idle' || this.phase.kind === 'maintenance' ? 'idle' : 'running'
  }

  /** Commit a phase and publish its externally visible status transition. */
  private setPhase(next: Phase): void {
    const previousStatus = this.status
    this.phase = next
    const status = this.status
    if (status !== previousStatus) {
      this.dispatch.emit('agent/status', { status })
    }
  }

  send(message: UserMessage, target: InboxTarget, wakeup: boolean): void {
    // Waking input cannot join an aborted activity, so it starts the next turn.
    // Captured before the insertion so a reentrant cancel from a splice observer cannot reclassify it.
    const wakingAfterAbort = wakeup && this.phase.kind !== 'idle' && this.phase.abort.signal.aborted
    const resolvedTarget = wakingAfterAbort ? 'next-turn' : target
    this.inbox.splice(resolvedTarget, Infinity, 0, [message])
    if (wakeup) this.wakeDriver(wakingAfterAbort)
  }

  followup(input: UserMessage): void {
    this.send(input, 'next-turn', true)
  }

  steer(input: UserMessage): void {
    this.send(input, 'next-step', true)
  }

  inject(input: UserMessage): void {
    this.send(input, 'next-step', false)
  }

  cancel(cause: AgentCancelCause, options: CancelOptions = {}): void {
    if (!options.keepInbox) {
      this.inbox.clear()
      if (this.phase.kind !== 'idle') this.phase.wakeRequested = false
    }
    if (this.phase.kind !== 'idle') this.phase.abort.abort(cause)
  }

  runMaintenance<T>(job: (signal: AbortSignal) => Promise<T>): Promise<T> {
    if (this.phase.kind !== 'idle') throw new Error(`agent "${this.id}" already has active work`)
    const done = Promise.withResolvers<void>()
    const maintenance: Phase = {
      kind: 'maintenance',
      abort: new AbortController(),
      lastTurn: this.phase.lastTurn,
      wakeRequested: false,
    }
    this.setPhase(maintenance)
    this.activityDone = done.promise
    return (async () => {
      try {
        return await job(maintenance.abort.signal)
      } finally {
        this.setPhase({ kind: 'idle', lastTurn: maintenance.lastTurn })
        if (maintenance.wakeRequested && this.inbox.hasPending) this.wakeDriver()
        done.resolve()
      }
    })()
  }

  /**
   * Start one driver, or latch its wake behind maintenance or an aborted
   * activity. A wake sent while idle always opens its turn boundary, even
   * when its message was cleared; only a latched replay is suppressed when
   * the queue no longer holds the wake.
   *
   * 【步骤 1：唤醒驱动器】
   * 负责状态机流转与防并发唤醒锁存（Latch）：
   * - 若当前已在 running 或 maintenance 状态：不重复拉起循环，仅打上 `wakeRequested = true` 锁存标记，待当前轮次结束后自动重放；
   * - 若当前处于 idle 状态：将状态更新为 running，并调用 `kick()` 开启后台异步执行。
   *
   * @param wakeAfterAbort - the {@link send} classification, captured before
   *   the inbox insertion so a reentrant cancel cannot reclassify it.
   */
  private wakeDriver(wakeAfterAbort = false): void {
    if (this.phase.kind !== 'idle') {
      // Maintenance and aborted drivers cannot deliver the wake: latch it for
      // replay at convergence. Live drivers claim queued work themselves;
      // disposal never latches, so teardown waits on no model turn.
      const reason = this.phase.abort.signal.reason as AgentCancelCause | undefined
      if (reason?.kind !== 'disposed' && (this.phase.kind === 'maintenance' || wakeAfterAbort)) {
        this.phase.wakeRequested = true
      }
      return
    }
    const driver = Promise.withResolvers<void>()
    this.activityDone = driver.promise
    this.setPhase({
      kind: 'running',
      abort: new AbortController(),
      turn: this.phase.lastTurn,
      step: 0,
      wakeRequested: false,
    })
    this.loopCtx.agents.withInitiator(this, () => this.kick()).then(driver.resolve, driver.reject)
  }

  async whenIdle(): Promise<void> {
    let activity: Promise<void>
    do {
      await (activity = this.activityDone)
    } while (activity !== this.activityDone)
  }

  /** Report one failure at its live boundary, then preserve it for driver containment. */
  private throwError(error: unknown): never {
    const turn = this.phase.kind === 'running' ? this.phase.turn : this.phase.lastTurn
    const step = this.phase.kind === 'running' ? this.phase.step : 0
    this.dispatch.emit('agent/error', { turn, step, error })
    throw error
  }

  /**
   * 【步骤 2：最外层驱动器循环】
   * 只要 `turn()` 返回 true（表示收件箱中还有未消费的 next-turn 排队消息），
   * 就会一直循环执行新的 Turn。
   * 当所有轮次执行完毕后，在 finally 块中将状态重置回 `idle`，并检查是否有新锁存的唤醒请求。
   */
  private async kick(): Promise<void> {
    try {
      while (await this.turn()) {}
    } catch (_error) {
      // Reported failures and cancellation are contained at the driver boundary.
    } finally {
      /* v8 ignore next -- kick owns a running phase until this driver boundary */
      if (this.phase.kind === 'running') {
        const { turn, wakeRequested } = this.phase
        this.setPhase({ kind: 'idle', lastTurn: turn })
        if (wakeRequested && this.inbox.hasPending) this.wakeDriver()
      }
    }
  }

  /**
   * 【步骤 3：单步前置准备（preStep）】
   * 在进入每一个 Step 前执行：
   * 1. 认领收件箱消息（从 next-turn 或 next-step 中 claim，并从队列中弹出）；
   * 2. 组装 System Prompt 与运行时注入的上下文切片（Context Sections）；
   * 3. 触发 `agent/pre-step` Waterfall 插件钩子（插件可在此拒绝执行或重写注入的消息）。
   */
  private async preStep(target: InboxTarget, position: { turn: number; step: number }): Promise<PreparedStep> {
    /* v8 ignore next -- private callers establish the running phase before proposing a step */
    if (this.phase.kind !== 'running') throw new Error(`agent "${this.id}": pre-step outside running phase`)
    const signal = this.phase.abort.signal
    const claimed = this.inbox.claim(target, position.turn)
    // 1. 动态拼装 System Prompt 与可用工具列表：
    // 调用全局 systemPrompt 服务，遍历所有插件注册的提示词片段、环境变量、Agent 身份与工具描述，
    // 生成包含 system、context、tools 的组装对象 assembly
    const assembly = await this.loopCtx.systemPrompt.assemble(assembleContextFor(this, signal))
    signal.throwIfAborted()

    // 2. 渲染动态上下文切片（Context Sections）：
    // 将插件提供的运行时上下文（如当前打开文件、工作区状态等）渲染为结构化片段
    const sections = renderContextSections(assembly)

    // 3. 计算增量上下文投影（Runtime Context Projection）：
    // 比对当前上下文与历史已发送的内容，若上下文有变化或需要更新，则生成一条注入给模型的上下文消息；
    // 若无变化则返回 undefined，避免在多步骤间无意义地重复灌入冗余提示词
    const context = this.runtimeContext.project(joinContextSections(sections), sections)

    // 4. 执行 `agent/pre-step` Waterfall 插件流水线：
    // 允许中间件/插件在正式迈入 Step 前介入决策：
    // - 插件可以返回 { kind: 'reject' } 拒绝/阻断本次 Step（如超出步数安全守卫、权限拦截）；
    // - 插件可以改写 messages 列表（如注入安全提醒、清洗输入）；
    // - 默认决策：{ kind: 'enter' } 正常准入，将认领的用户消息 claimed 与动态上下文 context 组合传入
    const decision = await this.dispatch.waterfall(
      'agent/pre-step', { messages: claimed, ...position, signal },
      (): Promise<PreStepDecision> => Promise.resolve<PreStepDecision>({
        kind: 'enter',
        messages: context === undefined ? claimed : [...claimed, context],
      }),
    )
    signal.throwIfAborted()
    return decision.kind === 'reject' ? decision : { ...decision, assembly }
  }

  /**
   * 【步骤 4：单个轮次执行（Turn）】
   * 一个 Turn 代表一次宏观的一问一答周期，包含：
   * 1. 写入 `turn/start` 持久化事件；
   * 2. 内部开启 `while (true)` 进行多 Step 推进循环；
   * 3. 每次迭代调用 `preStep` 认领输入；
   * 4. 写入 `step/start` 与用户消息；
   * 5. 调用 `step()` 执行大模型调用与工具执行；
   * 6. 检查 Turn 结束条件（模型回复完毕且无 next-step 插队）；
   * 7. 写入 `turn/end`，若收件箱仍有待处理消息则返回 true 开启新 Turn。
   */
  private async turn(): Promise<boolean> {
    if (this.phase.kind !== 'running') {
      this.throwError(new Error(`agent "${this.id}": turn without driver reservation`))
    }
    const phase = this.phase
    const { signal } = phase.abort
    signal.throwIfAborted()
    const turn = phase.turn + 1
    try {
      this.session.append('turn/start', { turn })
    } catch (error: unknown) {
      this.throwError(error)
    }
    phase.turn = turn
    let turnEnds: TurnEndReason | null = null
    let target: InboxTarget = 'next-turn'
    try {
      while (true) {
        signal.throwIfAborted()
        const step = phase.step + 1
        const decision = await this.preStep(target, { turn, step })
        if (decision.kind === 'reject') {
          turnEnds = { kind: 'blocked' }
          return false
        }
        if (turnEnds && decision.messages.length === 0) break
        // A removed waking message or an enter decision rewritten to empty
        // still owns the initial turn boundary, but it spends no model call.
        if (phase.step === 0 && decision.messages.length === 0) {
          turnEnds = { kind: 'completed' }
          return false
        }
        signal.throwIfAborted()
        this.session.append('step/start', { turn, step })
        phase.step = step
        try {
          for (const message of decision.messages) {
            this.session.append('user/message', message, { surfaceOp: 'append' })
          }
          // 【执行单个微观 Step】
          // 传入组装好的 Prompt 与上下文，发起大模型流式调用与工具执行。
          // 返回值 stepEnd:
          //   - null: 工具已执行，任务未结束，需要继续推进下一个 Step；
          //   - { kind: 'completed' }: 无工具调用，模型给出最终文本回复，本轮完成；
          //   - { kind: 'max-tokens' }: 模型生成达到最大 token 截断上限。
          const stepEnd = await this.step(decision.assembly)

          // 【max-tokens 粘滞性状态机（Sticky Outcome）】
          // 规则：在一个 Turn 的多次 Step 中，只要有任意一步触发了 token 超限（max-tokens），
          // 后续哪怕有步骤正常完成了，也绝对不能将整个 Turn 的结果“降级覆盖”为正常的 completed。
          // 必须保留这一截断事实，以便客户端 UI 准确提示用户或触发后续上下文整理。
          if (turnEnds === null || turnEnds.kind !== 'max-tokens') turnEnds = stepEnd
        } finally {
          // 【强保证：步骤闭合事件】
          // 无论上面的 step() 是成功完成、抛出异常（网络中断等）、还是被外部 cancel() 取消，
          // finally 块百分之百确保向会话追加写入 `step/end` 事件，
          // 保证与上面的 `step/start` 严格成对闭合，维持事件溯源（Event Sourcing）日志的完整性。
          this.session.append('step/end', { turn, step })
        }
        signal.throwIfAborted()
        if (turnEnds && this.inbox.nextStep.length === 0) {
          await this.dispatch.serial('agent/turn-stopping', { turn, signal })
          signal.throwIfAborted()
        }
        if (turnEnds && this.inbox.nextStep.length === 0) break
        target = 'next-step'
      }
    } catch (error: unknown) {
      if (signal.aborted) {
        turnEnds = { kind: 'aborted', reason: signal.reason as AgentCancelCause }
        throw error
      }
      // Every failure is structured: an `LlmError` keeps its facts, anything
      // else flattens to `errorChain` text under the `UNKNOWN` code.
      turnEnds = {
        kind: 'error',
        error: error instanceof LlmError
          ? error.failure
          : { message: errorChain(error), code: 'UNKNOWN' },
      }
      this.throwError(error)
    } finally {
      try {
        // oxlint-disable-next-line typescript/no-non-null-assertion -- every exit assigns a turn ending
        this.session.append('turn/end', { turn, reason: turnEnds! })
      } catch (error: unknown) {
        this.throwError(error)
      }
    }
    if (!this.inbox.hasPending) return false
    phase.abort = new AbortController()
    // A fresh controller makes a latch set on the old one stale: the live driver claims the queue itself.
    phase.wakeRequested = false
    phase.step = 0
    return true
  }

  /**
   * 【步骤 5：单个微观步骤执行（Step）—— ReAct 核心引擎】
   * 负责一次完整的「模型调用 -> 流式接收 -> 工具调度执行」微观循环：
   * 1. 调用 buildRequest 准备冻结的 LLM 请求参数与可用工具列表；
   * 2. 调用 llm.stream 发起流式请求，逐 Chunk 记录 assistant/chunk 事件并组装 Blocks；
   * 3. 错误捕获与请求恢复重试（支持 agent/request-error 钩子介入重试）；
   * 4. 提交完整的 assistant/message 事件写入 Session 日志；
   * 5. 核心分支判断：
   *    - 若模型未调用任何工具（纯文本回复）：返回 completed，本轮交互结束；
   *    - 若模型产生了 Tool Call：执行 executeToolCalls 并发调用工具，将工具结果写回会话，
   *      返回 null（告诉外层 Turn 循环继续开启下一个 Step 让模型根据工具结果继续执行！）。
   */
  private async step(assembly: PromptAssembly): Promise<StepEndReason | null> {
    /* v8 ignore next -- private callers establish the running phase before executing a step */
    if (this.phase.kind !== 'running') throw new Error(`agent "${this.id}": step outside running phase`)
    const { turn, step, abort: { signal } } = this.phase
    signal.throwIfAborted()
    const system = renderPrompt(assembly)

    while (true) {
      // 5.1 构建请求参数（合并配置、触发 agent/request waterfall 插件修改参数、记录 session header）
      const { request, preparedCall } = await this.buildRequest(
        turn, step, assembly.tools, system, this.session.deriveMessages(), signal,
      )
      const assembler = new BlockAssembler()
      const chunkSeqs: number[] = []
      try {
        // 5.2 发起大模型流式调用，实时将 Chunk 写入 Session Log
        const stream = preparedCall?.stream(request) ?? this.loopCtx.llm.stream(request)
        signal.throwIfAborted()
        for await (const chunk of stream) {
          signal.throwIfAborted()
          chunkSeqs.push(this.session.append('assistant/chunk', { turn, step, chunk }).seq)
          assembler.push(chunk)
        }
        signal.throwIfAborted()
      } catch (error: unknown) {
        // 5.3 异常或中断保护：若被外部 cancel，保留已接收到的半截消息内容并标记 interrupted
        if (signal.aborted) {
          const content = assembler.interruptedBlocks()
          if (content.length > 0) {
            this.session.append('assistant/message', {
              turn,
              step,
              message: createAssistantMessage({
                content,
                source: { provider: request.provider, model: request.model },
              }),
              interrupted: true,
              ...assembler.usage === undefined ? {} : { usage: assembler.usage },
            }, { surfaceOp: 'append', sourceEventSeqs: chunkSeqs })
          }
        }
        throw error
      }
      const finish = assembler.finish
      // 5.4 请求级错误恢复：触发 agent/request-error waterfall，允许中间件决定是否 retry
      if (finish.kind === 'error' || finish.kind === 'aborted') {
        const action = await this.dispatch.waterfall(
          'agent/request-error', {
            turn,
            step,
            provider: request.provider,
            failure: finish.failure,
            retryPolicy: preparedCall?.retryPolicy,
            signal,
          },
          () => Promise.resolve<RequestErrorAction>(undefined),
        )
        signal.throwIfAborted()
        if (action?.kind !== 'retry') {
          throw new LlmError(finish.failure.message, finish.failure.code, finish.failure)
        }
        continue
      }

      // 5.5 提交完整的 Assistant Message 并持久化到会话日志
      const message = createAssistantMessage({
        content: assembler.blocks(),
        source: {
          provider: request.provider,
          model: request.model,
          ...assembler.replayState !== undefined ? { replayState: assembler.replayState } : {},
        },
      })
      this.session.append(
        'assistant/message',
        {
          turn,
          step,
          message,
          ...assembler.usage === undefined ? {} : { usage: assembler.usage },
        },
        { surfaceOp: 'append', sourceEventSeqs: chunkSeqs },
      )
      if (finish.kind === 'max-tokens') return { kind: 'max-tokens' }

      // 5.6 检查工具调用（ReAct 的核心分水岭）：
      const toolCalls = message.content.filter(block => block.type === 'tool-call')
      // 分支 A：模型没有调用工具 -> 说明给出了最终文本回答，本轮任务结束
      if (toolCalls.length === 0) return { kind: 'completed' }

      // 分支 B：模型调用了工具 -> 并发调度执行工具组，并将工具执行结果写回会话
      const { concluded } = await executeToolCalls(
        this.loopCtx, turn, step, toolCalls, signal,
        context => this.inbox.splice('next-step', this.inbox.nextStep.length, 0, [context]),
      )
      // concluded 为 true（如某些终止工具声明结束）时返回 completed，否则返回 null 驱动外层 Turn 循环继续下一步！
      return concluded ? { kind: 'completed' } : null
    }
  }

  /**
   * Compose one frozen request and bind it to the adapter registration that
   * resolved its exact-model defaults.
   *
   * 【步骤 6：构建冻结的请求配置（buildRequest）】
   * 负责组装一次 LLM 调用所需的一切元数据与参数，包含 6 个关键步骤：
   * 1. 种子配置推导（Seed Config）：恢复路由选项与历史推理深度（reasoningEffort）；
   * 2. Waterfall 插件干预：触发 `agent/request` 钩子允许插件动态调整配置（如切换模型/调参）；
   * 3. 适配器准备（Prepare Call）：通过 llm.prepareCall 解析特定模型适配器的默认参数与上下文窗口；
   * 4. 会话请求头版本化（Request Header）：比对并持久化写入 `request/header` 事件（initial / resume / change）；
   * 5. 请求上下文同步：更新会话的 `request/context`（记录 provider、model、contextWindow 等）；
   * 6. 请求体深冻结（Deep Freeze）：生成不可变的 GenerateOptions 供后续流式执行。
   */
  private async buildRequest(
    turn: number,
    step: number,
    tools: GenerateOptions['tools'] & object,
    system: string,
    boundaryMessages: Message[],
    signal: AbortSignal,
  ): Promise<{ request: GenerateOptions; preparedCall?: PreparedLlmCall }> {
    const { session } = this

    // 6.1【推导种子配置 seedConfig】
    // - 若已记录过请求头（requestHeaderLogged），则从会话已持久化的头部提案演化；
    // - 若为首次调用，从当前实例 options 提取 provider/model/maxTokens；
    // - reasoningEffort 继承策略：仅当当前模型与会话历史模型完全相同，且此前不是由适配器默认值生成的，才予以继承。
    const persistedHeader = session.requestHeader()
    const persistedConfig = persistedHeader?.config
    const route = { provider: this.options.provider ?? '', model: this.options.model ?? '' }
    const reasoningEffort = persistedConfig?.provider === route.provider
      && persistedConfig.model === route.model
      && persistedHeader?.adapterDefaults?.reasoningEffort !== true
      ? persistedConfig.reasoningEffort
      : undefined
    const maxTokens = this.options.maxTokens
    const seedConfig = deepFreeze(structuredClone(
      this.requestHeaderLogged
        // oxlint-disable-next-line typescript/no-non-null-assertion -- the instance logged the header it now folds
        ? requestProposal(persistedHeader!)
        : {
          ...route,
          ...reasoningEffort === undefined ? {} : { reasoningEffort },
          ...maxTokens === undefined ? {} : { maxTokens },
        },
    ))

    // 6.2【触发 `agent/request` Waterfall 钩子】
    // 允许中间件/插件在底层发请求前，动态修改本次 LLM 请求配置（如根据剩余上下文修改 maxTokens，或切换 Fallback 模型）
    const proposedConfig = await this.dispatch.waterfall(
      'agent/request', { turn, step, signal },
      () => Promise.resolve(seedConfig),
    )
    signal.throwIfAborted()
    if (!proposedConfig.provider || !proposedConfig.model) {
      throw new Error(`agent "${this.id}" has no provider/model: set AgentOptions.provider and AgentOptions.model or supply both via the agent/request waterfall`)
    }

    // 6.3【LLM 适配器准备阶段】
    // 调用 llm.prepareCall 解析该模型适配器特有的默认配置（adapterDefaults，如特定的 reasoningEffort 默认值）
    let config: LlmCallConfig
    let preparedCall: PreparedLlmCall | undefined
    try {
      preparedCall = await this.loopCtx.llm.prepareCall(proposedConfig, signal)
      config = preparedCall.config
    } catch (error: unknown) {
      // 容错处理：若中间件拦截了路由或未注册终端适配器，则沿用 proposedConfig
      if (!(error instanceof LlmError) || error.code !== 'NO_ADAPTER') throw error
      config = proposedConfig
    }
    signal.throwIfAborted()

    // 6.4【构建并持久化规范化请求头（request/header）】
    // 审计与版本化要求：发送给模型的每一个系统提示词、工具列表、配置组合，必须作为不可变事件记录在 Session Log 中
    const header = canonicalHeader({
      config,
      ...preparedCall === undefined ? {} : { adapterDefaults: preparedCall.adapterDefaults },
      ...system ? { system } : {},
      ...tools.length > 0 ? { tools } : {},
    })
    const baseline = this.session.requestHeader()
    if (!this.requestHeaderLogged) {
      // 首次记录：新会话标记为 'initial'，恢复会话标记为 'resume'
      this.session.append('request/header', { header, reason: baseline === undefined ? 'initial' : 'resume' })
      this.requestHeaderLogged = true
    } else if (baseline === undefined || !headerEquals(baseline, header)) {
      // 后续检测到配置/Prompt/Tools 发生实质变更，标记为 'change' 增量记录
      this.session.append('request/header', { header, reason: 'change' })
    }

    // 6.5【同步会话请求上下文元数据（request/context）】
    // 记录 Provider、Model 及模型上下文窗口大小（contextWindow），供后续上下文裁剪与压缩插件参考
    const contextWindow = preparedCall?.context?.contextWindow
    const requestContext: RequestContext = {
      provider: config.provider,
      model: config.model,
      ...contextWindow === undefined ? {} : { contextWindow },
    }
    const previousContext = session.requestContext()
    if (previousContext?.provider !== requestContext.provider
      || previousContext.model !== requestContext.model
      || previousContext.contextWindow !== requestContext.contextWindow) {
      session.append('request/context', requestContext)
    }
    signal.throwIfAborted()

    // 6.6【生成深冻结的最终请求对象】
    // 包装为只读的 GenerateOptions，挂载消息历史、系统提示词、工具、SessionId 与取消信号，
    // 确保后续流程中没有任何环节能非法篡改已准备好的请求对象
    const request = markAgentLoopRequest(deepFreeze({
      ...header.config,
      messages: boundaryMessages,
      ...header.system !== undefined ? { system: header.system } : {},
      ...header.tools !== undefined ? { tools: header.tools } : {},
      sessionId: this.session.id,
      signal,
    }))
    return { request, ...preparedCall === undefined ? {} : { preparedCall } }
  }
}
