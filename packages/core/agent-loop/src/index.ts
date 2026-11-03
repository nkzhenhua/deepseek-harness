/**
 * Concrete agent-loop plugin: creates scoped ReactLoopAgents, publishes them
 * through the agent/session registries, and owns their ordered teardown.
 *
 * 核心 Agent 循环执行器插件入口：
 * 负责在 Cordis 微内核架构中注册 `agentLoop` 服务并实现 `AgentFactory` 接口。
 * 管理所有运行中的 ReactLoopAgent 实例生命周期、提供会话创建与恢复、
 * 融合多源取消信号，并协调并发工具调用等全局设置。
 *
 * @module @deepseek-ai/dsh-agent-loop
 */

import { Context, FiberState, Service } from '@deepseek-ai/cordis'
import { randomUUID } from 'node:crypto'
import z from '@deepseek-ai/schemastery'
import { emitAgentEvent } from '@deepseek-ai/dsh-agent'
import type {
  Agent,
  AgentFactory,
  AgentHandle,
  AgentOptions,
  AgentSetup,
  CreateAgentOptions,
  ResumeAgentOptions,
  SessionStartSource,
} from '@deepseek-ai/dsh-agent'
import { errorChain } from '@deepseek-ai/dsh-llm'
import { installSettingsSection, settingsNamespace } from '@deepseek-ai/dsh-settings'
import { SessionId, SessionPreparation } from '@deepseek-ai/dsh-session'
import type { Session, SessionHeader } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-system-prompt'
import type {} from '@deepseek-ai/dsh-tools'
import type { SessionPersistence } from '@deepseek-ai/dsh-session-persistence'
import { ReactLoopAgent } from './agent.ts'
import { DEFAULT_MAX_PARALLEL_TOOL_CALLS } from './constants.ts'

/** Fiber states that cannot own or serve a new lifecycle. 不可再服务新生命周期的 Cordis Fiber 状态集合 */
const INACTIVE_STATES: ReadonlySet<FiberState> = new Set([
  FiberState.UNLOADING,
  FiberState.DISPOSED,
  FiberState.FAILED,
])

/**
 * Factory-level ownership: live agent teardowns plus config startup work.
 *
 * 工厂级别的生命周期与所有权管理器：
 * 1. 维护工厂自身的就绪/销毁状态（`teardown` AbortController 与 `inactive` Promise）。
 * 2. 跟踪工厂层面启动任务（如启动时恢复配置的持久化会话）。
 * 3. 跟踪所有已创建 Agent 的析构器（`liveAgents`），在工厂注销时并发等待所有 Agent 优雅退出。
 */
class FactoryOwnership {
  /** 标识当前工厂是否仍在接收新的 Agent 创建请求 */
  private accepting = true
  /** 当工厂触发 dispose 开始卸载时发出 abort 信号，熔断一切进行中的创建与恢复 */
  private readonly teardown = new AbortController()
  /** 工厂变为非活动状态时 resolve 的状态信号，用于 Promise.race 快速退出等待 */
  private readonly inactive = Promise.withResolvers<void>()
  /** 正在运行中的 Agent 析构回调集合 */
  private readonly liveAgents = new Set<() => Promise<void>>()
  /** 尚未完成的启动/恢复异步任务集合 */
  private startupTasks = new Set<Promise<void>>()

  constructor(private readonly fiber: Context['fiber']) {}

  /** Aborts (reason: `agent loop is not active` error) when factory teardown begins. */
  get signal(): AbortSignal {
    return this.teardown.signal
  }

  /** 判断当前工厂是否处于活动状态（未进入卸载且 Fiber 状态正常） */
  isActive(): boolean {
    return this.accepting && !INACTIVE_STATES.has(this.fiber.state)
  }

  /**
   * 追踪单个活体 Agent 的逆序析构函数，直到其自身执行完成
   * @param dispose - Agent 的清理析构回调
   * @returns 取消追踪的清理闭包
   */
  track(dispose: () => Promise<void>): () => void {
    this.liveAgents.add(dispose)
    return () => { this.liveAgents.delete(dispose) }
  }

  /**
   * 追踪在 Agent 实例正式就绪前开始的配置启动任务
   * 任务完成后自动从集合中移除，避免内存累积
   */
  trackStartup(job: Promise<void>): void {
    this.startupTasks.add(job)
    const forget = () => { this.startupTasks.delete(job) }
    void job.then(forget, forget)
  }

  /** 将外部公开的 create/resume 任务挂载至启动追踪，工厂卸载时必须等待其彻底结算 */
  trackWrapper(job: Promise<unknown>): void {
    this.trackStartup(job.then(() => undefined, () => undefined))
  }

  /**
   * 等待指定异步任务完成；若工厂提前开始析构则立即停止等待并解除阻塞
   */
  async waitWhileActive(job: Promise<void>): Promise<void> {
    await Promise.race([job, this.inactive.promise])
  }

  /**
   * 优雅销毁工厂自身：
   * 1. 标记不再接收新请求并中止工厂 AbortSignal。
   * 2. 并发执行所有存活 Agent 的 dispose，并等待所有未完成的启动任务结算。
   */
  async dispose(): Promise<void> {
    this.accepting = false
    this.teardown.abort(new Error('agent loop is not active'))
    this.inactive.resolve()
    await Promise.all([
      ...[...this.liveAgents].map(dispose => dispose()),
      ...this.startupTasks,
    ])
  }
}

/**
 * Await `operation`, or throw the signal's reason as soon as it aborts.
 * 将异步操作包装为可被 AbortSignal 抢占熔断的形式；信号一旦中断立即抛出对应原因
 */
async function raceAbort<T>(operation: PromiseLike<T> | T, signal: AbortSignal, id: SessionId): Promise<T> {
  const toAbortError = (): Error => signal.reason instanceof Error
    ? signal.reason
    : new Error(`agent "${id}" creation aborted`, { cause: signal.reason })
  if (signal.aborted) throw toAbortError()
  const aborted = Promise.withResolvers<never>()
  const listener = (): void => { aborted.reject(toAbortError()) }
  signal.addEventListener('abort', listener, { once: true })
  try {
    return await Promise.race([Promise.resolve(operation), aborted.promise])
  } finally {
    signal.removeEventListener('abort', listener)
  }
}

/**
 * Start an abortable operation and release a value that arrives after cancellation.
 * 启动可中止操作，并提供迟到资源释放回调（releaseAbandoned）：
 * 避免当底层异步任务在 Abort 之后才成功完成时造成资源遗留泄漏
 */
async function raceAbortCall<T>(
  operation: () => PromiseLike<T> | T,
  signal: AbortSignal,
  id: SessionId,
  releaseAbandoned?: (value: T) => void,
): Promise<T> {
  if (signal.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new Error(`agent "${id}" creation aborted`, { cause: signal.reason })
  }
  const pending = Promise.resolve().then(operation)
  try {
    return await raceAbort(pending, signal, id)
  } catch (error: unknown) {
    // oxlint-disable-next-line typescript/no-unnecessary-condition -- the signal can abort while the operation is awaited.
    if (signal.aborted && releaseAbandoned !== undefined) {
      void pending.then(releaseAbandoned, () => undefined)
    }
    throw error
  }
}

/**
 * Resolve the deployment-wide scheduler cap at the owning config boundary.
 * 校验并解析每步最大并行工具调用数（必须为正整数）
 */
function resolveMaxParallelToolCalls(value: number | undefined): number {
  const maxParallelToolCalls = value ?? DEFAULT_MAX_PARALLEL_TOOL_CALLS
  if (!Number.isInteger(maxParallelToolCalls) || maxParallelToolCalls < 1) {
    throw new Error('maxParallelToolCalls must be a positive integer')
  }
  return maxParallelToolCalls
}

/**
 * Reject an output-token cap that cannot be represented exactly on the request wire.
 * 校验 AgentOptions 中的配置项合法性（如 maxTokens 必须是安全正整数）
 */
function assertAgentOptions(options: AgentOptions): void {
  if (options.maxTokens !== undefined
    && (!Number.isSafeInteger(options.maxTokens) || options.maxTokens <= 0)) {
    throw new TypeError('agent maxTokens must be a positive safe integer')
  }
}

/**
 * Prepared-but-unpublished agent resources sharing one memoized teardown.
 *
 * 已完成初始化装配但尚未向外部注册表发布的 Agent 包装对象：
 * 提供原子性的 publish 和 memoized 的逆序清理（dispose）能力。
 */
interface PreparedAgent {
  /** 构造完成的执行机内核实例 */
  agent: ReactLoopAgent
  /** 熔合了外部调用方、上下文 Fiber 卸载和工厂卸载的综合取消信号 */
  signal: AbortSignal
  /**
   * 将 Agent 登记到 session 和 agent 注册表，发布 session-start 事件，正式开启运行
   */
  publish(source: SessionStartSource): AgentHandle
  /**
   * 逆序析构：终止执行机、等待当前步收敛进入静默、注销注册表、释放 Cordis 作用域
   */
  dispose(): Promise<void>
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    /** 注入 Cordis 上下文的 agentLoop 服务句柄 */
    agentLoop: AgentLoop
    /**
     * Launcher-owned exact session identities for configured agents, keyed by
     * the agent's config `id` and set with `ctx.provide()` before any Loader
     * entry mounts (see {@link CONFIGURED_AGENT_IDENTITIES_KEY}). A launcher
     * owns identity because only it knows whether the session already exists,
     * while the `cordis.yml` row keeps the model route as ordinary patchable
     * config. An entry with no matching key keeps its configured identity.
     *
     * 启动器层显式分配的会话身份映射表，在插件加载前通过 `ctx.provide` 提供，
     * 以确保 CLI/ACP 可以精确重写配置中的 SessionId（区分新建与续传）。
     */
    configuredAgentIdentities?: ConfiguredAgentIdentities
  }
  interface Events {
    /**
     * A declarative agent entry failed before it could publish a live agent.
     * Consumers that buffer work for the configured identity use this
     * transient signal to reject that work instead of waiting forever. Normal
     * factory teardown suppresses failures from the cancelled startup attempt.
     *
     * 声明式配置的 Agent 启动失败事件：通知外部缓冲等待该 Agent 的消费者快速失败。
     * @param payload.sessionId - exact shared agent/session identity that failed startup.
     * @param payload.error - persistence, setup, or publication failure.
     * @mode emit
     */
    'agent-loop/config-start-failed'(payload: { sessionId: SessionId; error: unknown }): void
  }
}

export { DEFAULT_MAX_PARALLEL_TOOL_CALLS }

/**
 * One launcher-selected session identity for a configured agent. `resume`
 * distinguishes rehydrating existing persisted history from creating the
 * session fresh under that exact id, which the two config keys express as
 * `resumeSessionId` and `sessionId`.
 *
 * 启动器指定的 Agent 会话身份描述，通过 `resume` 标记区分是恢复历史还是全新创建。
 */
export interface LauncherAgentIdentity {
  /** Exact session id to create fresh or resume. */
  id: SessionId
  /** Resume existing persisted history instead of creating the session fresh. */
  resume: boolean
}

/** Launcher-selected identities keyed by the configured agent's `id`. */
export interface ConfiguredAgentIdentities extends Readonly<Record<string, LauncherAgentIdentity>> {}

/**
 * Context key a launcher sets before any Loader entry mounts
 * (`ctx.provide(CONFIGURED_AGENT_IDENTITIES_KEY, identities)`) to fix
 * configured agents' session identities without a config key, so an overlay
 * repointing the row's model route cannot drop them.
 */
export const CONFIGURED_AGENT_IDENTITIES_KEY = 'configuredAgentIdentities'

/**
 * Apply launcher-owned identities over the configured agents, replacing both
 * identity keys for every entry the launcher named so a config-supplied
 * identity can never survive alongside a launcher-supplied one.
 *
 * 将启动器提供的动态会话身份覆盖应用到静态声明配置上。
 * @param agents - the configured agent entries.
 * @param identities - launcher identities keyed by configured agent `id`, or `undefined`.
 * @returns the entries with launcher-owned identities applied.
 */
function applyLauncherIdentities(
  agents: Config['agents'],
  identities: ConfiguredAgentIdentities | undefined,
): Config['agents'] {
  if (identities === undefined) return agents
  return agents.map((agent) => {
    const identity = identities[agent.id]
    if (identity === undefined) return agent
    const { sessionId: _sessionId, resumeSessionId: _resumeSessionId, ...rest } = agent
    return identity.resume
      ? { ...rest, resumeSessionId: identity.id }
      : { ...rest, sessionId: identity.id }
  })
}

/** Settings namespace carrying the tool-call parallelism a user owns. 用户可配置的 settings 命名空间 */
export const AGENT_LOOP_SETTINGS_NAMESPACE = settingsNamespace('agent-loop')

/**
 * The agent-loop fields a user owns. Deliberately a strict subset of
 * {@link Config}: `agents` is a boot-time composition array consumed once when
 * the service starts, so a stored change could only look like it had an effect.
 * 用户在运行时可动态修改的设置项（仅允许修改最大工具调用并发度，不包含仅启动生效的 agents 列表）。
 */
export interface AgentLoopSettings {
  /** Maximum parallel-safe calls in flight per agent step. */
  maxParallelToolCalls: number
}

/** Schema of the agent-loop settings section. */
export const AGENT_LOOP_SETTINGS_SCHEMA: z<AgentLoopSettings> = z.object({
  maxParallelToolCalls: z.number().step(1).min(1).default(DEFAULT_MAX_PARALLEL_TOOL_CALLS),
})

/** Agent-loop plugin configuration. 插件配置结构定义 */
export interface Config {
  /**
   * Maximum parallel-safe calls in flight per agent step. `1` is serial;
   * omission defaults to {@link DEFAULT_MAX_PARALLEL_TOOL_CALLS}.
   * 每步调度中允许并发执行的最大工具调用数
   */
  maxParallelToolCalls?: number
  /** Agents created or resumed at plugin startup. 启动时由插件自动创建或恢复的 Agent 列表 */
  agents: (AgentOptions & {
    /** Stable config label used in logs and as the fresh combined-id prefix. 配置中该 Agent 的稳定标识名 */
    id: string
    /** Optional stable identity; remounts resume its materialized history, while first use creates it fresh. 可选的会话 ID */
    sessionId?: SessionId
    /** Optional workspace for a fresh session. 全新会话的工作目录路径 */
    cwd?: string
    /** Persisted session to resume instead of creating a fresh session. 需恢复的历史会话 ID */
    resumeSessionId?: SessionId
  })[]
}

/** Agent-loop configuration after defaults and load-time validation. */
type ResolvedConfig = Config & { maxParallelToolCalls: number }

/**
 * Reject self-contained identity conflicts before any configured agent starts.
 * 启动前执行严格校验：互斥检查（sessionId 与 resumeSessionId 不得同存）以及 SessionId 跨配置项唯一性。
 */
function validateConfiguredAgents(agents: Config['agents']): void {
  const exactIdentities = new Map<SessionId, string>()
  for (const { id, sessionId, resumeSessionId } of agents) {
    const hasResumeId = resumeSessionId !== undefined && resumeSessionId !== ''
    if (sessionId !== undefined && hasResumeId) {
      throw new Error(`agent "${id}": sessionId and resumeSessionId are mutually exclusive`)
    }
    const exactIdentity = hasResumeId ? resumeSessionId : sessionId
    if (exactIdentity === undefined) continue
    const firstId = exactIdentities.get(exactIdentity)
    if (firstId !== undefined) {
      throw new Error(`agents "${firstId}" and "${id}" use duplicate exact session identity "${exactIdentity}"`)
    }
    exactIdentities.set(exactIdentity, id)
  }
}

/**
 * Concrete agent factory and driver service.
 *
 * 核心驱动服务与 Agent 工厂实现类：
 * 作为 Cordis 服务挂载到根上下文，负责协调 Agent 的创建、恢复、参数热更新与生命周期销毁。
 */
export class AgentLoop extends Service implements AgentFactory {
  /** 声明当前服务依赖的 Cordis 服务列表 */
  static inject = ['agents', 'sessions', 'llm', 'tools', 'systemPrompt']

  /** Runtime schema for declarative agents. 运行时配置校验 Schema */
  static Config = z.object({
    maxParallelToolCalls: z.number().step(1).min(1).default(DEFAULT_MAX_PARALLEL_TOOL_CALLS),
    agents: z.array(z.object({
      id: z.string().required(),
      sessionId: z.string().min(1),
      provider: z.string(),
      model: z.string(),
      maxTokens: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER),
      cwd: z.string(),
      resumeSessionId: z.string(),
    })).default([]),
  }) as z<Config>

  /** Validated configuration owned by the agent-loop service. */
  readonly config: ResolvedConfig
  /** 工厂级生命周期管理器 */
  private readonly ownership: FactoryOwnership
  /** Plain holder prevents Cordis from re-tracing the factory's dependency context through a caller shadow. */
  private readonly runtime: { ctx: Context }

  constructor(ctx: Context, config: Config) {
    super(ctx, 'agentLoop')
    const entry: AgentLoopSettings = {
      maxParallelToolCalls: resolveMaxParallelToolCalls(config.maxParallelToolCalls),
    }
    let source: () => AgentLoopSettings = () => entry
    this.config = {
      ...config,
      agents: applyLauncherIdentities(config.agents, ctx.get(CONFIGURED_AGENT_IDENTITIES_KEY)),
      // Read through on every scheduler decision: `tool-calls.ts` destructures
      // this at the start of each group, so a committed change caps the next
      // group without disturbing the one in flight.
      // 每次工具调度决策时通过 getter 实时读取最新动态并发上限
      get maxParallelToolCalls() {
        return source().maxParallelToolCalls
      },
    }
    // 挂载动态设置区段，允许在运行时通过 Settings API 调整并发上限
    installSettingsSection(ctx, AGENT_LOOP_SETTINGS_NAMESPACE, AGENT_LOOP_SETTINGS_SCHEMA, entry, {
      validate: value => void resolveMaxParallelToolCalls(value.maxParallelToolCalls),
      setSource: (current) => {
        source = current
      },
      onChange: () => {},
    })
    // 启动前校验所有静态配置的 Agent
    validateConfiguredAgents(this.config.agents)
    this.ownership = new FactoryOwnership(ctx.fiber)
    this.runtime = { ctx }
    // 注册全局副作用：当本插件卸载时触发工厂所有权销毁，清理所有 Agent
    ctx.effect(() => () => this.ownership.dispose(), 'agentLoop.transactions()')
    // 将自身注册为全局 AgentFactory，供外部创建调用
    ctx.effect(() => ctx.agents.setFactory(this), 'agentLoop.setFactory()')
    // 向系统提示词服务注入环境变量插值
    ctx.systemPrompt.variable('provider', context => context.agent?.options.provider)
    ctx.systemPrompt.variable('model', context => context.agent?.options.model)
    ctx.systemPrompt.variable('cwd', context => context.agent?.session.header.cwd)

    // 逐一初始化配置声明中的所有 Agent
    for (const { id, sessionId, cwd, resumeSessionId, ...options } of this.config.agents) {
      const meta = cwd === undefined ? {} : { cwd }
      if (resumeSessionId === undefined || resumeSessionId === '') {
        // 未指定 resumeSessionId：全新创建会话
        const configuredId = sessionId ?? SessionId(`${id}-session-${randomUUID()}`)
        const persistence = sessionId === undefined ? undefined : ctx.get('sessionPersistence')
        if (persistence === undefined) {
          // 无持久化服务，直接就地创建内存会话
          this.create(configuredId, options, meta)
        } else {
          // 有持久化服务，尝试恢复（若插件重载残留）或新建并挂载启动追踪
          const startup = this.restoreOrCreateConfigured(ctx, persistence, configuredId, options, meta).catch((error: unknown) => {
            this.reportConfiguredStartupFailure(id, 'restore', configuredId, error)
          })
          this.ownership.trackStartup(startup)
        }
        continue
      }
      // 指定了 resumeSessionId：必须等待持久化服务就绪后恢复
      ctx.effect(() => {
        const fiber = ctx.inject(['sessionPersistence'], (childCtx: Context) => {
          void this.resumeWith(ctx, childCtx.sessionPersistence, {
            resumeSessionId,
            agentOptions: options,
          }).catch((error: unknown) => {
            this.reportConfiguredStartupFailure(id, 'resume', resumeSessionId, error)
          })
        })
        return fiber.dispose
      }, `agentLoop.resume(${id})`)
    }
  }

  /**
   * Report a contained declarative-start failure to identity-bound consumers.
   * 向等待声明式 Agent 启动的外部消费者广播启动失败事件，避免消费者陷入悬挂等待
   */
  private reportConfiguredStartupFailure(
    configId: string,
    action: 'restore' | 'resume',
    sessionId: SessionId,
    error: unknown,
  ): void {
    if (!this.ownership.isActive()) return
    this.ctx.logger.warn(`agent "${configId}": config-driven ${action} of "${sessionId}" failed: ${errorChain(error)}`)
    const args: unknown[] = ['agent-loop/config-start-failed', { sessionId, error }]
    for (const callback of this.ctx.events.dispatch('emit', args)) {
      try {
        const returned: unknown = callback(...args)
        void Promise.resolve(returned).catch((listenerError: unknown) => {
          this.ctx.logger.warn(`agent "${configId}": config-start-failed listener rejected: ${errorChain(listenerError)}`)
        })
      } catch (listenerError: unknown) {
        this.ctx.logger.warn(`agent "${configId}": config-start-failed listener threw: ${errorChain(listenerError)}`)
      }
    }
  }

  /**
   * Restore a materialized exact config identity on remount, or create it on first use.
   * 插件重载时优先恢复已有物理会话，若确实不存在则回退至全新创建
   */
  private async restoreOrCreateConfigured(
    ownerCtx: Context,
    persistence: SessionPersistence,
    sessionId: SessionId,
    agentOptions: AgentOptions,
    meta: Pick<SessionHeader, 'cwd'>,
  ): Promise<void> {
    // 等待旧的同名会话注销排空，避免注册表冲突
    await this.waitForDrainingConfiguredIdentity(ownerCtx, sessionId)
    if (!this.ownership.isActive()) return
    try {
      await this.resumeWith(ownerCtx, persistence, { resumeSessionId: sessionId, agentOptions })
      return
    } catch (error: unknown) {
      if (!this.ownership.isActive()) return
      // A load is the per-id serialization barrier for eager write-behind and
      // lifecycle retirement. Only a genuinely absent artifact falls back to
      // first creation; corruption and backend failures stay loud.
      // 只有在持久化列表中确认完全不存在该会话时才回退至全新创建，存储损坏或后端错误保持报错
      const exists = (await persistence.list()).some(header => header.id === sessionId)
      if (exists) throw error
    }
    this.create(sessionId, agentOptions, meta)
  }

  /**
   * Wait for a draining same-id lifecycle to finish registry teardown.
   * 当会话热重载时，异步等待正在排空注销的旧同名实例彻底离开注册表
   */
  private async waitForDrainingConfiguredIdentity(ownerCtx: Context, sessionId: SessionId): Promise<void> {
    // 若当前注册表中已无该身份占用，则直接返回
    if (ownerCtx.agents.get(sessionId) === undefined && ownerCtx.sessions.get(sessionId) === undefined) return

    const released = Promise.withResolvers<void>()
    const checkReleased = (): void => {
      if (ownerCtx.agents.get(sessionId) === undefined && ownerCtx.sessions.get(sessionId) === undefined) {
        released.resolve()
      }
    }
    // 监听 agent 和 session 的注销事件
    const disposeAgentListener = ownerCtx.on('agent/disposed', () => { checkReleased() })
    const disposeSessionListener = ownerCtx.on('session/disposed', checkReleased)
    try {
      checkReleased()
      await this.ownership.waitWhileActive(released.promise)
    } finally {
      disposeAgentListener()
      disposeSessionListener()
    }
  }

  /**
   * Construct the driver, scope, and one memoized reverse teardown for a new
   * agent. The teardown is registered with the factory and the owner fiber
   * BEFORE publication, so a mid-setup unload rolls everything back; `signal`
   * fuses caller cancellation with lifecycle teardown for setup awaits.
   *
   * 核心装配管线：
   * 构造 ReactLoopAgent 执行机内核，熔合三方取消信号（调用方取消、所有者 Fiber 卸载、工厂卸载），
   * 并在真正对外发布之前提前挂载具备幂等保证的逆序析构器（Reverse Teardown）。
   */
  private prepare(ownerCtx: Context, id: SessionId, options: AgentOptions, session: Session, callerSignal?: AbortSignal): PreparedAgent {
    assertAgentOptions(options)
    ownerCtx.fiber.assertActive()
    if (!this.ownership.isActive()) throw new Error('agent loop is not active')
    if (callerSignal?.aborted) {
      throw callerSignal.reason instanceof Error
        ? callerSignal.reason
        : new Error(`agent "${id}" creation aborted`, { cause: callerSignal.reason })
    }
    const loopCtx = this.runtime.ctx

    // 熔合三方取消信号源：外部 callerSignal、所有者 Fiber 销毁、工厂卸载
    const abort = new AbortController()
    const onCallerAbort = (): void => {
      abort.abort(callerSignal?.reason instanceof Error
        ? callerSignal.reason
        : new Error(`agent "${id}" creation aborted`, { cause: callerSignal?.reason }))
    }
    const onFactoryTeardown = (): void => { abort.abort(this.ownership.signal.reason) }
    callerSignal?.addEventListener('abort', onCallerAbort, { once: true })
    this.ownership.signal.addEventListener('abort', onFactoryTeardown, { once: true })

    let machine: ReactLoopAgent | undefined
    let detachSession: (() => void) | undefined
    let detachAgent: (() => void) | undefined
    let disposing: Promise<void> | undefined
    const machineReady = Promise.withResolvers<void>()

    // 逆序析构函数，使用 (disposing ??= ...) 确保幂等：
    // 1. 发出 disposed 信号中断 machine 执行。
    // 2. 等待 machine 进入静默状态并释放 private scope。
    // 3. 从 sessions 与 agents 注册表注销。
    // 4. 从工厂所有权追踪中解绑。
    const dispose = (ownerTriggered = false): Promise<void> => (disposing ??= (async () => {
      abort.abort(new Error(`agent "${id}" lifecycle disposed`))
      callerSignal?.removeEventListener('abort', onCallerAbort)
      this.ownership.signal.removeEventListener('abort', onFactoryTeardown)
      try {
        if (machine === undefined) await machineReady.promise
        if (machine !== undefined) {
          machine.cancel({ kind: 'disposed' })
          await machine.whenIdle()
          await machine.scope.dispose()
        }
      } finally {
        try {
          detachAgent?.()
          detachSession?.()
        } finally {
          untrack()
          if (!ownerTriggered) await unfollowOwner()
        }
      }
    })())
    const untrack = this.ownership.track(dispose)
    let unfollowOwner: () => Promise<void> | void
    try {
      // 绑定到所有者 Context 的生命周期，若所有者 Fiber 卸载则自动触发析构
      unfollowOwner = ownerCtx.effect(() => () => {
        if (disposing !== undefined) return
        abort.abort(new Error(`agent "${id}" setup aborted: owner disposed during setup`))
        return dispose(true)
      }, `agentLoop.lifecycle(${id})`)
    } catch (error: unknown) {
      untrack()
      callerSignal?.removeEventListener('abort', onCallerAbort)
      this.ownership.signal.removeEventListener('abort', onFactoryTeardown)
      throw error
    }

    const assertLive = (): void => {
      if (!abort.signal.aborted) return
      throw abort.signal.reason instanceof Error ? abort.signal.reason : new Error(String(abort.signal.reason))
    }
    try {
      const agent = machine = new ReactLoopAgent(loopCtx, id, options, session)
      machineReady.resolve()
      assertLive()

      return {
        agent,
        signal: abort.signal,
        publish: (source) => {
          assertLive()
          // 登记至注册表
          detachSession = agent.ctx.sessions.enter(session)
          detachAgent = loopCtx.agents.enter(agent, ownerCtx.agent)
          // 广播上线事件
          agent.ctx.sessions.announce(session)
          assertLive()
          loopCtx.agents.announce(agent)
          assertLive()
          // 派发 agent/session-start 扩展点事件，正式启动
          emitAgentEvent(loopCtx, agent, 'agent/session-start', { source })
          assertLive()
          return { agent, dispose }
        },
        dispose,
      }
    } catch (error: unknown) {
      machineReady.resolve()
      void dispose()
      throw error
    }
  }

  /**
   * Create an agent and session under one caller-supplied identity, owned by
   * the accessing fiber. Constructor-driven config calls mint a fresh combined
   * id before entering this boundary.
   *
   * 快捷/同步创建入口：
   * 准备内存 Session，装配 Agent 并直接以 'startup' 源发布上线。
   * @param id - shared agent/session identity.
   * @param options - concrete loop options.
   * @param meta - optional fresh-session workspace metadata.
   * @returns the published running agent.
   */
  create(id: SessionId, options: AgentOptions = {}, meta: Pick<SessionHeader, 'cwd'> = {}): Agent {
    using preparation = SessionPreparation.create(this.runtime.ctx.sessions.prepare(id, { meta }))
    const prepared = this.prepare(this.ctx, id, options, preparation.session)
    try {
      return prepared.publish('startup').agent
    } catch (error: unknown) {
      void prepared.dispose()
      throw error
    }
  }

  /**
   * Create an owned agent on a caller-supplied session id.
   *
   * 异步规范创建入口（实现 AgentFactory 接口）：
   * 支持通过 setup 回调在发布前注入插件或执行配置，并在外部 signal 取消时中断。
   * @param ownerCtx - caller context that structurally owns the lifecycle.
   * @param options - identities, session seed/metadata, loop options, setup, and cancellation.
   * @returns the published handle.
   */
  async createAgent(ownerCtx: Context, options: CreateAgentOptions): Promise<AgentHandle> {
    const preparation = SessionPreparation.create(this.runtime.ctx.sessions.prepare(options.sessionId, {
      ...options.seed === undefined ? {} : { seed: options.seed },
      ...options.meta === undefined ? {} : { meta: options.meta },
    }))
    const published = this.setupAndPublish(
      ownerCtx,
      options.sessionId,
      preparation,
      options.agentOptions ?? {},
      options.setup,
      options.signal,
      'startup',
    )
    this.ownership.trackWrapper(published)
    return published
  }

  /**
   * Prepare one Agent around an acquired Session, run setup, and publish it.
   * 执行完整的 装配 -> 运行 setup 回调 -> 提交设置 -> 发布上线 流水线
   */
  private async setupAndPublish(
    ownerCtx: Context,
    id: SessionId,
    preparation: SessionPreparation,
    agentOptions: AgentOptions,
    setup: AgentSetup | undefined,
    signal: AbortSignal | undefined,
    source: SessionStartSource,
  ): Promise<AgentHandle> {
    using ownedPreparation = preparation
    const session = ownedPreparation.session
    const prepared = this.prepare(ownerCtx, id, agentOptions, session, signal)
    try {
      const setupCommit = await raceAbort(setup?.(prepared.agent.ctx), prepared.signal, id)
      setupCommit?.commit()
      return prepared.publish(source)
    } catch (error: unknown) {
      await prepared.dispose()
      throw error
    }
  }

  /**
   * Resume an owned agent from the configured persistence service.
   *
   * 从持久化存储中恢复已有会话的 Agent（规范入口）：
   * @param ownerCtx - caller context that owns load, setup, and the live lifecycle.
   * @param options - persisted identity, loop options, setup, and cancellation.
   * @returns the published handle.
   */
  async resume(ownerCtx: Context, options: ResumeAgentOptions): Promise<AgentHandle> {
    const persistence = this.runtime.ctx.get('sessionPersistence')
    if (persistence === undefined) {
      throw new Error('cannot resume: session persistence is not configured (load a dsh-session-persistence backend)')
    }
    return this.resumeWith(ownerCtx, persistence, options)
  }

  /**
   * Resume through an explicit persistence handle used by the deferred config path.
   * 通过指定的 SessionPersistence 句柄异步加载并重水合会话
   */
  private resumeWith(
    ownerCtx: Context,
    persistence: SessionPersistence,
    options: ResumeAgentOptions,
  ): Promise<AgentHandle> {
    const id = options.resumeSessionId
    const published = (async () => {
      // 加载可能耗时较长：熔合调用方取消、所有者卸载和工厂卸载三方信号
      const ownerAbort = new AbortController()
      const unfollowOwner = ownerCtx.effect(() => () => {
        ownerAbort.abort(new Error(`agent "${id}" setup aborted: owner disposed during setup`))
      }, `agentLoop.resume-load(${id})`)
      const fused = AbortSignal.any([
        ...options.signal === undefined ? [] : [options.signal],
        ownerAbort.signal,
        this.ownership.signal,
      ])
      let preparation: SessionPreparation | undefined
      try {
        try {
          preparation = await raceAbortCall(
            () => persistence.prepare(id, fused),
            fused,
            id,
            (abandoned) => { abandoned[Symbol.dispose]() },
          )
        } finally {
          await unfollowOwner()
        }
        ownerCtx.fiber.assertActive()
        if (!this.ownership.isActive()) throw new Error('agent loop is not active')
        return await this.setupAndPublish(
          ownerCtx,
          id,
          preparation,
          options.agentOptions ?? {},
          options.setup,
          options.signal,
          'resume',
        )
      } finally {
        preparation?.[Symbol.dispose]()
      }
    })()
    this.ownership.trackWrapper(published)
    return published
  }
}

export default AgentLoop

