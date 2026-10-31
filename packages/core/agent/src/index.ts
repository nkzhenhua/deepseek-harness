/**
 * Agent service: live registry, factory delegation, and process-local
 * initiator scope. Concrete creation and driving belong to the loop.
 *
 * @module @deepseek-ai/dsh-agent
 */

import { Context, FiberState, getTraceable, Service, symbols } from '@deepseek-ai/cordis'
import type { Fiber } from '@deepseek-ai/cordis'
import { AsyncLocalStorage } from 'node:async_hooks'
import { isPromise } from 'node:util/types'
import { scopeTarget } from '@deepseek-ai/dsh-scope'
import type { Scoped } from '@deepseek-ai/dsh-scope'
import type { SessionEvent, SessionId } from '@deepseek-ai/dsh-session'
import type { TypertContext, TypertLookup } from '@deepseek-ai/dsh-typert-protocol'
import type { Agent, AgentOptions } from './runtime-types.ts'

export * from './runtime-types.ts'
export * from './types.ts'
export * from './inbox.ts'
export * from './consumed-work.ts'
export * from './model-selection.ts'
export { agentCarrier, agentEvents, assembleContextFor, emitAgentEvent } from './dispatch.ts'
export type { AgentEventDispatch, AgentSubjectEvent } from './dispatch.ts'

declare module '@deepseek-ai/dsh-typert-protocol' {
  interface TypertLookupMap {
    agent: TypertLookup<Agent, SessionId>
  }

  interface TypertContextMap {
    agent: TypertContext<SessionId>
  }
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    agents: AgentRegistry
    /**
     * The agent association installed as an own property on `Agent.ctx`, or
     * `undefined` on a plain context. Contexts derived from `Agent.ctx` inherit
     * the association; a deliberately nested scope may carry a nearer
     * `dsh-scope` tag while retaining it, so this field is DX context rather
     * than the scope resolver. {@link AgentRegistry} registers a root accessor
     * defaulting to `undefined`, and core packages below the agent layer use
     * `scopeOf()` for layer selection instead of reading this field.
     */
    agent?: Agent
  }
}

/**
 * Synchronous finalizer returned by unpublished Agent setup when its
 * contributions need validation at the exact publication commit point.
 */
export interface AgentSetupCommit {
  /**
   * Validate and commit the prepared setup immediately before publication.
   * @throws when publication must roll the unpublished Agent back.
   */
  commit(): void
}

/**
 * Compose an unpublished Agent scope and optionally return its publication commit.
 * @param agentCtx - unpublished Agent scope.
 * @returns an optional synchronous commit invoked after setup awaits settle and immediately before publication.
 */
export type AgentSetup = (
  agentCtx: Context,
) => AgentSetupCommit | Promise<AgentSetupCommit | void> | void

/**
 * Options for programmatically creating an agent through the registry factory
 * ({@link AgentRegistry.create}). The caller supplies the single live
 * `sessionId` shared by the agent registry and session log (e.g. an
 * ACP-generated id), plus optional session metadata (the validated `cwd`, fork
 * lineage); the factory creates the session and agent under that identity.
 */
export interface CreateAgentOptions {
  /** The live agent/session identity. */
  readonly sessionId: SessionId
  /**
   * Session creation metadata: validated absolute `cwd`, `parentSession`
   * fork lineage, the `seedLength` seed boundary, the coarse `origin`
   * classification, and the `delegationDepth` recursion budget. Mirrors the
   * `cwd`/`parentSession`/`seedLength`/`origin`/`delegationDepth` fields of
   * {@link CreateSessionOptions.meta} in dsh-session (the internal-only
   * `createdAt`, used when reconstructing a persisted session, is deliberately
   * excluded — a factory caller never sets it). This is durable session data,
   * so the session boundary validates and snapshots it before asynchronous
   * setup begins.
   */
  readonly meta?: {
    readonly cwd?: string
    readonly parentSession?: SessionId
    readonly seedLength?: number
    readonly origin?: 'subagent'
    readonly delegationDepth?: number
    readonly agentPreset?: string
  }
  /**
   * Initial replay/fork history. A fork supplies a balanced completed-turn
   * prefix of the parent's log. The complete seed must be contiguous from seq
   * 0, carry only lossless-JSON data, and contain no open turn/step or dangling
   * tool call. The factory passes it to the session's durable
   * validator/snapshot boundary before publication.
   */
  readonly seed?: readonly SessionEvent[]
  /** Per-agent options (model, …). */
  readonly agentOptions?: AgentOptions
  /** Optional creation-only cancellation signal; detached before the returned handle becomes visible. */
  readonly signal?: AbortSignal
  /**
   * Creation-time composition of the agent's scoped world. The factory awaits
   * setup after minting `agentCtx` but BEFORE inserting or announcing either
   * the session or agent, so observers can never see a partially configured
   * world. Setup may return an {@link AgentSetupCommit}; the factory invokes its
   * synchronous `commit()` after every setup await settles and immediately
   * before registry publication. This lets mutable provisioning revalidate at
   * the exact publication boundary. Everything registered through `agentCtx`
   * (scoped tools, prompt sections/variables, `restrict()`, listeners, awaited
   * child plugins) exists before `session/created`, `agent/created`,
   * `agent/session-start`, and the first prompt assembly. A setup
   * throw/rejection, commit throw, or owner disposal rolls the scope back
   * without publishing either id.
   *
   * **Setup composes, it never drives**: the callback is trusted same-process
   * code and receives the full scoped context, so this is a contract rather
   * than a runtime restriction. Drive the agent only after creation resolves.
   */
  readonly setup?: AgentSetup
}

/**
 * Options for resuming an agent on a persisted session
 * ({@link AgentRegistry.resume}).
 */
export interface ResumeAgentOptions {
  /** The persisted session id to load and use as the live agent/session identity. */
  readonly resumeSessionId: SessionId
  /** Per-agent options (model, …). */
  readonly agentOptions?: AgentOptions
  /** Optional creation-only cancellation signal for persistence load/setup; detached before return. */
  readonly signal?: AbortSignal
  /**
   * Resume-time composition of the agent's fresh scoped world. Persistence is
   * loaded first; the factory then mints `agentCtx` and awaits setup while the
   * reconstructed session and agent remain unpublished. The callback has the
   * same trusted composition-only contract and optional synchronous
   * publication commit as {@link CreateAgentOptions.setup}: all registrations
   * exist before either creation announcement, and rejection, commit failure,
   * or owner disposal rolls the transaction back without publishing either id.
   */
  readonly setup?: AgentSetup
}

/**
 * An owned agent plus its disposer, returned by {@link AgentRegistry.create} /
 * {@link AgentRegistry.resume}. The disposer is a CAPABILITY: among consumers,
 * only the holder can tear this agent down. The registered factory provider is
 * also a structural owner because the scoped agent depends on that provider's
 * service API; provider unload stops and drains every live handle it made.
 * `dispose()` stops the loop, awaits its exit, unregisters the agent, removes
 * its session from the store, and finally unwinds its scoped world.
 *
 * `ctx.agents.get(id)` still returns a bare {@link Agent} — the handle is
 * exposed only to the consumer owner that created it; the structural provider
 * reaches the same teardown internally. Config-created agents (the loop's own
 * startup) are owned by the loop fiber and never need a handle.
 */
export interface AgentHandle {
  agent: Agent
  dispose(): Promise<void>
}

/**
 * The agent-creation factory the loop implementation provides to the registry
 * via {@link AgentRegistry.setFactory}. Kept on the `dsh-agent` interface so
 * consumers (e.g. the ACP bridge) program against `ctx.agents` without
 * depending on the concrete `dsh-agent-loop` package.
 */
export interface AgentFactory {
  /**
   * Create a new agent on a caller-supplied session id. Async because creation
   * awaits unpublished setup, invokes its optional synchronous commit, inserts
   * both session and agent, emits their creation notifications in order, emits
   * `agent/session-start`, and only then starts the loop. The sequence is
   * rollback-covered, but notifications delivered before a later listener
   * failure remain observable; every agent or session creation announcement
   * that began is paired by `agent/disposed` or `session/disposed` during
   * rollback. The owner disposes the resolved handle to stop/drain,
   * unregister, remove the session, and unwind the scope.
   * The registry passes a context carrying the `create()` caller's fiber and
   * scope as `ownerCtx`. The implementation attaches the unpublished
   * transaction and resulting lifecycle to that owner; it must not infer
   * ownership from the factory object's registration context.
   * @param ownerCtx - caller-bound context that owns the transaction and live handle.
   * @param options - agent/session identity, configuration, and optional setup.
   * @returns the owned handle after setup, both announcements, and loop start complete.
   */
  createAgent(ownerCtx: Context, options: CreateAgentOptions): Promise<AgentHandle>
  /**
   * Prepare a persisted session and resume an agent on it. Async because it awaits
   * both `ctx.sessionPersistence.prepare` and the optional unpublished setup
   * transaction; must be called after that service exists (consumers inject
   * `sessionPersistence`). Publication follows the same setup-commit and
   * ordered boundary as {@link createAgent}.
   * @param ownerCtx - caller-bound context that owns load, setup, and the live handle.
   * @param options - persisted identity, configuration, and optional setup.
   * @returns the owned handle after setup, both announcements, and loop start complete.
   */
  resume(ownerCtx: Context, options: ResumeAgentOptions): Promise<AgentHandle>
}

/** Thrown when create/resume is called before an agent factory is registered. */
const NO_FACTORY_MESSAGE = 'no agent factory registered (load an agent-loop plugin)'
const NO_INITIATOR_MESSAGE = 'no initiating agent is active'
const DISPOSED_INITIATOR_MESSAGE = 'agent initiator scope is disposed'

/** All mutable lifecycle state for one exact registry entry. */
interface AgentEntry {
  readonly id: SessionId
  readonly agent: Agent
  /** Runtime creator-agent ownership; independent of durable session lineage. */
  readonly owner: Agent | undefined
  readonly carrier: Scoped<Agent>
  announced: boolean
  announcing: boolean
  detachRequested: boolean
}

/** One tracked boundary plus its inherited nesting chain. */
interface InitiatorRun {
  active: boolean
  readonly parent: InitiatorRun | undefined
}

/** Plain holder prevents Cordis from tracing the factory field before the caller context is known. */
interface FactorySlot {
  readonly target: AgentFactory
}

/**
 * Agent service (`ctx.agents`): tracks live agents and carries the initiating
 * Agent through one process-local asynchronous driver chain. Agent *creation*
 * is provided by whichever plugin implements the {@link AgentFactory}
 * (`@deepseek-ai/dsh-agent-loop`), registered via {@link setFactory}.
 *
 * Initiator methods provide same-process causal attribution only. Ambient
 * presence is neither liveness proof nor authorization; subjects and owners
 * remain explicit, as does identity at worker, process, persistence, and wire
 * boundaries. Returned Promise boundaries drain during teardown, except a
 * nested lineage that starts an owning-fiber unload is excluded from its own drain.
 *
 * ============================================================================
 * 【学习笔记：AgentRegistry 的核心职责与架构机制】
 *
 * 1. 这个类主要做什么？（定位与核心职责）
 *    - 它是 Cordis 框架下的全局核心服务之一，通过 `super(ctx, 'agents')` 挂载在根上下文，
 *      在整个进程中可以通过 `ctx.agents` 直接访问。
 *    - 职责一【注册表与存活实例目录】：
 *      维护进程内所有“活着的” Agent 实例（`store` 字典），追踪父子隶属树（`owner`，
 *      明确谁是主 Agent、谁是子 Agent/Subagent），并提供生命周期事件广播（`agent/created`、`agent/disposed`）。
 *    - 职责二【解耦 Agent 构建循环】：
 *      `AgentRegistry` 自身不实现大模型多轮对话、工具调用的轮次循环（Loop），具体实例化
 *      通过插槽委托给外部工厂（`factory: AgentFactory`，由 `@deepseek-ai/dsh-agent-loop` 注入）。
 *      这种设计解耦了“注册/查询/类型投射”与“运行循环驱动”，打破了循环依赖。
 *    - 职责三【因果归因与隐式上下文传递（initiators）】：
 *      基于 Node.js 原生 `AsyncLocalStorage`，在同一个进程内，跨越深层异步调用链（Promise / async-await），
 *      隐式传播“当前这一系列操作是由哪一个 Agent 发起的”，免去在所有底层函数参数中层层透传 `agent` 的污染。
 *    - 职责四【优雅排空停机与防死锁（Teardown Drain）】：
 *      当包含当前服务的插件或 Fiber 卸载时，自动拒绝新任务并等待所有进行中的异步操作优雅结算（Drain），
 *      且内置了“自卸载防死锁”机制（防止当前任务触发卸载时无限等待自己完成）。
 *
 * 2. 为什么需要 `initiators`（AsyncLocalStorage）？
 *    - 解决痛点：系统中有大量底层组件（如文件系统、Shell 执行器、Web 请求、日志系统、指标监控等）。
 *      当一个 Agent 触发某个底层操作时，这些底层组件需要记录“这是谁干的”（Causal Attribution）。
 *      如果通过函数入参层层显式传递 `agent`，整个代码库的函数签名都会被严重污染。
 *    - 运行原理：通过 `withInitiator(agent, () => ...)` 开启一个异步作用域。在这个函数及其调用的
 *      任何深层异步代码中，直接调用 `ctx.agents.currentInitiator()` 即可拿到发起者 Agent。
 *    - 安全界限：仅用于“同进程内的因果关联与可观测性（日志/追踪）”，严禁作为跨进程、跨工作区或安全鉴权的凭据。
 * ============================================================================
 */
export class AgentRegistry extends Service {
  /**
   * 进程内所有存活 Agent 的权威目录字典（SessionId -> AgentEntry）。
   * 记录了实例引用、所属父级（owner）、生命周期状态机标记（两阶段发布 announce / detach 等）。
   */
  private store = new Map<SessionId, AgentEntry>()

  /**
   * Agent 创建工厂插槽（延迟注入）。
   * 实际的 Agent 构建与运行由 `@deepseek-ai/dsh-agent-loop` 提供并通过 `setFactory()` 注入。
   * 用 FactorySlot 对象包裹是为了防止 Cordis 框架在调用者上下文明确前过早对 Service 属性建立追踪代理。
   */
  private factory: FactorySlot | undefined

  /**
   * 【核心成员：异步发起者追踪器】
   * 基于 Node.js 原生 AsyncLocalStorage。
   * 在当前同进程的异步调用栈上下文中，维护当前链路的发起者 Agent（即 Causal Initiator）。
   * 使得在任意深度的异步回调、工具调用或日志追踪中，无需层层显式传递 agent 参数即可追溯来源。
   */
  private readonly initiators = new AsyncLocalStorage<Agent | undefined>()

  /**
   * 【发起者调用链路链表】
   * 通过 AsyncLocalStorage 记录当前异步任务的嵌套链（`run.parent`），
   * 用于在优雅停机（Drain）时追踪在途任务，并在自卸载（Reentrant unload）时排除自身防止死锁。
   */
  private readonly initiatorRuns = new AsyncLocalStorage<InitiatorRun>()

  /** 发起者服务的生命周期状态：'active'（正常处理） | 'closing'（停止接收新边界，等待排空） | 'disposed'（已完全注销） */
  private initiatorState: 'active' | 'closing' | 'disposed' = 'active'

  /** 当前在途进行中的 initiator 异步任务计数（计数归 0 时触发 initiatorDrain resolve） */
  private activeInitiatorRuns = 0

  /** 优雅停机排空器：当服务准备注销且仍有在途异步任务时创建，待所有任务完成后 resolve */
  private initiatorDrain: PromiseWithResolvers<void> | undefined

  /** 服务停机过程的单例 Promise，确保注销逻辑只执行一次 */
  private initiatorDisposal: Promise<void> | undefined

  constructor(ctx: Context) {
    // 1. 将自身注册为 Cordis 服务，挂载到全局上下文上的名称为 'agents'（即通过 ctx.agents 访问）
    super(ctx, 'agents')

    // 2. 注入并集成 Typert RPC 类型运行时协议网关：
    // 当存在 typert 服务时，注册 Agent 与 SessionId 之间的 Host <-> Wire 自动编解码与查找映射
    ctx.inject(['typert'], (typeCtx) => {
      // 注册远程参数解析器：将 RPC 传输的 wire 参数 "agentId" (SessionId) 自动转换为内存中的宿主 Agent 实例
      typeCtx.typert.lookups.register('agent', {
        parameter: 'agent',
        wire: 'agentId',
        hostTypeSymbol: '@deepseek-ai/dsh-agent#Agent',
        wireTypeSymbol: '@deepseek-ai/dsh-session/types#SessionId',
        resolve: sessionId => this.get(sessionId),
      })
      // 注册宿主上下文解析器：根据 agentId 解析出该 Agent 绑定的专属 Cordis Context（agent.ctx）
      typeCtx.typert.contexts.registerHost('agent', {
        wire: 'agentId',
        wireTypeSymbol: '@deepseek-ai/dsh-session/types#SessionId',
        resolve: sessionId => this.get(sessionId)?.ctx,
      })
    })

    // 3. 定义全局通用的 `ctx.agent` 便捷访问器（DX 体验优化）：
    // 在 Cordis 中，访问 Context 上未声明的属性会直接抛错。
    // 这里在顶层上下文默认注册一个 getter 返回 undefined，保证普通插件上下文读取 `ctx.agent` 安全返回 undefined。
    // 而真正运行中的 Agent 会在自身的 `agent.ctx` 上以自身属性（Own Property）形式遮蔽（Shadow）此属性并返回其实例。
    ctx.accessor('agent', { get: () => undefined })

    // 4. 监听 Cordis 内部 Fiber 状态变更，提前进入停机准备：
    // 若当前服务的祖先生命周期 Fiber 开始卸载（UNLOADING），立即调用 closeInitiators() 拒绝新的 initiator 边界，
    // 在子级资源被完全拆除前尽早阻止新操作进入。
    ctx.on('internal/status', (fiber) => {
      if (fiber.state === FiberState.UNLOADING && this.hasLifecycleAncestor(fiber)) {
        this.closeInitiators()
      }
    })

    // 5. 声明本服务的生命周期清理 Effect（采用 Generator 逆序执行清理）：
    // Cordis 生成器 effect 的清理函数按照 yield 的逆序执行：
    // - 卸载时先执行第 2 个 yield：调用 this.closeInitiators()（拒收新任务）
    // - 接着执行第 1 个 yield：调用 this.disposeInitiators()（等待排空已存在的任务，并彻底注销 ALS）
    ctx.effect(function* (this: AgentRegistry) {
      yield () => this.disposeInitiators()
      yield () => { this.closeInitiators() }
    }.bind(this), 'agents.initiatorLifecycle()')
  }

  /**
   * Read the Agent that initiated the inherited asynchronous driver chain.
   * Use this optional form for logging, tracing, metrics, or host attribution
   * that also supports agentless calls. When a parent creates a child, setup
   * reports the causal parent while `agentCtx.agent` identifies the child.
   * @returns the inherited Agent, or `undefined` outside an initiator boundary
   *   and inside an explicit clearing boundary.
   * @throws when this service instance has been disposed.
   */
  currentInitiator(): Agent | undefined {
    this.assertInitiatorsReadable()
    return this.initiators.getStore()
  }

  /**
   * Read the initiating Agent and fail when no initiator boundary is active.
   * Use this for private helpers contractually below a driver, or for a
   * deployment-owned outbound request whose contract forbids agentless calls.
   * Generic or direct-call paths use optional lookup or explicit request fields.
   * @returns the inherited Agent.
   * @throws when no initiator is active or this service instance has been disposed.
   */
  requireInitiator(): Agent {
    const agent = this.currentInitiator()
    if (agent === undefined) throw new Error(NO_INITIATOR_MESSAGE)
    return agent
  }

  /**
   * Run an operation with one exact Agent as its process-local initiator. The
   * exact synchronous value or Promise returned by the operation is preserved.
   * Custom drivers and test harnesses wrap their complete returned foreground
   * lifetime.
   * A queue or wire receiver may establish this boundary only after validating
   * explicit identity and resolving the exact live Agent; this method does neither.
   * Detached work remains owned by the subsystem that starts it.
   * @param agent - initiating Agent to inherit; presence is neither liveness proof nor authorization.
   * @param operation - synchronous or asynchronous operation to invoke.
   * @returns the exact value returned by `operation`.
   * @throws when the initiator scope is closing/disposed, or when `operation` throws.
   */
  withInitiator<T>(agent: Agent, operation: () => T): T {
    return this.runWithInitiator(agent, operation)
  }

  /**
   * Run an operation inside a boundary that hides any inherited initiating
   * Agent. The exact synchronous value or Promise is preserved.
   * Use this while creating lazy shared timers, queue pumps, pool maintenance,
   * watchers, or exporters so they do not inherit the first Agent that happens
   * to initialize them. It clears only initiator attribution, not explicit
   * fields, and does not own or drain detached resources.
   * @param operation - synchronous or asynchronous operation to invoke without an initiator.
   * @returns the exact value returned by `operation`.
   * @throws when the initiator scope is closing/disposed, or when `operation` throws.
   */
  withoutInitiator<T>(operation: () => T): T {
    return this.runWithInitiator(undefined, operation)
  }

  /**
   * Register the agent-creation factory (the loop calls this on construction,
   * effect-scoped). A traced Cordis service is canonicalized to its concrete
   * target; each create/resume call is then traced through that caller's
   * context so ownership follows the caller without stacking proxy layers.
   * Throws if a factory is already registered. Returns the disposer; on
   * dispose the factory slot is cleared.
   * @param factory - the loop-owned factory {@link create}/{@link resume} delegate to.
   * @returns the disposer that clears the factory slot. The exact
   *   Cordis effect disposer (single-shot): composite (generator) effects may
   *   yield it directly — exact identity nests the teardown in order.
   */
  setFactory(factory: AgentFactory): () => void {
    const dispose = this.ctx.effect(() => {
      if (this.factory !== undefined) throw new Error('an agent factory is already registered')
      // Avoid stacking two Cordis shadow layers when a caller passes a Service
      // already read through a context. Calls are re-traced through their
      // actual owner context below.
      const target = (factory as AgentFactory & { [symbols.original]?: AgentFactory })[symbols.original] ?? factory
      this.factory = { target }
      return () => { this.factory = undefined }
    }, 'agents.setFactory()')
    // The exact cordis effect disposer (the agents.register() convention): a
    // caller's composite effect can yield it for in-order teardown; the
    // loop's constructor effect returns it directly, identity-nesting the
    // registration under that effect.
    // oxlint-disable-next-line typescript/no-misused-promises -- synchronous cleanup; direct return preserves disposer identity
    return dispose
  }

  /** Return the active creation factory. */
  private requireFactory(): FactorySlot {
    if (this.factory === undefined) throw new Error(NO_FACTORY_MESSAGE)
    return this.factory
  }

  /**
   * Create and publish a new agent through the registered factory.
   * Distinct from {@link register} (which records an already-constructed
   * agent): this constructs the agent and its session. Rejects if no factory is
   * registered or creation/setup fails. The resolved {@link AgentHandle} lets
   * the owner tear down exactly this agent.
   * @param options - shared identity, session seed/metadata, and agent options.
   * @returns the handle after setup, rollback-covered publication, and loop start complete.
   */
  async create(options: CreateAgentOptions): Promise<AgentHandle> {
    const ownerCtx = this.ctx
    // Re-trace a Service-backed factory through the accessing context
    // explicitly. This preserves AgentLoop's dependency origin while binding
    // its effects to ownerCtx; plain factories receive ownerCtx as an explicit
    // capability and need no Cordis tracker magic.
    const { target } = this.requireFactory()
    const receiver = getTraceable(ownerCtx, target)
    // oxlint-disable-next-line typescript/unbound-method -- Reflect.apply intentionally supplies the caller-traced receiver
    return Reflect.apply(target.createAgent, receiver, [ownerCtx, options])
  }

  /**
   * Load a persisted session and resume an agent on it through the registered
   * factory. Rejects if no factory is registered; the factory rejects if
   * session persistence is not configured or persistence/setup fails.
   * @param options - persisted identity, configuration, and optional setup.
   * @returns the handle after setup, rollback-covered publication, and loop start complete.
   */
  async resume(options: ResumeAgentOptions): Promise<AgentHandle> {
    const ownerCtx = this.ctx
    const { target } = this.requireFactory()
    const receiver = getTraceable(ownerCtx, target)
    // oxlint-disable-next-line typescript/unbound-method -- Reflect.apply intentionally supplies the caller-traced receiver
    return Reflect.apply(target.resume, receiver, [ownerCtx, options])
  }

  /**
   * Register a live agent. Throws if an agent with the same id is already
   * registered. Emits `agent/created` on registration and `agent/disposed`
   * when the calling fiber is disposed — both with the agent's scope carrier
   * (`scopeTarget(agent, agent)`): the subject is the agent in hand, so the
   * emits are scope-filtered regardless of which context invoked `register`
   * (calling through `agent.ctx` scopes EFFECTS; dispatch scoping always
   * requires passing the carrier). Returns the disposer.
   * @param agent - the already-constructed agent to record in the store.
   * @returns the EXACT Cordis effect disposer (single-shot; a repeat call
   *   returns undefined without awaiting an in-flight teardown). Exact
   *   identity is load-bearing: a composite (generator) effect that owns a
   *   teardown ORDER — the agent factory's lifecycle chain — must yield THIS
   *   function so Cordis nests the unregistration at that yield position;
   *   yielding a wrapper would leave it disposing as a concurrent sibling on
   *   owner unload, unregistering the agent (and emitting `agent/disposed`)
   *   while its final turn is still draining.
   */
  register(agent: Agent): () => void {
    const dispose = this.ctx.effect(function* (this: AgentRegistry) {
      yield this.enter(agent, this.ctx.agent)
      this.announce(agent)
    }.bind(this), 'agents.register()')
    // oxlint-disable-next-line typescript/no-misused-promises -- synchronous cleanup; direct return preserves disposer identity
    return dispose
  }

  /**
   * Insert an already-constructed agent without announcing it. This is the
   * advanced ordered-lifecycle primitive used by the async agent factory: it
   * first completes setup while the agent is unpublished, then assigns the
   * returned detach closure into its pre-installed composite teardown before
   * calling {@link announce}. Ordinary callers use {@link register}.
   * @param agent - the prepared, unpublished agent.
   * @param owner - live agent whose scoped context created this agent, or
   *   undefined for a top-level runtime root. This is runtime ownership, not
   *   the resumed session's durable parent lineage.
   * @returns an idempotent closure that removes this exact entry and emits
   *   `agent/disposed` with listener failures contained. When called from a
   *   synchronous `agent/created` listener, removal and disposal wait until
   *   that creation dispatch unwinds.
   */
  enter(agent: Agent, owner: Agent | undefined): () => void {
    const id = agent.id
    if (id !== agent.session.id) {
      throw new Error(`agent id "${id}" does not match session id "${agent.session.id}"`)
    }
    const carrier = scopeTarget(agent, agent)
    // This is the authoritative collision boundary. Concurrent create/resume
    // operations may both prepare, but only one exact entry can publish.
    if (this.store.has(id)) throw new Error(`agent "${id}" is already registered`)
    const entry: AgentEntry = {
      id,
      agent,
      owner,
      carrier,
      announced: false,
      announcing: false,
      detachRequested: false,
    }
    this.store.set(id, entry)
    let entered = true
    const detach = (): void => {
      if (!entered) return
      entered = false
      // Every callback reached by this creation dispatch must observe the same
      // live entry, and disposal must follow creation. A listener may own
      // the advanced detach capability, so make that ordering structural:
      // visibility and the paired disposal are deferred until announce()'s
      // synchronous dispatch has unwound.
      if (entry.announcing) {
        entry.detachRequested = true
        return
      }
      this.detachEntered(entry)
    }
    return detach
  }

  /** Remove one exact entered agent and emit its paired disposal when announced. */
  private detachEntered(entry: AgentEntry): void {
    entry.detachRequested = false
    // A stale capability can never delete a later same-id lifecycle. The
    // captured entry identity is the final boundary.
    /* v8 ignore next -- enter() rejects replacement while this single-shot detach capability is live. */
    if (this.store.get(entry.id) !== entry) return
    this.store.delete(entry.id)
    // An insertion rolled back before announce was never externally created,
    // so emitting disposed would invent an impossible lifecycle edge. Marking
    // happens before the created emit: if a later created listener throws,
    // earlier listeners may already have observed it and must see disposal.
    if (!entry.announced) return
    this.emitDisposed(entry)
  }

  /** Emit the paired disposal edge through the entry's stable carrier. */
  private emitDisposed(entry: AgentEntry): void {
    const args: unknown[] = [entry.carrier, 'agent/disposed', { agent: entry.agent }]
    for (const callback of this.ctx.events.dispatch('emit', args)) {
      try {
        const returned: unknown = callback(...args)
        void Promise.resolve(returned).catch((error: unknown) => {
          this.ctx.logger.warn(`agent "${entry.id}": agent/disposed listener rejected: ${String(error)}`)
        })
      } catch (error: unknown) {
        this.ctx.logger.warn(`agent "${entry.id}": agent/disposed listener threw: ${String(error)}`)
      }
    }
  }

  /**
   * Announce an agent previously inserted with {@link enter}.
   * @param agent - the live inserted agent to announce.
   * @throws if `agent` is not the exact live registry entry for its id, or its
   *   creation announcement already began (including a reentrant call from a
   *   creation listener).
   */
  announce(agent: Agent): void {
    const entry = this.store.get(agent.id)
    if (entry === undefined || entry.agent !== agent) {
      throw new Error(`agent "${agent.id}" is not live in this registry`)
    }
    if (entry.announced || entry.announcing) {
      throw new Error(`agent "${entry.id}" was already announced`)
    }
    // Mark before dispatch so a listener cannot recursively create a second
    // lifecycle edge; detach still pairs a partially delivered first edge.
    entry.announcing = true
    entry.announced = true
    const args: unknown[] = [entry.carrier, 'agent/created', { agent: entry.agent }]
    try {
      for (const callback of this.ctx.events.dispatch('emit', args)) {
        // A synchronous creation failure vetoes publication and rolls back.
        // Returned-promise rejection happens after this synchronous boundary, so
        // observe and report it instead of leaking an unhandled rejection.
        const returned: unknown = callback(...args)
        void Promise.resolve(returned).catch((error: unknown) => {
          this.ctx.logger.warn(`agent "${entry.id}": agent/created listener rejected: ${String(error)}`)
        })
      }
    } finally {
      entry.announcing = false
      if (entry.detachRequested) this.detachEntered(entry)
    }
  }

  /**
   * Look up a live agent.
   * @param id - the shared agent/session id to look up.
   * @returns the agent, or undefined when no live agent has that id.
   */
  get(id: SessionId): Agent | undefined {
    return this.store.get(id)?.agent
  }

  /**
   * Test whether a live agent was created through one exact parent agent's
   * scoped context. Runtime ownership is independent of durable session
   * lineage and remains unambiguous when unrelated providers reuse an id.
   * @param id - the candidate child agent's shared agent/session id.
   * @param owner - the expected runtime creator agent.
   * @returns true only while the exact child entry is live under that owner.
   */
  isOwnedBy(id: SessionId, owner: Agent): boolean {
    return this.store.get(id)?.owner === owner
  }

  /**
   * All live agents, in registration order.
   * @returns a fresh array; mutating it does not affect the registry.
   */
  list(): Agent[] {
    return [...this.store.values()].map(entry => entry.agent)
  }

  /**
   * All live top-level agents in registration order. A top-level agent was
   * created without an owning agent context; durable session lineage does not
   * affect this runtime relation, so a resumed fork may still be a root.
   * @returns a fresh array; mutating it does not affect the registry.
   */
  roots(): Agent[] {
    return [...this.store.values()]
      .filter(entry => entry.owner === undefined)
      .map(entry => entry.agent)
  }

  /**
   * Reject new initiator boundaries while inherited continuations drain.
   * 停止接收新的 withInitiator 边界，准备进行优雅排空。
   */
  private closeInitiators(): void {
    if (this.initiatorState === 'active') this.initiatorState = 'closing'
  }

  /**
   * Wait for returned-Promise boundaries, then invalidate retained references.
   * 优雅停机注销：
   * 1. 阻止新任务（closeInitiators）
   * 2. 剔除因自卸载而重入的链路，防止自等待死锁（releaseReentrantInitiatorRuns）
   * 3. 等待所有在途异步操作排空完毕（await initiatorDrain.promise）
   * 4. 彻底禁用 AsyncLocalStorage，释放引用避免内存泄露
   */
  private disposeInitiators(): Promise<void> {
    return (this.initiatorDisposal ??= (async () => {
      this.closeInitiators()
      this.releaseReentrantInitiatorRuns()
      if (this.activeInitiatorRuns !== 0) {
        this.initiatorDrain ??= Promise.withResolvers<void>()
        await this.initiatorDrain.promise
      }
      this.initiatorState = 'disposed'
      this.initiators.disable()
      this.initiatorRuns.disable()
    })())
  }

  /**
   * Establish one tracked initiator or clearing boundary.
   * 在 AsyncLocalStorage 中绑定指定 agent（或 undefined 清除发起者），
   * 同时记录 InitiatorRun 链表并增加在途计数（activeInitiatorRuns）。
   * 若 operation 返回 Promise，会在其 settled 时自动扣减计数以支持优雅排空。
   */
  private runWithInitiator<T>(agent: Agent | undefined, operation: () => T): T {
    if (this.initiatorState !== 'active') throw new Error(DISPOSED_INITIATOR_MESSAGE)
    const run: InitiatorRun = {
      active: true,
      parent: this.initiatorRuns.getStore(),
    }
    this.activeInitiatorRuns += 1
    let result: T
    try {
      result = this.initiatorRuns.run(run, () => this.initiators.run(agent, operation))
    } catch (error: unknown) {
      this.releaseInitiatorRun(run)
      throw error
    }
    if (isPromise(result)) {
      try {
        void Promise.prototype.then.call(
          result,
          () => { this.releaseInitiatorRun(run) },
          () => { this.releaseInitiatorRun(run) },
        )
      } catch {
        // A branded Promise may expose a failing @@species. Observer setup did
        // not attach, so preserve the exact return without leaking the run.
        this.releaseInitiatorRun(run)
      }
    } else {
      this.releaseInitiatorRun(run)
    }
    return result
  }

  /** Whether one unloading fiber owns this service's lifecycle. */
  private hasLifecycleAncestor(candidate: Fiber): boolean {
    let fiber = this.ctx.fiber
    while (true) {
      if (fiber === candidate) return true
      const parent = fiber.parent.fiber
      if (parent === fiber) return false
      fiber = parent
    }
  }

  private assertInitiatorsReadable(): void {
    if (this.initiatorState === 'disposed') throw new Error(DISPOSED_INITIATOR_MESSAGE)
  }

  /** Exclude the boundary chain that initiated this teardown from its own drain. */
  private releaseReentrantInitiatorRuns(): void {
    let run = this.initiatorRuns.getStore()
    while (run !== undefined) {
      this.releaseInitiatorRun(run)
      run = run.parent
    }
  }

  private releaseInitiatorRun(run: InitiatorRun): void {
    if (!run.active) return
    run.active = false
    this.activeInitiatorRuns -= 1
    if (this.activeInitiatorRuns !== 0) return
    this.initiatorDrain?.resolve()
    this.initiatorDrain = undefined
  }
}

export default AgentRegistry
