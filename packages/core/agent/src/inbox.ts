/**
 * Incremental projection of durable agent inbox events.
 *
 * @module @deepseek-ai/dsh-agent/inbox
 */

import type { MessageId } from '@deepseek-ai/dsh-llm'
import type {
  Session,
  SessionEventMap,
  UserMessage,
} from '@deepseek-ai/dsh-session'
import type { InboxTarget } from './types.ts'

/** Mutable state privately owned by an {@link Inbox}. */
type InboxState = Record<InboxTarget, UserMessage[]>

/** Live notifications committed by inbox mutations. */
export interface InboxNotifications {
  /** Publish one inserted message. */
  inserted(message: UserMessage): void
  /** Publish one discarded message. */
  discarded(message: UserMessage): void
  /** Publish one claimed message inside its owning turn. */
  claimed(message: UserMessage, turn: number): void
}

/**
 * A replay-once projection that incrementally consumes later inbox splices.
 *
 * ============================================================================
 * 【学习笔记：Agent 运转模型与收件箱设计机制】
 *
 * Q1: Agent 内部多步任务（Step 1 -> 2 -> 3...）是怎么驱动的？是靠消息队列吗？
 * A1: 绝对不是靠消息队列！
 *     - `Inbox` 里存的全部是 `UserMessage`（外部输入），Agent 不会在里面给自己写“子任务”。
 *     - 步骤推进的核心是：代码里的 `while (true)` 循环 + 大模型的对话上下文（Message History）+ Tool 结果：
 *       1. Step 1: LLM 拿到用户输入，决定先写脚本 -> 输出 ToolCall: write_file("script.py")。
 *       2. 系统本地执行工具，产生 ToolResult: "写入成功"，追加到会话上下文。
 *       3. Step 2: 循环继续。LLM 看到历史记录里上一步脚本已写好，自主决定写测试 -> 输出 ToolCall: write_file("test.py")。
 *       4. Step 3/4/5: 工具结果驱动模型继续执行，直到最终模型认为无需再调工具，输出纯文本回复，Turn 结束。
 *     - 结论：多 Step 自主执行全靠“上下文 + 工具结果自循环”，在此期间如果没有外部打扰，`next-step` 队列全程为空！
 *
 * Q2: 系统怎么知道用户的消息是“不打扰当前任务的排队（Queue）”还是“立刻影响的干预（Steer）”？Agent 会单独调一次 LLM 来判断吗？
 * A2: 完全不会调 LLM！采用的是“零延迟、确定性”的前端手势策略与用户设置（详见 ui-conversation / submission-policy）：
 *     - 普通回车（Enter，默认行为）：
 *       由配置 `busyEnter: 'queue'` 决定，打上 `mode: 'queue'` 标签 -> 后端调用 `agent.followup()` 放入 `next-turn` 队列，
 *       不打扰当前运行中的任务，老老实实排队等当前 Turn 结束后开启新 Turn。
 *     - 加速快捷键（Cmd/Ctrl + Enter 或 UI 上的 Steer 按钮）：
 *       打上 `mode: 'steer'` 标签 -> 后端调用 `agent.steer()` 放入 `next-step` 队列，
 *       Agent 会在下一个 step 边界通过 `claim('next-step')` 立刻将其取出塞给大模型，实现即时转向。
 *     - 排队提升（Promote to Steer）：
 *       用户在界面的排队列表中，还可以随时手动将某条排队消息从 `next-turn` 挪到 `next-step`。
 *     - 为什么坚决不用 LLM 猜测：避免 1~3 秒网络延迟、避免分类误判搞乱正在跑的代码、避免浪费 Token 额度。
 * ============================================================================
 */
export class Inbox {
  private readonly state: InboxState = { 'next-turn': [], 'next-step': [] }

  constructor(
    private readonly session: Session,
    private readonly notifications: InboxNotifications,
  ) {
    for (const event of session.events.slice(session.header.seedLength ?? 0)) {
      if (event.type !== 'agent/inbox/spliced') continue
      try {
        this.apply(event.data)
      } catch (error: unknown) {
        throw new Error(
          `invalid persisted inbox splice at session seq ${event.seq}`,
          { cause: error },
        )
      }
    }
  }

  /**
   * Prompts awaiting individual turns.
   *
   * 【学习笔记：任务排队队列 (FIFO)】
   * - 存放等待在开启新轮次时执行的独立 Prompt 任务（通常由 `agent.followup()` 提交）。
   * - 哪怕 Agent 正在忙碌执行当前任务，用户普通回车发送的消息也会先进入此队列排队（可以同时积攒多条）。
   * - 每个 Turn 开启时，仅从队头消费 1 条。
   */
  get nextTurn(): readonly UserMessage[] {
    return this.state['next-turn']
  }

  /**
   * Input awaiting the next step boundary.
   *
   * 【学习笔记：步间即时干预/注入队列】
   * - 存放需要在当前任务执行中途立刻插话/生效的消息（由 `agent.steer()` 实时转向或 `agent.inject()` 静默注入）。
   * - 注意：Agent 内部子任务的推进（Step 1 -> Step 2 ...）依靠的是 LLM 对话上下文与 Tool 结果，
   *   该队列平日里保持为空；只有外部用户/系统强行插话时才会有数据。
   * - 与 claim() 区别：此处为只读 getter，不改变队列也不产生持久化事件。
   */
  get nextStep(): readonly UserMessage[] {
    return this.state['next-step']
  }

  /** Whether either pending-message list contains work. */
  get hasPending(): boolean {
    return this.nextTurn.length > 0 || this.nextStep.length > 0
  }

  /** Durably cancel all pending input, clearing next-step before next-turn. */
  clear(): void {
    this.splice('next-step', 0, this.nextStep.length, [])
    this.splice('next-turn', 0, this.nextTurn.length, [])
  }

  /**
   * Remove and return the complete batch proposed for one step, publishing
   * each claimed message. The durable splices are pure deletions.
   * @param target - whether this boundary also consumes one queued turn.
   * @param turn - turn that will own the claimed batch.
   * @returns next-step input followed by the queued turn, when requested.
   * @internal - The agent loop's step-boundary operation, not a plugin extension point.
   *
   * 【学习笔记：步骤边界领取/认领核心函数】
   * 在 Agent Loop 准备执行每一个 step（调用 preStep）时被调用：
   * 1. 为什么清空 nextStep：
   *    所有堆积的 next-step 干预消息都是对“当下这个 step”的实时环境补充，
   *    必须打包一次性喂给大模型，避免信息滞后，因此传 `this.nextStep.length` 一次性全部取出。
   * 2. 为什么 nextTurn 只取 1 条：
   *    当 target === 'next-turn'（新轮次边界）时，`0, 1` 从队头弹出正好 1 条独立任务开启新轮次，
   *    其余排队任务继续保留在 nextTurn 中等待后续轮次。
   * 3. 为什么传 false (discardRemoved = false)：
   *    此处消息被取出是正常“认领消费（claim）”，属于纯删除 splice，不记录 outcome: 'canceled'，
   *    不触发 discarded 通知，而是随后触发 notifications.claimed(message, turn)。
   */
  claim(target: InboxTarget, turn: number): UserMessage[] {
    const claimed = this.mutate(
      'next-step',
      0,
      this.nextStep.length,
      [],
      false,
    )
    if (target === 'next-turn') {
      claimed.push(...this.mutate('next-turn', 0, 1, [], false))
    }
    for (const message of claimed) this.notifications.claimed(message, turn)
    return claimed
  }

  /**
   * Append one message to a pending list and durably record the insertion.
   * @param target - pending list to extend.
   * @param message - message to append.
   * @throws if the message identity is already pending.
   */
  append(target: InboxTarget, message: UserMessage): void {
    this.splice(target, this.state[target].length, 0, [message])
  }

  /**
   * Prepend one message to a pending list and durably record the insertion.
   * @param target - pending list to extend.
   * @param message - message to prepend.
   * @throws if the message identity is already pending.
   */
  prepend(target: InboxTarget, message: UserMessage): void {
    this.splice(target, 0, 0, [message])
  }

  /**
   * Replace one pending message in place, possibly changing its identity. A
   * successful replacement publishes the old message as discarded and the new
   * message as inserted.
   * @param messageId - identity of the pending message to replace.
   * @param newMessage - replacement message.
   * @returns whether the message was still pending.
   * @throws if the replacement duplicates another pending message identity.
   */
  replace(messageId: MessageId, newMessage: UserMessage): boolean {
    const location = this.locate(messageId)
    if (location === undefined) return false
    this.splice(location.target, location.index, 1, [newMessage])
    return true
  }

  /**
   * Remove one pending message and durably record its cancellation.
   * @param messageId - identity of the pending message to remove.
   * @returns whether the message was still pending.
   */
  remove(messageId: MessageId): boolean {
    const location = this.locate(messageId)
    if (location === undefined) return false
    this.splice(location.target, location.index, 1, [])
    return true
  }

  /**
   * Apply standard splice semantics and durably record the normalized result.
   * The durable event commits before the live projection mutates, so synchronous
   * `session/event` observers see the pre-splice lists and can reconstruct the
   * removed messages from the normalized coordinates.
   * @param target - pending list to mutate.
   * @param start - splice position.
   * @param deleteCount - maximum number of messages to remove.
   * @param inserted - messages to insert at the resolved position.
   * @returns messages removed by the splice.
   *
   * 【学习笔记：万能原子变更原语 (Splice)】
   * 1. 为什么又是 deleteCount 又是 inserted：
   *    借用 JS 原生 `Array.prototype.splice` 设计，任何队列变更本质上都是：
   *    “在指定位置删掉 N 个元素，并插入 M 个新元素”。
   *    - append: 删 0 个，末尾插入 [msg]
   *    - prepend: 删 0 个，开头插入 [msg]
   *    - remove: 删 1 个，插入 []
   *    - replace: 删 1 个旧的，原地插入 [newMsg]（原子替换）
   *    - clear: 删全部，插入 []
   * 2. 事件溯源与回放（Event Sourcing）：
   *    系统只需记录一种统一的持久化事件 `agent/inbox/spliced`。重启会话时直接把
   *    start/removedCount/inserted 给原生 array.splice() 执行，即可极简且确定性地还原队列。
   * 3. 为什么传 true (discardRemoved = true)：
   *    这里的删除是用户主动撤回/清空/替换，传入 true 会将移除的消息标记为 outcome: 'canceled'，
   *    并触发 notifications.discarded(message)。
   */
  splice(
    target: InboxTarget,
    start: number,
    deleteCount: number,
    inserted: UserMessage[],
  ): UserMessage[] {
    return this.mutate(target, start, deleteCount, inserted, true)
  }

  /** Locate one pending identity across both owned lists. */
  private locate(
    messageId: MessageId,
  ): { target: InboxTarget; index: number } | undefined {
    for (const target of ['next-turn', 'next-step'] as const) {
      const index = this.state[target].findIndex(
        message => message.id === messageId,
      )
      if (index >= 0) return { target, index }
    }
    return undefined
  }

  /** Commit one normalized mutation and publish its live notifications. */
  private mutate(
    target: InboxTarget,
    start: number,
    deleteCount: number,
    inserted: UserMessage[],
    discardRemoved: boolean,
  ): UserMessage[] {
    const inbox = this.state[target]
    const truncatedStart = Math.trunc(start)
    const offset = Number.isNaN(truncatedStart) ? 0 : truncatedStart
    const actualStart =
      offset < 0
        ? Math.max(inbox.length + offset, 0)
        : Math.min(offset, inbox.length)
    const truncatedDeleteCount = Math.trunc(deleteCount)
    const actualDeleteCount = Math.min(
      Math.max(
        Number.isNaN(truncatedDeleteCount) ? 0 : truncatedDeleteCount,
        0,
      ),
      inbox.length - actualStart,
    )
    if (actualDeleteCount === 0 && inserted.length === 0) return []
    const outcome =
      discardRemoved && actualDeleteCount > 0
        ? ('canceled' as const)
        : undefined
    const splice = {
      target,
      start: actualStart,
      ...(actualDeleteCount === 0 ? {} : { removedCount: actualDeleteCount }),
      inserted,
      ...(outcome === undefined ? {} : { outcome }),
    }
    this.validate(splice)
    const event = this.session.append('agent/inbox/spliced', splice)
    const removed = inbox.splice(
      actualStart,
      actualDeleteCount,
      ...event.data.inserted,
    )
    if (discardRemoved) {
      for (const message of removed) this.notifications.discarded(message)
    }
    for (const message of event.data.inserted)
      this.notifications.inserted(message)
    return removed
  }

  /** Apply one normalized durable splice to the projection. */
  private apply(splice: SessionEventMap['agent/inbox/spliced']): UserMessage[] {
    this.validate(splice)
    const inbox = this.state[splice.target]
    return inbox.splice(
      splice.start,
      splice.removedCount ?? 0,
      ...splice.inserted,
    )
  }

  /** Validate one normalized splice against the current projection. */
  private validate(splice: SessionEventMap['agent/inbox/spliced']): void {
    const inbox = this.state[splice.target]
    const removedCount = splice.removedCount ?? 0
    if (
      !Number.isSafeInteger(splice.start) ||
      splice.start < 0 ||
      splice.start > inbox.length ||
      !Number.isSafeInteger(removedCount) ||
      removedCount < 0 ||
      splice.start + removedCount > inbox.length
    ) {
      throw new Error('invalid inbox splice')
    }
    const candidate = inbox.toSpliced(
      splice.start,
      removedCount,
      ...splice.inserted,
    )
    const ids = new Set<string>()
    for (const message of splice.target === 'next-turn'
      ? [...candidate, ...this.nextStep]
      : [...this.nextTurn, ...candidate]) {
      if (ids.has(message.id))
        throw new Error(`message "${message.id}" is already pending`)
      ids.add(message.id)
    }
  }
}
