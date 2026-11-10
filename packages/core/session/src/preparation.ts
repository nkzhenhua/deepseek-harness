/**
 * Ownership of one unpublished Session before registry publication.
 * @module @deepseek-ai/dsh-session/preparation
 */

import type { Session } from './index.ts'

/** Options for a preparation whose provider retains unpublished state. */
export interface SessionPreparationOptions {
  /** Release provider-owned state when the Session was not published. */
  readonly release?: () => void
}

/**
 * One exact unpublished Session and the provider state that keeps it usable.
 * Disposal is synchronous and idempotent. Providers decide whether release
 * returns the Session to a cache or discards it; publication may consume that
 * state before disposal, making the callback a no-op.
 *
 * 【会话预备资源包装器】
 * 封装一个尚未正式注册发布到 SessionStore 的 Session 实例及其关联的提供者状态。
 * 实现了 Disposable 接口（支持 using 语法），用于在会话构建、校验或注册发布流程失败时，
 * 保证同步且幂等地释放（release）未发布资源，避免孤儿状态或未提交状态泄露。
 */
export class SessionPreparation implements Disposable {
  /** 标记是否已经释放，防止重复调用 release 回调 */
  private released = false

  /** The exact Session to use for setup and publication. / 当前准备好的裸 Session 实例 */
  readonly session: Session

  private constructor(
    session: Session,
    private readonly options: SessionPreparationOptions,
  ) {
    this.session = session
  }

  /**
   * Wrap an unpublished Session in one preparation lifetime.
   * @param session - exact unpublished Session.
   * @param options - optional provider release behavior.
   * @returns a preparation disposed after publication or rollback.
   */
  static create(session: Session, options?: SessionPreparationOptions): SessionPreparation {
    return new SessionPreparation(session, options ?? {})
  }

  /** Release provider state once when this preparation leaves its caller. / 离开作用域或销毁时同步调用 release 释放提供者保留的资源 */
  [Symbol.dispose](): void {
    if (this.released) return
    this.released = true
    this.options.release?.()
  }
}
