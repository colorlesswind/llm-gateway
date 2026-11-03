"""入口并发限流。

只解决一件事：限制「同一时刻正在处理中」的请求数，超过上限直接拒绝（429），
不排队等待 —— LLM 请求本身就很慢，排队只会把客户端超时叠起来。

为什么不用 ``asyncio.Semaphore``：

- 本场景要的是「满了立刻拒绝」而不是「排队等待」，而 Semaphore 没有
  ``acquire_nowait``；用 ``locked()`` + ``acquire()`` 会引入 check-then-set
  竞态（``locked()`` 返回 False 之后，名额仍可能被别的协程抢走）；
- 单事件循环内，:meth:`InflightLimiter.try_acquire` 里「判断 + 自增」之间
  没有 ``await``，是原子的，既正确又没有额外开销。

注意：计数器是**进程内**的。多 worker / 多副本部署时，实际总上限是
``max_concurrent_requests × 实例数``；如果需要全局上限，得换成 Redis 之类
的外部计数。
"""

from __future__ import annotations

from typing import Self


class Lease:
    """一次占用的凭证。

    ``release`` 是幂等的：重复调用不会把计数打乱，因此调用方不必纠结
    「这条路径是不是已经还过了」。
    """

    __slots__ = ("_limiter", "_released")

    def __init__(self, limiter: InflightLimiter) -> None:
        self._limiter = limiter
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._limiter._inflight -= 1

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class InflightLimiter:
    """固定容量的并发计数器：满则返回 ``None``，不排队。"""

    __slots__ = ("_inflight", "_limit")

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError(f"limit 必须为正整数，收到 {limit!r}")
        self._limit = limit
        self._inflight = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def inflight(self) -> int:
        """当前占用中的请求数（用于测试与可观测性）。"""
        return self._inflight

    def try_acquire(self) -> Lease | None:
        if self._inflight >= self._limit:
            return None
        self._inflight += 1
        return Lease(self)


__all__ = ["InflightLimiter", "Lease"]
