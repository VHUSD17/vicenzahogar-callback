"""
async_pool
----------
Ejecuta N corrutinas con concurrencia acotada, reintentos con
backoff exponencial + jitter, y cancelación limpia.

Uso:
    results = await run(tasks, workers=8, retries=3)

Diseñado para ser leído en 30 segundos y probado en 30 más.
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeAlias, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)

Task: TypeAlias = Callable[[], Awaitable[T]]


@dataclass(frozen=True, slots=True)
class Retry:
    attempts: int = 3
    base: float = 0.25
    cap: float = 8.0
    jitter: float = 0.1

    def delay(self, attempt: int) -> float:
        backoff = min(self.cap, self.base * 2 ** (attempt - 1))
        return backoff + random.uniform(0, self.jitter)


@dataclass(frozen=True, slots=True)
class Outcome(Generic[T]):
    index: int
    value: T | None
    error: BaseException | None
    elapsed: float

    @property
    def ok(self) -> bool:
        return self.error is None


async def _attempt(task: Task[T], policy: Retry) -> T:
    last: BaseException | None = None
    for n in range(1, policy.attempts + 1):
        try:
            return await task()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last = exc
            if n == policy.attempts:
                break
            await asyncio.sleep(policy.delay(n))
            log.debug("retry %d/%d after %r", n, policy.attempts, exc)
    assert last is not None
    raise last


async def run(
    tasks: Iterable[Task[T]],
    *,
    workers: int = 8,
    retries: "int | Retry" = 3,
) -> list[Outcome[T]]:
    policy = retries if isinstance(retries, Retry) else Retry(attempts=retries)
    sem = asyncio.Semaphore(workers)
    jobs = list(enumerate(tasks))
    results: list[Outcome[T] | None] = [None] * len(jobs)

    async def worker(i: int, task: Task[T]) -> None:
        async with sem:
            started = monotonic()
            try:
                value = await _attempt(task, policy)
                results[i] = Outcome(i, value, None, monotonic() - started)
            except BaseException as exc:
                results[i] = Outcome(i, None, exc, monotonic() - started)
                if isinstance(exc, asyncio.CancelledError):
                    raise

    async with asyncio.TaskGroup() as tg:
        for i, task in jobs:
            tg.create_task(worker(i, task), name=f"pool-{i}")

    return [r for r in results if r is not None]


# --- demo / smoke-test ----------------------------------------------------
async def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    flaky_counter = {"n": 0}

    async def ok(i: int) -> int:
        await asyncio.sleep(0.05)
        return i * i

    async def flaky() -> str:
        flaky_counter["n"] += 1
        if flaky_counter["n"] < 3:
            raise RuntimeError("bad luck")
        return "finally!"

    async def bad() -> None:
        raise ValueError("nope")

    tasks: list[Task] = [lambda i=i: ok(i) for i in range(6)]
    tasks += [flaky, bad]

    outcomes = await run(tasks, workers=3, retries=Retry(attempts=4, base=0.05))
    for o in outcomes:
        mark = "✓" if o.ok else "✗"
        payload = o.value if o.ok else f"{type(o.error).__name__}: {o.error}"
        log.info("%s [%d] %.3fs  %s", mark, o.index, o.elapsed, payload)


if __name__ == "__main__":
    asyncio.run(_demo())
