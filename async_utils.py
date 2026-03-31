from __future__ import annotations

import asyncio
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def _run_sync_guarded(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    try:
        return func(*args, **kwargs)
    except StopIteration as e:
        # Python 3.12: StopIteration нельзя пробрасывать в asyncio.Future.
        raise RuntimeError("Background sync function raised StopIteration") from e


async def safe_to_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return await asyncio.to_thread(_run_sync_guarded, func, *args, **kwargs)
