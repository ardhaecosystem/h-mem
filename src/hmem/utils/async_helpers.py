"""Helpers for bridging sync and async code."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")


def run_sync(coro: Awaitable[T]) -> T:
    """Run an async coroutine from a sync context.

    Works correctly inside Jupyter notebooks and other environments
    that already have a running event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if not loop.is_running():
        return loop.run_until_complete(coro)

    # Inside a running loop (e.g. Jupyter) → execute in a fresh thread loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
