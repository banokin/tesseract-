"""
Параллельная обработка запросов от нескольких клиентов.

- ThreadPoolExecutor задаётся как default executor цикла событий → asyncio.to_thread
  использует ограниченный пул потоков вместо неограниченного по умолчанию.
- Семафор ограничивает число одновременных «тяжёлых» задач (OCR / docx), чтобы
  при наплыве пользователей не исчерпать RAM и CPU.

Переменные окружения:
- OCR_THREAD_POOL_WORKERS — число потоков в пуле (по умолчанию: min(32, CPU+4)).
- OCR_MAX_CONCURRENT_JOBS — одновременных тяжёлых задач (по умолчанию: как workers).
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI


def _default_pool_workers() -> int:
    return min(32, (os.cpu_count() or 1) + 4)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    workers = int(os.environ.get("OCR_THREAD_POOL_WORKERS", str(_default_pool_workers())))
    workers = max(1, workers)
    max_jobs = int(os.environ.get("OCR_MAX_CONCURRENT_JOBS", str(workers)))
    max_jobs = max(1, max_jobs)

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr")
    loop = asyncio.get_running_loop()
    loop.set_default_executor(executor)
    app.state.ocr_semaphore = asyncio.Semaphore(max_jobs)

    try:
        yield
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


async def run_blocking(request: Any, fn: Any, /, *args: Any) -> Any:
    """Выполнить синхронную функцию в пуле потоков с опциональным лимитом параллелизма."""
    sem = getattr(request.app.state, "ocr_semaphore", None)
    if sem is not None:
        async with sem:
            return await asyncio.to_thread(fn, *args)
    return await asyncio.to_thread(fn, *args)
