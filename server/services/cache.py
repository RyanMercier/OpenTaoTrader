import asyncio
import time
from typing import Any, Callable, Coroutine, Optional


class TTLCache:
    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is not None:
            value, expires_at = entry
            if time.monotonic() < expires_at:
                return value
            self._store.pop(key, None)
        return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        self._store[key] = (value, time.monotonic() + ttl)

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Coroutine[Any, Any, Any]],
        ttl: int,
    ) -> Any:
        cached = await self.get(key)
        if cached is not None:
            return cached

        lock = await self._get_lock(key)
        async with lock:
            # Double-check after acquiring lock
            cached = await self.get(key)
            if cached is not None:
                return cached

            value = await factory()
            await self.set(key, value, ttl)
            return value

    async def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()
