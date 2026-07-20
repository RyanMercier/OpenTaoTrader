import asyncio
from typing import Awaitable, Callable, Optional, TypeVar

from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.core.chain_data.stake_info import StakeInfo
from bittensor.utils.balance import Balance

from api.config import settings
from api.services.cache import TTLCache

T = TypeVar("T")


class ChainClient:
    def __init__(self, cache: TTLCache):
        self._cache = cache
        self._subtensor: Optional[AsyncSubtensor] = None

    async def startup(self):
        if settings.subtensor_endpoint:
            self._subtensor = AsyncSubtensor(network=settings.subtensor_endpoint)
        else:
            self._subtensor = AsyncSubtensor(network=settings.bittensor_network)
        await self._subtensor.__aenter__()

    async def shutdown(self):
        if self._subtensor:
            await self._subtensor.__aexit__(None, None, None)
            self._subtensor = None

    async def _call(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        timeout: Optional[float] = None,
    ) -> T:
        """Run one RPC with a hard timeout so a single stuck call can't block
        the event loop. Default timeout comes from ``settings.rpc_timeout``."""
        return await asyncio.wait_for(
            factory(),
            timeout=timeout if timeout is not None else settings.rpc_timeout,
        )

    async def get_metagraph(self, netuid: int, force_refresh: bool = False):
        key = f"metagraph:{netuid}"
        if force_refresh:
            await self._cache.invalidate(key)
        return await self._cache.get_or_set(
            key,
            lambda: self._call(lambda: self._subtensor.metagraph(netuid=netuid)),
            ttl=settings.cache_ttl_metagraph,
        )

    async def get_dynamic_info(self, netuid: int, force_refresh: bool = False):
        key = f"dynamic_info:{netuid}"
        if force_refresh:
            await self._cache.invalidate(key)
        return await self._cache.get_or_set(
            key,
            lambda: self._call(lambda: self._subtensor.subnet(netuid=netuid)),
            ttl=settings.cache_ttl_dynamic_info,
        )

    async def get_balance(self, coldkey_ss58: str) -> Balance:
        return await self._cache.get_or_set(
            f"balance:{coldkey_ss58}",
            lambda: self._call(lambda: self._subtensor.get_balance(coldkey_ss58)),
            ttl=settings.cache_ttl_balance,
        )

    async def get_stake_info_for_coldkey(
        self, coldkey_ss58: str
    ) -> list[StakeInfo]:
        return await self._cache.get_or_set(
            f"stake_info:{coldkey_ss58}",
            lambda: self._call(
                lambda: self._subtensor.get_stake_info_for_coldkey(coldkey_ss58)
            ),
            ttl=settings.cache_ttl_balance,
        )

    async def get_current_block(self) -> int:
        return await self._call(lambda: self._subtensor.get_current_block())

    async def get_all_subnets_info(self):
        return await self._cache.get_or_set(
            "all_subnets",
            lambda: self._call(lambda: self._subtensor.all_subnets()),
            ttl=settings.cache_ttl_dynamic_info,
        )
