"""Live trading. Same signal pipeline as PaperTrader, real extrinsics on
execute.

The CLI launcher owns the wallet: coldkey is decrypted in the CLI process
and the unlocked Wallet is passed in here. The FastAPI server never sees
it. ``_execute_buy`` and ``_execute_sell`` override the paper hooks to
submit ``add_stake`` / ``unstake`` against the chain, with a pre-flight
that refreshes pool reserves, re-checks slippage, and verifies the
coldkey balance before signing. Stake before/after is queried so the
in-memory tracker matches what actually happened on chain.

If realized P&L for the calendar day breaches ``daily_loss_limit_pct``
the trader trips the kill switch: it pauses the portfolio in the DB and
refuses further entries until the operator manually resumes.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from .amm import buy_alpha, sell_alpha
from .config import TradingConfig
from .models import Direction, Position, Trade
from .paper_trader import PaperTrader
from .portfolio import PortfolioTracker

logger = logging.getLogger(__name__)


class LiveTrader(PaperTrader):
    def __init__(
        self,
        portfolio_id: int,
        config: TradingConfig,
        portfolio: PortfolioTracker,
        api_client,
        database,
        chain_client,
        wallet,
        hotkey_name: str | None = None,
    ):
        super().__init__(
            portfolio_id=portfolio_id,
            config=config,
            portfolio=portfolio,
            api_client=api_client,
            database=database,
        )
        # LiveTrader keeps its own chain client for signing extrinsics
        # and for the pre-flight pool refresh + balance check (latency
        # matters here; HTTP would add hops).
        self.chain_client = chain_client
        self.wallet = wallet
        self.hotkey_name = hotkey_name or getattr(wallet, "hotkey_str", None) or "default"
        self._kill_switch_tripped = False
        # Daily P&L tracking (UTC date -> realized loss tao). Reset at midnight.
        self._day_start_value_tao: float | None = None
        self._day_key: str | None = None

    async def _refresh_pool(self, netuid: int) -> tuple[float, float, int]:
        """Force a fresh pool read. The cached value the feature engine
        used can be a few minutes stale; we want real-time state for the
        slippage check."""
        dyn = await self.chain_client.get_dynamic_info(netuid, force_refresh=True)
        block = await self.chain_client.get_current_block()
        return float(dyn.tao_in), float(dyn.alpha_in), int(block)

    async def _coldkey_balance_tao(self) -> float:
        addr = self.wallet.coldkeypub.ss58_address
        bal = await self.chain_client.get_balance(addr)
        return float(bal.tao) if hasattr(bal, "tao") else float(bal)

    async def _stake_alpha(self, netuid: int) -> float:
        """Alpha the configured hotkey currently holds on this netuid.
        Sampled before and after each trade to learn what actually moved."""
        try:
            stakes = await self.chain_client.get_stake_info_for_coldkey(
                self.wallet.coldkeypub.ss58_address
            )
        except Exception:
            return 0.0
        target = self.wallet.hotkey.ss58_address
        for s in stakes or []:
            if int(getattr(s, "netuid", -1)) == netuid and getattr(s, "hotkey_ss58", "") == target:
                return float(s.stake)
        return 0.0

    def _check_kill_switch(self, now: datetime, current_value: float) -> bool:
        """True if today's loss has exceeded the configured cap. Resets at
        UTC midnight; the first call on a new day records the baseline."""
        day = now.strftime("%Y-%m-%d")
        if self._day_key != day:
            self._day_key = day
            self._day_start_value_tao = current_value
            return False
        if self._day_start_value_tao is None or self._day_start_value_tao <= 0:
            return False
        loss_pct = (self._day_start_value_tao - current_value) / self._day_start_value_tao
        return loss_pct >= self.config.daily_loss_limit_pct

    async def _execute_buy(self, signal, amount, snapshot, hotkey_id):
        if self._kill_switch_tripped:
            logger.warning(
                "Live: kill-switch tripped, refusing buy on SN%d", signal.netuid
            )
            return None, {}

        netuid = signal.netuid
        # Pre-flight: refresh pool, re-check slippage at REAL state.
        try:
            tao_in, alpha_in, block = await self._refresh_pool(netuid)
        except Exception as e:
            logger.warning("Live: pool refresh failed for SN%d: %s", netuid, e)
            return None, {}
        if tao_in <= 0 or alpha_in <= 0:
            return None, {}
        sim = buy_alpha(amount, tao_in, alpha_in)
        if sim["slippage_pct"] > self.config.max_slippage_pct:
            logger.info(
                "Live: SN%d slippage %.2f%% over cap, skipping",
                netuid, sim["slippage_pct"] * 100,
            )
            return None, {}

        # Balance check (leave a 0.01 TAO buffer for fees).
        try:
            free = await self._coldkey_balance_tao()
        except Exception as e:
            logger.warning("Live: balance check failed: %s", e)
            return None, {}
        if free < amount + 0.01:
            logger.warning(
                "Live: insufficient free balance %.4f TAO for buy of %.4f",
                free, amount,
            )
            return None, {}

        # Snapshot stake-before, submit, snapshot stake-after.
        before = await self._stake_alpha(netuid)
        ext_hash, ok = await _submit_add_stake(
            self.chain_client, self.wallet, netuid, amount,
        )
        if not ok:
            logger.error("Live: add_stake failed on SN%d", netuid)
            return None, {}
        try:
            after = await self._stake_alpha(netuid)
        except Exception:
            after = before + sim["alpha_received"]  # best-effort

        alpha_received = max(after - before, 0.0)
        if alpha_received <= 0:
            logger.error(
                "Live: extrinsic accepted but no alpha delta on SN%d", netuid
            )
            return None, {}

        # Mutate in-memory tracker to match reality.
        effective_price = amount / alpha_received if alpha_received > 0 else sim["effective_price"]
        spot_price = sim["spot_price"]
        slippage_pct = (effective_price / spot_price - 1.0) if spot_price > 0 else 0.0

        trade = Trade(
            id=str(uuid.uuid4()),
            timestamp=snapshot.timestamp,
            block=block,
            netuid=netuid,
            direction=Direction.BUY,
            strategy=signal.strategy,
            tao_amount=amount,
            alpha_amount=alpha_received,
            spot_price=spot_price,
            effective_price=effective_price,
            slippage_pct=slippage_pct,
            signal_strength=signal.strength,
            hotkey_id=hotkey_id,
        )
        self.portfolio.free_tao -= amount
        self.portfolio.positions[netuid] = Position(
            netuid=netuid,
            entry_time=snapshot.timestamp,
            entry_block=block,
            entry_price=effective_price,
            alpha_amount=alpha_received,
            tao_invested=amount,
            strategy=signal.strategy,
            hotkey_id=hotkey_id,
        )
        self.portfolio.trades.append(trade)
        self.portfolio.hotkey_cooldowns[hotkey_id] = block

        return trade, {"extrinsic_hash": ext_hash, "executed_block": block}

    async def _execute_sell(self, netuid, snapshot, reason, strategy):
        if self._kill_switch_tripped:
            logger.warning(
                "Live: kill-switch tripped, refusing sell on SN%d", netuid
            )
            return None, {}

        position = self.portfolio.positions.get(netuid)
        if position is None:
            return None, {}

        try:
            tao_in, alpha_in, block = await self._refresh_pool(netuid)
        except Exception as e:
            logger.warning("Live: pool refresh failed for SN%d: %s", netuid, e)
            return None, {}
        if tao_in <= 0 or alpha_in <= 0:
            return None, {}
        sim = sell_alpha(position.alpha_amount, tao_in, alpha_in)
        if sim["slippage_pct"] > self.config.max_slippage_pct:
            # Sells get more leeway: closing a position is sometimes
            # urgent (drain detected). Log but continue.
            logger.warning(
                "Live: SN%d sell slippage %.2f%% above max, proceeding anyway",
                netuid, sim["slippage_pct"] * 100,
            )

        before_tao = await self._coldkey_balance_tao()
        ext_hash, ok = await _submit_unstake(
            self.chain_client, self.wallet, netuid, position.alpha_amount,
        )
        if not ok:
            logger.error("Live: unstake failed on SN%d", netuid)
            return None, {}
        try:
            after_tao = await self._coldkey_balance_tao()
        except Exception:
            after_tao = before_tao + sim["tao_received"]

        tao_received = max(after_tao - before_tao, 0.0)
        if tao_received <= 0:
            logger.error(
                "Live: extrinsic accepted but no TAO delta on SN%d", netuid
            )
            return None, {}

        spot_price = sim["spot_price"]
        effective_price = tao_received / position.alpha_amount if position.alpha_amount > 0 else 0
        slippage_pct = (1.0 - effective_price / spot_price) if spot_price > 0 else 0.0
        pnl_tao = tao_received - position.tao_invested
        pnl_pct = (pnl_tao / position.tao_invested) if position.tao_invested > 0 else 0.0
        hold_hours = position.hold_duration_hours(snapshot.timestamp)

        trade = Trade(
            id=str(uuid.uuid4()),
            timestamp=snapshot.timestamp,
            block=block,
            netuid=netuid,
            direction=Direction.SELL,
            strategy=strategy,
            tao_amount=tao_received,
            alpha_amount=position.alpha_amount,
            spot_price=spot_price,
            effective_price=effective_price,
            slippage_pct=slippage_pct,
            signal_strength=1.0,
            hotkey_id=position.hotkey_id,
            entry_price=position.entry_price,
            pnl_tao=pnl_tao,
            pnl_pct=pnl_pct,
            hold_duration_hours=hold_hours,
            entry_strategy=position.strategy,
        )
        self.portfolio.free_tao += tao_received
        del self.portfolio.positions[netuid]
        self.portfolio.trades.append(trade)
        self.portfolio.hotkey_cooldowns[position.hotkey_id] = block

        # Daily kill-switch evaluation after every realized loss/gain.
        now = datetime.now(timezone.utc)
        current_value = self.portfolio.free_tao + sum(
            p.tao_invested for p in self.portfolio.positions.values()
        )
        if self._check_kill_switch(now, current_value):
            self._kill_switch_tripped = True
            await self.database.set_paper_portfolio_active(self.portfolio_id, False)
            logger.error(
                "Live: portfolio %d hit daily loss limit %.1f%%, kill-switch tripped",
                self.portfolio_id, self.config.daily_loss_limit_pct * 100,
            )

        return trade, {"extrinsic_hash": ext_hash, "executed_block": block}


def _alpha_balance(amount: float, netuid: int):
    """Build a Balance carrying ``amount`` of subnet ``netuid``'s alpha
    token. ``set_unit`` is available on bittensor >= 8 and tags the
    Balance so the chain interprets it as alpha rather than TAO."""
    from bittensor.utils.balance import Balance
    bal = Balance.from_tao(amount)
    if hasattr(bal, "set_unit"):
        return bal.set_unit(netuid)
    return bal


async def _submit_add_stake(chain_client, wallet, netuid: int, amount_tao: float) -> tuple[str | None, bool]:
    """Submit add_stake. Returns ``(extrinsic_hash, success)``. The
    extrinsic hash is ``None`` for now; the high-level SDK call only
    returns a bool. To capture the hash we'd drop down to the
    substrate-interface submit_extrinsic path."""
    try:
        from bittensor.utils.balance import Balance
    except Exception:
        logger.exception("bittensor SDK unavailable for live execution")
        return None, False
    try:
        success = await chain_client._subtensor.add_stake(
            wallet=wallet,
            netuid=netuid,
            hotkey_ss58=wallet.hotkey.ss58_address,
            amount=Balance.from_tao(amount_tao),
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        return None, bool(success)
    except Exception:
        logger.exception("add_stake call raised")
        return None, False


async def _submit_unstake(chain_client, wallet, netuid: int, alpha_amount: float) -> tuple[str | None, bool]:
    """Submit unstake. ``alpha_amount`` is in the subnet's alpha token,
    not TAO; we tag the Balance with ``set_unit(netuid)`` so the SDK
    treats it correctly."""
    try:
        success = await chain_client._subtensor.unstake(
            wallet=wallet,
            netuid=netuid,
            hotkey_ss58=wallet.hotkey.ss58_address,
            amount=_alpha_balance(alpha_amount, netuid),
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        return None, bool(success)
    except Exception:
        logger.exception("unstake call raised")
        return None, False
