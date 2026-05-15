"""Paper-trading runner.

One ``PaperTrader`` per portfolio, kept alive by the FastAPI lifespan
task in OpenTaoTrader's server. Each cycle: pull recent snapshots from
the upstream OpenTaoAPI, compute features, run the configured
strategies, simulate buys and sells against AMM math, write trades and
a value-history row to SQLite.

State lives in ``paper_portfolios`` / ``paper_positions`` /
``paper_trades`` / ``paper_value_history`` in the trader's own SQLite.
``PortfolioTracker`` is in-memory and is rehydrated from those tables
on startup; persistence is incremental so a crash mid-cycle loses at
most the trades from that cycle.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from .config import TradingConfig
from .features import FeatureEngine
from .models import (
    Features,
    Position,
    Signal,
    Snapshot,
    StrategyName,
    Trade,
    get_regime,
)
from .portfolio import PortfolioTracker
from .risk import RiskManager
from .strategies import (
    DrainDetector,
    STRATEGIES,
    load_external_strategies,
)
from .strategies.base import Strategy

logger = logging.getLogger(__name__)


class PaperTrader:
    """One paper-trading runner per portfolio. The instance is long-lived
    inside the FastAPI lifespan task; ``run_once`` advances by one cycle.
    """

    def __init__(
        self,
        portfolio_id: int,
        config: TradingConfig,
        portfolio: PortfolioTracker,
        api_client,
        database,
    ):
        self.portfolio_id = portfolio_id
        self.config = config
        self.portfolio = portfolio
        self.api_client = api_client
        self.database = database

        self.feature_engine = FeatureEngine()
        self.risk = RiskManager(config)
        self.strategies: list[Strategy] = []
        self.drain_detector: Optional[DrainDetector] = None
        self._init_strategies()

    def _init_strategies(self) -> None:
        if self.config.external_strategy_paths:
            load_external_strategies(":".join(self.config.external_strategy_paths))

        # drain_exit always runs as a safety net regardless of config.
        wanted = set(self.config.strategies or [])
        wanted.add("drain_exit")

        for key in sorted(wanted):
            cls = STRATEGIES.get(key)
            if cls is None:
                logger.warning(
                    "Paper portfolio %d: strategy %s not registered, skipping",
                    self.portfolio_id, key,
                )
                continue
            inst = cls(self.config)
            if isinstance(inst, DrainDetector):
                self.drain_detector = inst
            self.strategies.append(inst)

        logger.info(
            "Paper portfolio %d strategies: %s",
            self.portfolio_id,
            [s.name().value for s in self.strategies],
        )

    async def run_once(self) -> dict:
        """Advance one trading cycle. Returns a small status dict for
        the runner to log. Errors propagate to the caller."""
        # Bail if the upstream API's poller is stale; trading on dead
        # data is worse than skipping the cycle.
        stale, detail = await self.api_client.is_stale()
        if stale:
            logger.info("Skipping cycle, upstream API is stale: %s", detail)
            return {"skipped": "upstream_stale"}

        history = await self._load_history()
        if not history:
            logger.info(
                "Paper portfolio %d: no subnet history yet, skipping cycle",
                self.portfolio_id,
            )
            return {"skipped": "no_history"}

        current_snaps: dict[int, Snapshot] = {n: buf[-1] for n, buf in history.items()}

        features_map: dict[int, Features] = {}
        for netuid, buf in history.items():
            if len(buf) < self.config.min_snapshots:
                continue
            feats = self.feature_engine.compute(buf, len(buf) - 1, current_snaps)
            features_map[netuid] = feats
            if self.drain_detector is not None:
                self.drain_detector.update(netuid, feats)

        exits: list[Signal] = []
        entries: list[Signal] = []
        for netuid, feats in features_map.items():
            snap = current_snaps[netuid]
            for strat in self.strategies:
                if not strat.can_run_in_regime(snap.regime):
                    continue
                if netuid in self.portfolio.positions:
                    sig = strat.generate_exit_signal(
                        netuid, feats, snap, self.portfolio.positions[netuid]
                    )
                    if sig is not None:
                        exits.append(sig)
                else:
                    sig = strat.generate_entry_signal(netuid, feats, snap)
                    if sig is not None:
                        entries.append(sig)

        # Sells first (drain_exit ahead of others), then buys by strength.
        exits.sort(
            key=lambda s: (
                0 if s.strategy == StrategyName.DRAIN_EXIT else 1,
                -s.strength,
            )
        )
        executed: list[tuple[Trade, dict]] = []
        exited = set()
        for sig in exits:
            if sig.netuid in exited or sig.netuid not in self.portfolio.positions:
                continue
            snap = current_snaps.get(sig.netuid)
            if snap is None:
                continue
            position = self.portfolio.positions[sig.netuid]
            cooldown_end = (
                self.portfolio.hotkey_cooldowns.get(position.hotkey_id, 0)
                + self.config.blocks_per_cooldown
            )
            if snap.block < cooldown_end:
                continue
            trade, meta = await self._execute_sell(
                sig.netuid, snap, sig.reason, sig.strategy
            )
            if trade is not None:
                executed.append((trade, meta))
                exited.add(sig.netuid)

        entries.sort(key=lambda s: -s.strength)
        now = datetime.now(timezone.utc)
        state = self.portfolio.get_state(now, current_snaps)
        for sig in entries:
            snap = current_snaps.get(sig.netuid)
            if snap is None:
                continue
            allowed, reason, amount = self.risk.check_entry(sig, state, snap)
            if not allowed:
                continue
            hotkey = self.portfolio.get_available_hotkey(snap.block)
            if hotkey is None:
                continue
            trade, meta = await self._execute_buy(sig, amount, snap, hotkey)
            if trade is not None:
                executed.append((trade, meta))
                state = self.portfolio.get_state(now, current_snaps)

        for trade, meta in executed:
            await self.database.insert_paper_trade(
                self.portfolio_id, trade,
                extrinsic_hash=meta.get("extrinsic_hash"),
                executed_block=meta.get("executed_block"),
            )
        await self.database.replace_paper_positions(
            self.portfolio_id, self.portfolio.positions
        )
        await self.database.insert_paper_value_history(
            self.portfolio_id,
            timestamp=now.isoformat(),
            free_tao=state.free_tao,
            total_value_tao=state.total_value_tao,
            total_pnl_tao=state.total_pnl_tao,
            drawdown_pct=state.drawdown_pct,
            num_open_positions=len(state.positions),
        )
        await self.database.update_paper_portfolio_runtime(
            self.portfolio_id,
            peak_value=self.portfolio.peak_value,
            free_tao=self.portfolio.free_tao,
            hotkey_cooldowns=self.portfolio.hotkey_cooldowns,
            last_cycle_at=now.isoformat(),
        )

        return {
            "trades": len(executed),
            "open_positions": len(self.portfolio.positions),
            "value_tao": state.total_value_tao,
            "pnl_pct": state.total_pnl_pct,
        }

    # Execute hooks. LiveTrader overrides these to submit real
    # extrinsics; the meta dict carries the on-chain fields when present.

    async def _execute_buy(self, signal, amount, snapshot, hotkey_id):
        trade = self.portfolio.execute_buy(signal, amount, snapshot, hotkey_id)
        return trade, {}

    async def _execute_sell(self, netuid, snapshot, reason, strategy):
        trade = self.portfolio.execute_sell(netuid, snapshot, reason, strategy)
        return trade, {}

    async def _load_history(self) -> dict[int, list[Snapshot]]:
        """Pull recent subnet snapshots from the upstream API. The
        api_client keeps a rolling buffer fed by SSE so subsequent
        cycles get sub-second freshness; we just snapshot it under the
        client's lock."""
        async with self.api_client._buffer_lock:
            buffer_copy = {
                n: list(buf) for n, buf in self.api_client.snapshot_buffer.items()
            }

        history: dict[int, list[Snapshot]] = {}
        for n, rows in buffer_copy.items():
            if n in self.config.exclude_netuids:
                continue
            snaps = [_row_to_snapshot(r, n) for r in rows]
            if snaps:
                history[n] = snaps
        return history


def _parse_ts(s: str) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)


def _row_to_snapshot(row: dict, netuid: int) -> Snapshot:
    """Convert a raw row from the OpenTaoAPI history or SSE feed into a
    Snapshot. SSE events carry fewer fields than history rows, so the
    metagraph counters default to 0 when missing."""
    ts_str = row.get("timestamp", "") or ""
    return Snapshot(
        block=int(row.get("block", 0) or 0),
        timestamp=_parse_ts(ts_str),
        netuid=netuid,
        alpha_price_tao=float(row.get("alpha_price_tao", 0.0) or 0.0),
        tao_price_usd=float(row.get("tao_price_usd", 0.0) or 0.0),
        tao_in=float(row.get("tao_in", 0.0) or 0.0),
        alpha_in=float(row.get("alpha_in", 0.0) or 0.0),
        total_stake=float(row.get("total_stake", 0.0) or 0.0),
        emission_rate=float(row.get("emission_rate", 0.0) or 0.0),
        validator_count=int(row.get("validator_count", 0) or 0),
        neuron_count=int(row.get("neuron_count", 0) or 0),
        regime=get_regime(ts_str),
    )


async def hydrate_portfolio(database, portfolio_id: int, config: TradingConfig) -> PortfolioTracker:
    """Build a fresh PortfolioTracker from the DB. Loads runtime fields
    (free_tao, peak_value, hotkey_cooldowns) and open positions. Trades
    stay in the DB and are queried for reporting; we don't pull the full
    log into memory."""
    portfolio = PortfolioTracker(config)
    runtime = await database.get_paper_portfolio_runtime(portfolio_id)
    if runtime:
        if runtime.get("free_tao") is not None:
            portfolio.free_tao = float(runtime["free_tao"])
        if runtime.get("peak_value") is not None:
            portfolio.peak_value = float(runtime["peak_value"])
        cooldowns_json = runtime.get("hotkey_cooldowns_json")
        if cooldowns_json:
            try:
                cd = json.loads(cooldowns_json)
                portfolio.hotkey_cooldowns = {int(k): int(v) for k, v in cd.items()}
            except Exception:
                pass

    rows = await database.list_paper_positions(portfolio_id)
    for row in rows:
        try:
            portfolio.positions[int(row["netuid"])] = Position(
                netuid=int(row["netuid"]),
                entry_time=_parse_ts(row["entry_time"]),
                entry_block=int(row["entry_block"]),
                entry_price=float(row["entry_price"]),
                alpha_amount=float(row["alpha_amount"]),
                tao_invested=float(row["tao_invested"]),
                strategy=_strategy_from_value(row["strategy"]),
                hotkey_id=int(row["hotkey_id"]),
            )
        except Exception:
            logger.exception("Skipping malformed position row %s", row)
    return portfolio


def _strategy_from_value(value: str) -> StrategyName:
    """Map a stored string back to a StrategyName. External keys fall
    back to ``EXTERNAL``; their original string is kept in the trade row
    for attribution."""
    try:
        return StrategyName(value)
    except ValueError:
        return StrategyName.EXTERNAL
