"""Historical replay engine.

The loop is: for each timestamp on the unified timeline, advance each subnet's
snapshot pointer if a new snapshot exists, compute causal features, run exit
strategies first (freeing hotkeys and capital), then run entry strategies in
signal-strength order.

Rules enforced:
  - causality (features use data up to and including the current index)
  - mandatory AMM slippage on all trades
  - per-hotkey cooldowns
  - sells before buys per timestep
  - position sizing = min(all constraints) * signal strength
  - force-close all open positions at the end of the run
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from .config import TradingConfig
from .data import DataLoader
from .features import FeatureEngine
from .models import (
    Direction,
    Features,
    Signal,
    Snapshot,
    StrategyName,
    Trade,
)
from .portfolio import PortfolioTracker
from .risk import RiskManager
from .strategies import (
    STRATEGIES,
    DrainDetector,
    MeanReversionStrategy,
    MomentumStrategy,
    StakeVelocityStrategy,
    load_external_strategies,
)
from .strategies.base import Strategy


# Built-in StrategyName mapping; external strategies live in STRATEGIES
# under their own keys and are looked up directly.
STRATEGY_KEYS = {
    "stake_velocity": StrategyName.STAKE_VELOCITY,
    "mean_reversion": StrategyName.MEAN_REVERSION,
    "momentum": StrategyName.MOMENTUM,
    "drain_exit": StrategyName.DRAIN_EXIT,
}


@dataclass
class BacktestResult:
    config: TradingConfig
    start_time: str
    end_time: str
    regime_filter: Optional[str]
    strategies_used: list[str]

    initial_capital: float
    final_value: float
    total_return_pct: float
    total_return_tao: float
    annualized_return_pct: float
    max_drawdown_pct: float
    max_drawdown_tao: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    total_trades: int
    buy_trades: int
    sell_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    profit_factor: float
    avg_hold_hours: float
    median_hold_hours: float

    strategy_stats: dict = field(default_factory=dict)

    avg_entry_slippage_pct: float = 0.0
    avg_exit_slippage_pct: float = 0.0
    total_slippage_cost_tao: float = 0.0
    max_slippage_trade_pct: float = 0.0

    blocked_trades: int = 0
    blocked_reasons: dict = field(default_factory=dict)

    portfolio_values: list = field(default_factory=list)
    drawdown_series: list = field(default_factory=list)
    trades: list = field(default_factory=list)

    monthly_returns: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["config"] = asdict(self.config)
        d["trades"] = [t.to_dict() if isinstance(t, Trade) else t for t in self.trades]
        return d


class Backtester:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data = DataLoader(config.db_path)
        self.feature_engine = FeatureEngine()
        self.risk = RiskManager(config)
        self.portfolio = PortfolioTracker(config)
        self.strategies: list[Strategy] = []
        self.drain_detector: Optional[DrainDetector] = None
        self._init_strategies()

        self.blocked_trades = 0
        self.blocked_reasons: dict[str, int] = defaultdict(int)

    def _init_strategies(self) -> None:
        # Always include the four built-ins as instances. External
        # strategies are loaded into the registry on demand and added below.
        self.strategies.append(StakeVelocityStrategy(self.config))
        self.strategies.append(MeanReversionStrategy(self.config))
        self.strategies.append(MomentumStrategy(self.config))
        self.drain_detector = DrainDetector(self.config)
        self.strategies.append(self.drain_detector)

        # External strategies registered in the global STRATEGIES dict but
        # not already added above. The registry decorator ran at import.
        builtin_keys = {"stake_velocity", "mean_reversion", "momentum", "drain_exit"}
        if self.config.external_strategy_paths:
            load_external_strategies(":".join(self.config.external_strategy_paths))
        for key, cls in STRATEGIES.items():
            if key in builtin_keys:
                continue
            self.strategies.append(cls(self.config))

    # --------------------------------------------------------------------

    def run(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        netuids: Optional[list[int]] = None,
        regime: Optional[str] = None,
        strategies: Optional[list[str]] = None,
    ) -> BacktestResult:
        if strategies:
            allowed = {STRATEGY_KEYS[s] for s in strategies if s in STRATEGY_KEYS}
            # Always keep drain_detector as a safety exit
            allowed.add(StrategyName.DRAIN_EXIT)
            self.strategies = [s for s in self.strategies if s.name() in allowed]

        # 1. Load data
        all_snapshots = self.data.load_all_snapshots(start=start, end=end, netuids=netuids)

        # 2. Filter subnets
        filtered = {}
        for netuid, snaps in all_snapshots.items():
            if netuid in self.config.exclude_netuids:
                continue
            if len(snaps) < self.config.min_snapshots:
                continue
            latest = snaps[-1]
            if latest.tao_in < self.config.min_pool_depth_tao:
                continue
            filtered[netuid] = snaps
        all_snapshots = filtered

        if not all_snapshots:
            raise RuntimeError("No subnets passed filters, check db and config")

        # 3. Build unified timeline (sorted unique timestamps across all subnets)
        timestamps: set = set()
        for snaps in all_snapshots.values():
            for s in snaps:
                timestamps.add(s.timestamp)
        unified = sorted(timestamps)

        # 4. Maintain per-netuid pointer: index of the latest snapshot at or
        # before the current timestamp.
        pointers: dict[int, int] = {n: -1 for n in all_snapshots}

        started_at = unified[0]
        ended_at = unified[-1]

        # Warmup: skip first min_snapshots ticks so features can stabilize.
        warmup = self.config.min_snapshots

        # Monthly return tracking
        monthly_first_value: dict[str, float] = {}
        monthly_last_value: dict[str, float] = {}

        for tick_idx, ts in enumerate(unified):
            # Advance pointers for each subnet to the latest snapshot at or before ts
            for netuid, snaps in all_snapshots.items():
                p = pointers[netuid]
                while p + 1 < len(snaps) and snaps[p + 1].timestamp <= ts:
                    p += 1
                pointers[netuid] = p

            if tick_idx < warmup:
                continue

            # Build current snapshot dict (for portfolio valuation)
            current_snaps: dict[int, Snapshot] = {}
            for netuid, snaps in all_snapshots.items():
                p = pointers[netuid]
                if p >= 0:
                    current_snaps[netuid] = snaps[p]

            # Apply regime filter, skip ticks outside the desired regime
            if regime is not None:
                if not current_snaps:
                    continue
                # Use any one snapshot's regime (all snapshots at the same tick
                # share the same regime since it's time-based)
                sample = next(iter(current_snaps.values()))
                if sample.regime != regime:
                    # Still record portfolio value for continuity
                    state = self.portfolio.get_state(ts, current_snaps)
                    self._update_monthly(ts, state.total_value_tao, monthly_first_value, monthly_last_value)
                    continue

            # Compute features for each subnet with enough history
            features_map: dict[int, Features] = {}
            all_subnets_context = current_snaps  # snapshot dict for cross-subnet features
            for netuid, snaps in all_snapshots.items():
                p = pointers[netuid]
                if p < 0:
                    continue
                feats = self.feature_engine.compute(snaps, p, all_subnets_context)
                features_map[netuid] = feats
                # Update drain detector streak (chronological per-tick update)
                if self.drain_detector is not None:
                    self.drain_detector.update(netuid, feats)

            # Collect signals
            exit_signals, entry_signals = self._collect_signals(
                features_map, current_snaps
            )

            # --- Execute exits first ---
            # Urgency: drain_exit first, then by strength
            exit_signals.sort(
                key=lambda s: (
                    0 if s.strategy == StrategyName.DRAIN_EXIT else 1,
                    -s.strength,
                )
            )
            exits_executed_netuids: set = set()
            for sig in exit_signals:
                if sig.netuid in exits_executed_netuids:
                    continue
                if sig.netuid not in self.portfolio.positions:
                    continue
                snap = current_snaps.get(sig.netuid)
                if snap is None:
                    continue
                # Rate limit: sells also consume the hotkey
                position = self.portfolio.positions[sig.netuid]
                # The hotkey that owns this position, check its cooldown
                cooldown_end = self.portfolio.hotkey_cooldowns.get(position.hotkey_id, 0) + self.config.blocks_per_cooldown
                if snap.block < cooldown_end:
                    self.blocked_trades += 1
                    self.blocked_reasons[f"exit_cooldown_hk{position.hotkey_id}"] += 1
                    continue
                trade = self.portfolio.execute_sell(
                    sig.netuid, snap, sig.reason, sig.strategy
                )
                if trade is not None:
                    exits_executed_netuids.add(sig.netuid)

            # Also apply force-close from risk manager (max hold)
            for netuid, position in list(self.portfolio.positions.items()):
                if netuid in exits_executed_netuids:
                    continue
                snap = current_snaps.get(netuid)
                feats = features_map.get(netuid)
                if snap is None or feats is None:
                    continue
                should_exit, reason = self.risk.check_exit(position, feats, snap)
                if should_exit:
                    cooldown_end = self.portfolio.hotkey_cooldowns.get(position.hotkey_id, 0) + self.config.blocks_per_cooldown
                    if snap.block < cooldown_end:
                        self.blocked_trades += 1
                        self.blocked_reasons[f"forced_exit_cooldown_hk{position.hotkey_id}"] += 1
                        continue
                    self.portfolio.execute_sell(netuid, snap, reason, StrategyName.HOLD_TIMEOUT)
                    exits_executed_netuids.add(netuid)

            # --- Execute entries (strongest first) ---
            entry_signals.sort(key=lambda s: -s.strength)

            # Snapshot portfolio state for sizing
            state = self.portfolio.get_state(ts, current_snaps)
            for sig in entry_signals:
                snap = current_snaps.get(sig.netuid)
                if snap is None:
                    continue
                allowed, reason, amount = self.risk.check_entry(sig, state, snap)
                if not allowed:
                    self.blocked_trades += 1
                    self.blocked_reasons[f"risk:{reason}"] += 1
                    continue
                hotkey = self.portfolio.get_available_hotkey(snap.block)
                if hotkey is None:
                    self.blocked_trades += 1
                    self.blocked_reasons["no_hotkey_available"] += 1
                    continue
                self.portfolio.execute_buy(sig, amount, snap, hotkey)
                # Refresh state for the next entry
                state = self.portfolio.get_state(ts, current_snaps)

            # Record portfolio state (already done via get_state calls; ensure
            # one final record per tick exists)
            final_state = self.portfolio.get_state(ts, current_snaps)
            self._update_monthly(
                ts, final_state.total_value_tao, monthly_first_value, monthly_last_value
            )

        # 6. Force-close all positions at the last available snapshot
        if self.portfolio.positions:
            # Build the final snapshot dict from latest per-subnet
            final_snaps: dict[int, Snapshot] = {}
            for netuid, snaps in all_snapshots.items():
                if snaps:
                    final_snaps[netuid] = snaps[-1]
            for netuid in list(self.portfolio.positions.keys()):
                snap = final_snaps.get(netuid)
                if snap is None:
                    continue
                self.portfolio.execute_sell(
                    netuid, snap, "end-of-backtest force close", StrategyName.HOLD_TIMEOUT
                )
            # Final state after force-closes
            self.portfolio.get_state(ended_at, final_snaps)

        return self._build_result(started_at, ended_at, regime, monthly_first_value, monthly_last_value)

    # --------------------------------------------------------------------

    def _collect_signals(
        self,
        features_map: dict[int, Features],
        snapshots: dict[int, Snapshot],
    ) -> tuple[list[Signal], list[Signal]]:
        exits: list[Signal] = []
        entries: list[Signal] = []
        for netuid, feats in features_map.items():
            snap = snapshots.get(netuid)
            if snap is None:
                continue
            for strat in self.strategies:
                if not strat.can_run_in_regime(snap.regime):
                    continue
                if netuid in self.portfolio.positions:
                    position = self.portfolio.positions[netuid]
                    sig = strat.generate_exit_signal(netuid, feats, snap, position)
                    if sig is not None:
                        exits.append(sig)
                else:
                    sig = strat.generate_entry_signal(netuid, feats, snap)
                    if sig is not None:
                        entries.append(sig)
        return exits, entries

    def _update_monthly(
        self,
        ts: datetime,
        value: float,
        first_map: dict[str, float],
        last_map: dict[str, float],
    ) -> None:
        key = ts.strftime("%Y-%m")
        if key not in first_map:
            first_map[key] = value
        last_map[key] = value

    def _build_result(
        self,
        start_ts: datetime,
        end_ts: datetime,
        regime_filter: Optional[str],
        monthly_first: dict[str, float],
        monthly_last: dict[str, float],
    ) -> BacktestResult:
        c = self.config
        trades = list(self.portfolio.trades)
        buys = [t for t in trades if t.direction == Direction.BUY]
        sells = [t for t in trades if t.direction == Direction.SELL]

        # P&L is realized on sells
        wins = [t for t in sells if (t.pnl_tao or 0) > 0]
        losses = [t for t in sells if (t.pnl_tao or 0) < 0]

        initial = c.initial_capital_tao
        # Final value: if no positions, free_tao; otherwise value from history
        final_value = self.portfolio.value_history[-1].total_value_tao if self.portfolio.value_history else initial
        total_return_tao = final_value - initial
        total_return_pct = total_return_tao / initial if initial > 0 else 0.0

        # Annualized return
        span_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1.0)
        if span_days > 0 and (1.0 + total_return_pct) > 0:
            annualized = (1.0 + total_return_pct) ** (365.0 / span_days) - 1.0
        else:
            annualized = 0.0

        # Drawdown from value history
        max_dd_pct = 0.0
        max_dd_tao = 0.0
        peak = initial
        drawdown_series: list[tuple[str, float]] = []
        for state in self.portfolio.value_history:
            if state.total_value_tao > peak:
                peak = state.total_value_tao
            dd_tao = state.total_value_tao - peak
            dd_pct = dd_tao / peak if peak > 0 else 0.0
            if dd_pct < max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_tao = dd_tao
            drawdown_series.append((state.timestamp.isoformat(), dd_pct))

        # Sharpe / Sortino, from 30-min portfolio returns, annualized (48*365 steps/yr)
        values = [s.total_value_tao for s in self.portfolio.value_history]
        returns: list[float] = []
        for i in range(1, len(values)):
            prev = values[i - 1]
            if prev > 0:
                returns.append(values[i] / prev - 1.0)
        sharpe = _annualized_sharpe(returns, periods_per_year=48 * 365)
        sortino = _annualized_sortino(returns, periods_per_year=48 * 365)
        calmar = (annualized / abs(max_dd_pct)) if max_dd_pct < 0 else 0.0

        # Trade stats
        sell_returns = [(t.pnl_pct or 0.0) for t in sells]
        win_returns = [t.pnl_pct or 0.0 for t in wins]
        loss_returns = [t.pnl_pct or 0.0 for t in losses]
        win_rate = (len(wins) / len(sells)) if sells else 0.0
        avg_win = (sum(win_returns) / len(win_returns)) if win_returns else 0.0
        avg_loss = (sum(loss_returns) / len(loss_returns)) if loss_returns else 0.0
        best_trade = max(sell_returns) if sell_returns else 0.0
        worst_trade = min(sell_returns) if sell_returns else 0.0
        gross_wins = sum(t.pnl_tao for t in wins if t.pnl_tao is not None)
        gross_losses = -sum(t.pnl_tao for t in losses if t.pnl_tao is not None)
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)
        holds = [t.hold_duration_hours or 0.0 for t in sells]
        avg_hold = (sum(holds) / len(holds)) if holds else 0.0
        median_hold = _median(holds) if holds else 0.0

        # Per-strategy stats
        per_strategy: dict[str, dict] = defaultdict(lambda: {
            "buys": 0, "sells": 0, "wins": 0, "losses": 0,
            "total_pnl_tao": 0.0, "avg_return_pct": 0.0, "win_rate": 0.0,
        })
        strat_sell_pnls: dict[str, list[float]] = defaultdict(list)
        for t in buys:
            per_strategy[t.strategy.value]["buys"] += 1
        for t in sells:
            # Attribute sells to the originating entry strategy, not the
            # strategy that triggered the exit. Falls back to t.strategy if
            # we lost the entry_strategy reference.
            key = (t.entry_strategy or t.strategy).value
            per_strategy[key]["sells"] += 1
            if (t.pnl_tao or 0) > 0:
                per_strategy[key]["wins"] += 1
            elif (t.pnl_tao or 0) < 0:
                per_strategy[key]["losses"] += 1
            per_strategy[key]["total_pnl_tao"] += (t.pnl_tao or 0.0)
            strat_sell_pnls[key].append(t.pnl_pct or 0.0)
        for key, stats in per_strategy.items():
            pnls = strat_sell_pnls[key]
            if pnls:
                stats["avg_return_pct"] = sum(pnls) / len(pnls)
            if stats["sells"] > 0:
                stats["win_rate"] = stats["wins"] / stats["sells"]

        # Slippage stats
        entry_slips = [t.slippage_pct for t in buys]
        exit_slips = [t.slippage_pct for t in sells]
        avg_entry_slip = (sum(entry_slips) / len(entry_slips)) if entry_slips else 0.0
        avg_exit_slip = (sum(exit_slips) / len(exit_slips)) if exit_slips else 0.0
        # Approximate total slippage cost: entry_slip_pct * tao_amount for buys,
        # exit_slip_pct * (tao_received / (1 - slip)) - tao_received for sells.
        slip_cost = 0.0
        max_slip = 0.0
        for t in buys:
            slip_cost += t.tao_amount * max(t.slippage_pct, 0.0)
            max_slip = max(max_slip, t.slippage_pct)
        for t in sells:
            if t.slippage_pct < 1.0:
                slip_cost += (t.tao_amount / (1 - t.slippage_pct) - t.tao_amount) if t.slippage_pct > 0 else 0.0
            max_slip = max(max_slip, t.slippage_pct)

        # Portfolio value time series
        portfolio_values = [
            (s.timestamp.isoformat(), s.total_value_tao)
            for s in self.portfolio.value_history
        ]

        # Monthly returns
        monthly_returns: dict[str, float] = {}
        for month, first_v in monthly_first.items():
            last_v = monthly_last.get(month, first_v)
            if first_v > 0:
                monthly_returns[month] = (last_v - first_v) / first_v

        return BacktestResult(
            config=c,
            start_time=start_ts.isoformat(),
            end_time=end_ts.isoformat(),
            regime_filter=regime_filter,
            strategies_used=[s.name().value for s in self.strategies],
            initial_capital=initial,
            final_value=final_value,
            total_return_pct=total_return_pct,
            total_return_tao=total_return_tao,
            annualized_return_pct=annualized,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_tao=max_dd_tao,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=len(trades),
            buy_trades=len(buys),
            sell_trades=len(sells),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            best_trade_pct=best_trade,
            worst_trade_pct=worst_trade,
            profit_factor=profit_factor,
            avg_hold_hours=avg_hold,
            median_hold_hours=median_hold,
            strategy_stats=dict(per_strategy),
            avg_entry_slippage_pct=avg_entry_slip,
            avg_exit_slippage_pct=avg_exit_slip,
            total_slippage_cost_tao=slip_cost,
            max_slippage_trade_pct=max_slip,
            blocked_trades=self.blocked_trades,
            blocked_reasons=dict(self.blocked_reasons),
            portfolio_values=portfolio_values,
            drawdown_series=drawdown_series,
            trades=trades,
            monthly_returns=monthly_returns,
        )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _annualized_sharpe(returns: list[float], periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def _annualized_sortino(returns: list[float], periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return 0.0
    d_mean = 0.0
    d_var = sum((r - d_mean) ** 2 for r in downside) / len(downside)
    d_std = math.sqrt(d_var) if d_var > 0 else 0.0
    if d_std == 0:
        return 0.0
    return (mean / d_std) * math.sqrt(periods_per_year)
