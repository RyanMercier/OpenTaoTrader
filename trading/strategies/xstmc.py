"""Cross-Sectional Short-Term Momentum Continuation (XSTMC).

Enters when a subnet's 90-minute return is in the top-K across the active
universe at the same tick — concentrating capital on the single best
opportunity rather than firing on every absolute-threshold hit.

The cross-subnet view is built lazily via a module-level per-tick cache:
each call to ``generate_entry_signal`` registers its 90m score, and the
later calls in the same tick can read the full ranking. The cache keys by
snapshot timestamp; stale entries age out as new ticks arrive.

Exits on time target (``stmc_hold_bars`` × 30min), stop-loss, or take-profit.
"""

from __future__ import annotations

from typing import Optional

from ..models import (
    Direction,
    Features,
    Position,
    Signal,
    Snapshot,
    StrategyName,
)
from . import register_strategy
from .base import Strategy


_TICK_CACHE: dict = {"ts": None, "ranks": {}}


def _update_cross_sectional_rank(
    timestamp, netuid: int, score: Optional[float]
) -> None:
    if timestamp != _TICK_CACHE["ts"]:
        _TICK_CACHE["ts"] = timestamp
        _TICK_CACHE["ranks"] = {}
    if score is not None:
        _TICK_CACHE["ranks"][netuid] = score


def _current_rank(netuid: int) -> Optional[tuple[int, int]]:
    ranks = _TICK_CACHE["ranks"]
    if not ranks or netuid not in ranks:
        return None
    ordered = sorted(ranks.items(), key=lambda kv: -kv[1])
    for i, (n, _) in enumerate(ordered):
        if n == netuid:
            return i + 1, len(ordered)
    return None


@register_strategy("xstmc")
class XSTMCStrategy(Strategy):
    """Cross-sectional 90m momentum: enter only if rank <= xstmc_top_k."""

    def name(self) -> StrategyName:
        return StrategyName.XSTMC

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        pm_90 = features.price_momentum_90m
        pm_24 = features.price_momentum_24h
        depth = features.pool_depth_tao

        _update_cross_sectional_rank(snapshot.timestamp, netuid, pm_90)

        if pm_90 is None or depth is None:
            return None
        if pm_90 <= c.stmc_entry_threshold:
            return None
        if depth <= c.stmc_min_pool_depth:
            return None
        if pm_24 is not None and pm_24 > c.stmc_max_entry_pm_24h:
            return None

        rank_info = _current_rank(netuid)
        top_k = getattr(c, "xstmc_top_k", 2)
        if rank_info is not None:
            rank, total = rank_info
            if rank > top_k:
                return None
            rank_strength = 1.0 - (rank - 1) / max(total, 1)
        else:
            rank_strength = 0.5

        rng = max(c.stmc_strong_threshold - c.stmc_entry_threshold, 1e-9)
        raw_strength = min(max((pm_90 - c.stmc_entry_threshold) / rng, 0.0), 1.0)
        strength = max(0.5, min(1.0, 0.5 * raw_strength + 0.5 * rank_strength))

        rank_str = (
            f"rank {rank_info[0]}/{rank_info[1]}" if rank_info else "rank unk"
        )
        reason = (
            f"XSTMC entry SN{netuid}: 90m {pm_90*100:+.2f}% "
            f"({rank_str}), strength {strength:.2f}"
        )
        return Signal(
            timestamp=snapshot.timestamp,
            netuid=netuid,
            direction=Direction.BUY,
            strategy=self.name(),
            strength=strength,
            reason=reason,
            features=features.to_dict(),
        )

    def generate_exit_signal(
        self,
        netuid: int,
        features: Features,
        snapshot: Snapshot,
        position: Position,
    ) -> Optional[Signal]:
        c = self.config
        if position.strategy != self.name():
            return None

        hours_held = position.hold_duration_hours(snapshot.timestamp)
        target_hours = (c.stmc_hold_bars * 30.0) / 60.0
        if hours_held >= target_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"XSTMC time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )

        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized_pct = position.unrealized_pnl_pct(
                snapshot.tao_in, snapshot.alpha_in
            )
            if unrealized_pct <= c.stmc_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"XSTMC stop-loss SN{netuid}: {unrealized_pct*100:+.2f}%",
                    features=features.to_dict(),
                )
            if unrealized_pct >= c.stmc_take_profit_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"XSTMC take-profit SN{netuid}: {unrealized_pct*100:+.2f}%",
                    features=features.to_dict(),
                )

        return None
