"""Emission-Yield Carry (EYC).

For each subnet, yield = emission_rate / total_stake. Subnets in the top
``eyc_top_k`` of the cross-sectional yield distribution at each tick are
treated as buyable; the highest yield gets the strongest signal.

Hypothesis: subnets with the highest emission-per-staked-tao rate are
under-staked relative to their rewards stream, and tend to attract stake
inflows over the following 3-7 days as participants rebalance.

The cross-sectional view is built lazily via a per-tick cache, mirroring
xstmc's pattern: each call registers its yield score, later calls in the
same tick read the full ranking.
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


_TICK: dict = {"ts": None, "yields": {}}


def _push_yield(ts, netuid: int, yld: Optional[float]) -> None:
    if ts != _TICK["ts"]:
        _TICK["ts"] = ts
        _TICK["yields"] = {}
    if yld is not None and yld > 0:
        _TICK["yields"][netuid] = yld


def _rank_of(netuid: int) -> Optional[tuple[int, int]]:
    yields = _TICK["yields"]
    if netuid not in yields:
        return None
    ordered = sorted(yields.items(), key=lambda kv: -kv[1])
    for i, (n, _) in enumerate(ordered):
        if n == netuid:
            return i + 1, len(ordered)
    return None


@register_strategy("eyc")
class EmissionYieldCarryStrategy(Strategy):
    """Buy top-K yield subnets; hold 3-7d unless stop hits."""

    def name(self) -> StrategyName:
        return StrategyName.EYC

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        depth = features.pool_depth_tao
        if snapshot.total_stake <= 0 or snapshot.emission_rate <= 0:
            _push_yield(snapshot.timestamp, netuid, None)
            return None
        yld = snapshot.emission_rate / snapshot.total_stake
        _push_yield(snapshot.timestamp, netuid, yld)

        if depth is None or depth <= c.eyc_min_pool_depth:
            return None
        rank_info = _rank_of(netuid)
        if rank_info is None:
            return None
        rank, total = rank_info
        if rank > c.eyc_top_k:
            return None
        if total < c.eyc_min_universe:
            return None

        # Skip if already overheated short-term — let it cool.
        pm_24 = features.price_momentum_24h
        if pm_24 is not None and pm_24 > c.eyc_max_entry_pm_24h:
            return None

        rank_strength = 1.0 - (rank - 1) / max(c.eyc_top_k, 1)
        strength = max(0.5, min(1.0, rank_strength))
        reason = (
            f"EYC entry SN{netuid}: yield={yld*1e6:.2f}e-6, rank {rank}/{total}"
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
        if hours_held >= c.eyc_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"EYC time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )
        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized = position.unrealized_pnl_pct(snapshot.tao_in, snapshot.alpha_in)
            if unrealized <= c.eyc_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"EYC stop SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
            if unrealized >= c.eyc_take_profit_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"EYC take-profit SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )

        # Yield rerank exit: if we drop out of top-K, close.
        if snapshot.total_stake > 0 and snapshot.emission_rate > 0:
            cur_yld = snapshot.emission_rate / snapshot.total_stake
            _push_yield(snapshot.timestamp, netuid, cur_yld)
            rank_info = _rank_of(netuid)
            if rank_info is not None and rank_info[0] > c.eyc_top_k * 2:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"EYC rank-exit SN{netuid}: dropped to {rank_info[0]}/{rank_info[1]}",
                    features=features.to_dict(),
                )
        return None
