"""Stake-Flow Breakout (SFB).

Detects abnormally large stake inflows into a subnet — defined as a
total_stake delta whose z-score against a rolling 7d window of deltas is
above ``sfb_entry_z``. The thesis is that whales who add stake before
price reacts are positioning ahead of news/utility/halving moves; we ride
the resulting price drift for a 24-48h window.

Per-subnet rolling deltas are tracked in a module-level cache keyed by
netuid. The cache is bounded; old entries fall out as new snapshots arrive.
Entries are sized by signal z-score, capped at strength=1.0 at z=4.
"""

from __future__ import annotations

from collections import deque
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


# Per-subnet ring buffer of (timestamp, total_stake_delta) over the last
# sfb_window_bars samples. Keys are netuid.
_STAKE_DELTAS: dict[int, deque] = {}
_LAST_STAKE: dict[int, float] = {}


def _push_delta(netuid: int, ts, stake: float, window: int) -> Optional[float]:
    """Record the latest stake reading; return the per-bar delta, or None
    if this is the first observation. Maintains a per-subnet deque sized
    to ``window`` so older deltas drop off automatically."""
    prev = _LAST_STAKE.get(netuid)
    _LAST_STAKE[netuid] = stake
    if prev is None:
        return None
    delta = stake - prev
    dq = _STAKE_DELTAS.setdefault(netuid, deque(maxlen=window))
    dq.append(delta)
    return delta


def _delta_zscore(netuid: int, delta: float) -> Optional[float]:
    dq = _STAKE_DELTAS.get(netuid)
    if dq is None or len(dq) < 20:
        return None
    n = len(dq)
    mean = sum(dq) / n
    var = sum((d - mean) ** 2 for d in dq) / n
    std = var ** 0.5
    if std <= 0:
        return None
    return (delta - mean) / std


@register_strategy("sfb")
class StakeFlowBreakoutStrategy(Strategy):
    """Buy when total_stake inflow z-score breaches sfb_entry_z."""

    def name(self) -> StrategyName:
        return StrategyName.SFB

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        depth = features.pool_depth_tao
        # Fall back to tao_in (AMM-side staked TAO) when total_stake is
        # unpopulated — historical rows had 0 in that column. tao_in delta
        # IS a valid stake-flow proxy: a sudden tao_in jump = somebody
        # adding alpha at a fast clip, which is the same signal.
        stake_signal = snapshot.total_stake if snapshot.total_stake > 0 else snapshot.tao_in
        delta = _push_delta(netuid, snapshot.timestamp, stake_signal, c.sfb_window_bars)
        if delta is None or depth is None:
            return None
        if depth <= c.sfb_min_pool_depth:
            return None
        # Reject sells; we want inflows only.
        if delta <= 0:
            return None
        z = _delta_zscore(netuid, delta)
        if z is None or z < c.sfb_entry_z:
            return None
        # Chase guard: skip if we already missed a big move.
        pm_24 = features.price_momentum_24h
        if pm_24 is not None and pm_24 > c.sfb_max_entry_pm_24h:
            return None

        strength = min(max((z - c.sfb_entry_z) / 2.0 + 0.5, 0.5), 1.0)
        reason = (
            f"SFB entry SN{netuid}: stake Δ={delta:+.1f} TAO, z={z:.2f} "
            f"(depth {depth:.0f}, pm24h {0 if pm_24 is None else pm_24*100:+.1f}%)"
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
        if hours_held >= c.sfb_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"SFB time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )
        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized = position.unrealized_pnl_pct(snapshot.tao_in, snapshot.alpha_in)
            if unrealized <= c.sfb_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"SFB stop SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
            if unrealized >= c.sfb_take_profit_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"SFB take-profit SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
        return None
