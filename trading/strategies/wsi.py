"""Whale Stake-Inflow (W1) — total_stake jump percentile detector.

For each subnet, maintain a 30-day rolling window of per-bar deltas in
``total_stake`` (or ``tao_in`` as fallback when total_stake isn't yet
populated by the collector). At each tick, if the latest delta is above
the ``wsi_percentile``-th percentile of the historical window AND
positive, that's a whale-sized inflow — buy and hold 48h.

This is a true whale signal in spirit: at the subnet level, sudden
stake jumps that fall outside the historical noise band almost always
reflect a single large coldkey moving funds. We can't see WHO moved them
without per-wallet tracking, but we can see WHEN someone did.
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


_DELTAS: dict[int, deque] = {}
_LAST: dict[int, float] = {}


def _push_delta(netuid: int, val: float, maxlen: int) -> Optional[float]:
    prev = _LAST.get(netuid)
    _LAST[netuid] = val
    if prev is None:
        return None
    delta = val - prev
    dq = _DELTAS.setdefault(netuid, deque(maxlen=maxlen))
    dq.append(delta)
    return delta


def _percentile(deltas: deque, pct: float) -> Optional[float]:
    if len(deltas) < 50:
        return None
    sorted_vals = sorted(deltas)
    idx = int(len(sorted_vals) * pct)
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


@register_strategy("wsi")
class WhaleStakeInflowStrategy(Strategy):
    """Buy when stake inflow blows past the historical percentile."""

    def name(self) -> StrategyName:
        return StrategyName.WSI

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        depth = features.pool_depth_tao
        stake_signal = snapshot.total_stake if snapshot.total_stake > 0 else snapshot.tao_in
        delta = _push_delta(netuid, stake_signal, c.wsi_window_bars)
        if delta is None or depth is None:
            return None
        if depth <= c.wsi_min_pool_depth:
            return None
        if delta <= 0:
            return None
        threshold = _percentile(_DELTAS[netuid], c.wsi_percentile)
        if threshold is None or delta < threshold:
            return None
        # Don't chase if already heated.
        pm_24 = features.price_momentum_24h
        if pm_24 is not None and pm_24 > c.wsi_max_entry_pm_24h:
            return None

        ratio = delta / max(threshold, 1e-9)
        strength = min(max((ratio - 1.0) / 2.0 + 0.5, 0.5), 1.0)
        reason = (
            f"WSI entry SN{netuid}: stake Δ={delta:+.1f} > p{c.wsi_percentile*100:.0f}={threshold:+.1f}, "
            f"depth {depth:.0f}"
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
        if hours_held >= c.wsi_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"WSI time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )
        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized = position.unrealized_pnl_pct(snapshot.tao_in, snapshot.alpha_in)
            if unrealized <= c.wsi_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"WSI stop SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
            if unrealized >= c.wsi_take_profit_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"WSI take-profit SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
        return None
