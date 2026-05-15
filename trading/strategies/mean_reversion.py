"""Buy statistically cheap subnets with a positive liquidity trend.

Z-score entry, mean-revert exit. The positive-velocity filter stops the
strategy catching falling knives during a death spiral.

This is the simplest non-trivial reference strategy. Copy this file as a
starting point for your own strategies.
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


@register_strategy("mean_reversion")
class MeanReversionStrategy(Strategy):
    """Buy when 7-day z-score is sufficiently negative and stake velocity is non-negative."""

    def name(self) -> StrategyName:
        return StrategyName.MEAN_REVERSION

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        z = features.price_zscore_7d
        depth = features.pool_depth_tao
        vol = features.price_volatility_7d

        if z is None or depth is None or vol is None:
            return None
        if z >= c.mr_entry_zscore:
            return None
        if depth <= c.min_pool_depth_tao:
            return None
        if vol <= 0.01:
            return None
        if c.mr_require_positive_velocity:
            sv = features.stake_velocity_24h
            if sv is None or sv < 0:
                return None

        strength = min(abs(z) / 3.0, 1.0)
        reason = (
            f"Mean reversion entry on SN{netuid}: price z-score {z:.2f} "
            f"(7d vol {vol*100:.2f}%, depth {depth:.1f} TAO)"
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
        self, netuid: int, features: Features, snapshot: Snapshot, position: Position
    ) -> Optional[Signal]:
        c = self.config
        hours_held = position.hold_duration_hours(snapshot.timestamp)
        z = features.price_zscore_7d

        if hours_held >= c.default_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=StrategyName.HOLD_TIMEOUT,
                strength=1.0,
                reason=f"Default hold period ({c.default_hold_hours}h) exceeded on SN{netuid}",
                features=features.to_dict(),
            )
        if z is not None and z > c.mr_exit_zscore:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"Mean reversion exit on SN{netuid}: z-score {z:.2f} back above {c.mr_exit_zscore}",
                features=features.to_dict(),
            )
        return None
