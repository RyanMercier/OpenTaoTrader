"""Ride steady climbs backed by real stake growth, not pure price pumps."""

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


@register_strategy("momentum")
class MomentumStrategy(Strategy):
    """72h price + stake velocity momentum, gated by volatility ceiling."""

    def name(self) -> StrategyName:
        return StrategyName.MOMENTUM

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        pm_72 = features.price_momentum_72h
        sv_72 = features.stake_velocity_72h
        vol = features.price_volatility_7d
        depth = features.pool_depth_tao

        if pm_72 is None or sv_72 is None or vol is None or depth is None:
            return None
        if pm_72 <= c.mo_min_price_gain:
            return None
        if sv_72 <= c.mo_min_velocity:
            return None
        if vol >= c.mo_max_volatility:
            return None
        if depth <= c.min_pool_depth_tao:
            return None

        strength = min(pm_72 / 0.15, 1.0)
        reason = (
            f"Momentum entry on SN{netuid}: price +{pm_72*100:.2f}%/72h, "
            f"stake velocity +{sv_72*100:.2f}%/72h, vol {vol*100:.2f}%"
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
        pm_24 = features.price_momentum_24h

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
        if pm_24 is not None and pm_24 < -0.03:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=min(abs(pm_24) / 0.05, 1.0),
                reason=f"Momentum reversed on SN{netuid}: price {pm_24*100:.2f}%/24h",
                features=features.to_dict(),
            )
        return None
