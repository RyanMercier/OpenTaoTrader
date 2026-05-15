"""Stake-velocity front-running: enter when tao_in is growing faster than
price, betting on the emissions -> stake -> price feedback loop.

Only runs under Taoflow regimes, where tao_in actually drives emissions.
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
    REGIME_TAOFLOW_PREHALVING,
    REGIME_TAOFLOW_POSTHALVING,
)
from . import register_strategy
from .base import Strategy


@register_strategy("stake_velocity")
class StakeVelocityStrategy(Strategy):
    """Bittensor-specific: stake growth leading price under Taoflow."""

    def name(self) -> StrategyName:
        return StrategyName.STAKE_VELOCITY

    def can_run_in_regime(self, regime: str) -> bool:
        return regime in (REGIME_TAOFLOW_PREHALVING, REGIME_TAOFLOW_POSTHALVING)

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        if not self.can_run_in_regime(snapshot.regime):
            return None
        c = self.config
        sv_24 = features.stake_velocity_24h
        pm_24 = features.price_momentum_24h
        div = features.velocity_price_divergence
        sv_72 = features.stake_velocity_72h
        depth = features.pool_depth_tao

        if sv_24 is None or pm_24 is None or div is None or sv_72 is None or depth is None:
            return None
        if sv_24 <= c.sv_velocity_threshold:
            return None
        if pm_24 >= c.sv_price_lag_threshold:
            return None
        if div <= 0:
            return None
        if sv_72 <= 0:
            return None
        if depth <= c.sv_min_pool_depth:
            return None

        strength = min(div / 0.05, 1.0)
        reason = (
            f"Stake velocity {sv_24*100:.2f}% outpacing price {pm_24*100:.2f}% "
            f"on SN{netuid} (divergence: {div*100:.2f}%, 72h velocity: {sv_72*100:.2f}%)"
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
        sv_24 = features.stake_velocity_24h
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
        if sv_24 is not None and sv_24 < -0.02:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=min(abs(sv_24) / 0.05, 1.0),
                reason=f"Pool draining on SN{netuid}: stake velocity {sv_24*100:.2f}%/24h",
                features=features.to_dict(),
            )
        if pm_24 is not None and pm_24 > 0.10:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"Take profit on SN{netuid}: price +{pm_24*100:.2f}% in 24h",
                features=features.to_dict(),
            )
        return None
