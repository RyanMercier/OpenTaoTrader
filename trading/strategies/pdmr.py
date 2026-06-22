"""Pool-Depth Mean Reversion (PDMR).

Tighter cousin of the built-in mean_reversion strategy: requires a stronger
oversold reading (z < pdmr_entry_z, default -2.0) AND positive pool-depth
growth (tao_in_change_24h > 0). The thesis is that price oversold + LPs
adding liquidity is a much higher-conviction signal than price oversold
alone — LPs adding liquidity into weakness usually means they have a
view, and that view tends to be right.

Exits on z reverting above pdmr_exit_z, hold timeout, or stop-loss.
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


@register_strategy("pdmr")
class PoolDepthMeanReversionStrategy(Strategy):
    """Z < -2 AND liquidity growing → buy. Tighter than vanilla MR."""

    def name(self) -> StrategyName:
        return StrategyName.PDMR

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        z = features.price_zscore_7d
        depth = features.pool_depth_tao
        depth_chg = features.pool_depth_change_24h
        vol = features.price_volatility_7d
        if z is None or depth is None or depth_chg is None or vol is None:
            return None
        if z >= c.pdmr_entry_z:
            return None
        if depth <= c.pdmr_min_pool_depth:
            return None
        if depth_chg < c.pdmr_min_depth_change:
            return None
        if vol <= 0.005:
            return None
        strength = min(max(abs(z) / 3.0, 0.5), 1.0)
        reason = (
            f"PDMR entry SN{netuid}: z={z:.2f}, depth Δ24h={depth_chg*100:+.1f}%, "
            f"vol {vol*100:.2f}%, depth {depth:.0f}"
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
        z = features.price_zscore_7d

        if hours_held >= c.pdmr_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"PDMR time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )
        if z is not None and z > c.pdmr_exit_z:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"PDMR mean-revert exit SN{netuid}: z={z:.2f}",
                features=features.to_dict(),
            )
        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized = position.unrealized_pnl_pct(snapshot.tao_in, snapshot.alpha_in)
            if unrealized <= c.pdmr_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"PDMR stop SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
        return None
