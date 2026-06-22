"""Liquidity-Adjusted Momentum (LAM).

Standard 24h momentum, but the signal strength is weighted by
1/sqrt(slippage_for_one_TAO). The hypothesis: in deep pools the momentum
signal is reliable (clean order flow); in thin pools it's dominated by
single-trader noise and should be discounted.

We compute slippage_for_one_tao analytically from the AMM constant-product
formula (no extra RPC). The strategy is otherwise a vanilla intraday
momentum trade with a time-based exit and a stop-loss.
"""

from __future__ import annotations

from math import sqrt
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


def _slippage_pct_for_one_tao(tao_in: float, alpha_in: float) -> Optional[float]:
    """Slippage paid to buy 1 TAO worth of alpha at current pool state.
    Constant-product: new_alpha_in = k / (tao_in + 1); alpha_received =
    alpha_in - new_alpha_in. Effective price = 1 / alpha_received.
    Returns slippage_pct = (effective - spot) / spot."""
    if tao_in <= 0 or alpha_in <= 0:
        return None
    spot = tao_in / alpha_in  # TAO per alpha
    k = tao_in * alpha_in
    new_tao_in = tao_in + 1.0
    new_alpha_in = k / new_tao_in
    alpha_recv = alpha_in - new_alpha_in
    if alpha_recv <= 0:
        return None
    effective = 1.0 / alpha_recv
    return (effective - spot) / spot


@register_strategy("lam")
class LiquidityAdjustedMomentumStrategy(Strategy):
    """24h momentum × 1/sqrt(slippage) for cleaner signal in deep pools."""

    def name(self) -> StrategyName:
        return StrategyName.LAM

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        pm = features.price_momentum_24h
        depth = features.pool_depth_tao
        vol = features.price_volatility_7d
        if pm is None or depth is None or vol is None:
            return None
        if depth <= c.lam_min_pool_depth:
            return None
        if pm <= c.lam_entry_threshold:
            return None
        if vol > c.lam_max_vol:
            return None

        slip = _slippage_pct_for_one_tao(snapshot.tao_in, snapshot.alpha_in)
        if slip is None or slip <= 0:
            return None
        # Weight: 1/sqrt(slip) normalized so that a 1% slippage gives weight 1.0
        # Deep pool (lower slip) → higher weight; thin pool → lower weight.
        liq_weight = min(sqrt(0.01 / slip), 1.5)

        raw = min(max(pm / c.lam_strong_threshold, 0.0), 1.0)
        strength = max(0.5, min(1.0, raw * liq_weight))

        reason = (
            f"LAM entry SN{netuid}: 24h pm {pm*100:+.2f}%, slip(1τ) {slip*100:.3f}%, "
            f"liq_weight {liq_weight:.2f}, depth {depth:.0f}"
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
        if hours_held >= c.lam_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"LAM time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )
        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized = position.unrealized_pnl_pct(snapshot.tao_in, snapshot.alpha_in)
            if unrealized <= c.lam_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"LAM stop SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
            if unrealized >= c.lam_take_profit_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"LAM take-profit SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
        return None
