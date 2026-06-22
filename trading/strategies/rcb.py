"""Range Compression Breakout (RCB).

Looks for subnets whose 24h price range has compressed to a small fraction
of the 7d range (squeeze), then enters when price breaks above the 24h high.

This is a classic technical-analysis pattern. Volatility compression
precedes volatility expansion; if you catch the break in the right
direction you ride the expansion before the rest of the market notices.

Per-subnet rolling 24h and 7d high/low ring buffers live in module-level
caches. The strategy gets called once per netuid per tick.
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


# 48 bars × 30min = 24h; 336 bars × 30min = 7d.
_24H_BARS = 48
_7D_BARS = 336

# Per-subnet price ring buffers.
_PRICE_HIST: dict[int, deque] = {}


def _push_price(netuid: int, price: float) -> None:
    dq = _PRICE_HIST.setdefault(netuid, deque(maxlen=_7D_BARS))
    dq.append(price)


def _range(prices, lookback: int) -> Optional[tuple[float, float]]:
    if len(prices) < lookback:
        return None
    window = list(prices)[-lookback:]
    return min(window), max(window)


@register_strategy("rcb")
class RangeCompressionBreakoutStrategy(Strategy):
    """Squeeze → break above 24h high → ride expansion."""

    def name(self) -> StrategyName:
        return StrategyName.RCB

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        depth = features.pool_depth_tao
        price = snapshot.alpha_price_tao
        _push_price(netuid, price)

        if depth is None or depth <= c.rcb_min_pool_depth:
            return None
        if price <= 0:
            return None
        hist = _PRICE_HIST[netuid]
        r24 = _range(hist, _24H_BARS)
        r7d = _range(hist, _7D_BARS)
        if r24 is None or r7d is None:
            return None
        lo24, hi24 = r24
        lo7d, hi7d = r7d
        if hi7d <= lo7d:
            return None
        # Compression ratio: 24h range as a fraction of 7d range. <0.3 = squeezed.
        rng24 = hi24 - lo24
        rng7d = hi7d - lo7d
        compression = rng24 / rng7d
        if compression > c.rcb_compression_max:
            return None
        # Breakout filter: current price must be at or above the 24h high.
        # Allow a small tolerance band so we don't miss it by one bar.
        if price < hi24 * (1.0 - c.rcb_breakout_tolerance):
            return None
        # Also require that the breakout level is above the 7d midpoint —
        # otherwise we're just bouncing in a downtrend.
        mid7d = (hi7d + lo7d) / 2.0
        if price < mid7d:
            return None

        # Stronger compression = stronger signal.
        strength = max(0.5, min(1.0, 1.0 - compression / c.rcb_compression_max))
        reason = (
            f"RCB entry SN{netuid}: compression {compression:.2f}, "
            f"price {price:.4f} ≥ 24h hi {hi24:.4f}, 7d range [{lo7d:.4f},{hi7d:.4f}]"
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
        if hours_held >= c.rcb_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"RCB time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )
        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized = position.unrealized_pnl_pct(snapshot.tao_in, snapshot.alpha_in)
            if unrealized <= c.rcb_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"RCB stop SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
            if unrealized >= c.rcb_take_profit_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"RCB take-profit SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
        return None
