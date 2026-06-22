"""RLB — RL-Baseline Trend Follower.

Port of the trend-following baseline from
https://github.com/ZiadFrancis/Reinforcement_Trading_Part_2

That repo trains a PPO agent on XAUUSD M1 data with a MultiDiscrete
action space (direction × SL bucket × TP bucket). The agent is judged
against a tunable trend-following baseline whose logic is straightforward
and well-suited to a non-ML port. Translated to our setting:

  - LONG signal: short EMA above long EMA AND current price is far above
    the short EMA in volatility-normalized units.
  - Hold the position until take-profit, stop-loss, max-hold, or until the
    trend flips (EMA crossover the other way).
  - Long-only (the AMM doesn't support shorts).

The "ATR" in the original is replaced with realized-volatility × price as
a TAO-equivalent volatility scale — we don't have OHLC bars per snapshot.
The TP/SL brackets correspond to the action-space SL/TP indices in the
PPO env, just hardcoded to one bucket pair rather than chosen by the
policy. Run with rlb_threshold sweeps to find the best bucket per regime.

Per-subnet EMAs are tracked in a module-level cache and computed
incrementally with a standard recursive formula so the per-tick cost
stays O(1) per subnet.
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


# Per-subnet EMA cache. We track two windows (short/long) and the
# total number of samples seen, so we can apply the recursive EMA update.
_EMA: dict[int, dict] = {}


def _ema_update(netuid: int, price: float, short_span: int, long_span: int) -> tuple[Optional[float], Optional[float]]:
    """Update both EMAs for this subnet with the new price; return (short, long).
    Both EMAs need >= span observations before they are considered "warm";
    we return None for an EMA that isn't warm yet."""
    state = _EMA.setdefault(netuid, {"n": 0, "short": None, "long": None})
    state["n"] += 1
    alpha_s = 2.0 / (short_span + 1)
    alpha_l = 2.0 / (long_span + 1)
    if state["short"] is None:
        state["short"] = price
    else:
        state["short"] = alpha_s * price + (1 - alpha_s) * state["short"]
    if state["long"] is None:
        state["long"] = price
    else:
        state["long"] = alpha_l * price + (1 - alpha_l) * state["long"]
    short_v = state["short"] if state["n"] >= short_span else None
    long_v = state["long"] if state["n"] >= long_span else None
    return short_v, long_v


@register_strategy("rlb")
class RLBTrendStrategy(Strategy):
    """EMA crossover + vol-normalized distance, with TP/SL bracket exits."""

    def name(self) -> StrategyName:
        return StrategyName.RLB

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        c = self.config
        price = snapshot.alpha_price_tao
        depth = features.pool_depth_tao
        vol = features.price_volatility_7d
        if price <= 0:
            return None
        ema_s, ema_l = _ema_update(netuid, price, c.rlb_ema_short, c.rlb_ema_long)
        if depth is None or depth <= c.rlb_min_pool_depth:
            return None
        if vol is None or vol <= 0:
            return None
        if ema_s is None or ema_l is None:
            return None
        # ATR proxy: realized vol × price gives a TAO-denominated scale.
        atr_proxy = vol * price
        if atr_proxy <= 0:
            return None
        # Normalized distance from short EMA, in vol-scale units.
        dist_atr = (price - ema_s) / atr_proxy
        # Trend filter (EMA crossover) + distance gate.
        if ema_s <= ema_l:
            return None
        if dist_atr <= c.rlb_threshold_atr:
            return None
        # Chase guard so we don't enter after a parabolic move already done.
        pm_24 = features.price_momentum_24h
        if pm_24 is not None and pm_24 > c.rlb_max_entry_pm_24h:
            return None

        # Map distance to signal strength. Saturate at 3× threshold.
        raw = min(max((dist_atr - c.rlb_threshold_atr) / (2 * c.rlb_threshold_atr), 0.0), 1.0)
        strength = max(0.5, min(1.0, 0.5 + 0.5 * raw))

        reason = (
            f"RLB long SN{netuid}: ema20 {ema_s:.4f} > ema50 {ema_l:.4f}, "
            f"dist/ATR {dist_atr:.2f}, vol {vol*100:.2f}%"
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
        if hours_held >= c.rlb_hold_hours:
            return Signal(
                timestamp=snapshot.timestamp,
                netuid=netuid,
                direction=Direction.SELL,
                strategy=self.name(),
                strength=1.0,
                reason=f"RLB time-exit SN{netuid}: held {hours_held:.1f}h",
                features=features.to_dict(),
            )
        # Trend flip exit: if EMA crossover reverses, take the trade off.
        state = _EMA.get(netuid)
        if state and state["short"] is not None and state["long"] is not None:
            if state["short"] < state["long"]:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"RLB trend-flip exit SN{netuid}: ema crossed down",
                    features=features.to_dict(),
                )
        if snapshot.tao_in > 0 and snapshot.alpha_in > 0 and position.tao_invested > 0:
            unrealized = position.unrealized_pnl_pct(snapshot.tao_in, snapshot.alpha_in)
            if unrealized <= c.rlb_stop_loss_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"RLB stop SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
            if unrealized >= c.rlb_take_profit_pct:
                return Signal(
                    timestamp=snapshot.timestamp,
                    netuid=netuid,
                    direction=Direction.SELL,
                    strategy=self.name(),
                    strength=1.0,
                    reason=f"RLB take-profit SN{netuid}: {unrealized*100:+.2f}%",
                    features=features.to_dict(),
                )
        return None
