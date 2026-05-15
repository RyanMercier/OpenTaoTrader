"""Core data structures for the trading system.

Lightweight dataclasses, no ORM, no pydantic. The Snapshot/Features/Signal/
Trade/Position quartet flows through the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


REGIME_PRICE_BASED = "price_based"
REGIME_TAOFLOW_PREHALVING = "taoflow_prehalving"
REGIME_TAOFLOW_POSTHALVING = "taoflow_posthalving"


def get_regime(timestamp_str: str) -> str:
    """Classify an ISO timestamp into an emission regime.

    The cutoff dates come from the Taoflow rollout and the December 2025
    halving. Strings comparable lexicographically because ISO 8601.
    """
    if timestamp_str < "2025-11-01T00:00:00":
        return REGIME_PRICE_BASED
    elif timestamp_str < "2025-12-12T00:00:00":
        return REGIME_TAOFLOW_PREHALVING
    else:
        return REGIME_TAOFLOW_POSTHALVING


class Direction(Enum):
    BUY = "buy"
    SELL = "sell"


class StrategyName(Enum):
    """Built-in strategy identifiers. External strategies pick their own
    string keys via ``@register_strategy("...")`` and are not enumerated
    here. Trades from external strategies store the raw key in
    ``Trade.strategy_key`` and use ``EXTERNAL`` for typed code paths."""
    STAKE_VELOCITY = "stake_velocity"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    DRAIN_EXIT = "drain_exit"
    HOLD_TIMEOUT = "hold_timeout"
    MANUAL = "manual"
    EXTERNAL = "external"


@dataclass
class Snapshot:
    block: int
    timestamp: datetime
    netuid: int
    alpha_price_tao: float
    tao_price_usd: float
    tao_in: float
    alpha_in: float
    total_stake: float
    emission_rate: float
    validator_count: int
    neuron_count: int
    regime: str


@dataclass
class Features:
    netuid: int
    timestamp: datetime

    stake_velocity_24h: float | None = None
    stake_velocity_72h: float | None = None
    stake_velocity_7d: float | None = None

    # Short-term price momentum windows (for intraday strategies)
    price_momentum_30m: float | None = None   # single-bar return
    price_momentum_60m: float | None = None   # 2-bar momentum
    price_momentum_90m: float | None = None   # 3-bar momentum (sweet spot per research)
    price_momentum_2h: float | None = None    # 4-bar momentum
    price_momentum_24h: float | None = None
    price_momentum_72h: float | None = None
    price_momentum_7d: float | None = None

    velocity_price_divergence: float | None = None

    price_zscore_7d: float | None = None
    price_zscore_30d: float | None = None

    pool_depth_tao: float | None = None
    pool_depth_change_24h: float | None = None
    alpha_in_change_24h: float | None = None

    price_volatility_7d: float | None = None
    price_volatility_30d: float | None = None

    relative_pool_rank: float | None = None

    regime: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Signal:
    timestamp: datetime
    netuid: int
    direction: Direction
    strategy: StrategyName
    strength: float
    reason: str
    features: dict = field(default_factory=dict)


@dataclass
class Trade:
    id: str
    timestamp: datetime
    block: int
    netuid: int
    direction: Direction
    strategy: StrategyName
    tao_amount: float
    alpha_amount: float
    spot_price: float
    effective_price: float
    slippage_pct: float
    signal_strength: float
    hotkey_id: int

    entry_price: float | None = None
    pnl_tao: float | None = None
    pnl_pct: float | None = None
    hold_duration_hours: float | None = None
    # For sell trades, the originating entry strategy. Allows per-strategy
    # P&L attribution even when the sell is triggered by a different strategy
    # (e.g. drain_exit or hold_timeout closing a momentum position).
    entry_strategy: "StrategyName | None" = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "block": self.block,
            "netuid": self.netuid,
            "direction": self.direction.value,
            "strategy": self.strategy.value,
            "tao_amount": self.tao_amount,
            "alpha_amount": self.alpha_amount,
            "spot_price": self.spot_price,
            "effective_price": self.effective_price,
            "slippage_pct": self.slippage_pct,
            "signal_strength": self.signal_strength,
            "hotkey_id": self.hotkey_id,
            "entry_price": self.entry_price,
            "pnl_tao": self.pnl_tao,
            "pnl_pct": self.pnl_pct,
            "hold_duration_hours": self.hold_duration_hours,
            "entry_strategy": self.entry_strategy.value if self.entry_strategy else None,
        }


@dataclass
class Position:
    netuid: int
    entry_time: datetime
    entry_block: int
    entry_price: float
    alpha_amount: float
    tao_invested: float
    strategy: StrategyName
    hotkey_id: int

    def current_value_tao(self, tao_in: float, alpha_in: float) -> float:
        """Exit value after slippage using current pool state."""
        if alpha_in <= 0 or tao_in <= 0 or self.alpha_amount <= 0:
            return 0.0
        k = tao_in * alpha_in
        new_alpha_in = alpha_in + self.alpha_amount
        new_tao_in = k / new_alpha_in
        return tao_in - new_tao_in

    def unrealized_pnl_tao(self, tao_in: float, alpha_in: float) -> float:
        return self.current_value_tao(tao_in, alpha_in) - self.tao_invested

    def unrealized_pnl_pct(self, tao_in: float, alpha_in: float) -> float:
        if self.tao_invested <= 0:
            return 0.0
        return self.unrealized_pnl_tao(tao_in, alpha_in) / self.tao_invested

    def hold_duration_hours(self, current_time: datetime) -> float:
        return (current_time - self.entry_time).total_seconds() / 3600

    def to_dict(self) -> dict:
        return {
            "netuid": self.netuid,
            "entry_time": self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else self.entry_time,
            "entry_block": self.entry_block,
            "entry_price": self.entry_price,
            "alpha_amount": self.alpha_amount,
            "tao_invested": self.tao_invested,
            "strategy": self.strategy.value,
            "hotkey_id": self.hotkey_id,
        }


@dataclass
class PortfolioState:
    timestamp: datetime
    free_tao: float
    positions: dict[int, Position]
    total_value_tao: float
    total_pnl_tao: float
    total_pnl_pct: float
    num_trades: int
    peak_value_tao: float
    drawdown_pct: float
