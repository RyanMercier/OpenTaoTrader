"""Trader API response shapes. Subset of the monorepo's models that
covers only paper/live portfolio CRUD and strategy listing.
"""
from pydantic import BaseModel, Field


class PaperPortfolioCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    initial_capital_tao: float = Field(default=100.0, gt=0, le=1_000_000)
    strategies: list[str] = Field(
        default_factory=lambda: ["stake_velocity", "mean_reversion", "momentum"],
        description="Registered strategy keys. drain_exit is always added by the runner."
    )
    poll_interval_seconds: int = Field(default=1800, ge=60, le=86400)
    max_positions: int = Field(default=10, ge=1, le=100)
    max_single_position_pct: float = Field(default=0.20, gt=0, le=1.0)
    reserve_pct: float = Field(default=0.20, ge=0.0, lt=1.0)
    max_position_pct_of_pool: float = Field(default=0.02, gt=0, le=0.5)
    max_slippage_pct: float = Field(default=0.03, gt=0, le=0.5)
    num_hotkeys: int = Field(default=1, ge=1, le=64)
    external_strategy_paths: list[str] = Field(default_factory=list)
    # Per-strategy parameter overrides (e.g. lam_stop_loss_pct, rcb_hold_hours).
    # Anything here is setattr'd onto the per-portfolio TradingConfig if the
    # field exists on the dataclass. Unknown keys are silently dropped at
    # config-build time so a typo can't break the runner.
    extra_config: dict = Field(default_factory=dict)


class PaperPortfolio(BaseModel):
    id: int
    name: str
    initial_capital_tao: float
    active: bool
    created_at: str
    mode: str = "paper"
    wallet_name: str | None = None
    hotkey_name: str | None = None
    last_cycle_at: str | None = None
    free_tao: float | None = None
    peak_value: float | None = None
    strategies: list[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)


class PaperPosition(BaseModel):
    netuid: int
    entry_block: int
    entry_time: str
    entry_price: float
    alpha_amount: float
    tao_invested: float
    strategy: str
    hotkey_id: int
    # Mark-to-market against the latest pool state, computed via the
    # constant-product AMM exit formula. None when the live snapshot
    # isn't available (e.g. seeding still in progress).
    current_price: float | None = None
    current_value_tao: float | None = None
    unrealized_pnl_tao: float | None = None
    unrealized_pnl_pct: float | None = None


class PaperTrade(BaseModel):
    id: str
    timestamp: str
    block: int
    netuid: int
    direction: str
    strategy: str
    tao_amount: float
    alpha_amount: float
    spot_price: float
    effective_price: float
    slippage_pct: float
    signal_strength: float | None = None
    hotkey_id: int | None = None
    entry_price: float | None = None
    pnl_tao: float | None = None
    pnl_pct: float | None = None
    hold_duration_hours: float | None = None
    entry_strategy: str | None = None
    extrinsic_hash: str | None = None
    executed_block: int | None = None


class PaperValuePoint(BaseModel):
    timestamp: str
    free_tao: float
    total_value_tao: float
    total_pnl_tao: float
    drawdown_pct: float
    num_open_positions: int
    benchmark_value_tao: float | None = None


class PaperValueHistory(BaseModel):
    portfolio_id: int
    hours: int
    points: list[PaperValuePoint]
    benchmark_universe: list[int] = []
    benchmark_anchor_timestamp: str | None = None


class PaperPortfolioStats(BaseModel):
    portfolio_id: int
    mode: str = "paper"
    initial_capital_tao: float
    current_value_tao: float
    total_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    cycles: int
    cadence_seconds: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float | None = None
    avg_hold_hours: float


class StrategyDescriptor(BaseModel):
    name: str
    source: str
    doc: str = ""
