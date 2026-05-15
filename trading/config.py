"""Configuration for the trading system.

All tunable parameters live in TradingConfig. Defaults are reasonable starting
points, tune per regime and per strategy.
"""

from dataclasses import dataclass, field


@dataclass
class TradingConfig:
    """All configurable parameters for the trading system."""

    # === Data Source ===
    db_path: str = "data/opentao.db"
    opentao_api_url: str = "http://localhost:8009"

    # === Universe Filter ===
    min_pool_depth_tao: float = 50.0
    min_subnet_age_days: int = 14
    min_snapshots: int = 48
    exclude_netuids: list[int] = field(default_factory=lambda: [0])

    # === Position Sizing ===
    max_position_pct_of_pool: float = 0.02
    max_slippage_pct: float = 0.03

    # === Portfolio Risk ===
    initial_capital_tao: float = 100.0
    max_positions: int = 10
    max_single_position_pct: float = 0.20
    reserve_pct: float = 0.20
    daily_loss_limit_pct: float = 0.10

    # === Rate Limit ===
    blocks_per_cooldown: int = 360
    num_hotkeys: int = 1

    # === Hold Period ===
    default_hold_hours: int = 168
    max_hold_hours: int = 720

    # === Stake Velocity Strategy ===
    sv_velocity_window_hours: int = 24
    sv_velocity_threshold: float = 0.03
    sv_price_lag_threshold: float = 0.01
    sv_extended_window_hours: int = 72
    sv_min_pool_depth: float = 100.0

    # === Mean Reversion Strategy ===
    mr_zscore_window_hours: int = 168
    mr_entry_zscore: float = -1.5
    mr_exit_zscore: float = 0.0
    mr_require_positive_velocity: bool = True

    # === Momentum Strategy ===
    mo_price_window_hours: int = 72
    mo_min_price_gain: float = 0.05
    mo_min_velocity: float = 0.02
    mo_max_volatility: float = 0.10

    # === Drain Detection ===
    dd_velocity_threshold: float = -0.03
    dd_consecutive_epochs: int = 3
    # Don't force-close positions younger than this on a 24h drain reading;
    # short-hold strategies have their own intraday stops.
    dd_min_hold_hours: float = 6.0

    # === External strategies ===
    # Colon-separated list of file/directory paths whose @register_strategy
    # decorators populate the global registry. Same shape as
    # OPENTAO_EXTERNAL_STRATEGIES env var; the runner will OR them.
    external_strategy_paths: list[str] = field(default_factory=list)

    # === Paper Trading ===
    paper_poll_interval_seconds: int = 1800
    # Strategies enabled for this paper portfolio. Names must exist in the
    # registry (built-in keys: stake_velocity, mean_reversion, momentum,
    # drain_exit, plus any external keys loaded). drain_exit is always
    # added by the runner as a safety net regardless of this list.
    strategies: list[str] = field(default_factory=lambda: [
        "stake_velocity", "mean_reversion", "momentum",
    ])
