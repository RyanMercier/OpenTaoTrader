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

    # === Cross-Sectional STMC (XSTMC) ===
    # Short-term momentum continuation, gated by cross-subnet ranking. Enters
    # when a subnet is in the top-K by 90m return; exits on hold target,
    # stop-loss, or take-profit.
    stmc_entry_threshold: float = 0.003       # 90m return must exceed +0.30%
    stmc_strong_threshold: float = 0.01       # 1% saturates the strength bar
    stmc_max_entry_pm_24h: float = 0.15       # skip if already +15% on 24h (chase guard)
    stmc_min_pool_depth: float = 500.0        # tighter floor — deep pools only
    stmc_hold_bars: int = 8                   # 8×30min = ~4h target hold
    stmc_stop_loss_pct: float = -0.03         # intraday stop at -3%
    stmc_take_profit_pct: float = 0.05        # take 5% gain off the table
    xstmc_top_k: int = 2                      # only buy if rank <=K cross-subnet

    # === Stake-Flow Breakout (SFB) ===
    sfb_window_bars: int = 336                # 7d at 30-min bars
    sfb_entry_z: float = 2.0                  # delta must be 2σ above rolling mean
    sfb_min_pool_depth: float = 200.0
    sfb_max_entry_pm_24h: float = 0.20        # don't chase if already +20% in 24h
    sfb_hold_hours: float = 36.0
    sfb_stop_loss_pct: float = -0.05
    sfb_take_profit_pct: float = 0.08

    # === Emission-Yield Carry (EYC) ===
    eyc_top_k: int = 5                        # buy top-5 by yield
    eyc_min_universe: int = 30                # require at least 30 subnets ranked
    eyc_min_pool_depth: float = 300.0
    eyc_max_entry_pm_24h: float = 0.10
    eyc_hold_hours: float = 96.0              # ~4 days
    eyc_stop_loss_pct: float = -0.06
    eyc_take_profit_pct: float = 0.12

    # === Pool-Depth Mean Reversion (PDMR) ===
    pdmr_entry_z: float = -2.0
    pdmr_exit_z: float = 0.0
    pdmr_min_pool_depth: float = 200.0
    pdmr_min_depth_change: float = 0.01       # require +1% liquidity growth in 24h
    pdmr_hold_hours: float = 48.0
    pdmr_stop_loss_pct: float = -0.06

    # === Cross-Sectional Momentum + Vol Brake (XMVB) ===
    xmvb_top_k: int = 5
    xmvb_min_universe: int = 30
    xmvb_min_pool_depth: float = 300.0
    xmvb_max_vol_pct: float = 0.80            # exclude top-quintile by 7d vol
    xmvb_hold_hours: float = 168.0            # 7d
    xmvb_stop_loss_pct: float = -0.08

    # === Liquidity-Adjusted Momentum (LAM) ===
    lam_entry_threshold: float = 0.02         # 24h pm > 2% — apples-to-apples better than 0.005 sweep candidate
    lam_strong_threshold: float = 0.08
    lam_min_pool_depth: float = 500.0
    lam_max_vol: float = 0.20
    lam_hold_hours: float = 24.0
    lam_stop_loss_pct: float = -0.05
    lam_take_profit_pct: float = 0.10

    # === Range Compression Breakout (RCB) ===
    rcb_compression_max: float = 0.30         # 24h range / 7d range < 0.30 = squeezed
    rcb_breakout_tolerance: float = 0.003     # price within 0.3% of 24h hi counts as a break
    rcb_min_pool_depth: float = 200.0
    rcb_hold_hours: float = 18.0              # sweep winner: 18h beat 12h by 6.7× on OOS
    rcb_stop_loss_pct: float = -0.04
    rcb_take_profit_pct: float = 0.08

    # === Pair Mean Reversion (PMR) ===
    pmr_recompute_bars: int = 96              # recompute partner pairs every 48h
    pmr_min_corr: float = 0.6                 # partner must have rho > 0.6
    pmr_entry_z: float = 2.0                  # |spread z| > 2σ
    pmr_min_pool_depth: float = 200.0
    pmr_hold_hours: float = 72.0
    pmr_stop_loss_pct: float = -0.06
    pmr_take_profit_pct: float = 0.10

    # === RLB — RL Baseline Trend Follower ===
    # Port of the EMA+ATR trend-follow baseline from
    # github.com/ZiadFrancis/Reinforcement_Trading_Part_2 — same signal
    # logic, bracket exits, hold-until-flip behaviour.
    rlb_ema_short: int = 20
    rlb_ema_long: int = 50
    rlb_threshold_atr: float = 0.7            # entry: (close-ema20)/atr > thr
    rlb_min_pool_depth: float = 300.0
    rlb_max_entry_pm_24h: float = 0.20
    rlb_hold_hours: float = 72.0
    rlb_stop_loss_pct: float = -0.04          # SL bracket (idx ≈ 2)
    rlb_take_profit_pct: float = 0.08         # TP bracket (idx ≈ 3, ~2R)

    # === RLPPO — Trained PPO Policy Wrapper ===
    # Path to a .zip saved by trading.rl.train_ppo. If unset, the wrapper
    # falls back to models/ppo_bittensor.zip when present, else stays silent.
    rlppo_model_path: str = ""
    rlppo_min_pool_depth: float = 300.0

    # === Whale Stake-Inflow (W1) ===
    wsi_window_bars: int = 1440               # 30d at 30-min cadence
    wsi_percentile: float = 0.97              # 97th percentile delta threshold
    wsi_min_pool_depth: float = 200.0
    wsi_max_entry_pm_24h: float = 0.15
    wsi_hold_hours: float = 48.0
    wsi_stop_loss_pct: float = -0.06
    wsi_take_profit_pct: float = 0.10

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
