"""Gymnasium env for Bittensor subnet trading via PPO.

Adaptation of the bracket-trading env in
https://github.com/ZiadFrancis/Reinforcement_Trading_Part_2

Key changes vs the upstream FX env:
  - Long-only (the AMM has no shorts). Action space collapses to
    Discrete(3) = {hold, enter_long, exit_long}.
  - Execution uses the constant-product AMM math we already have in
    trading/amm.py, so slippage and fills match what the live trader
    and backtester see.
  - One episode = one subnet's price/feature timeseries, walked once.
  - Reward = step PnL in TAO, normalized by initial_capital (per the
    upstream's risk-normalized reward design — much more stable for PPO).
  - Observation is built from the same Features dataclass the rest of
    the trader uses, plus a few env-state features (position flag,
    unrealized PnL%, hours-held). All inputs are clipped + normalized.

We expose a make_env() factory so VecEnv can spin up parallel copies
across different subnets — that's how we get multi-subnet training
in one PPO run without paying the wall-clock cost of training one
model per subnet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..amm import buy_alpha, sell_alpha
from ..features import FeatureEngine
from ..models import Snapshot


# Feature columns we feed to the policy. Order matters — the policy is
# trained on this exact order, so changing it requires retraining.
_FEATURE_COLS = (
    "stake_velocity_24h",
    "stake_velocity_72h",
    "price_momentum_30m",
    "price_momentum_60m",
    "price_momentum_90m",
    "price_momentum_24h",
    "price_momentum_72h",
    "price_momentum_7d",
    "price_zscore_7d",
    "price_zscore_30d",
    "pool_depth_change_24h",
    "alpha_in_change_24h",
    "price_volatility_7d",
    "price_volatility_30d",
    "relative_pool_rank",
)
# Plus 3 env-state features appended: [position_flag, unrealized_pnl_pct, hours_held/24]
OBS_DIM = len(_FEATURE_COLS) + 3


def _safe(v, default=0.0):
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f):
        return default
    return f


@dataclass
class EnvConfig:
    initial_capital: float = 100.0
    max_steps: Optional[int] = None
    min_pool_depth: float = 300.0
    # Action 1 (enter) sizes the position at this fraction of free capital.
    position_fraction: float = 0.5
    # Hard stop-loss to terminate episode early on catastrophic loss.
    stop_loss_pct: float = -0.20
    # Reward shaping: small penalty per held step to discourage do-nothing
    # policies that just sit on cash for the whole episode.
    idle_penalty_per_step: float = 0.0


class BittensorSubnetEnv(gym.Env):
    """One subnet, one episode. The policy buys/sells with the AMM.

    Observation: causal features at the current bar + current position state.
    Action: 0=hold, 1=enter long (if flat), 2=exit (if holding).
    Reward: per-step change in portfolio value, divided by initial_capital.
    """

    metadata = {"render_modes": []}

    def __init__(self, snapshots: list[Snapshot], cfg: Optional[EnvConfig] = None):
        super().__init__()
        if not snapshots:
            raise ValueError("snapshots must be non-empty")
        self.snapshots = snapshots
        self.cfg = cfg or EnvConfig()
        self.fe = FeatureEngine()

        # Pre-compute features for every bar once — features are causal so
        # they're stable per (subnet, bar) pair.
        self._features = [None] * len(snapshots)
        for i in range(len(snapshots)):
            self._features[i] = self.fe.compute(snapshots, i)

        self.action_space = spaces.Discrete(3)
        # Observations live on the real line after clipping; we use a wide
        # Box and rely on training to learn the scale.
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self._reset_state()

    # ----- gym interface -------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        snap = self.snapshots[self.i]
        prev_value = self._portfolio_value(snap)
        info: dict = {}

        # Apply action
        if action == 1 and self.alpha_held == 0.0 and snap.tao_in > 0 and snap.alpha_in > 0:
            dx = self.free_tao * self.cfg.position_fraction
            if dx >= 0.1 and snap.tao_in >= self.cfg.min_pool_depth:
                result = buy_alpha(dx, snap.tao_in, snap.alpha_in)
                self.alpha_held = result["alpha_received"]
                self.entry_tao = dx
                self.entry_step = self.i
                self.free_tao -= dx
                info["entered"] = True
        elif action == 2 and self.alpha_held > 0.0:
            result = sell_alpha(self.alpha_held, snap.tao_in, snap.alpha_in)
            self.free_tao += result["tao_received"]
            pnl_pct = (result["tao_received"] - self.entry_tao) / max(self.entry_tao, 1e-9)
            info["exited"] = True
            info["trade_pnl_pct"] = pnl_pct
            self.alpha_held = 0.0
            self.entry_tao = 0.0
            self.entry_step = -1

        # Advance one bar
        self.i += 1
        terminated = False
        truncated = False
        # Force-close on episode end
        if self.i >= len(self.snapshots) - 1:
            last = self.snapshots[-1]
            if self.alpha_held > 0:
                result = sell_alpha(self.alpha_held, last.tao_in, last.alpha_in)
                self.free_tao += result["tao_received"]
                self.alpha_held = 0.0
            terminated = True
        elif self.cfg.max_steps is not None and (self.i - self.start_i) >= self.cfg.max_steps:
            truncated = True

        # Hard stop on catastrophic drawdown
        cur_value = self._portfolio_value(self.snapshots[min(self.i, len(self.snapshots)-1)])
        if cur_value / self.cfg.initial_capital - 1.0 <= self.cfg.stop_loss_pct:
            terminated = True

        # Reward: normalized step PnL with optional idle penalty
        reward = (cur_value - prev_value) / self.cfg.initial_capital
        if action == 0 and self.alpha_held == 0.0:
            reward -= self.cfg.idle_penalty_per_step

        info["portfolio_value"] = cur_value
        return self._obs(), float(reward), terminated, truncated, info

    # ----- internals -----------------------------------------------------

    def _reset_state(self) -> None:
        self.i = 0
        self.start_i = 0
        self.free_tao = self.cfg.initial_capital
        self.alpha_held = 0.0
        self.entry_tao = 0.0
        self.entry_step = -1

    def _obs(self) -> np.ndarray:
        idx = min(self.i, len(self.snapshots) - 1)
        snap = self.snapshots[idx]
        feats = self._features[idx]
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        for j, name in enumerate(_FEATURE_COLS):
            vec[j] = _safe(getattr(feats, name, None))
        # Env-state features
        position_flag = 1.0 if self.alpha_held > 0 else 0.0
        if self.alpha_held > 0 and snap.tao_in > 0 and snap.alpha_in > 0 and self.entry_tao > 0:
            from ..amm import sell_alpha as _sa
            exit_proj = _sa(self.alpha_held, snap.tao_in, snap.alpha_in)
            unrealized = (exit_proj["tao_received"] - self.entry_tao) / self.entry_tao
        else:
            unrealized = 0.0
        if self.entry_step >= 0:
            held_hours = (idx - self.entry_step) * 0.5  # 30-min bars
        else:
            held_hours = 0.0
        vec[len(_FEATURE_COLS)] = position_flag
        vec[len(_FEATURE_COLS) + 1] = float(np.clip(unrealized * 10.0, -5.0, 5.0))
        vec[len(_FEATURE_COLS) + 2] = float(np.clip(held_hours / 24.0, 0.0, 5.0))
        # Clip features to obs bounds — most are in [-3, 3] after normalization
        # in FeatureEngine; clip protects against rare outliers.
        return np.clip(vec, -5.0, 5.0)

    def _portfolio_value(self, snap: Snapshot) -> float:
        if self.alpha_held <= 0 or snap.tao_in <= 0 or snap.alpha_in <= 0:
            return self.free_tao
        result = sell_alpha(self.alpha_held, snap.tao_in, snap.alpha_in)
        return self.free_tao + result["tao_received"]


def make_env_factory(snapshots: list[Snapshot], cfg: EnvConfig):
    """Returns a thunk that builds a fresh env — for VecEnv parallelism."""
    def _thunk():
        return BittensorSubnetEnv(snapshots, cfg)
    return _thunk
