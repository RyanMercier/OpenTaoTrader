"""RLPPO — wraps a trained PPO policy as a Strategy.

The policy was trained with trading/rl/train_ppo.py against the
BittensorSubnetEnv (long-only Discrete(3) action space). At inference
time we reconstruct the same observation vector this strategy is being
called with, ask the policy for an action, and translate it into our
Signal interface:

  policy action 1 (enter long) → BUY signal
  policy action 2 (exit)       → SELL signal (for our own positions)
  policy action 0 (hold)       → no signal

The observation MUST match training exactly — same feature order, same
clipping, same env-state additions (position flag, unrealized pnl,
hours held). The env code is the source of truth; we duplicate the
formula here rather than instantiate a gym env per tick (too slow).

Loading the model is lazy — we import stable_baselines3 + torch only
when the strategy is first instantiated. That keeps the trader's
plain-backtest path free of the RL dependency footprint when no PPO
strategy is enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..amm import sell_alpha
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
from ..rl.bittensor_env import _FEATURE_COLS, OBS_DIM, _safe


@register_strategy("rlppo")
class RLPPOStrategy(Strategy):
    """Inference wrapper around a saved PPO .zip from train_ppo.py."""

    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._model_path = self._resolve_model_path()
        # Per-subnet entry-time tracking, mirroring what the env tracked.
        self._entry_step: dict[int, int] = {}
        self._entry_tao: dict[int, float] = {}
        self._step_count: dict[int, int] = {}

    def name(self) -> StrategyName:
        return StrategyName.RLPPO

    def can_run_in_regime(self, regime: str) -> bool:
        return True

    # ---- model loading --------------------------------------------------

    def _resolve_model_path(self) -> Optional[Path]:
        candidate = getattr(self.config, "rlppo_model_path", None)
        if candidate:
            p = Path(candidate)
            if p.exists():
                return p
            zipped = p.with_suffix(".zip")
            if zipped.exists():
                return zipped
        # Fallback to the conventional path used by train_ppo.py
        default = Path("models/ppo_bittensor.zip")
        if default.exists():
            return default
        return None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._model_path is None or not self._model_path.exists():
            raise RuntimeError(
                "RLPPO model not found. Train one first with "
                "`python -m trading.rl.train_ppo ...` and put the .zip at "
                "models/ppo_bittensor.zip or set rlppo_model_path in config."
            )
        from stable_baselines3 import PPO
        self._model = PPO.load(str(self._model_path), device="auto")

    # ---- observation construction (must match env exactly) -------------

    def _build_obs(
        self,
        netuid: int,
        features: Features,
        snapshot: Snapshot,
        position: Optional[Position],
    ) -> np.ndarray:
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        for j, name in enumerate(_FEATURE_COLS):
            vec[j] = _safe(getattr(features, name, None))

        position_flag = 1.0 if position is not None else 0.0
        unrealized = 0.0
        held_hours = 0.0
        if position is not None and snapshot.tao_in > 0 and snapshot.alpha_in > 0:
            proj = sell_alpha(position.alpha_amount, snapshot.tao_in, snapshot.alpha_in)
            unrealized = (proj["tao_received"] - position.tao_invested) / max(position.tao_invested, 1e-9)
            held_hours = position.hold_duration_hours(snapshot.timestamp)

        vec[len(_FEATURE_COLS)] = position_flag
        vec[len(_FEATURE_COLS) + 1] = float(np.clip(unrealized * 10.0, -5.0, 5.0))
        vec[len(_FEATURE_COLS) + 2] = float(np.clip(held_hours / 24.0, 0.0, 5.0))
        return np.clip(vec, -5.0, 5.0)

    # ---- Strategy interface --------------------------------------------

    def generate_entry_signal(
        self, netuid: int, features: Features, snapshot: Snapshot
    ) -> Optional[Signal]:
        depth = features.pool_depth_tao
        if depth is None or depth <= getattr(self.config, "rlppo_min_pool_depth", 300.0):
            return None
        try:
            self._ensure_model()
        except RuntimeError:
            # Model missing: stay silent so the rest of the ensemble works.
            return None
        obs = self._build_obs(netuid, features, snapshot, position=None)
        action, _ = self._model.predict(obs, deterministic=True)
        if int(action) != 1:
            return None
        return Signal(
            timestamp=snapshot.timestamp,
            netuid=netuid,
            direction=Direction.BUY,
            strategy=self.name(),
            strength=1.0,
            reason=f"RLPPO enter SN{netuid}",
            features=features.to_dict(),
        )

    def generate_exit_signal(
        self,
        netuid: int,
        features: Features,
        snapshot: Snapshot,
        position: Position,
    ) -> Optional[Signal]:
        if position.strategy != self.name():
            return None
        try:
            self._ensure_model()
        except RuntimeError:
            return None
        obs = self._build_obs(netuid, features, snapshot, position=position)
        action, _ = self._model.predict(obs, deterministic=True)
        if int(action) != 2:
            return None
        return Signal(
            timestamp=snapshot.timestamp,
            netuid=netuid,
            direction=Direction.SELL,
            strategy=self.name(),
            strength=1.0,
            reason=f"RLPPO exit SN{netuid}",
            features=features.to_dict(),
        )
