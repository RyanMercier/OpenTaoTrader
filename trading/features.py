"""Rolling features computed from snapshot history.

Causality rule: at index i, only snapshots[0:i+1] may be read. No feature here
peeks at the future. Window sizes are computed in snapshots assuming 30-minute
resolution (48/day).
"""

from __future__ import annotations

import math
from bisect import bisect_left
from typing import Optional

from .models import Features, Snapshot


SNAPSHOTS_PER_DAY = 48  # 30-minute resolution


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(var) if var > 0 else 0.0


def _pct_change(current: float, past: float) -> Optional[float]:
    if past is None or past == 0:
        return None
    return (current - past) / past


def _rolling_return_std(snapshots: list[Snapshot], start: int, end: int) -> Optional[float]:
    """Std of 30-minute returns over snapshots[start:end+1]. end inclusive."""
    if end - start < 2:
        return None
    rets = []
    for i in range(start + 1, end + 1):
        prev_p = snapshots[i - 1].alpha_price_tao
        cur_p = snapshots[i].alpha_price_tao
        if prev_p and prev_p > 0:
            rets.append((cur_p / prev_p) - 1.0)
    if len(rets) < 2:
        return None
    return _std(rets)


def _zscore(values: list[float], current: float) -> Optional[float]:
    if len(values) < 3:
        return None
    mean = sum(values) / len(values)
    std = _std(values)
    if std <= 0:
        return 0.0
    return (current - mean) / std


class FeatureEngine:
    """Computes trading features from a single subnet's history.

    All computations are strictly causal. Features that lack enough history
    are left as None.
    """

    def __init__(self, snapshots_per_day: int = SNAPSHOTS_PER_DAY):
        self.spd = snapshots_per_day
        self.w_24h = snapshots_per_day
        self.w_72h = snapshots_per_day * 3
        self.w_7d = snapshots_per_day * 7
        self.w_30d = snapshots_per_day * 30

    def compute(
        self,
        snapshots: list[Snapshot],
        current_idx: int,
        all_subnets_at_time: Optional[dict[int, Snapshot]] = None,
    ) -> Features:
        if current_idx < 0 or current_idx >= len(snapshots):
            raise IndexError(f"current_idx {current_idx} out of range for {len(snapshots)} snapshots")

        cur = snapshots[current_idx]
        feats = Features(
            netuid=cur.netuid,
            timestamp=cur.timestamp,
            pool_depth_tao=cur.tao_in,
            regime=cur.regime,
        )

        # Stake velocity (tao_in pct change)
        feats.stake_velocity_24h = self._lookback_pct(snapshots, current_idx, self.w_24h, attr="tao_in")
        feats.stake_velocity_72h = self._lookback_pct(snapshots, current_idx, self.w_72h, attr="tao_in")
        feats.stake_velocity_7d = self._lookback_pct(snapshots, current_idx, self.w_7d, attr="tao_in")

        # Price momentum, short windows first (for intraday strategies).
        # Each "bar" is 30 min, so 1=30m, 2=60m, 3=90m, 4=120m.
        feats.price_momentum_30m = self._lookback_pct(snapshots, current_idx, 1, attr="alpha_price_tao")
        feats.price_momentum_60m = self._lookback_pct(snapshots, current_idx, 2, attr="alpha_price_tao")
        feats.price_momentum_90m = self._lookback_pct(snapshots, current_idx, 3, attr="alpha_price_tao")
        feats.price_momentum_2h = self._lookback_pct(snapshots, current_idx, 4, attr="alpha_price_tao")
        feats.price_momentum_24h = self._lookback_pct(snapshots, current_idx, self.w_24h, attr="alpha_price_tao")
        feats.price_momentum_72h = self._lookback_pct(snapshots, current_idx, self.w_72h, attr="alpha_price_tao")
        feats.price_momentum_7d = self._lookback_pct(snapshots, current_idx, self.w_7d, attr="alpha_price_tao")

        # Divergence between 24h stake velocity and 24h price momentum
        if feats.stake_velocity_24h is not None and feats.price_momentum_24h is not None:
            feats.velocity_price_divergence = feats.stake_velocity_24h - feats.price_momentum_24h

        # Z-scores on price
        if current_idx + 1 >= self.w_7d:
            window = [s.alpha_price_tao for s in snapshots[current_idx + 1 - self.w_7d : current_idx + 1] if s.alpha_price_tao]
            if window:
                feats.price_zscore_7d = _zscore(window, cur.alpha_price_tao)
        if current_idx + 1 >= self.w_30d:
            window = [s.alpha_price_tao for s in snapshots[current_idx + 1 - self.w_30d : current_idx + 1] if s.alpha_price_tao]
            if window:
                feats.price_zscore_30d = _zscore(window, cur.alpha_price_tao)

        # Pool depth change (tao_in)
        feats.pool_depth_change_24h = feats.stake_velocity_24h
        feats.alpha_in_change_24h = self._lookback_pct(snapshots, current_idx, self.w_24h, attr="alpha_in")

        # Volatility
        if current_idx + 1 >= self.w_7d:
            feats.price_volatility_7d = _rolling_return_std(
                snapshots, current_idx + 1 - self.w_7d, current_idx
            )
        if current_idx + 1 >= self.w_30d:
            feats.price_volatility_30d = _rolling_return_std(
                snapshots, current_idx + 1 - self.w_30d, current_idx
            )

        # Cross-subnet relative rank
        if all_subnets_at_time:
            depths = sorted(
                s.tao_in for s in all_subnets_at_time.values()
                if s.tao_in is not None and s.tao_in > 0
            )
            if depths:
                rank = bisect_left(depths, cur.tao_in) / len(depths)
                feats.relative_pool_rank = rank

        return feats

    def _lookback_pct(
        self,
        snapshots: list[Snapshot],
        current_idx: int,
        window: int,
        attr: str,
    ) -> Optional[float]:
        """Pct change between snapshots[current_idx] and snapshots[current_idx-window]."""
        past_idx = current_idx - window
        if past_idx < 0:
            return None
        past_val = getattr(snapshots[past_idx], attr)
        cur_val = getattr(snapshots[current_idx], attr)
        if past_val is None or cur_val is None:
            return None
        return _pct_change(cur_val, past_val)
