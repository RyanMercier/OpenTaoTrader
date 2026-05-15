"""Monte Carlo analysis for the backtester.

Three flavors, all pure wrappers around Backtester.run(), so any new strategy
that plugs into the backtester is automatically supported:

1. Random-window bootstrap: pick N random (start, end) subwindows of a
   target length and run the backtest on each. Gives a distribution of
   returns / Sharpe / drawdown for a given capital and strategy set.

2. Netuid subsampling: the live data only covers 4 subnets, so overfitting
   to specific netuids is a real risk. This runs the backtest against every
   non-empty subset of netuids.

3. Parameter sweep: vary one TradingConfig attribute across a grid and
   record the resulting metrics. Use to check strategy robustness to
   threshold choices.

No lookahead is introduced, each MC run is an ordinary causal backtest.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

from .backtester import Backtester, BacktestResult
from .config import TradingConfig
from .data import DataLoader


@dataclass
class MCRun:
    label: str
    start: str
    end: str
    netuids: Optional[list[int]]
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    final_value: float


@dataclass
class MCSummary:
    """Aggregate stats over many runs. All numbers are in the same units as
    the underlying metric (fraction for pct fields, raw for Sharpe)."""
    n: int
    mean: float
    median: float
    std: float
    p05: float
    p25: float
    p75: float
    p95: float
    min_: float
    max_: float

    @classmethod
    def from_values(cls, values: list[float]) -> "MCSummary":
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        s = sorted(values)
        return cls(
            n=len(s),
            mean=statistics.fmean(s),
            median=statistics.median(s),
            std=statistics.pstdev(s) if len(s) > 1 else 0.0,
            p05=_percentile(s, 0.05),
            p25=_percentile(s, 0.25),
            p75=_percentile(s, 0.75),
            p95=_percentile(s, 0.95),
            min_=s[0],
            max_=s[-1],
        )


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = p * (len(sorted_values) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def _run_to_mc(result: BacktestResult, label: str, netuids: Optional[list[int]]) -> MCRun:
    return MCRun(
        label=label,
        start=result.start_time,
        end=result.end_time,
        netuids=netuids,
        total_return_pct=result.total_return_pct,
        annualized_return_pct=result.annualized_return_pct,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown_pct=result.max_drawdown_pct,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
        final_value=result.final_value,
    )


class MonteCarloRunner:
    def __init__(self, base_config: TradingConfig):
        self.base_config = base_config
        self.loader = DataLoader(base_config.db_path)

    # ------------------------------------------------------------------

    def random_window_bootstrap(
        self,
        num_runs: int,
        window_days: int,
        strategies: Optional[list[str]] = None,
        seed: int = 42,
    ) -> list[MCRun]:
        """Sample num_runs random sub-windows of length window_days (in days)
        from the full data range and backtest each. Returns one MCRun per
        sample.
        """
        rng = random.Random(seed)
        lo_str, hi_str = self.loader.get_data_range()
        if not lo_str or not hi_str:
            return []
        lo = _parse_any(lo_str)
        hi = _parse_any(hi_str)
        total_days = (hi - lo).total_seconds() / 86400
        if total_days <= window_days:
            # Only one meaningful window exists
            return [self._run_once(lo, hi, None, "full", strategies)]

        runs: list[MCRun] = []
        for i in range(num_runs):
            max_offset_days = total_days - window_days
            offset = rng.uniform(0, max_offset_days)
            start = lo + timedelta(days=offset)
            end = start + timedelta(days=window_days)
            label = f"win{i+1}"
            try:
                runs.append(self._run_once(start, end, None, label, strategies))
            except Exception as e:
                print(f"  {label} failed: {e}")
        return runs

    def netuid_subsampling(
        self,
        strategies: Optional[list[str]] = None,
        max_subset_size: int = 4,
    ) -> list[MCRun]:
        """Run the backtest against every non-empty subset of available
        netuids up to max_subset_size. Catches strategies that only work on
        a specific subnet.
        """
        netuids = [
            n for n in self.loader.get_available_netuids()
            if n not in self.base_config.exclude_netuids
        ]
        runs: list[MCRun] = []
        for size in range(1, min(max_subset_size, len(netuids)) + 1):
            for subset in itertools.combinations(netuids, size):
                label = "+".join(str(n) for n in subset)
                try:
                    runs.append(self._run_once(None, None, list(subset), label, strategies))
                except Exception as e:
                    print(f"  {label} failed: {e}")
        return runs

    def parameter_sweep(
        self,
        param_name: str,
        values: list,
        strategies: Optional[list[str]] = None,
    ) -> list[MCRun]:
        """Vary one TradingConfig attribute across a grid. param_name must be
        an attribute on TradingConfig.
        """
        runs: list[MCRun] = []
        if not hasattr(self.base_config, param_name):
            raise AttributeError(f"TradingConfig has no attribute {param_name}")
        for v in values:
            cfg = copy.deepcopy(self.base_config)
            setattr(cfg, param_name, v)
            bt = Backtester(cfg)
            try:
                res = bt.run(strategies=strategies)
                runs.append(_run_to_mc(res, f"{param_name}={v}", None))
            except Exception as e:
                print(f"  {param_name}={v} failed: {e}")
        return runs

    # ------------------------------------------------------------------

    def _run_once(
        self,
        start,
        end,
        netuids: Optional[list[int]],
        label: str,
        strategies: Optional[list[str]],
    ) -> MCRun:
        cfg = copy.deepcopy(self.base_config)
        bt = Backtester(cfg)
        start_s = start.isoformat() if isinstance(start, datetime) else start
        end_s = end.isoformat() if isinstance(end, datetime) else end
        res = bt.run(start=start_s, end=end_s, netuids=netuids, strategies=strategies)
        return _run_to_mc(res, label, netuids)


def _parse_any(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# =====================================================================


def summarize_runs(runs: list[MCRun]) -> dict[str, MCSummary]:
    """Compute per-metric summary over a list of runs."""
    return {
        "total_return_pct": MCSummary.from_values([r.total_return_pct for r in runs]),
        "annualized_return_pct": MCSummary.from_values([r.annualized_return_pct for r in runs]),
        "sharpe_ratio": MCSummary.from_values([r.sharpe_ratio for r in runs]),
        "max_drawdown_pct": MCSummary.from_values([r.max_drawdown_pct for r in runs]),
        "win_rate": MCSummary.from_values([r.win_rate for r in runs]),
        "total_trades": MCSummary.from_values([float(r.total_trades) for r in runs]),
        "final_value": MCSummary.from_values([r.final_value for r in runs]),
    }


def print_mc_report(runs: list[MCRun], title: str = "Monte Carlo Results") -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"  N runs: {len(runs)}")
    if not runs:
        print("  (no runs)")
        return
    s = summarize_runs(runs)
    print()
    print(f"  {'metric':<25} {'mean':>10} {'median':>10} {'std':>10} {'p05':>10} {'p95':>10}")
    for name, summ in s.items():
        mul = 100 if "pct" in name or name == "win_rate" else 1
        print(
            f"  {name:<25} {summ.mean*mul:>10.3f} {summ.median*mul:>10.3f} "
            f"{summ.std*mul:>10.3f} {summ.p05*mul:>10.3f} {summ.p95*mul:>10.3f}"
        )
    print()

    # Profitability odds
    profitable = sum(1 for r in runs if r.total_return_pct > 0)
    print(f"  P(positive return) = {profitable}/{len(runs)} = {profitable/len(runs)*100:.1f}%")
    pos_sharpe = sum(1 for r in runs if r.sharpe_ratio > 1.0)
    print(f"  P(Sharpe > 1.0)    = {pos_sharpe}/{len(runs)} = {pos_sharpe/len(runs)*100:.1f}%")
    pos_sharpe2 = sum(1 for r in runs if r.sharpe_ratio > 2.0)
    print(f"  P(Sharpe > 2.0)    = {pos_sharpe2}/{len(runs)} = {pos_sharpe2/len(runs)*100:.1f}%")
    print()


def save_mc_json(runs: list[MCRun], path: str) -> None:
    import json
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "runs": [asdict(r) for r in runs],
        "summary": {k: asdict(v) for k, v in summarize_runs(runs).items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
