"""Round 3 — break the +12.52% ceiling.

Findings so far:
  - LAM is the alpha (+9.19% solo)
  - LAM+RCB = +12.52% (RCB adds uncorrelated diversification)
  - Parameter tweaks on either strategy mostly hurt; ensemble is well-tuned
  - Strategy doesn't scale beyond 100 TAO

Round 3 hypothesis: a DIFFERENT diversifier (not RCB) might add more.
Pair LAM with every other strategy, then try triplets.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trading.backtester import Backtester
from trading.config import TradingConfig


DB_PATH = "/home/ryan/bittensor/OpenTaoAPI/TaoOpenAPI/data/opentao.db"
START = "2026-04-15"
END = "2026-06-21"


def run_one(name: str, strategies: list[str], overrides: dict = None) -> dict:
    cfg = TradingConfig(db_path=DB_PATH)
    cfg.rcb_hold_hours = 18.0
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    bt = Backtester(cfg)
    t0 = time.time()
    res = bt.run(start=START, end=END, strategies=strategies)
    elapsed = time.time() - t0
    real_ret_pct = (res.final_value / res.initial_capital - 1) * 100
    from datetime import datetime
    s = datetime.fromisoformat(res.start_time.replace("Z","+00:00"))
    e = datetime.fromisoformat(res.end_time.replace("Z","+00:00"))
    days = (e - s).total_seconds() / 86400
    annual = ((1 + real_ret_pct/100) ** (365/days) - 1) * 100 if days > 0 else float("nan")
    return {
        "name": name,
        "strategies": ",".join(strategies),
        "return_pct": real_ret_pct,
        "annual_pct": annual,
        "sharpe": res.sharpe_ratio,
        "sortino": res.sortino_ratio,
        "calmar": res.calmar_ratio,
        "max_dd_pct": res.max_drawdown_pct * 100,
        "trades": res.buy_trades + res.sell_trades,
        "pf": res.profit_factor,
        "final": res.final_value,
        "elapsed_s": elapsed,
    }


EXPERIMENTS = [
    # Reference
    ("BASELINE (LAM+RCB)", ["lam", "rcb"], None),
    ("LAM solo (ref)", ["lam"], None),

    # === Pair LAM with every other strategy ===
    ("LAM + momentum", ["lam", "momentum"], None),
    ("LAM + xstmc", ["lam", "xstmc"], None),
    ("LAM + mean_reversion", ["lam", "mean_reversion"], None),
    ("LAM + stake_velocity", ["lam", "stake_velocity"], None),
    ("LAM + sfb", ["lam", "sfb"], None),
    ("LAM + pdmr", ["lam", "pdmr"], None),
    ("LAM + xmvb", ["lam", "xmvb"], None),
    ("LAM + pmr", ["lam", "pmr"], None),
    ("LAM + wsi", ["lam", "wsi"], None),
    ("LAM + eyc", ["lam", "eyc"], None),
    ("LAM + rlb", ["lam", "rlb"], None),

    # === Triplets with the LAM+RCB winner ===
    ("LAM+RCB + momentum", ["lam", "rcb", "momentum"], None),
    ("LAM+RCB + xstmc", ["lam", "rcb", "xstmc"], None),
    ("LAM+RCB + mean_reversion", ["lam", "rcb", "mean_reversion"], None),
    ("LAM+RCB + pmr", ["lam", "rcb", "pmr"], None),
    ("LAM+RCB + wsi", ["lam", "rcb", "wsi"], None),
    ("LAM+RCB + xmvb", ["lam", "rcb", "xmvb"], None),
    ("LAM+RCB + pdmr", ["lam", "rcb", "pdmr"], None),

    # === The everything ensemble ===
    ("ALL rule-based (no rlppo)",
     ["lam", "rcb", "momentum", "mean_reversion", "stake_velocity",
      "xstmc", "sfb", "pdmr", "xmvb", "pmr", "wsi", "rlb"], None),

    # === LAM tuning variants we haven't tried ===
    ("LAM lam_min_pool=1000", ["lam", "rcb"], {"lam_min_pool_depth": 1000.0}),
    ("LAM lam_min_pool=200", ["lam", "rcb"], {"lam_min_pool_depth": 200.0}),
    ("LAM no chase-guard", ["lam", "rcb"], {"lam_max_entry_pm_24h": 999.0}),
    ("LAM strict chase-guard", ["lam", "rcb"], {"lam_max_entry_pm_24h": 0.05}),
]


def main():
    print(f"Running {len(EXPERIMENTS)} experiments (process-isolated)...")
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {}
        for name, strategies, overrides in EXPERIMENTS:
            fut = pool.submit(run_one, name, strategies, overrides)
            futures[fut] = name
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                print(f"  ✓ {res['name']}: ret={res['return_pct']:+.2f}% "
                      f"sortino={res['sortino']:+.2f} ({res['elapsed_s']:.0f}s)",
                      flush=True)
            except Exception as e:
                print(f"  ✗ {futures[fut]}: {e}", flush=True)

    results.sort(key=lambda r: (-r["sortino"], -r["calmar"]))

    print("\n" + "=" * 110)
    print(f'{"experiment":<32} {"ret%":>7} {"annual%":>9} {"sortino":>8} {"calmar":>7} {"DD%":>7} {"trades":>6} {"PF":>5}')
    print("-" * 110)
    for r in results:
        print(f'{r["name"]:<32} {r["return_pct"]:>+7.2f} {r["annual_pct"]:>+9.1f} '
              f'{r["sortino"]:>+8.2f} {r["calmar"]:>+7.2f} {r["max_dd_pct"]:>+7.2f} '
              f'{r["trades"]:>6} {r["pf"]:>5.2f}')

    Path("results/experiments_round3.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: results/experiments_round3.json")


if __name__ == "__main__":
    main()
