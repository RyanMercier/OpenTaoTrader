"""Round 6 — what works best at 3 TAO capital?

At 100 TAO, LAM solo wins walk-forward 4/4 with +9.61% mean per fold.
At 3 TAO the constraints flip:
  - Default max_positions=10 → 0.3 TAO/slot, below MIN_TRADE_TAO=0.1 after
    signal-strength scaling on most signals
  - reserve_pct=0.20 burns 0.6 TAO before trading even starts
  - max_position_pct_of_pool=0.02 still small in absolute TAO

This round sweeps the risk knobs alongside the strategies to find the
combination that actually trades enough to compound at micro-capital.
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
CAPITAL = 3.0


def run_one(name: str, strategies: list[str], overrides: dict = None) -> dict:
    cfg = TradingConfig(db_path=DB_PATH)
    cfg.initial_capital_tao = CAPITAL
    cfg.rcb_hold_hours = 18.0
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    bt = Backtester(cfg)
    t0 = time.time()
    res = bt.run(start=START, end=END, strategies=strategies)
    elapsed = time.time() - t0
    real_ret = (res.final_value / res.initial_capital - 1) * 100
    from datetime import datetime
    s = datetime.fromisoformat(res.start_time.replace("Z", "+00:00"))
    e = datetime.fromisoformat(res.end_time.replace("Z", "+00:00"))
    days = (e - s).total_seconds() / 86400
    annual = ((1 + real_ret/100) ** (365/days) - 1) * 100 if days > 0 else float("nan")
    blocked = res.blocked_reasons or {}
    top_block = max(blocked.items(), key=lambda x: x[1]) if blocked else ("none", 0)
    return {
        "name": name,
        "return_pct": real_ret,
        "annual_pct": annual,
        "sharpe": res.sharpe_ratio,
        "sortino": res.sortino_ratio,
        "calmar": res.calmar_ratio,
        "max_dd_pct": res.max_drawdown_pct * 100,
        "trades": res.buy_trades + res.sell_trades,
        "pf": res.profit_factor,
        "final": res.final_value,
        "top_blocked_reason": top_block[0],
        "top_blocked_count": top_block[1],
        "blocked_total": res.blocked_trades,
        "elapsed_s": elapsed,
    }


EXPERIMENTS = [
    # === Baselines at default config but 3 TAO capital ===
    ("LAM solo (defaults)", ["lam"], None),
    ("LAM+RCB (defaults)", ["lam", "rcb"], None),
    ("LAM+WSI (defaults)", ["lam", "wsi"], None),

    # === Lower MIN_TRADE_TAO so signals don't get filtered ===
    ("LAM min_trade=0.05", ["lam"], {"min_trade_tao": 0.05}),
    ("LAM min_trade=0.02", ["lam"], {"min_trade_tao": 0.02}),
    ("LAM min_trade=0.01", ["lam"], {"min_trade_tao": 0.01}),

    # === Concentrate: fewer slots → more TAO per slot ===
    ("LAM max_pos=1, min=0.05", ["lam"],
     {"max_positions": 1, "min_trade_tao": 0.05}),
    ("LAM max_pos=2, min=0.05", ["lam"],
     {"max_positions": 2, "min_trade_tao": 0.05}),
    ("LAM max_pos=3, min=0.05", ["lam"],
     {"max_positions": 3, "min_trade_tao": 0.05}),
    ("LAM max_pos=5, min=0.05", ["lam"],
     {"max_positions": 5, "min_trade_tao": 0.05}),

    # === Lower reserve so all 3 TAO actually deploys ===
    ("LAM no reserve, min=0.05", ["lam"],
     {"reserve_pct": 0.0, "min_trade_tao": 0.05}),
    ("LAM no reserve, max_pos=2, min=0.05", ["lam"],
     {"reserve_pct": 0.0, "max_positions": 2, "min_trade_tao": 0.05}),
    ("LAM no reserve, max_pos=3, min=0.05", ["lam"],
     {"reserve_pct": 0.0, "max_positions": 3, "min_trade_tao": 0.05}),
    ("LAM no reserve, max_pos=1, conc=1.0, min=0.05", ["lam"],
     {"reserve_pct": 0.0, "max_positions": 1,
      "max_single_position_pct": 1.0, "min_trade_tao": 0.05}),

    # === Raise pool depth cap so positions can be bigger fraction of pool ===
    ("LAM max_pos=2, pool_cap=0.05, min=0.05", ["lam"],
     {"max_positions": 2, "max_position_pct_of_pool": 0.05,
      "min_trade_tao": 0.05}),
    ("LAM aggressive: max_pos=1, pool_cap=0.10, conc=1.0, no_reserve, min=0.05",
     ["lam"], {"max_positions": 1, "max_position_pct_of_pool": 0.10,
               "max_single_position_pct": 1.0, "reserve_pct": 0.0,
               "min_trade_tao": 0.05}),

    # === Ensembles under the best-found risk config ===
    ("LAM+RCB tuned", ["lam", "rcb"],
     {"reserve_pct": 0.0, "max_positions": 3, "min_trade_tao": 0.05}),
    ("LAM+WSI tuned", ["lam", "wsi"],
     {"reserve_pct": 0.0, "max_positions": 3, "min_trade_tao": 0.05}),
    ("LAM+RLB tuned", ["lam", "rlb"],
     {"reserve_pct": 0.0, "max_positions": 3, "min_trade_tao": 0.05}),

    # === Other solo strategies under the best risk config (just in case) ===
    ("RCB solo tuned", ["rcb"],
     {"reserve_pct": 0.0, "max_positions": 2, "min_trade_tao": 0.05}),
    ("momentum solo tuned", ["momentum"],
     {"reserve_pct": 0.0, "max_positions": 2, "min_trade_tao": 0.05}),
    ("RLB solo tuned", ["rlb"],
     {"reserve_pct": 0.0, "max_positions": 2, "min_trade_tao": 0.05}),
]


def main():
    print(f"Running {len(EXPERIMENTS)} experiments at {CAPITAL} TAO capital...")
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_one, n, s, o): n for n, s, o in EXPERIMENTS}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                print(f"  ✓ {r['name']}: ret={r['return_pct']:+.2f}% "
                      f"trades={r['trades']} sortino={r['sortino']:+.2f} "
                      f"top_block={r['top_blocked_reason'][:30]}({r['top_blocked_count']}) "
                      f"({r['elapsed_s']:.0f}s)", flush=True)
            except Exception as e:
                print(f"  ✗ {futures[fut]}: {e}", flush=True)

    results.sort(key=lambda r: (-r["return_pct"], -r["sortino"]))

    print("\n" + "=" * 120)
    print(f'{"config":<54} {"ret%":>7} {"annual%":>9} {"sortino":>8} {"DD%":>7} {"trades":>6} {"PF":>5}  blocked')
    print("-" * 120)
    for r in results:
        print(f'{r["name"]:<54} {r["return_pct"]:>+7.2f} {r["annual_pct"]:>+9.1f} '
              f'{r["sortino"]:>+8.2f} {r["max_dd_pct"]:>+7.2f} '
              f'{r["trades"]:>6} {r["pf"]:>5.2f}  {r["top_blocked_reason"][:25]}({r["top_blocked_count"]})')

    Path("results/experiments_round6_3tao.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: results/experiments_round6_3tao.json")


if __name__ == "__main__":
    main()
