"""Side-car enhanced paper-trading dashboard.

Adds equal-weight buy-and-hold benchmark, Sharpe ratio, and win-rate to the
view without touching the running paper trader. The trader continues to write
``data/paper_dashboard.html`` every cycle; this script reads that file (to
inherit the in-memory equity curve), fetches historical prices from
OpenTaoAPI for the benchmark, and writes ``data/paper_dashboard_enhanced.html``.

Usage:
    python -m trading.dashboard_extras                 # one-shot
    python -m trading.dashboard_extras --loop 60       # regenerate every 60s
    python -m trading.dashboard_extras --api-url http://localhost:3000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen


DEFAULT_DASHBOARD_PATH = "data/paper_dashboard.html"
DEFAULT_TRADES_PATH = "data/paper_trades.json"
DEFAULT_STATE_PATH = "data/paper_state.json"
DEFAULT_OUT_PATH = "data/paper_dashboard_enhanced.html"
DEFAULT_API_URL = "http://localhost:3000"
DEFAULT_HISTORY_HOURS = 240  # 10 days, covers the running test window comfortably
EXCLUDE_NETUIDS = {0}
CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js"
# date-fns adapter, required for `type: 'time'` x-axes in Chart.js 4.x
CHART_JS_DATEFNS_CDN = "https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"


# ----------------------------- I/O helpers --------------------------------

def _load_running_data(html_path: str) -> dict:
    """Pull the embedded JSON payload out of the running dashboard."""
    with open(html_path, "r") as f:
        html = f.read()
    m = re.search(r"const DATA = (\{.*?\});\n", html, re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not locate DATA blob in {html_path}")
    return json.loads(m.group(1))


def _load_trades(trades_path: str) -> list[dict]:
    if not os.path.exists(trades_path):
        return []
    with open(trades_path, "r") as f:
        return json.load(f)


def _load_state(state_path: str) -> dict:
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r") as f:
        return json.load(f)


def _http_json(url: str, timeout: int = 20):
    with urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# --------------------------- benchmark math --------------------------------

def _parse_ts(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _list_subnets(api_url: str) -> list[int]:
    data = _http_json(f"{api_url.rstrip('/')}/api/v1/subnets")
    items = data.get("subnets", []) if isinstance(data, dict) else data
    return [int(it["netuid"]) for it in items
            if isinstance(it, dict) and "netuid" in it
            and int(it["netuid"]) not in EXCLUDE_NETUIDS]


def _fetch_history(api_url: str, netuid: int, hours: int) -> list[dict]:
    url = f"{api_url.rstrip('/')}/api/v1/history/{netuid}/snapshots?hours={hours}"
    try:
        return _http_json(url, timeout=15)
    except Exception:
        return []


def _compute_benchmark_series(
    api_url: str,
    start_time: datetime,
    initial_capital: float,
    hours: int,
) -> tuple[list[dict], int]:
    """Equal-weight basket of every (active, ex-root) subnet that has a price
    at or after ``start_time``. Baseline per subnet is the first price >=
    start_time. Returned points are spaced at 30-minute snapshot bars.
    """
    netuids = _list_subnets(api_url)
    if not netuids:
        return [], 0

    BAR_SECONDS = 1800
    # TAO-pool-weighted basket, mirrors the trader's universe. Excludes
    # subnets that didn't exist at start, and tiny pools the trader wouldn't
    # touch anyway. Weighting by initial pool depth keeps the benchmark from
    # being dominated by anomalies in shallow pools (e.g. drain events).
    MIN_BASELINE_PRICE = 1e-6
    MIN_POOL_TAO = 50.0  # matches TradingConfig.min_pool_depth_tao
    per_subnet: dict[int, list[tuple[float, float]]] = {}  # netuid -> [(epoch, price)] post-start
    baselines: dict[int, float] = {}
    weights: dict[int, float] = {}
    skipped_new = 0
    skipped_tiny = 0
    start_epoch = start_time.timestamp()

    for n in netuids:
        rows = _fetch_history(api_url, n, hours)
        all_clean: list[tuple[float, float, float]] = []  # epoch, price, tao_in
        for row in rows:
            try:
                price = float(row.get("alpha_price_tao") or 0.0)
                if price <= 0:
                    continue
                tao_in = float(row.get("tao_in") or 0.0)
                ts = _parse_ts(row.get("timestamp") or "")
            except Exception:
                continue
            all_clean.append((ts.timestamp(), price, tao_in))
        if len(all_clean) < 2:
            continue
        all_clean.sort()

        # Baseline = latest snapshot at-or-before start_time. If none, the
        # subnet didn't trade yet, skip.
        baseline_price = None
        baseline_pool = 0.0
        for epoch, price, tao_in in all_clean:
            if epoch <= start_epoch:
                baseline_price = price
                baseline_pool = tao_in
            else:
                break
        if baseline_price is None or baseline_price < MIN_BASELINE_PRICE:
            skipped_new += 1
            continue
        if baseline_pool < MIN_POOL_TAO:
            skipped_tiny += 1
            continue

        post = [(e, p) for e, p, _ in all_clean if e >= start_epoch]
        if not post:
            # Brand-new test, no post-start snapshot yet. Seed with the
            # baseline so the benchmark starts at parity; it will fill in as
            # new bars arrive over the next ~30 min.
            post = [(start_epoch, baseline_price)]
        per_subnet[n] = post
        baselines[n] = baseline_price
        weights[n] = baseline_pool
    if skipped_new:
        print(f"  skipped {skipped_new} subnets that didn't exist at test start")
    if skipped_tiny:
        print(f"  skipped {skipped_tiny} subnets with pool < {MIN_POOL_TAO} TAO at start")

    if not per_subnet:
        return [], 0

    # Build a union timeline at 30-min granularity
    bars = set()
    for series in per_subnet.values():
        for epoch, _ in series:
            bars.add(int(epoch // BAR_SECONDS))
    sorted_bars = sorted(bars)

    # Two-pointer scan per subnet for O(N) aggregation. TAO-weighted basket
    # (weights fixed at start = pool depth in TAO). Per-subnet ratios are
    # winsorized to [RATIO_FLOOR, RATIO_CAP], pool drain/emission events on
    # bittensor produce 100x+ price spikes that don't reflect realizable
    # buy-and-hold P&L (you can't actually exit at those quotes), and a single
    # outlier would otherwise swamp the basket.
    RATIO_CAP = 5.0
    RATIO_FLOOR = 0.2
    cursors = {n: 0 for n in per_subnet}
    out: list[dict] = []
    for bar in sorted_bars:
        bar_end = (bar + 1) * BAR_SECONDS
        weighted_sum = 0.0
        active_weight = 0.0
        for n, series in per_subnet.items():
            i = cursors[n]
            while i + 1 < len(series) and series[i + 1][0] < bar_end:
                i += 1
            cursors[n] = i
            epoch, price = series[i]
            if epoch < bar_end:
                ratio = max(RATIO_FLOOR, min(RATIO_CAP, price / baselines[n]))
                w = weights[n]
                weighted_sum += w * ratio
                active_weight += w
        if active_weight > 0:
            ts_iso = datetime.fromtimestamp(bar * BAR_SECONDS, tz=timezone.utc).isoformat()
            out.append({"t": ts_iso, "v": (weighted_sum / active_weight) * initial_capital})
    return out, len(per_subnet)


# --------------------------- statistics ------------------------------------

def _hourly_returns(value_series: list[dict]) -> list[float]:
    """Sample value_series to hourly buckets (last value per hour) and
    return per-hour pct returns."""
    buckets: dict[int, float] = {}
    for p in value_series:
        try:
            ts = _parse_ts(p["t"])
        except Exception:
            continue
        h = int(ts.timestamp() // 3600)
        buckets[h] = float(p["v"])
    keys = sorted(buckets)
    if len(keys) < 2:
        return []
    rets = []
    prev = buckets[keys[0]]
    for k in keys[1:]:
        v = buckets[k]
        if prev > 0:
            rets.append(v / prev - 1)
        prev = v
    return rets


def _sharpe(value_series: list[dict]) -> float:
    rets = _hourly_returns(value_series)
    if len(rets) < 5:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return (mean / sd) * math.sqrt(8760)  # annualize hourly


def _win_rate(trades: list[dict]) -> tuple[float, int, int]:
    sells = [t for t in trades if t.get("direction") == "sell"]
    if not sells:
        return 0.0, 0, 0
    wins = sum(1 for t in sells if (t.get("pnl_tao") or 0) > 0)
    return wins / len(sells), wins, len(sells)


def _avg_win_loss(trades: list[dict]) -> tuple[float, float, float]:
    """Avg winning trade pnl%, avg losing trade pnl%, profit factor."""
    win_pnls = [t.get("pnl_tao") or 0 for t in trades
                if t.get("direction") == "sell" and (t.get("pnl_tao") or 0) > 0]
    loss_pnls = [t.get("pnl_tao") or 0 for t in trades
                 if t.get("direction") == "sell" and (t.get("pnl_tao") or 0) < 0]
    win_pcts = [t.get("pnl_pct") or 0 for t in trades
                if t.get("direction") == "sell" and (t.get("pnl_tao") or 0) > 0]
    loss_pcts = [t.get("pnl_pct") or 0 for t in trades
                 if t.get("direction") == "sell" and (t.get("pnl_tao") or 0) < 0]
    avg_win = sum(win_pcts) / len(win_pcts) if win_pcts else 0.0
    avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0
    gross_win = sum(win_pnls)
    gross_loss = abs(sum(loss_pnls))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    return avg_win, avg_loss, pf


# --------------------------- HTML rendering --------------------------------

def _render_html(
    base_data: dict,
    trades: list[dict],
    state: dict,
    benchmark_series: list[dict],
    benchmark_count: int,
) -> str:
    value_series = base_data.get("value_series", [])
    initial_capital = value_series[0]["v"] if value_series else 0.0
    final_value = value_series[-1]["v"] if value_series else 0.0
    total_return_pct = (final_value / initial_capital - 1) if initial_capital > 0 else 0.0

    bh_initial = benchmark_series[0]["v"] if benchmark_series else initial_capital
    bh_final = benchmark_series[-1]["v"] if benchmark_series else initial_capital
    bh_return_pct = (bh_final / bh_initial - 1) if bh_initial > 0 else 0.0
    alpha = total_return_pct - bh_return_pct

    sharpe = _sharpe(value_series)
    win_rate, wins, sells = _win_rate(trades)
    avg_win, avg_loss, pf = _avg_win_loss(trades)

    # Downsample equity curve to the same 30-min bar grid as the benchmark so
    # both share an x-axis (avoids needing a Chart.js time-scale adapter).
    BAR_SECONDS = 1800
    eq_buckets: dict[int, float] = {}
    for p in value_series:
        try:
            ts = _parse_ts(p["t"])
        except Exception:
            continue
        eq_buckets[int(ts.timestamp() // BAR_SECONDS)] = float(p["v"])
    bh_buckets = {
        int(_parse_ts(p["t"]).timestamp() // BAR_SECONDS): float(p["v"])
        for p in benchmark_series
    }
    bars_sorted = sorted(set(eq_buckets) | set(bh_buckets))
    chart_labels = [
        datetime.fromtimestamp(b * BAR_SECONDS, tz=timezone.utc).strftime("%m-%d %H:%M")
        for b in bars_sorted
    ]
    last_eq = None
    chart_strategy = []
    for b in bars_sorted:
        if b in eq_buckets:
            last_eq = eq_buckets[b]
        chart_strategy.append(last_eq)
    last_bh = None
    chart_benchmark = []
    for b in bars_sorted:
        if b in bh_buckets:
            last_bh = bh_buckets[b]
        chart_benchmark.append(last_bh)

    # Render trades from the full JSON log (the source dashboard truncates to
    # the last 50). Newest first, capped to TRADE_TABLE_LIMIT for page weight.
    TRADE_TABLE_LIMIT = 200
    trades_sorted = sorted(
        trades,
        key=lambda t: t.get("timestamp", ""),
        reverse=True,
    )[:TRADE_TABLE_LIMIT]
    trade_rows = [
        {
            "timestamp": t.get("timestamp"),
            "netuid": t.get("netuid"),
            "direction": t.get("direction"),
            "strategy": t.get("strategy"),
            "tao": t.get("tao_amount") or 0.0,
            "alpha": t.get("alpha_amount") or 0.0,
            "slip": t.get("slippage_pct") or 0.0,
            "pnl_pct": t.get("pnl_pct"),
        }
        for t in trades_sorted
    ]

    payload = {
        "now": base_data.get("now"),
        "state": base_data.get("state", {}),
        "value_series": value_series,
        "benchmark_series": benchmark_series,
        "chart": {
            "labels": chart_labels,
            "strategy": chart_strategy,
            "benchmark": chart_benchmark,
        },
        "positions": base_data.get("positions", []),
        "pending": base_data.get("pending", []),
        "trades": trade_rows,
        "stats_extra": {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return_pct,
            "bh_return_pct": bh_return_pct,
            "alpha_vs_bh": alpha,
            "bh_initial": bh_initial,
            "bh_final": bh_final,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "wins": wins,
            "sells": sells,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": pf,
            "benchmark_count": benchmark_count,
            "open_positions": len(base_data.get("positions", [])),
            "total_trades": len(trades),
        },
    }
    data_json = json.dumps(payload)
    now_iso = datetime.now(timezone.utc).isoformat()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Portfolio · OpenTaoAPI</title>
<script src="{CHART_JS_CDN}"></script>
<style>
  *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}

  :root {{
    --bg: #0a0a0a;
    --bg-raised: #111;
    --bg-hover: #0d0d0d;
    --border: #1e1e1e;
    --border-soft: #141414;
    --text: #c8c8c8;
    --text-dim: #666;
    --text-bright: #e8e8e8;
    --accent: #00d4aa;
    --accent-soft: rgba(0, 212, 170, 0.08);
    --accent-border: rgba(0, 212, 170, 0.15);
    --red: #e84855;
    --red-soft: rgba(232, 72, 85, 0.08);
    --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }}

  .container-wide {{ max-width: 1200px; margin: 0 auto; padding: 0 20px; }}

  /* Header */
  header {{ border-bottom: 1px solid var(--border); padding: 16px 0; }}
  header .inner {{
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
  }}
  .logo {{
    font-family: var(--mono); font-size: 15px; font-weight: 600;
    color: var(--text-bright); letter-spacing: -0.3px;
  }}
  .logo span {{ color: var(--accent); }}
  .nav-links {{ display: flex; gap: 16px; }}
  .nav-links a {{
    font-size: 13px; color: var(--text-dim); text-decoration: none;
  }}
  .nav-links a:hover {{ color: var(--text); }}
  .nav-links a.active {{ color: var(--text-bright); }}

  /* Page title row */
  .page-head {{
    padding: 24px 0 12px; display: flex; align-items: baseline;
    justify-content: space-between; gap: 16px; flex-wrap: wrap;
  }}
  .page-title {{
    font-size: 20px; font-weight: 600; color: var(--text-bright); letter-spacing: -0.2px;
  }}
  .page-title .meta {{
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
    margin-left: 12px; font-weight: 400; letter-spacing: 0;
  }}
  .updated {{ font-family: var(--mono); font-size: 11px; color: var(--text-dim); }}

  /* Summary bar (top-line) */
  .summary-bar {{
    padding: 18px 0; display: flex; gap: 40px; flex-wrap: wrap;
    align-items: baseline; border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }}
  .summary-stat {{ display: flex; flex-direction: column; gap: 2px; }}
  .summary-label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--text-dim);
  }}
  .summary-value {{
    font-family: var(--mono); font-size: 22px; font-weight: 600;
    color: var(--text-bright);
  }}
  .summary-sub {{ font-family: var(--mono); font-size: 12px; color: var(--text-dim); }}
  .summary-value.up {{ color: var(--accent); }}
  .summary-value.down {{ color: var(--red); }}

  /* Stats grid (smaller secondary cards) */
  .stats-grid {{
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 0;
    border-bottom: 1px solid var(--border);
  }}
  .stat-cell {{
    padding: 14px 16px; border-right: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 4px;
  }}
  .stat-cell:last-child {{ border-right: none; }}
  .stat-label {{
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--text-dim);
  }}
  .stat-value {{ font-family: var(--mono); font-size: 15px; color: var(--text-bright); }}
  .stat-value.up {{ color: var(--accent); }}
  .stat-value.down {{ color: var(--red); }}
  .stat-sub {{ font-family: var(--mono); font-size: 10px; color: var(--text-dim); }}

  /* Section panels */
  .section {{ padding: 28px 0 12px; }}
  .section-title {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--text-dim); margin-bottom: 12px; display: flex;
    align-items: baseline; justify-content: space-between;
  }}
  .section-title .sub {{
    font-family: var(--mono); color: var(--text-dim); text-transform: none;
    letter-spacing: 0; font-size: 11px;
  }}

  /* Chart panel */
  .chart-card {{
    background: var(--bg-raised); border: 1px solid var(--border);
    padding: 16px;
  }}
  canvas {{ max-height: 360px; }}

  /* Tables */
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); }}
  table.data {{ width: 100%; border-collapse: collapse; min-width: 600px; }}
  table.data th {{
    text-align: right; font-weight: 400; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-dim);
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    background: var(--bg); white-space: nowrap;
  }}
  table.data th:first-child, table.data td:first-child {{ text-align: left; padding-left: 16px; }}
  table.data th:last-child, table.data td:last-child {{ padding-right: 16px; }}
  table.data td {{
    text-align: right; font-family: var(--mono); font-size: 12px;
    padding: 9px 14px; border-bottom: 1px solid var(--border-soft);
    color: var(--text); white-space: nowrap;
  }}
  table.data tbody tr:hover td {{ background: var(--bg-hover); }}
  table.data tr:last-child td {{ border-bottom: none; }}
  .sn-badge {{
    font-family: var(--mono); font-size: 10px; color: var(--text-dim);
    background: var(--bg-raised); border: 1px solid var(--border);
    padding: 1px 5px; min-width: 42px; display: inline-block; text-align: center;
  }}
  .pill {{
    font-family: var(--mono); font-size: 10px; padding: 1px 6px;
    border: 1px solid var(--border); color: var(--text-dim); display: inline-block;
  }}
  .pill.buy  {{ color: var(--accent); border-color: var(--accent-border); background: var(--accent-soft); }}
  .pill.sell {{ color: var(--red);    border-color: rgba(232,72,85,0.2); background: var(--red-soft); }}
  .up {{ color: var(--accent); }}
  .down {{ color: var(--red); }}
  .empty {{ padding: 24px 16px; color: var(--text-dim); font-size: 12px; text-align: center; }}

  /* Footer */
  footer {{ border-top: 1px solid var(--border); padding: 16px 0; text-align: center; margin-top: 40px; }}
  footer p {{ font-size: 11px; color: var(--text-dim); }}
  footer a {{ color: var(--text-dim); text-decoration: none; }}
  footer a:hover {{ color: var(--text); }}

  @media (max-width: 900px) {{
    .stats-grid {{ grid-template-columns: repeat(3, 1fr); }}
    .stat-cell:nth-child(3n) {{ border-right: none; }}
    .stat-cell {{ border-bottom: 1px solid var(--border); }}
  }}
  @media (max-width: 600px) {{
    .summary-bar {{ gap: 24px; }}
    .summary-value {{ font-size: 18px; }}
  }}
</style>
</head>
<body>

<header>
  <div class="container-wide inner">
    <span class="logo">Tao<span>Trader</span></span>
    <span class="updated">Paper trading · auto-refresh 30s</span>
  </div>
</header>

<div class="container-wide">

  <div class="page-head">
    <div class="page-title">
      Paper Trading
      <span class="meta" id="metaLine"></span>
    </div>
    <div class="updated">Last update {now_iso}</div>
  </div>

  <div class="summary-bar">
    <div class="summary-stat">
      <div class="summary-label">Portfolio Value</div>
      <div class="summary-value" id="sumValue"></div>
      <div class="summary-sub" id="sumValueSub"></div>
    </div>
    <div class="summary-stat">
      <div class="summary-label">Total Return</div>
      <div class="summary-value" id="sumReturn"></div>
      <div class="summary-sub" id="sumReturnSub"></div>
    </div>
    <div class="summary-stat">
      <div class="summary-label">vs Pool-Weighted B&amp;H</div>
      <div class="summary-value" id="sumAlpha"></div>
      <div class="summary-sub" id="sumAlphaSub"></div>
    </div>
    <div class="summary-stat">
      <div class="summary-label">Sharpe (ann.)</div>
      <div class="summary-value" id="sumSharpe"></div>
      <div class="summary-sub">hourly returns</div>
    </div>
    <div class="summary-stat">
      <div class="summary-label">Win Rate</div>
      <div class="summary-value" id="sumWin"></div>
      <div class="summary-sub" id="sumWinSub"></div>
    </div>
  </div>

  <div class="stats-grid" id="statsGrid"></div>

  <div class="section">
    <div class="section-title">
      Equity Curve
      <span class="sub">strategy vs TAO-pool-weighted subnet basket (winsorized 0.2x-5x)</span>
    </div>
    <div class="chart-card"><canvas id="valueChart"></canvas></div>
  </div>

  <div class="section">
    <div class="section-title">Open Positions <span class="sub" id="posCount"></span></div>
    <div class="table-wrap" id="positions"></div>
  </div>

  <div class="section">
    <div class="section-title">Pending Signals <span class="sub" id="pendCount"></span></div>
    <div class="table-wrap" id="pending"></div>
  </div>

  <div class="section">
    <div class="section-title">Trade Log <span class="sub" id="tradeCount"></span></div>
    <div class="table-wrap" id="trades"></div>
  </div>

</div>

<footer>
  <div class="container-wide">
    <p>Powered by <a href="https://bittensor.com">Bittensor</a></p>
  </div>
</footer>

<script>
const DATA = {data_json};

const fmtPct = x => (x*100).toFixed(2) + '%';
const fmtPctSigned = x => (x>=0?'+':'') + (x*100).toFixed(2) + '%';
const fmtTao = x => 'τ' + Number(x).toFixed(3);
const fmtTaoFull = x => 'τ' + Number(x).toFixed(6);
const upDown = x => x >= 0 ? 'up' : 'down';
const escapeHtml = s => String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));

function fmtTimeShort(iso) {{
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  const mm = String(d.getMonth()+1).padStart(2,'0');
  const dd = String(d.getDate()).padStart(2,'0');
  const hh = String(d.getHours()).padStart(2,'0');
  const mn = String(d.getMinutes()).padStart(2,'0');
  return `${{mm}}/${{dd}} ${{hh}}:${{mn}}`;
}}

function renderHero() {{
  const e = DATA.stats_extra;
  const s = DATA.state || {{}};

  // Meta line: duration + trade count
  const labels = (DATA.chart || {{labels:[]}}).labels;
  const span = labels.length ? `${{labels[0]}} -> ${{labels[labels.length-1]}}` : '';
  document.getElementById('metaLine').textContent =
    `${{span}} · ${{e.total_trades}} trades`;

  document.getElementById('sumValue').textContent = fmtTao(s.total_value_tao || e.final_value);
  document.getElementById('sumValueSub').textContent = `start ${{fmtTao(e.initial_capital)}}`;

  const ret = e.total_return_pct;
  const retEl = document.getElementById('sumReturn');
  retEl.textContent = fmtPctSigned(ret);
  retEl.className = 'summary-value ' + upDown(ret);
  document.getElementById('sumReturnSub').textContent =
    `${{ret>=0?'+':''}}${{(s.total_value_tao - e.initial_capital).toFixed(3)}} TAO`;

  const a = e.alpha_vs_bh;
  const aEl = document.getElementById('sumAlpha');
  aEl.textContent = fmtPctSigned(a);
  aEl.className = 'summary-value ' + upDown(a);
  document.getElementById('sumAlphaSub').textContent =
    `B&H ${{fmtPctSigned(e.bh_return_pct)}} · ${{e.benchmark_count}} subnets`;

  document.getElementById('sumSharpe').textContent = e.sharpe.toFixed(2);

  const wr = e.win_rate;
  document.getElementById('sumWin').textContent = (wr*100).toFixed(1) + '%';
  document.getElementById('sumWinSub').textContent = `${{e.wins}}/${{e.sells}} closed`;
}}

function renderStatsGrid() {{
  const e = DATA.stats_extra;
  const s = DATA.state || {{}};
  const cells = [
    {{ label: 'Free TAO',       v: fmtTao(s.free_tao || 0) }},
    {{ label: 'Open Positions', v: e.open_positions }},
    {{ label: 'Avg Win',        v: fmtPctSigned(e.avg_win_pct), tone: 'up' }},
    {{ label: 'Avg Loss',       v: fmtPctSigned(e.avg_loss_pct), tone: 'down' }},
    {{ label: 'Profit Factor',  v: isFinite(e.profit_factor) ? e.profit_factor.toFixed(2) : '∞',
       tone: e.profit_factor >= 1 ? 'up' : 'down' }},
    {{ label: 'Max Drawdown',   v: fmtPct(s.drawdown_pct || 0), tone: 'down',
       sub: `peak ${{fmtTao(s.peak_value_tao || 0)}}` }},
  ];
  document.getElementById('statsGrid').innerHTML = cells.map(c => `
    <div class="stat-cell">
      <div class="stat-label">${{c.label}}</div>
      <div class="stat-value ${{c.tone || ''}}">${{c.v}}</div>
      ${{c.sub ? `<div class="stat-sub">${{c.sub}}</div>` : ''}}
    </div>
  `).join('');
}}

function renderValue() {{
  const c = DATA.chart || {{labels:[], strategy:[], benchmark:[]}};
  if (!c.labels.length) return;
  const initial = DATA.stats_extra.initial_capital;
  // Reformat labels to compact MM/DD HH:MM
  const labels = c.labels.map(l => l.replace(/-/g, '/'));
  new Chart(document.getElementById('valueChart'), {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [
        {{
          label: 'Strategy',
          data: c.strategy,
          borderColor: '#00d4aa',
          backgroundColor: 'rgba(0,212,170,0.08)',
          borderWidth: 1.5,
          fill: true,
          pointRadius: 0,
          tension: 0.15,
        }},
        {{
          label: 'Pool-weighted B&H',
          data: c.benchmark,
          borderColor: '#888',
          borderWidth: 1.2,
          borderDash: [5,4],
          pointRadius: 0,
          fill: false,
          tension: 0.15,
        }},
        {{
          label: 'Initial',
          data: c.labels.map(_ => initial),
          borderColor: '#2a2a2a',
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        }},
      ]
    }},
    options: {{
      animation: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          position: 'top', align: 'end',
          labels: {{ color: '#c8c8c8', font: {{ size: 11, family: '-apple-system, sans-serif' }},
                   boxWidth: 10, boxHeight: 2 }},
        }},
        tooltip: {{
          backgroundColor: '#0a0a0a', borderColor: '#1e1e1e', borderWidth: 1,
          titleColor: '#e8e8e8', bodyColor: '#c8c8c8',
          titleFont: {{ family: 'SF Mono, monospace', size: 11 }},
          bodyFont:  {{ family: 'SF Mono, monospace', size: 11 }},
          callbacks: {{ label: ctx => `${{ctx.dataset.label}}: τ${{ctx.parsed.y.toFixed(4)}}` }},
        }},
      }},
      scales: {{
        x: {{
          ticks: {{ autoSkip: true, maxTicksLimit: 10, color: '#666',
                  font: {{ size: 10, family: 'SF Mono, monospace' }} }},
          grid: {{ color: '#141414', drawBorder: false }},
        }},
        y: {{
          ticks: {{ color: '#666', font: {{ size: 10, family: 'SF Mono, monospace' }},
                  callback: v => 'τ' + v.toFixed(2) }},
          grid: {{ color: '#141414', drawBorder: false }},
        }},
      }},
    }},
  }});
}}

function renderTable(root, rows, cols, emptyText) {{
  const el = document.getElementById(root);
  if (!rows || !rows.length) {{
    el.innerHTML = `<div class="empty">${{emptyText || 'No rows.'}}</div>`;
    return;
  }}
  const head = `<tr>${{cols.map(c => `<th>${{c.h}}</th>`).join('')}}</tr>`;
  const body = rows.map(r =>
    `<tr>${{cols.map(c => `<td>${{c.fmt ? c.fmt(r) : escapeHtml(r[c.k])}}</td>`).join('')}}</tr>`
  ).join('');
  el.innerHTML = `<table class="data"><thead>${{head}}</thead><tbody>${{body}}</tbody></table>`;
}}

renderHero();
renderStatsGrid();
renderValue();

document.getElementById('posCount').textContent =
  DATA.positions.length ? `${{DATA.positions.length}} open` : '';
renderTable('positions', DATA.positions, [
  {{ h: 'SN',       fmt: r => `<span class="sn-badge">SN ${{r.netuid}}</span>` }},
  {{ h: 'Strategy', fmt: r => `<span class="pill">${{escapeHtml(r.strategy)}}</span>` }},
  {{ h: 'Entry',    fmt: r => fmtTaoFull(r.entry_price) }},
  {{ h: 'Current',  fmt: r => fmtTaoFull(r.current_price) }},
  {{ h: 'Alpha',    fmt: r => r.alpha_amount.toFixed(2) }},
  {{ h: 'Invested', fmt: r => fmtTao(r.tao_invested) }},
  {{ h: 'P&L',      fmt: r => `<span class="${{r.pnl_tao>=0?'up':'down'}}">${{r.pnl_tao>=0?'+':''}}${{r.pnl_tao.toFixed(3)}}τ</span>` }},
  {{ h: 'P&L %',    fmt: r => `<span class="${{r.pnl_pct>=0?'up':'down'}}">${{fmtPctSigned(r.pnl_pct)}}</span>` }},
  {{ h: 'Hold',     fmt: r => r.hold_hours.toFixed(1) + 'h' }},
], 'No open positions.');

document.getElementById('pendCount').textContent =
  DATA.pending.length ? `${{DATA.pending.length}}` : '';
renderTable('pending', DATA.pending, [
  {{ h: 'SN',       fmt: r => `<span class="sn-badge">SN ${{r.netuid}}</span>` }},
  {{ h: 'Side',     fmt: r => `<span class="pill ${{r.direction}}">${{r.direction}}</span>` }},
  {{ h: 'Strategy', fmt: r => `<span class="pill">${{escapeHtml(r.strategy)}}</span>` }},
  {{ h: 'Strength', fmt: r => r.strength.toFixed(2) }},
  {{ h: 'Reason',   fmt: r => `<span style="color:var(--text-dim)">${{escapeHtml(r.reason)}}</span>` }},
], 'No pending signals.');

document.getElementById('tradeCount').textContent =
  DATA.trades.length ? `${{DATA.trades.length}} most recent` : '';
renderTable('trades', DATA.trades, [
  {{ h: 'Time',     fmt: r => fmtTimeShort(r.timestamp) }},
  {{ h: 'SN',       fmt: r => `<span class="sn-badge">SN ${{r.netuid}}</span>` }},
  {{ h: 'Side',     fmt: r => `<span class="pill ${{r.direction}}">${{r.direction}}</span>` }},
  {{ h: 'Strategy', fmt: r => `<span class="pill">${{escapeHtml(r.strategy)}}</span>` }},
  {{ h: 'TAO',      fmt: r => 'τ' + r.tao.toFixed(3) }},
  {{ h: 'Alpha',    fmt: r => r.alpha.toFixed(2) }},
  {{ h: 'Slip',     fmt: r => (r.slip*100).toFixed(3) + '%' }},
  {{ h: 'P&L %',    fmt: r => r.pnl_pct == null ? '<span style="color:var(--text-dim)">, </span>'
                              : `<span class="${{r.pnl_pct>=0?'up':'down'}}">${{fmtPctSigned(r.pnl_pct)}}</span>` }},
], 'No trades yet.');
</script>
</body>
</html>
"""


# --------------------------- driver ----------------------------------------

def generate_once(
    api_url: str,
    dashboard_path: str,
    trades_path: str,
    state_path: str,
    out_path: str,
    history_hours: int,
) -> None:
    base_data = _load_running_data(dashboard_path)
    trades = _load_trades(trades_path)
    state = _load_state(state_path)

    value_series = base_data.get("value_series") or []
    if not value_series:
        raise RuntimeError("Source dashboard has no value_series, paper trader not running yet?")
    start_time = _parse_ts(value_series[0]["t"])
    initial_capital = float(value_series[0]["v"])

    bench, bench_count = _compute_benchmark_series(
        api_url, start_time, initial_capital, history_hours,
    )
    html = _render_html(base_data, trades, state, bench, bench_count)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Wrote {out_path}  "
          f"(pool-weighted B&H across {bench_count} subnets, {len(bench)} pts)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--dashboard", default=DEFAULT_DASHBOARD_PATH)
    parser.add_argument("--trades", default=DEFAULT_TRADES_PATH)
    parser.add_argument("--state", default=DEFAULT_STATE_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--history-hours", type=int, default=DEFAULT_HISTORY_HOURS)
    parser.add_argument("--loop", type=int, default=0,
                        help="If > 0, regenerate every N seconds in a loop.")
    args = parser.parse_args()

    if args.loop <= 0:
        generate_once(args.api_url, args.dashboard, args.trades, args.state,
                      args.out, args.history_hours)
        return 0

    while True:
        try:
            generate_once(args.api_url, args.dashboard, args.trades, args.state,
                          args.out, args.history_hours)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] error: {e}", file=sys.stderr)
        time.sleep(args.loop)


if __name__ == "__main__":
    sys.exit(main())
