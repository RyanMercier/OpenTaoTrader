"""CLI entry point for the trading system.

Subcommands:
  backtest       run a historical replay
  paper          start live paper trading loop
  paper-status   print current paper state
  info           show what data is available
  scan           compute features and list active signals without trading
  dashboard      generate an HTML dashboard (backtest or paper)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from .config import TradingConfig


def _apply_args_to_config(args, config: TradingConfig) -> TradingConfig:
    if getattr(args, "db", None):
        config.db_path = args.db
    if getattr(args, "api_url", None):
        config.opentao_api_url = args.api_url
    if getattr(args, "capital", None) is not None:
        config.initial_capital_tao = args.capital
    if getattr(args, "max_positions", None) is not None:
        config.max_positions = args.max_positions
    if getattr(args, "hotkeys", None) is not None:
        config.num_hotkeys = args.hotkeys
    if getattr(args, "external_strategies", None):
        config.external_strategy_paths = list(args.external_strategies)
    return config


def _normalize_date(s: str | None) -> str | None:
    if not s:
        return None
    if "T" in s:
        return s
    # Bare YYYY-MM-DD, expand to midnight UTC
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return f"{s}T00:00:00"
    except ValueError:
        return s


def cmd_info(args) -> int:
    from .data import DataLoader
    config = _apply_args_to_config(args, TradingConfig())
    loader = DataLoader(config.db_path)
    print(f"Database: {config.db_path}")
    netuids = loader.get_available_netuids()
    print(f"Netuids with data: {len(netuids)}")
    if not netuids:
        print("No data.")
        return 1
    counts = loader.get_snapshot_counts()
    lo, hi = loader.get_data_range()
    print(f"Time range: {lo}  ->  {hi}")
    print()
    print(f"{'netuid':>7} {'snapshots':>10} {'first':>26} {'last':>26}")
    for n in netuids:
        rng = loader.get_data_range(n)
        print(f"{n:>7} {counts.get(n, 0):>10}  {rng[0][:25]:>26}  {rng[1][:25]:>26}")
    return 0


def cmd_backtest(args) -> int:
    from .backtester import Backtester
    from .dashboard import generate_backtest_dashboard, open_dashboard
    from .report import print_backtest_report, save_backtest_json

    config = _apply_args_to_config(args, TradingConfig())
    start = _normalize_date(args.start)
    end = _normalize_date(args.end)
    netuids = None
    if args.netuids:
        netuids = [int(x) for x in args.netuids.split(",") if x.strip()]
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    bt = Backtester(config)
    result = bt.run(start=start, end=end, netuids=netuids, regime=args.regime, strategies=strategies)
    print_backtest_report(result)

    output_json = args.output or "results/backtest_result.json"
    save_backtest_json(result, output_json)
    print(f"\nJSON saved: {output_json}")

    if args.dashboard:
        dashboard_path = output_json.replace(".json", "_dashboard.html")
        if dashboard_path == output_json:
            dashboard_path = output_json + ".html"
        generate_backtest_dashboard(result, dashboard_path)
        print(f"Dashboard:   {dashboard_path}")
        if args.open:
            open_dashboard(dashboard_path)
    return 0


def cmd_paper(args) -> int:
    from .paper_trader import PaperTrader
    config = _apply_args_to_config(args, TradingConfig())

    # Extra sizing/risk overrides for paper, not in _apply_args_to_config
    if getattr(args, "max_positions", None) is not None:
        config.max_positions = args.max_positions
    if getattr(args, "conc", None) is not None:
        config.max_single_position_pct = args.conc
    if getattr(args, "reserve", None) is not None:
        config.reserve_pct = args.reserve
    if getattr(args, "poll_interval", None) is not None:
        config.paper_poll_interval_seconds = args.poll_interval
    if getattr(args, "state_file", None):
        config.paper_state_file = args.state_file
    if getattr(args, "trade_log", None):
        config.paper_trade_log = args.trade_log
    if getattr(args, "dashboard_path", None):
        config.paper_dashboard_path = args.dashboard_path

    strategies = None
    if getattr(args, "strategies", None):
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    if args.resume:
        print("Resuming from saved paper state.")
    elif os.path.exists(config.paper_state_file):
        print(f"Warning: paper state exists at {config.paper_state_file} but --resume not set.")
        print("         Delete the file or pass --resume to continue. Starting fresh would overwrite.")
        return 1
    trader = PaperTrader(config, allowed_strategies=strategies)
    trader.run_loop()
    return 0


def cmd_paper_status(args) -> int:
    from .paper_trader import PaperTrader
    from .report import print_paper_status
    config = _apply_args_to_config(args, TradingConfig())
    if not os.path.exists(config.paper_state_file):
        print(f"No paper state at {config.paper_state_file}")
        return 1
    trader = PaperTrader(config)
    # Use the most recent buffer snapshots (loaded from DB)
    current = {n: buf[-1] for n, buf in trader._snapshot_buffer.items() if buf}
    print_paper_status(trader.portfolio, current)
    return 0


def cmd_scan(args) -> int:
    """Compute features for every subnet at the latest available snapshot and
    print any strategy signals that fire. Read-only, no trades."""
    from .data import DataLoader
    from .features import FeatureEngine
    from .strategies import (
        StakeVelocityStrategy, MeanReversionStrategy, MomentumStrategy,
        DrainDetector, STRATEGIES, load_external_strategies,
    )

    config = _apply_args_to_config(args, TradingConfig())
    loader = DataLoader(config.db_path)
    engine = FeatureEngine()

    strategies = [
        StakeVelocityStrategy(config),
        MeanReversionStrategy(config),
        MomentumStrategy(config),
        DrainDetector(config),
    ]
    if config.external_strategy_paths:
        load_external_strategies(":".join(config.external_strategy_paths))
    builtin_keys = {"stake_velocity", "mean_reversion", "momentum", "drain_exit"}
    for key, cls in STRATEGIES.items():
        if key in builtin_keys:
            continue
        strategies.append(cls(config))

    netuids = loader.get_available_netuids()
    all_snaps = loader.load_all_snapshots(netuids=netuids)
    current = {n: s[-1] for n, s in all_snaps.items() if s}

    fired = []
    for n, snaps in all_snaps.items():
        if n in config.exclude_netuids:
            continue
        if len(snaps) < config.min_snapshots:
            continue
        feats = engine.compute(snaps, len(snaps) - 1, current)
        snap = snaps[-1]
        for s in strategies:
            if isinstance(s, DrainDetector):
                s.update(n, feats)
        for s in strategies:
            if not s.can_run_in_regime(snap.regime):
                continue
            sig = s.generate_entry_signal(n, feats, snap)
            if sig:
                fired.append(sig)

    print("\nSignal scan, latest snapshot per subnet")
    print("-" * 72)
    if not fired:
        print("No active entry signals.")
        return 0
    fired.sort(key=lambda s: -s.strength)
    for sig in fired:
        print(f"[{sig.strength:.2f}] SN{sig.netuid} {sig.strategy.value}: {sig.reason}")
    return 0


def cmd_dashboard(args) -> int:
    from .dashboard import generate_backtest_dashboard, generate_paper_dashboard, open_dashboard

    if args.paper:
        # Regenerate paper dashboard from saved state
        from .config import TradingConfig as _TC
        from .paper_trader import PaperTrader
        config = _TC()
        if getattr(args, "db", None):
            config.db_path = args.db
        if not os.path.exists(config.paper_state_file):
            print(f"No paper state at {config.paper_state_file}")
            return 1
        trader = PaperTrader(config)
        current = {n: buf[-1] for n, buf in trader._snapshot_buffer.items() if buf}
        out = args.output or "data/paper_dashboard.html"
        path = generate_paper_dashboard(trader.portfolio, current_snapshots=current, output_path=out)
        print(f"Dashboard: {path}")
        if args.open:
            open_dashboard(path)
        return 0

    if not args.input:
        print("Provide --input <backtest_result.json> or --paper")
        return 1
    with open(args.input, "r") as f:
        data = json.load(f)

    # Rehydrate just enough of BacktestResult for the dashboard.
    from .models import Trade, Direction, StrategyName
    trades = []
    for t in data.get("trades", []):
        try:
            trades.append(Trade(
                id=t["id"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                block=t["block"],
                netuid=t["netuid"],
                direction=Direction(t["direction"]),
                strategy=StrategyName(t["strategy"]),
                tao_amount=t["tao_amount"],
                alpha_amount=t["alpha_amount"],
                spot_price=t["spot_price"],
                effective_price=t["effective_price"],
                slippage_pct=t["slippage_pct"],
                signal_strength=t["signal_strength"],
                hotkey_id=t["hotkey_id"],
                entry_price=t.get("entry_price"),
                pnl_tao=t.get("pnl_tao"),
                pnl_pct=t.get("pnl_pct"),
                hold_duration_hours=t.get("hold_duration_hours"),
                entry_strategy=StrategyName(t["entry_strategy"]) if t.get("entry_strategy") else None,
            ))
        except Exception:
            continue

    # Minimal reconstruction: drop unsupported fields, pass through dict-like
    # BacktestResult via a simple namespace sharing the same attributes the
    # dashboard reads.
    from types import SimpleNamespace
    result = SimpleNamespace(**data)
    result.trades = trades
    out = args.output or args.input.replace(".json", "_dashboard.html")
    path = generate_backtest_dashboard(result, out)
    print(f"Dashboard: {path}")
    if args.open:
        open_dashboard(path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bittensor Subnet Trading System")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Run historical backtest")
    bt.add_argument("--start", type=str, default=None)
    bt.add_argument("--end", type=str, default=None)
    bt.add_argument("--netuids", type=str, default=None)
    bt.add_argument("--regime", type=str, choices=["price_based","taoflow_prehalving","taoflow_posthalving"])
    bt.add_argument("--capital", type=float, default=None)
    bt.add_argument("--max-positions", dest="max_positions", type=int, default=None)
    bt.add_argument("--hotkeys", type=int, default=None)
    bt.add_argument("--strategies", type=str, default=None)
    bt.add_argument("--output", type=str, default=None)
    bt.add_argument("--dashboard", action="store_true", default=True)
    bt.add_argument("--open", action="store_true")
    bt.add_argument("--db", type=str, default=None)

    pp = sub.add_parser("paper", help="Start paper trading")
    pp.add_argument("--capital", type=float, default=None)
    pp.add_argument("--resume", action="store_true")
    pp.add_argument("--dashboard", action="store_true", default=True)
    pp.add_argument("--api-url", dest="api_url", type=str, default=None)
    pp.add_argument("--db", type=str, default=None)
    pp.add_argument("--strategies", type=str, default=None, help="Comma list, e.g. momentum,xstmc")
    pp.add_argument("--hotkeys", type=int, default=None)
    pp.add_argument("--max-positions", dest="max_positions", type=int, default=None)
    pp.add_argument("--conc", type=float, default=None, help="max_single_position_pct (0-1)")
    pp.add_argument("--reserve", type=float, default=None, help="reserve_pct (0-1)")
    pp.add_argument("--poll-interval", dest="poll_interval", type=int, default=None, help="seconds between polls")
    pp.add_argument("--state-file", dest="state_file", type=str, default=None)
    pp.add_argument("--trade-log", dest="trade_log", type=str, default=None)
    pp.add_argument("--dashboard-path", dest="dashboard_path", type=str, default=None)

    ps = sub.add_parser("paper-status", help="Show paper trading status")
    ps.add_argument("--db", type=str, default=None)

    inf = sub.add_parser("info", help="Show available data summary")
    inf.add_argument("--db", type=str, default=None)

    sc = sub.add_parser("scan", help="Scan for current signals")
    sc.add_argument("--db", type=str, default=None)
    sc.add_argument("--api-url", dest="api_url", type=str, default=None)

    dd = sub.add_parser("dashboard", help="Generate visual dashboard")
    dd.add_argument("--input", type=str, default=None)
    dd.add_argument("--paper", action="store_true")
    dd.add_argument("--output", type=str, default=None)
    dd.add_argument("--open", action="store_true")
    dd.add_argument("--db", type=str, default=None)

    lv = sub.add_parser("live", help="Start LIVE trading on a paper portfolio")
    lv.add_argument("--portfolio", type=str, required=True,
                    help="Portfolio name (must already exist; create via API or paper run first)")
    lv.add_argument("--wallet", dest="wallet_name", type=str, required=True,
                    help="Bittensor wallet name on disk (~/.bittensor/wallets/<name>/)")
    lv.add_argument("--hotkey", dest="hotkey_name", type=str, default="default",
                    help="Hotkey name on the wallet (default: 'default')")
    lv.add_argument("--db", type=str, default=None,
                    help="Path to opentaotrader.db (default: from settings)")
    lv.add_argument("--api-url", dest="api_url", type=str, default=None,
                    help="URL of the OpenTaoAPI instance to read snapshots from")
    lv.add_argument("--poll-interval", dest="poll_interval", type=int, default=None,
                    help="Override paper_poll_interval_seconds for this run")
    lv.add_argument("--no-confirm", dest="no_confirm", action="store_true",
                    help="Skip the [y/N] confirmation prompt before the first trade")
    lv.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="Run the strategy pipeline but skip extrinsic submission")

    cmp = sub.add_parser("compare", help="Generate multi-config comparison dashboard")
    cmp.add_argument("--summary", type=str, default="results/compare/summary.json")
    cmp.add_argument("--output", type=str, default="results/comparison_dashboard.html")
    cmp.add_argument("--open", action="store_true")

    mc = sub.add_parser("mc", help="Monte Carlo analysis over the backtester")
    mc.add_argument("--mode", type=str, choices=["bootstrap", "netuids", "sweep"], default="bootstrap")
    mc.add_argument("--runs", type=int, default=50, help="Bootstrap runs (bootstrap mode)")
    mc.add_argument("--window-days", dest="window_days", type=int, default=60)
    mc.add_argument("--seed", type=int, default=42)
    mc.add_argument("--param", type=str, default=None, help="TradingConfig attribute to sweep")
    mc.add_argument("--values", type=str, default=None, help="Comma-separated grid of values for --param")
    mc.add_argument("--value-type", type=str, choices=["float","int","str"], default="float")
    mc.add_argument("--capital", type=float, default=None)
    mc.add_argument("--strategies", type=str, default=None)
    mc.add_argument("--hotkeys", type=int, default=None)
    mc.add_argument("--output", type=str, default=None)
    mc.add_argument("--db", type=str, default=None)

    args = parser.parse_args()

    if args.command == "backtest":
        return cmd_backtest(args)
    if args.command == "paper":
        return cmd_paper(args)
    if args.command == "paper-status":
        return cmd_paper_status(args)
    if args.command == "info":
        return cmd_info(args)
    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "dashboard":
        return cmd_dashboard(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "mc":
        return cmd_mc(args)
    if args.command == "live":
        return cmd_live(args)
    return 1


def cmd_live(args) -> int:
    """Start the live runner. Loads the wallet locally, prompts for the
    coldkey password, then loops calling LiveTrader.run_once. Keys never
    leave this process. Use Ctrl+C to stop."""
    import asyncio
    return asyncio.run(_run_live(args))


async def _run_live(args) -> int:
    import asyncio
    import getpass
    import json as _json

    from server.config import settings
    from server.services.cache import TTLCache
    from server.services.chain_client import ChainClient
    from server.database import Database
    from server.api_client import OpenTaoAPIClient

    db_path = args.db or settings.database_path
    api_url = args.api_url or settings.opentao_api_url
    database = Database(db_path)
    cache = TTLCache()
    chain_client = ChainClient(cache)
    api_client = OpenTaoAPIClient(api_url)

    print(f"Opening trader database: {db_path}")
    print(f"Reading data from:       {api_url}")
    await database.startup()
    await chain_client.startup()
    await api_client.startup()
    await api_client.seed_snapshots(hours=720)
    await api_client.start_sse()

    try:
        # 1. Look up the portfolio.
        portfolios = await database.list_paper_portfolios(active_only=False)
        match = next((p for p in portfolios if p["name"] == args.portfolio), None)
        if not match:
            print(f"Portfolio {args.portfolio!r} not found. Create it first via the web UI or API.", file=sys.stderr)
            return 2

        # 2. Promote (or verify) live mode + persist wallet/hotkey on the row.
        await database.set_paper_portfolio_mode(
            match["id"], "live",
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
        )

        # 3. Load + unlock the wallet.
        try:
            import bittensor_wallet  # type: ignore
        except ImportError:
            print("bittensor_wallet not installed. Install with: pip install bittensor", file=sys.stderr)
            return 3
        wallet = bittensor_wallet.Wallet(name=args.wallet_name, hotkey=args.hotkey_name)
        password = getpass.getpass(f"Coldkey password for wallet {args.wallet_name!r}: ")
        try:
            # Newer SDKs expose unlock_coldkey(); older ones decrypt on
            # first .coldkey access. Try both so this works across versions.
            if hasattr(wallet, "unlock_coldkey"):
                wallet.unlock_coldkey(password)
            else:
                _ = wallet.coldkey
        except Exception as e:
            print(f"Failed to unlock coldkey: {e}", file=sys.stderr)
            return 4
        ck_addr = wallet.coldkeypub.ss58_address
        hk_addr = wallet.hotkey.ss58_address
        print(f"Coldkey: {ck_addr}")
        print(f"Hotkey:  {hk_addr}")

        # 4. Build the trader.
        config_dict = _json.loads(match["config_json"]) if match.get("config_json") else {}
        config = TradingConfig()
        config.db_path = db_path
        config.initial_capital_tao = float(match["initial_capital_tao"])
        for attr in ("strategies", "max_positions", "max_single_position_pct",
                     "reserve_pct", "max_position_pct_of_pool", "max_slippage_pct",
                     "num_hotkeys", "external_strategy_paths",
                     "paper_poll_interval_seconds"):
            if attr in config_dict and config_dict[attr] is not None:
                setattr(config, attr, config_dict[attr])
        if args.poll_interval:
            config.paper_poll_interval_seconds = int(args.poll_interval)

        from .paper_trader import hydrate_portfolio
        portfolio = await hydrate_portfolio(database, match["id"], config)

        if args.dry_run:
            print(">>> DRY RUN: signals will be evaluated but no extrinsics submitted")
            from .paper_trader import PaperTrader
            trader = PaperTrader(
                portfolio_id=match["id"],
                config=config,
                portfolio=portfolio,
                api_client=api_client,
                database=database,
            )
        else:
            from .live_trader import LiveTrader
            trader = LiveTrader(
                portfolio_id=match["id"],
                config=config,
                portfolio=portfolio,
                api_client=api_client,
                database=database,
                chain_client=chain_client,
                wallet=wallet,
                hotkey_name=args.hotkey_name,
            )

        # 5. Confirm.
        if not args.no_confirm and not args.dry_run:
            print()
            print("=" * 60)
            print("LIVE TRADING CONFIRMATION")
            print("=" * 60)
            print(f"Portfolio:     {match['name']} (id={match['id']})")
            print(f"Capital:       {match['initial_capital_tao']} TAO (allocated)")
            print(f"Free balance:  {portfolio.free_tao} TAO")
            print(f"Strategies:    {config.strategies}")
            print(f"Cadence:       {config.paper_poll_interval_seconds}s")
            print(f"Daily loss cap:{config.daily_loss_limit_pct * 100:.1f}%")
            print(f"Max slippage:  {config.max_slippage_pct * 100:.2f}%")
            print(f"Max pos/pool:  {config.max_position_pct_of_pool * 100:.2f}%")
            print()
            print("Real TAO will move on-chain when strategies fire. Ctrl+C to stop.")
            ans = input("Type 'go' to continue: ").strip().lower()
            if ans != "go":
                print("Aborted.")
                return 5

        print()
        print(f"Live trader running. Cycle every {config.paper_poll_interval_seconds}s. Ctrl+C to stop.")
        print()

        # 6. Loop.
        try:
            while True:
                # Check active flag in DB; respect web pause.
                row = await database.get_paper_portfolio(match["id"])
                if not row or not row["active"]:
                    print("Portfolio is paused (active=0). Sleeping 60s before re-check.")
                    await asyncio.sleep(60)
                    continue
                try:
                    result = await trader.run_once()
                    print(f"[{datetime.now().isoformat(timespec='seconds')}] cycle: {result}")
                except Exception as e:
                    print(f"Cycle error: {e}", file=sys.stderr)
                await asyncio.sleep(config.paper_poll_interval_seconds)
        except KeyboardInterrupt:
            print()
            print("Stopping. Final state already persisted from last cycle.")
            return 0
    finally:
        for closer in (chain_client.shutdown, api_client.shutdown, database.shutdown):
            try:
                await closer()
            except Exception:
                pass
    return 0


def cmd_compare(args) -> int:
    from .compare_dashboard import generate_comparison_dashboard
    from .dashboard import open_dashboard
    path = generate_comparison_dashboard(args.summary, args.output)
    print(f"Dashboard: {path}")
    if args.open:
        open_dashboard(path)
    return 0


def cmd_mc(args) -> int:
    from .montecarlo import MonteCarloRunner, print_mc_report, save_mc_json
    config = _apply_args_to_config(args, TradingConfig())
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    runner = MonteCarloRunner(config)

    if args.mode == "bootstrap":
        runs = runner.random_window_bootstrap(
            num_runs=args.runs,
            window_days=args.window_days,
            strategies=strategies,
            seed=args.seed,
        )
        title = f"Bootstrap: {args.runs} runs x {args.window_days}d windows"
    elif args.mode == "netuids":
        runs = runner.netuid_subsampling(strategies=strategies)
        title = "Netuid subsampling"
    elif args.mode == "sweep":
        if not args.param or not args.values:
            print("Sweep mode requires --param and --values")
            return 1
        cast = {"float": float, "int": int, "str": str}[args.value_type]
        values = [cast(v.strip()) for v in args.values.split(",")]
        runs = runner.parameter_sweep(args.param, values, strategies=strategies)
        title = f"Parameter sweep: {args.param} across {values}"
    else:
        return 1

    print_mc_report(runs, title=title)
    out = args.output or "results/mc_runs.json"
    save_mc_json(runs, out)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
