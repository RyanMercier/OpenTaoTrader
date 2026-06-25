"""OpenTaoTrader FastAPI app.

Hosts the paper-trading dashboard at ``/paper`` and the management API at
``/api/v1/paper/...``. Lifespan owns:

  - The HTTP+SSE client to the upstream OpenTaoAPI (seeds snapshots,
    keeps a rolling buffer fed by SSE).
  - The trader's own SQLite (paper trading state).
  - The paper-trading runner (gated by ``PAPER_TRADING_ENABLED``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.api_client import OpenTaoAPIClient
from server.config import settings
from server.database import Database
from server.routes.paper import router as paper_router, init_paper_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

database = Database(settings.database_path)
api_client = OpenTaoAPIClient(settings.opentao_api_url)


# --- Paper trading runner -----------------------------------------

_paper_traders: dict = {}   # portfolio_id -> PaperTrader instance


async def _paper_trader_runner():
    """Advance every active paper portfolio one cycle when its interval
    elapses. ``_paper_traders`` keeps trader instances alive between
    cycles so position state and strategy internal counters survive."""
    if not settings.paper_trading_enabled:
        logger.info("Paper trader disabled (PAPER_TRADING_ENABLED=false)")
        return

    # Lazy import: keep the trading package off the import path for
    # instances that just serve the dashboard.
    from trading.paper_trader import PaperTrader, hydrate_portfolio
    from trading.strategies import load_external_strategies

    if settings.opentao_external_strategies:
        load_external_strategies(settings.opentao_external_strategies)

    logger.info("Paper trader runner started")

    while True:
        try:
            portfolios = await database.list_paper_portfolios(active_only=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Paper trader: failed to load portfolios")
            await asyncio.sleep(60)
            continue

        now_ts = time.time()
        # If nothing is due, recheck in 60s so newly-created portfolios
        # get a quick first cycle. The for-loop tightens this further
        # when portfolios are pending their next interval.
        next_due_in = 60.0

        for row in portfolios:
            pid = row["id"]
            cfg_dict = {}
            try:
                cfg_dict = json.loads(row["config_json"])
            except Exception:
                logger.warning("Paper portfolio %d has unparseable config_json", pid)

            interval = int(cfg_dict.get("paper_poll_interval_seconds", 1800))

            last = row.get("last_cycle_at")
            if last:
                try:
                    last_ts = datetime.fromisoformat(last).timestamp()
                except Exception:
                    last_ts = 0.0
                age = now_ts - last_ts
                if age < interval:
                    next_due_in = min(next_due_in, max(60.0, interval - age))
                    continue

            trader = _paper_traders.get(pid)
            if trader is None:
                config = _build_trading_config(cfg_dict, row)
                portfolio = await hydrate_portfolio(database, pid, config)
                trader = PaperTrader(
                    portfolio_id=pid,
                    config=config,
                    portfolio=portfolio,
                    api_client=api_client,
                    database=database,
                )
                _paper_traders[pid] = trader

            try:
                result = await asyncio.wait_for(trader.run_once(), timeout=180)
                logger.info("Paper portfolio %d cycle: %s", pid, result)
            except asyncio.TimeoutError:
                logger.warning("Paper portfolio %d cycle timed out", pid)
                await database.update_paper_portfolio_runtime(
                    pid,
                    peak_value=trader.portfolio.peak_value,
                    free_tao=trader.portfolio.free_tao,
                    hotkey_cooldowns=trader.portfolio.hotkey_cooldowns,
                    last_cycle_at=datetime.now(timezone.utc).isoformat(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Paper portfolio %d cycle failed", pid)
                try:
                    await database.update_paper_portfolio_runtime(
                        pid,
                        peak_value=trader.portfolio.peak_value,
                        free_tao=trader.portfolio.free_tao,
                        hotkey_cooldowns=trader.portfolio.hotkey_cooldowns,
                        last_cycle_at=datetime.now(timezone.utc).isoformat(),
                    )
                except Exception:
                    logger.exception("Paper portfolio %d: failed to update last_cycle_at", pid)

            next_due_in = min(next_due_in, float(interval))

        # Drop traders for portfolios that are no longer active.
        active_ids = {p["id"] for p in portfolios}
        for stale_id in list(_paper_traders):
            if stale_id not in active_ids:
                _paper_traders.pop(stale_id, None)

        await asyncio.sleep(max(15.0, min(next_due_in, 600.0)))


def _build_trading_config(cfg_dict: dict, portfolio_row: dict):
    """Build a TradingConfig from a portfolio row plus its stored
    ``config_json``. Defaults come from the dataclass; per-portfolio
    overrides win."""
    from trading.config import TradingConfig
    config = TradingConfig()
    config.initial_capital_tao = float(
        portfolio_row.get("initial_capital_tao", config.initial_capital_tao)
    )
    KNOWN_RISK_ATTRS = {
        "strategies", "max_positions", "max_single_position_pct",
        "reserve_pct", "max_position_pct_of_pool", "max_slippage_pct",
        "num_hotkeys", "external_strategy_paths",
        "paper_poll_interval_seconds",
    }
    # Curated risk knobs first.
    for attr in KNOWN_RISK_ATTRS:
        if attr in cfg_dict and cfg_dict[attr] is not None:
            setattr(config, attr, cfg_dict[attr])
    # Per-portfolio strategy parameter overrides (lam_stop_loss_pct etc).
    # Only apply if the key names a real TradingConfig field — silently drop
    # typos so a malformed POST can't break the runner.
    for attr, value in cfg_dict.items():
        if attr in KNOWN_RISK_ATTRS or value is None:
            continue
        if hasattr(config, attr):
            setattr(config, attr, value)
    return config


async def _paper_trader_supervisor():
    while True:
        try:
            await _paper_trader_runner()
            logger.info("Paper trader runner exited cleanly; supervisor stopping")
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Paper trader runner crashed; restarting in 30s")
            await asyncio.sleep(30)


# --- Lifespan ------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    await database.startup()
    await api_client.startup()
    init_paper_router(database, api_client)

    # Seed initial history + start the SSE consumer. If the upstream API
    # is down at boot we still come up; the runner will skip cycles
    # until /health reports the upstream is healthy.
    try:
        await api_client.seed_snapshots(hours=720)
    except Exception:
        logger.exception("Initial snapshot seed failed; will retry over SSE")
    await api_client.start_sse()

    paper_task = asyncio.create_task(_paper_trader_supervisor())

    yield

    paper_task.cancel()
    try:
        await paper_task
    except asyncio.CancelledError:
        pass
    await api_client.shutdown()
    await database.shutdown()


# --- FastAPI app ---------------------------------------------------

app = FastAPI(
    title="OpenTaoTrader",
    description="Paper + live trading runner that reads Bittensor data from OpenTaoAPI.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(paper_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "upstream_api": settings.opentao_api_url,
        "paper_trading_enabled": settings.paper_trading_enabled,
        "active_traders": len(_paper_traders),
    }


# --- Static frontend ----------------------------------------------

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(FRONTEND_DIR / "paper.html")


@app.get("/paper", include_in_schema=False)
async def paper_index():
    return FileResponse(FRONTEND_DIR / "paper.html")


@app.get("/paper/{portfolio_id}", include_in_schema=False)
async def paper_detail(portfolio_id: int):
    return FileResponse(FRONTEND_DIR / "paper.html")
