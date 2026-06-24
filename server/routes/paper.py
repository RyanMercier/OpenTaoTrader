"""Paper trading routes.

Reads return whatever the runner has persisted; writes create or pause
portfolios. The runner that actually advances cycles is gated by
``PAPER_TRADING_ENABLED`` so a public instance can host the dashboard
without running anyone's bot.
"""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from server.models.schemas import (
    PaperPortfolio,
    PaperPortfolioCreate,
    PaperPortfolioStats,
    PaperPosition,
    PaperTrade,
    PaperValueHistory,
    PaperValuePoint,
    StrategyDescriptor,
)
from server.database import Database
from server.api_client import OpenTaoAPIClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["paper"])

_db: Database | None = None
_api_client: OpenTaoAPIClient | None = None


def init_paper_router(db: Database, api_client: OpenTaoAPIClient) -> None:
    global _db, _api_client
    _db = db
    _api_client = api_client


def _row_to_portfolio(row: dict) -> PaperPortfolio:
    cfg = {}
    strategies: list[str] = []
    raw = row.get("config_json")
    if raw:
        try:
            cfg = json.loads(raw)
            strategies = list(cfg.get("strategies") or [])
        except Exception:
            cfg = {}
    return PaperPortfolio(
        id=row["id"],
        name=row["name"],
        initial_capital_tao=row["initial_capital_tao"],
        active=bool(row["active"]),
        created_at=row["created_at"],
        mode=row.get("mode") or "paper",
        wallet_name=row.get("wallet_name"),
        hotkey_name=row.get("hotkey_name"),
        last_cycle_at=row.get("last_cycle_at"),
        free_tao=row.get("free_tao"),
        peak_value=row.get("peak_value"),
        strategies=strategies,
        config=cfg,
    )


@router.post(
    "/paper/portfolios",
    response_model=PaperPortfolio,
    status_code=201,
    summary="Create a paper-trading portfolio",
)
async def create_paper_portfolio(req: PaperPortfolioCreate):
    if not _db:
        raise HTTPException(status_code=503, detail="Database not available")

    config_dict = {
        "initial_capital_tao": req.initial_capital_tao,
        "strategies": req.strategies,
        "paper_poll_interval_seconds": req.poll_interval_seconds,
        "max_positions": req.max_positions,
        "max_single_position_pct": req.max_single_position_pct,
        "reserve_pct": req.reserve_pct,
        "max_position_pct_of_pool": req.max_position_pct_of_pool,
        "max_slippage_pct": req.max_slippage_pct,
        "num_hotkeys": req.num_hotkeys,
        "external_strategy_paths": req.external_strategy_paths,
    }
    try:
        portfolio_id = await _db.create_paper_portfolio(
            name=req.name,
            initial_capital_tao=req.initial_capital_tao,
            config_json=json.dumps(config_dict),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        # Likely UNIQUE constraint on name.
        raise HTTPException(status_code=400, detail=str(e))

    row = await _db.get_paper_portfolio(portfolio_id)
    if not row:
        raise HTTPException(status_code=500, detail="Portfolio created but lookup failed")
    return _row_to_portfolio(row)


@router.get(
    "/paper/portfolios",
    response_model=list[PaperPortfolio],
    summary="List paper-trading portfolios",
)
async def list_paper_portfolios():
    if not _db:
        raise HTTPException(status_code=503, detail="Database not available")
    rows = await _db.list_paper_portfolios(active_only=False)
    return [_row_to_portfolio(r) for r in rows]


@router.get(
    "/paper/portfolios/{portfolio_id}",
    response_model=PaperPortfolio,
)
async def get_paper_portfolio(portfolio_id: int):
    if not _db:
        raise HTTPException(status_code=503, detail="Database not available")
    row = await _db.get_paper_portfolio(portfolio_id)
    if not row:
        raise HTTPException(status_code=404, detail="Paper portfolio not found")
    return _row_to_portfolio(row)


def _mark_to_market(row: dict) -> dict:
    """Add current_price / current_value_tao / unrealized_pnl_* to a position
    row using the latest snapshot held by the api_client (if any). Constant-
    product exit math, same model as the live trader and backtester."""
    if _api_client is None:
        return row
    buf = _api_client.snapshot_buffer.get(row["netuid"])
    if not buf:
        return row
    latest = buf[-1]
    tao_in = float(latest.get("tao_in") or 0.0)
    alpha_in = float(latest.get("alpha_in") or 0.0)
    alpha = float(row["alpha_amount"])
    invested = float(row["tao_invested"])
    if tao_in <= 0 or alpha_in <= 0 or alpha <= 0 or invested <= 0:
        return row
    # AMM exit: simulate selling our alpha back into the pool.
    k = tao_in * alpha_in
    new_alpha_in = alpha_in + alpha
    new_tao_in = k / new_alpha_in
    tao_received = tao_in - new_tao_in
    pnl_tao = tao_received - invested
    row["current_price"] = tao_in / alpha_in
    row["current_value_tao"] = tao_received
    row["unrealized_pnl_tao"] = pnl_tao
    row["unrealized_pnl_pct"] = pnl_tao / invested
    return row


@router.get(
    "/paper/portfolios/{portfolio_id}/positions",
    response_model=list[PaperPosition],
)
async def get_paper_positions(portfolio_id: int):
    if not _db:
        raise HTTPException(status_code=503, detail="Database not available")
    if not await _db.get_paper_portfolio(portfolio_id):
        raise HTTPException(status_code=404, detail="Paper portfolio not found")
    rows = await _db.list_paper_positions(portfolio_id)
    enriched = [_mark_to_market(dict(r)) for r in rows]
    return [PaperPosition(**r) for r in enriched]


@router.get(
    "/paper/portfolios/{portfolio_id}/trades",
    response_model=list[PaperTrade],
)
async def get_paper_trades(
    portfolio_id: int,
    limit: int = 200,
    closed_only: bool = False,
):
    """Closed trades = ``direction == "sell"`` (the round-trip realised on
    the exit leg). Buys are open-position events; with ``closed_only=true``
    they're filtered out so the table is purely realised P&L."""
    if not _db:
        raise HTTPException(status_code=503, detail="Database not available")
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit out of range (1..5000)")
    if not await _db.get_paper_portfolio(portfolio_id):
        raise HTTPException(status_code=404, detail="Paper portfolio not found")
    rows = await _db.list_paper_trades(portfolio_id, limit=limit)
    if closed_only:
        rows = [r for r in rows if r.get("direction") == "sell"]
    return [PaperTrade(**r) for r in rows]


@router.get(
    "/paper/portfolios/{portfolio_id}/history",
    response_model=PaperValueHistory,
)
async def get_paper_history(
    portfolio_id: int, hours: int = 168, limit: int = 5000
):
    if not _db:
        raise HTTPException(status_code=503, detail="Database not available")
    if hours < 1 or hours > 8760:
        raise HTTPException(status_code=400, detail="hours out of range")

    portfolio_row = await _db.get_paper_portfolio(portfolio_id)
    if not portfolio_row:
        raise HTTPException(status_code=404, detail="Paper portfolio not found")

    rows = await _db.get_paper_value_history(portfolio_id, hours=hours, limit=limit)
    if not rows:
        return PaperValueHistory(portfolio_id=portfolio_id, hours=hours, points=[])

    # Pull the portfolio's universe filter from its stored config so the
    # benchmark mirrors what the trader could have traded.
    initial_capital_tao = float(portfolio_row["initial_capital_tao"])
    config = {}
    if portfolio_row.get("config_json"):
        try:
            config = json.loads(portfolio_row["config_json"])
        except Exception:
            config = {}
    exclude = config.get("exclude_netuids") or [0]
    min_depth = float(config.get("min_pool_depth_tao") or 50.0)

    timestamps = [r["timestamp"] for r in rows]
    anchor_ts = await _db.get_paper_anchor_timestamp(portfolio_id) or timestamps[0]
    benchmark_values: list[float] = []
    universe: list[int] = []
    if _api_client is not None:
        try:
            benchmark_values, universe = await _api_client.compute_benchmark_series(
                timestamps=timestamps,
                anchor_ts=anchor_ts,
                initial_capital_tao=initial_capital_tao,
                exclude_netuids=exclude,
                min_pool_depth_tao=min_depth,
            )
        except Exception:
            logger.exception("Benchmark fetch failed; returning history without it")
    points: list[PaperValuePoint] = []
    for i, r in enumerate(rows):
        bench = benchmark_values[i] if i < len(benchmark_values) else None
        points.append(PaperValuePoint(
            timestamp=r["timestamp"],
            free_tao=r["free_tao"],
            total_value_tao=r["total_value_tao"],
            total_pnl_tao=r["total_pnl_tao"],
            drawdown_pct=r["drawdown_pct"],
            num_open_positions=r["num_open_positions"],
            benchmark_value_tao=bench,
        ))
    return PaperValueHistory(
        portfolio_id=portfolio_id,
        hours=hours,
        points=points,
        benchmark_universe=universe,
        benchmark_anchor_timestamp=anchor_ts,
    )


@router.get(
    "/paper/portfolios/{portfolio_id}/stats",
    response_model=PaperPortfolioStats,
    summary="Headline metrics: Sharpe, win rate, drawdown, return vs benchmark",
)
async def get_paper_stats(portfolio_id: int):
    if not _db:
        raise HTTPException(status_code=503, detail="Database not available")
    portfolio_row = await _db.get_paper_portfolio(portfolio_id)
    if not portfolio_row:
        raise HTTPException(status_code=404, detail="Paper portfolio not found")

    config = {}
    if portfolio_row.get("config_json"):
        try:
            config = json.loads(portfolio_row["config_json"])
        except Exception:
            config = {}
    cadence = int(config.get("paper_poll_interval_seconds") or 1800)
    exclude = config.get("exclude_netuids") or [0]
    min_depth = float(config.get("min_pool_depth_tao") or 50.0)

    stats = await _db.compute_paper_portfolio_stats(
        portfolio_id,
        cadence_seconds=cadence,
        api_client=_api_client,
        exclude_netuids=exclude,
        min_pool_depth_tao=min_depth,
    )
    if stats is None:
        raise HTTPException(status_code=404, detail="Paper portfolio not found")
    return PaperPortfolioStats(**stats)


@router.post("/paper/portfolios/{portfolio_id}/pause")
async def pause_paper_portfolio(portfolio_id: int):
    if not _db:
        raise HTTPException(status_code=503, detail="Database not available")
    if not await _db.set_paper_portfolio_active(portfolio_id, False):
        raise HTTPException(status_code=404, detail="Paper portfolio not found")
    return {"id": portfolio_id, "active": False}


@router.post("/paper/portfolios/{portfolio_id}/resume")
async def resume_paper_portfolio(portfolio_id: int):
    if not _db:
        raise HTTPException(status_code=503, detail="Database not available")
    if not await _db.set_paper_portfolio_active(portfolio_id, True):
        raise HTTPException(status_code=404, detail="Paper portfolio not found")
    return {"id": portfolio_id, "active": True}


@router.get(
    "/trading/strategies",
    response_model=list[StrategyDescriptor],
    summary="List registered trading strategies (built-in + external)",
)
async def list_trading_strategies():
    """Returns the contents of the strategy registry. External strategies
    show their file path under ``source``; built-ins show ``builtin``.
    Importing the registry triggers the built-in decorators."""
    from trading.strategies import list_strategies, load_external_strategies
    # Best-effort: pick up the env-var external paths so they appear in
    # the listing even before the runner has touched them.
    load_external_strategies()
    return [StrategyDescriptor(**s) for s in list_strategies()]
