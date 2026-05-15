"""Trader-only SQLite. Holds paper/live trading state (portfolios,
positions, trades, value history). Subnet snapshots come from OpenTaoAPI
over HTTP, not from here.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    initial_capital_tao REAL NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    mode TEXT NOT NULL DEFAULT 'paper',
    wallet_name TEXT,
    hotkey_name TEXT,
    free_tao REAL,
    peak_value REAL,
    hotkey_cooldowns_json TEXT,
    last_cycle_at TEXT
);

CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    netuid INTEGER NOT NULL,
    entry_block INTEGER NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    alpha_amount REAL NOT NULL,
    tao_invested REAL NOT NULL,
    strategy TEXT NOT NULL,
    hotkey_id INTEGER NOT NULL,
    FOREIGN KEY (portfolio_id) REFERENCES paper_portfolios(id)
);
CREATE INDEX IF NOT EXISTS idx_paper_positions_portfolio
    ON paper_positions(portfolio_id);

CREATE TABLE IF NOT EXISTS paper_trades (
    id TEXT PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    block INTEGER NOT NULL,
    netuid INTEGER NOT NULL,
    direction TEXT NOT NULL,
    strategy TEXT NOT NULL,
    tao_amount REAL NOT NULL,
    alpha_amount REAL NOT NULL,
    spot_price REAL NOT NULL,
    effective_price REAL NOT NULL,
    slippage_pct REAL NOT NULL,
    signal_strength REAL,
    hotkey_id INTEGER,
    entry_price REAL,
    pnl_tao REAL,
    pnl_pct REAL,
    hold_duration_hours REAL,
    entry_strategy TEXT,
    extrinsic_hash TEXT,
    executed_block INTEGER,
    FOREIGN KEY (portfolio_id) REFERENCES paper_portfolios(id)
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_portfolio_ts
    ON paper_trades(portfolio_id, timestamp);

CREATE TABLE IF NOT EXISTS paper_value_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    free_tao REAL NOT NULL,
    total_value_tao REAL NOT NULL,
    total_pnl_tao REAL NOT NULL,
    drawdown_pct REAL NOT NULL,
    num_open_positions INTEGER NOT NULL,
    FOREIGN KEY (portfolio_id) REFERENCES paper_portfolios(id)
);
CREATE INDEX IF NOT EXISTS idx_paper_value_history_portfolio_ts
    ON paper_value_history(portfolio_id, timestamp);
"""


class Database:
    def __init__(self, db_path: str):
        self._path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        # Writes are serialized; reads are unguarded (SQLite handles them).
        self._write_lock = asyncio.Lock()

    async def startup(self):
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info("Trader database ready at %s", self._path)

    async def shutdown(self):
        if self._db:
            await self._db.close()
            self._db = None

    # --- Paper portfolios ---

    async def create_paper_portfolio(
        self,
        name: str,
        initial_capital_tao: float,
        config_json: str,
        created_at: str,
    ) -> int:
        async with self._write_lock:
            cursor = await self._db.execute(
                """INSERT INTO paper_portfolios
                       (name, initial_capital_tao, config_json, created_at,
                        active, free_tao, peak_value, hotkey_cooldowns_json,
                        last_cycle_at)
                   VALUES (?, ?, ?, ?, 1, ?, ?, NULL, NULL)""",
                (name, initial_capital_tao, config_json, created_at,
                 initial_capital_tao, initial_capital_tao),
            )
            await self._db.commit()
            return cursor.lastrowid

    async def list_paper_portfolios(self, active_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM paper_portfolios"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY id ASC"
        cursor = await self._db.execute(sql)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_paper_portfolio(self, portfolio_id: int) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM paper_portfolios WHERE id = ?", (portfolio_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_paper_portfolio_runtime(self, portfolio_id: int) -> dict | None:
        cursor = await self._db.execute(
            """SELECT free_tao, peak_value, hotkey_cooldowns_json, last_cycle_at
               FROM paper_portfolios WHERE id = ?""",
            (portfolio_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_paper_portfolio_runtime(
        self,
        portfolio_id: int,
        peak_value: float,
        free_tao: float,
        hotkey_cooldowns: dict,
        last_cycle_at: str,
    ) -> None:
        cooldowns_json = json.dumps({str(k): int(v) for k, v in hotkey_cooldowns.items()})
        async with self._write_lock:
            await self._db.execute(
                """UPDATE paper_portfolios
                   SET free_tao = ?, peak_value = ?,
                       hotkey_cooldowns_json = ?, last_cycle_at = ?
                   WHERE id = ?""",
                (free_tao, peak_value, cooldowns_json, last_cycle_at, portfolio_id),
            )
            await self._db.commit()

    async def set_paper_portfolio_active(
        self, portfolio_id: int, active: bool
    ) -> bool:
        async with self._write_lock:
            cursor = await self._db.execute(
                "UPDATE paper_portfolios SET active = ? WHERE id = ?",
                (1 if active else 0, portfolio_id),
            )
            await self._db.commit()
            return cursor.rowcount > 0

    async def set_paper_portfolio_mode(
        self,
        portfolio_id: int,
        mode: str,
        wallet_name: str | None = None,
        hotkey_name: str | None = None,
    ) -> bool:
        if mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {mode!r}")
        if mode == "live" and not wallet_name:
            raise ValueError("wallet_name required for mode='live'")
        async with self._write_lock:
            cursor = await self._db.execute(
                """UPDATE paper_portfolios
                   SET mode = ?, wallet_name = ?, hotkey_name = ?
                   WHERE id = ?""",
                (mode, wallet_name, hotkey_name, portfolio_id),
            )
            await self._db.commit()
            return cursor.rowcount > 0

    # --- Positions ---

    async def list_paper_positions(self, portfolio_id: int) -> list[dict]:
        cursor = await self._db.execute(
            """SELECT netuid, entry_block, entry_time, entry_price,
                      alpha_amount, tao_invested, strategy, hotkey_id
               FROM paper_positions WHERE portfolio_id = ?
               ORDER BY netuid ASC""",
            (portfolio_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def replace_paper_positions(
        self, portfolio_id: int, positions: dict
    ) -> None:
        async with self._write_lock:
            await self._db.execute(
                "DELETE FROM paper_positions WHERE portfolio_id = ?",
                (portfolio_id,),
            )
            for netuid, pos in positions.items():
                entry_time = (
                    pos.entry_time.isoformat()
                    if hasattr(pos.entry_time, "isoformat")
                    else str(pos.entry_time)
                )
                strategy = (
                    pos.strategy.value
                    if hasattr(pos.strategy, "value")
                    else str(pos.strategy)
                )
                await self._db.execute(
                    """INSERT INTO paper_positions
                         (portfolio_id, netuid, entry_block, entry_time,
                          entry_price, alpha_amount, tao_invested, strategy,
                          hotkey_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        portfolio_id, int(netuid), int(pos.entry_block),
                        entry_time, float(pos.entry_price),
                        float(pos.alpha_amount), float(pos.tao_invested),
                        strategy, int(pos.hotkey_id),
                    ),
                )
            await self._db.commit()

    # --- Trades ---

    async def insert_paper_trade(
        self,
        portfolio_id: int,
        trade,
        extrinsic_hash: str | None = None,
        executed_block: int | None = None,
    ) -> None:
        ts = (
            trade.timestamp.isoformat()
            if hasattr(trade.timestamp, "isoformat")
            else str(trade.timestamp)
        )
        async with self._write_lock:
            await self._db.execute(
                """INSERT INTO paper_trades
                     (id, portfolio_id, timestamp, block, netuid, direction,
                      strategy, tao_amount, alpha_amount, spot_price,
                      effective_price, slippage_pct, signal_strength,
                      hotkey_id, entry_price, pnl_tao, pnl_pct,
                      hold_duration_hours, entry_strategy,
                      extrinsic_hash, executed_block)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.id, portfolio_id, ts, int(trade.block),
                    int(trade.netuid),
                    trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
                    trade.strategy.value if hasattr(trade.strategy, "value") else str(trade.strategy),
                    float(trade.tao_amount), float(trade.alpha_amount),
                    float(trade.spot_price), float(trade.effective_price),
                    float(trade.slippage_pct),
                    float(trade.signal_strength) if trade.signal_strength is not None else None,
                    int(trade.hotkey_id) if trade.hotkey_id is not None else None,
                    float(trade.entry_price) if trade.entry_price is not None else None,
                    float(trade.pnl_tao) if trade.pnl_tao is not None else None,
                    float(trade.pnl_pct) if trade.pnl_pct is not None else None,
                    float(trade.hold_duration_hours) if trade.hold_duration_hours is not None else None,
                    trade.entry_strategy.value if (trade.entry_strategy and hasattr(trade.entry_strategy, "value")) else None,
                    extrinsic_hash,
                    executed_block,
                ),
            )
            await self._db.commit()

    async def list_paper_trades(
        self, portfolio_id: int, limit: int = 200
    ) -> list[dict]:
        cursor = await self._db.execute(
            """SELECT id, timestamp, block, netuid, direction, strategy,
                      tao_amount, alpha_amount, spot_price, effective_price,
                      slippage_pct, signal_strength, hotkey_id,
                      entry_price, pnl_tao, pnl_pct, hold_duration_hours,
                      entry_strategy, extrinsic_hash, executed_block
               FROM paper_trades WHERE portfolio_id = ?
               ORDER BY datetime(timestamp) DESC LIMIT ?""",
            (portfolio_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def count_paper_trades(self, portfolio_id: int) -> int:
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE portfolio_id = ?",
            (portfolio_id,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # --- Value history ---

    async def insert_paper_value_history(
        self,
        portfolio_id: int,
        timestamp: str,
        free_tao: float,
        total_value_tao: float,
        total_pnl_tao: float,
        drawdown_pct: float,
        num_open_positions: int,
    ) -> None:
        async with self._write_lock:
            await self._db.execute(
                """INSERT INTO paper_value_history
                     (portfolio_id, timestamp, free_tao, total_value_tao,
                      total_pnl_tao, drawdown_pct, num_open_positions)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (portfolio_id, timestamp, float(free_tao),
                 float(total_value_tao), float(total_pnl_tao),
                 float(drawdown_pct), int(num_open_positions)),
            )
            await self._db.commit()

    async def get_paper_value_history(
        self, portfolio_id: int, hours: int = 168, limit: int = 5000
    ) -> list[dict]:
        cursor = await self._db.execute(
            """SELECT timestamp, free_tao, total_value_tao, total_pnl_tao,
                      drawdown_pct, num_open_positions
               FROM paper_value_history
               WHERE portfolio_id = ?
                 AND datetime(timestamp) >= datetime('now', ?)
               ORDER BY datetime(timestamp) ASC
               LIMIT ?""",
            (portfolio_id, f"-{hours} hours", limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_paper_anchor_timestamp(self, portfolio_id: int) -> str | None:
        cursor = await self._db.execute(
            """SELECT timestamp FROM paper_value_history
               WHERE portfolio_id = ?
               ORDER BY datetime(timestamp) ASC LIMIT 1""",
            (portfolio_id,),
        )
        row = await cursor.fetchone()
        return row["timestamp"] if row else None

    # --- Stats + benchmark ---

    async def compute_paper_portfolio_stats(
        self,
        portfolio_id: int,
        cadence_seconds: int = 1800,
        api_client=None,
        exclude_netuids: list[int] | None = None,
        min_pool_depth_tao: float = 50.0,
    ) -> dict | None:
        """Headline metrics over a portfolio's full lifetime. Same shape as
        the monorepo's stats endpoint; benchmark series is computed via
        ``api_client`` (an OpenTaoAPI HTTP client)."""
        portfolio = await self.get_paper_portfolio(portfolio_id)
        if not portfolio:
            return None

        initial = float(portfolio["initial_capital_tao"])
        cursor = await self._db.execute(
            """SELECT timestamp, total_value_tao, drawdown_pct
               FROM paper_value_history WHERE portfolio_id = ?
               ORDER BY datetime(timestamp) ASC""",
            (portfolio_id,),
        )
        history = [dict(r) for r in await cursor.fetchall()]

        cursor = await self._db.execute(
            """SELECT direction, pnl_tao, pnl_pct, hold_duration_hours
               FROM paper_trades WHERE portfolio_id = ?""",
            (portfolio_id,),
        )
        trade_rows = [dict(r) for r in await cursor.fetchall()]
        total_trades = len(trade_rows)
        sells = [t for t in trade_rows
                 if t["direction"] == "sell" and t["pnl_tao"] is not None]
        winning = [t for t in sells if (t["pnl_tao"] or 0) > 0]
        losing = [t for t in sells if (t["pnl_tao"] or 0) <= 0]
        win_rate = (len(winning) / len(sells)) if sells else 0.0
        avg_win_pct = (
            sum(float(t["pnl_pct"]) for t in winning) / len(winning)
        ) if winning else 0.0
        avg_loss_pct = (
            sum(float(t["pnl_pct"]) for t in losing) / len(losing)
        ) if losing else 0.0
        gross_wins = sum(float(t["pnl_tao"]) for t in winning) if winning else 0.0
        gross_losses = abs(sum(float(t["pnl_tao"]) for t in losing)) if losing else 0.0
        profit_factor: float | None
        if gross_losses > 0:
            profit_factor = gross_wins / gross_losses
        elif gross_wins > 0:
            profit_factor = None
        else:
            profit_factor = 0.0
        hold_hours = [
            float(t["hold_duration_hours"]) for t in sells
            if t["hold_duration_hours"] is not None
        ]
        avg_hold_hours = (sum(hold_hours) / len(hold_hours)) if hold_hours else 0.0

        if not history:
            current = float(portfolio.get("free_tao") or initial)
            return {
                "portfolio_id": portfolio_id,
                "mode": portfolio.get("mode", "paper"),
                "initial_capital_tao": initial,
                "current_value_tao": current,
                "total_return_pct": (current - initial) / initial if initial > 0 else 0.0,
                "benchmark_return_pct": 0.0,
                "alpha_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "cycles": 0,
                "cadence_seconds": cadence_seconds,
                "total_trades": total_trades,
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": win_rate,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct,
                "profit_factor": profit_factor,
                "avg_hold_hours": avg_hold_hours,
            }

        values = [float(r["total_value_tao"]) for r in history]
        current = values[-1]
        total_return = (current - initial) / initial if initial > 0 else 0.0

        returns: list[float] = []
        for i in range(1, len(values)):
            prev = values[i - 1]
            if prev > 0:
                returns.append(values[i] / prev - 1)

        sharpe = 0.0
        sortino = 0.0
        if len(returns) >= 2:
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std_r = var_r ** 0.5
            periods_per_year = 31536000.0 / max(cadence_seconds, 1)
            ann = periods_per_year ** 0.5
            if std_r > 0:
                sharpe = (mean_r / std_r) * ann
            downside = [r for r in returns if r < 0]
            if downside:
                d_var = sum(r * r for r in downside) / len(downside)
                d_std = d_var ** 0.5
                if d_std > 0:
                    sortino = (mean_r / d_std) * ann

        max_dd_pct = 0.0
        for r in history:
            dd = float(r["drawdown_pct"] or 0.0)
            if abs(dd) > max_dd_pct:
                max_dd_pct = abs(dd)

        bench_current = initial
        if api_client is not None:
            try:
                bench_values, _ = await api_client.compute_benchmark_series(
                    timestamps=[history[-1]["timestamp"]],
                    anchor_ts=history[0]["timestamp"],
                    initial_capital_tao=initial,
                    exclude_netuids=exclude_netuids,
                    min_pool_depth_tao=min_pool_depth_tao,
                )
                if bench_values:
                    bench_current = bench_values[0]
            except Exception:
                logger.exception("Benchmark fetch failed; defaulting to initial")
        bench_return = (bench_current - initial) / initial if initial > 0 else 0.0
        alpha = total_return - bench_return

        return {
            "portfolio_id": portfolio_id,
            "mode": portfolio.get("mode", "paper"),
            "initial_capital_tao": initial,
            "current_value_tao": current,
            "total_return_pct": total_return,
            "benchmark_return_pct": bench_return,
            "alpha_pct": alpha,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd_pct,
            "cycles": len(history),
            "cadence_seconds": cadence_seconds,
            "total_trades": total_trades,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": win_rate,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "profit_factor": profit_factor,
            "avg_hold_hours": avg_hold_hours,
        }
