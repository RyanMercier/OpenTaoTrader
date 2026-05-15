"""Synchronous read-only loader over the OpenTaoAPI snapshot database.

The trading system never writes to this database. It is treated as an
append-only source of truth that the OpenTaoAPI process owns.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from .models import Snapshot, get_regime


def _parse_ts(s: str) -> datetime:
    """Parse ISO 8601 timestamps produced by OpenTaoAPI. Tolerates trailing Z
    and timezone offsets."""
    if s is None:
        return datetime.min
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Some backfilled rows may have no tz; try trimming microseconds
        try:
            return datetime.fromisoformat(s.split(".")[0])
        except Exception:
            return datetime.min


def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
    ts_str = row["timestamp"] or ""
    return Snapshot(
        block=row["block"],
        timestamp=_parse_ts(ts_str),
        netuid=row["netuid"],
        alpha_price_tao=row["alpha_price_tao"] or 0.0,
        tao_price_usd=row["tao_price_usd"] or 0.0,
        tao_in=row["tao_in"] or 0.0,
        alpha_in=row["alpha_in"] or 0.0,
        total_stake=row["total_stake"] or 0.0,
        emission_rate=row["emission_rate"] or 0.0,
        validator_count=row["validator_count"] or 0,
        neuron_count=row["neuron_count"] or 0,
        regime=get_regime(ts_str),
    )


class DataLoader:
    """Read-only access to subnet_snapshots.

    Opens a new connection per call, SQLite is fast enough for this and it
    avoids thread-safety pitfalls.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def load_snapshots(
        self,
        netuid: int,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[Snapshot]:
        q = (
            "SELECT block, timestamp, netuid, alpha_price_tao, tao_price_usd, "
            "tao_in, alpha_in, total_stake, emission_rate, "
            "validator_count, neuron_count "
            "FROM subnet_snapshots WHERE netuid = ?"
        )
        params: list = [netuid]
        if start:
            q += " AND timestamp >= ?"
            params.append(start)
        if end:
            q += " AND timestamp <= ?"
            params.append(end)
        q += " ORDER BY block ASC"
        with self._connect() as conn:
            rows = conn.execute(q, params).fetchall()
        return [_row_to_snapshot(r) for r in rows]

    def load_all_snapshots(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        netuids: Optional[list[int]] = None,
    ) -> dict[int, list[Snapshot]]:
        if netuids is None:
            netuids = self.get_available_netuids()
        return {n: self.load_snapshots(n, start, end) for n in netuids}

    def get_available_netuids(self) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT netuid FROM subnet_snapshots ORDER BY netuid"
            ).fetchall()
        return [r["netuid"] for r in rows]

    def get_data_range(self, netuid: Optional[int] = None) -> tuple[str, str]:
        with self._connect() as conn:
            if netuid is not None:
                row = conn.execute(
                    "SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi "
                    "FROM subnet_snapshots WHERE netuid = ?",
                    (netuid,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi "
                    "FROM subnet_snapshots"
                ).fetchone()
        lo = row["lo"] if row else None
        hi = row["hi"] if row else None
        return (lo or "", hi or "")

    def get_snapshot_counts(self) -> dict[int, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT netuid, COUNT(*) AS c FROM subnet_snapshots GROUP BY netuid"
            ).fetchall()
        return {r["netuid"]: r["c"] for r in rows}

    def get_closest_snapshot(
        self, netuid: int, timestamp: datetime
    ) -> Optional[Snapshot]:
        ts = timestamp.isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT block, timestamp, netuid, alpha_price_tao, tao_price_usd, "
                "tao_in, alpha_in, total_stake, emission_rate, "
                "validator_count, neuron_count "
                "FROM subnet_snapshots "
                "WHERE netuid = ? AND timestamp <= ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (netuid, ts),
            ).fetchone()
        return _row_to_snapshot(row) if row else None

    def get_all_netuids_at_time(
        self, timestamp: datetime
    ) -> dict[int, Snapshot]:
        """Latest snapshot for each subnet at or before timestamp.

        Uses a windowed query, one row per netuid, the most recent one not
        exceeding the cutoff.
        """
        ts = timestamp.isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.block, s.timestamp, s.netuid, s.alpha_price_tao, "
                "s.tao_price_usd, s.tao_in, s.alpha_in, s.total_stake, "
                "s.emission_rate, s.validator_count, s.neuron_count "
                "FROM subnet_snapshots s "
                "INNER JOIN ( "
                "  SELECT netuid, MAX(timestamp) AS max_ts "
                "  FROM subnet_snapshots "
                "  WHERE timestamp <= ? "
                "  GROUP BY netuid "
                ") latest ON s.netuid = latest.netuid AND s.timestamp = latest.max_ts",
                (ts,),
            ).fetchall()
        return {r["netuid"]: _row_to_snapshot(r) for r in rows}
