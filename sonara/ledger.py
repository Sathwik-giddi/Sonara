"""The quota ledger — the heart of the product, per the design doc.

Sonara runs at $0 forever. That makes "how many free requests are left, where"
a load-bearing question rather than an operational detail, so it gets real
storage with real rules:

  * caps come from a dated config file, because free-tier limits drift and a
    number without an as_of date is a guess wearing a lab coat
  * a live 429/402 OVERRIDES the predicted cap immediately - the provider is
    always right, the config is only ever an estimate
  * `total` windows (NVIDIA NIM's one-time trial credits) never reset

Deliberately simple: SQLite, no ORM, no server. It has to survive a laptop
being closed mid-sentence.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "ledger.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS burn (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    provider    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    task        TEXT,
    ok          INTEGER NOT NULL,
    status      INTEGER,
    latency_ms  REAL
);
CREATE INDEX IF NOT EXISTS burn_provider_ts ON burn(provider, ts);

-- Exhaustion is separate from burn on purpose: it is set by a live 429/402,
-- not inferred from counting. Ground truth beats prediction.
CREATE TABLE IF NOT EXISTS exhausted (
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    until_ts    REAL NOT NULL,
    reason      TEXT,
    PRIMARY KEY (provider, model)
);
"""


@dataclass(frozen=True)
class Usage:
    minute: int
    day: int
    total: int


def _utc_day_start() -> float:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


class Ledger:
    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---------- writing ----------

    def record(self, provider: str, model: str, *, task: str | None = None,
               ok: bool, status: int | None = None, latency_ms: float | None = None) -> None:
        self.db.execute(
            "INSERT INTO burn (ts, provider, model, task, ok, status, latency_ms) "
            "VALUES (?,?,?,?,?,?,?)",
            (time.time(), provider, model, task, int(ok), status, latency_ms),
        )
        self.db.commit()

    def mark_exhausted(self, provider: str, model: str, *,
                       retry_after_s: float | None, reason: str = "") -> None:
        """A live 429/402 arrived. Believe it over any predicted cap.

        retry_after_s comes from the provider's header when present. When absent
        we back off to the next UTC day, which is the common free-tier reset and
        errs toward not hammering a provider that just said no.
        """
        until = time.time() + retry_after_s if retry_after_s else _utc_day_start() + 86_400
        self.db.execute(
            "INSERT INTO exhausted (provider, model, until_ts, reason) VALUES (?,?,?,?) "
            "ON CONFLICT(provider, model) DO UPDATE SET until_ts=excluded.until_ts, reason=excluded.reason",
            (provider, model, until, reason[:200]),
        )
        self.db.commit()

    # ---------- reading ----------

    def is_exhausted(self, provider: str, model: str) -> bool:
        row = self.db.execute(
            "SELECT until_ts FROM exhausted WHERE provider=? AND model=?", (provider, model)
        ).fetchone()
        if not row:
            return False
        if row[0] <= time.time():
            self.db.execute("DELETE FROM exhausted WHERE provider=? AND model=?", (provider, model))
            self.db.commit()
            return False
        return True

    def usage(self, provider: str, model: str | None = None) -> Usage:
        now = time.time()
        where = "provider=?" + ("" if model is None else " AND model=?")
        args: tuple = (provider,) if model is None else (provider, model)

        def count(extra: str, more: tuple = ()) -> int:
            sql = f"SELECT COUNT(*) FROM burn WHERE {where} AND ok=1 {extra}"
            return self.db.execute(sql, args + more).fetchone()[0]

        return Usage(
            minute=count("AND ts > ?", (now - 60,)),
            day=count("AND ts > ?", (_utc_day_start(),)),
            total=count(""),
        )

    def has_headroom(self, provider: str, model: str, caps: dict | None) -> bool:
        """Predicted headroom. Advisory only - `is_exhausted` is the real gate."""
        if self.is_exhausted(provider, model):
            return False
        if not caps:
            return True
        u = self.usage(provider, model)
        for window, used in (("minute", u.minute), ("day", u.day), ("total", u.total)):
            limit = caps.get(window)
            if isinstance(limit, int) and used >= limit:
                return False
        return True

    # ---------- reporting ----------

    def burn_table(self, days: int = 7) -> list[tuple]:
        """Per-provider burn, the raw material for the M2 burn table."""
        since = time.time() - days * 86_400
        return self.db.execute(
            "SELECT provider, model, COUNT(*) AS calls, "
            "       SUM(ok) AS ok, ROUND(AVG(latency_ms)) AS avg_ms "
            "FROM burn WHERE ts > ? GROUP BY provider, model ORDER BY calls DESC",
            (since,),
        ).fetchall()

    def close(self) -> None:
        self.db.close()
