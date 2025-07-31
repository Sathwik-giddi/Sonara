"""Background work — the difference between a command line and an assistant.

Everything so far was REACTIVE: you speak, it answers, it waits. However good the
answers get, that is an interface, not an assistant. A real assistant works while you
are not looking - chasing the thing you asked about an hour ago, watching the clock,
and interrupting you when it matters.

Three kinds of autonomous work, all persisted so they survive a restart:

  REMINDER   speak at a time.                    "remind me at six"
  FOLLOW-UP  DO something at a time, then report. "check the weather at seven and tell me"
  WATCH      poll a condition, speak when it becomes true, then stop.
             "tell me when it stops raining"

The last two are what a human assistant actually does. A reminder makes YOU do the work
at the right time; a follow-up means the work is already done when you are told.

Design notes that matter:
  - jobs live in SQLite, not memory, so closing the laptop does not cancel your day
  - the worker never speaks directly; it queues announcements and the voice loop plays
    them between turns, so it can never talk over you or over itself
  - a watch that never becomes true expires, because an assistant that checks the same
    thing forever is a memory leak with a personality
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "jobs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT NOT NULL,          -- followup | watch
    tool      TEXT NOT NULL,
    args      TEXT NOT NULL,
    due_ts    REAL,                   -- followup: when to run
    every_s   REAL,                   -- watch: poll interval
    until_ts  REAL,                   -- watch: give up after this
    condition TEXT,                   -- watch: substring that must appear in the result
    say       TEXT,                   -- what to announce
    created   REAL NOT NULL,
    done      INTEGER NOT NULL DEFAULT 0,
    last_run  REAL
);
"""


@dataclass
class Announcement:
    text: str
    job_id: int
    kind: str


def speakable(tool: str, result) -> str:
    """Turn a tool result into something a person would say out loud.

    The worker speaks WITHOUT a model in the loop - that keeps background work free and
    instant - so the formatting has to happen here. Without it Sonara announced
    "{'place': 'Bengaluru, India', 'temperature_c': 24.6, ...}" aloud, which is the same
    plumbing-leak bug as before, just arriving down a different pipe.
    """
    if isinstance(result, dict):
        if "temperature_c" in result:
            r = result
            return (f"it's {r['temperature_c']:.0f} degrees and {r.get('conditions','')} "
                    f"in {r.get('place','')}").strip()
        if "summary" in result:
            return str(result["summary"])[:220]
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "title" in first:
            return f"top result: {first['title']}. {str(first.get('snippet',''))[:160]}"
        return "; ".join(str(x) for x in result[:3])[:220]
    return str(result)[:220]


class Scheduler:
    """Runs jobs on its own clock and queues what should be said."""

    def __init__(self, executor, *, path: Path | str = DB, tick_s: float = 5.0) -> None:
        self.executor = executor
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.lock = threading.Lock()
        self.out: Queue[Announcement] = Queue()
        self.tick_s = tick_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- scheduling ----------

    def follow_up(self, tool: str, args: dict, when_ts: float, say: str = "") -> int:
        """Do something later, then report the result. The work is DONE when you hear."""
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO jobs (kind, tool, args, due_ts, say, created) "
                "VALUES ('followup',?,?,?,?,?)",
                (tool, json.dumps(args), when_ts, say, time.time()))
            self.db.commit()
            return cur.lastrowid

    def watch(self, tool: str, args: dict, condition: str, *, every_s: float = 300,
              for_hours: float = 12, say: str = "") -> int:
        """Poll until the result contains `condition`, announce once, then stop."""
        now = time.time()
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO jobs (kind, tool, args, every_s, until_ts, condition, say, created) "
                "VALUES ('watch',?,?,?,?,?,?,?)",
                (tool, json.dumps(args), every_s, now + for_hours * 3600,
                 condition, say, now))
            self.db.commit()
            return cur.lastrowid

    def pending(self) -> list[tuple]:
        return self.db.execute(
            "SELECT id, kind, tool, due_ts, condition FROM jobs WHERE done=0 "
            "ORDER BY COALESCE(due_ts, created)").fetchall()

    def cancel(self, job_id: int) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE jobs SET done=1 WHERE id=? AND done=0", (job_id,))
            self.db.commit()
            return cur.rowcount > 0

    # ---------- the worker ----------

    def _run_one(self, job_id: int, kind: str, tool: str, args_json: str,
                 condition: str | None, say: str) -> None:
        args = json.loads(args_json or "{}")
        try:
            out = self.executor.execute(tool, args)
        except Exception as e:  # noqa: BLE001 - a bad job must never kill the worker
            with self.lock:
                self.db.execute("UPDATE jobs SET done=1 WHERE id=?", (job_id,))
                self.db.commit()
            self.out.put(Announcement(
                f"I tried to {say or tool} but it failed: {type(e).__name__}.", job_id, kind))
            return

        result = speakable(tool, out.result) if out.ok else f"failed: {out.error}"

        if kind == "followup":
            with self.lock:
                self.db.execute("UPDATE jobs SET done=1, last_run=? WHERE id=?",
                                (time.time(), job_id))
                self.db.commit()
            lead = say or f"about {tool}"
            self.out.put(Announcement(f"You asked me to check {lead}. {result}", job_id, kind))
            return

        # watch: only speak when the condition is actually met
        with self.lock:
            self.db.execute("UPDATE jobs SET last_run=? WHERE id=?", (time.time(), job_id))
            self.db.commit()
        if condition and condition.lower() in result.lower():
            with self.lock:
                self.db.execute("UPDATE jobs SET done=1 WHERE id=?", (job_id,))
                self.db.commit()
            self.out.put(Announcement(say or f"That thing you asked me to watch: {result}",
                                      job_id, kind))

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            rows = self.db.execute(
                "SELECT id, kind, tool, args, due_ts, every_s, until_ts, condition, say, "
                "last_run FROM jobs WHERE done=0").fetchall()
            for (jid, kind, tool, args, due, every, until, cond, say, last) in rows:
                if kind == "followup" and due and now >= due:
                    self._run_one(jid, kind, tool, args, cond, say)
                elif kind == "watch":
                    if until and now > until:
                        # Expire quietly. An assistant that polls the same thing for
                        # ever is a memory leak with a personality.
                        with self.lock:
                            self.db.execute("UPDATE jobs SET done=1 WHERE id=?", (jid,))
                            self.db.commit()
                    elif not last or now - last >= (every or 300):
                        self._run_one(jid, kind, tool, args, cond, say)
            self._stop.wait(self.tick_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sonara-bg")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def drain(self) -> list[Announcement]:
        """Whatever the worker wants said. The voice loop plays these between turns,
        so background work can never talk over the person."""
        items = []
        while not self.out.empty():
            items.append(self.out.get())
        return items
