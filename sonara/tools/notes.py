"""Pack: notes and reminders.

Class L data, by the design doc's classification: note BODIES and the memory store
never reach a hosted tier. So recall returns structured rows and the assistant is
expected to read them back locally - the hosted model gets the intent, not the diary.

Reminders live in SQLite with an absolute due time, so they survive a laptop closing
mid-sentence and are re-armed on boot. A reminder that only exists in RAM is a
reminder you will miss exactly once, memorably.
"""

from __future__ import annotations

import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from .base import Risk, registry

PACK = "notes"
ROOT = Path(__file__).resolve().parents[2]
VAULT = ROOT / "data" / "notes"
DB = ROOT / "data" / "reminders.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    text    TEXT NOT NULL,
    due_ts  REAL NOT NULL,
    created REAL NOT NULL,
    done    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS reminders_due ON reminders(due_ts, done);
"""


def _db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    return con


def parse_when(when: str) -> float:
    """Turn spoken time into an absolute timestamp.

    Deliberately small and explicit rather than a natural-language date library:
    the failure mode of a clever parser is a reminder silently set for the wrong
    year, which is worse than refusing. Anything it cannot parse raises.
    """
    s = (when or "").strip().lower()
    now = datetime.now()

    if m := re.match(r"in (\d+) (second|minute|hour|day)s?", s):
        n, unit = int(m.group(1)), m.group(2)
        return (now + timedelta(**{f"{unit}s": n})).timestamp()

    if m := re.match(r"(?:at )?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s):
        hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        elif ampm is None and hour < 8:
            hour += 12  # "at 6" in the evening is the common case
        target = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target.timestamp()

    if s in ("tomorrow", "tomorrow morning"):
        return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0).timestamp()

    raise ValueError(f"could not understand the time: {when!r}")


@registry.tool(
    name="set_reminder", pack=PACK,
    description="Set a reminder for a specific time, e.g. 'at 6 PM', 'in 20 minutes', 'tomorrow'.",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "what to be reminded of"},
            "when": {"type": "string", "description": "when, e.g. 'at 6 pm' or 'in 20 minutes'"},
        },
        "required": ["text", "when"],
    },
)
def set_reminder(text: str, when: str) -> str:
    due = parse_when(when)
    con = _db()
    con.execute("INSERT INTO reminders (text, due_ts, created) VALUES (?,?,?)",
                (text, due, time.time()))
    con.commit()
    con.close()
    return f"reminder set for {datetime.fromtimestamp(due):%A %I:%M %p}".replace(" 0", " ")


@registry.tool(
    name="list_reminders", pack=PACK,
    description="List reminders that have not fired yet.",
    parameters={"type": "object", "properties": {}},
)
def list_reminders() -> list[dict]:
    con = _db()
    rows = con.execute(
        "SELECT id, text, due_ts FROM reminders WHERE done=0 ORDER BY due_ts"
    ).fetchall()
    con.close()
    return [{"id": r[0], "text": r[1],
             "due": datetime.fromtimestamp(r[2]).strftime("%a %I:%M %p")} for r in rows]


@registry.tool(
    name="due_reminders", pack=PACK,
    description="Get reminders that are due now and mark them fired. Used by the tray loop.",
    parameters={"type": "object", "properties": {}},
)
def due_reminders() -> list[str]:
    con = _db()
    now = time.time()
    rows = con.execute("SELECT id, text FROM reminders WHERE done=0 AND due_ts<=?",
                       (now,)).fetchall()
    if rows:
        con.executemany("UPDATE reminders SET done=1 WHERE id=?", [(r[0],) for r in rows])
        con.commit()
    con.close()
    return [r[1] for r in rows]


@registry.tool(
    name="add_note", pack=PACK,
    description="Append a note to today's note file.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "the note to save"}},
        "required": ["text"],
    },
)
def add_note(text: str) -> str:
    VAULT.mkdir(parents=True, exist_ok=True)
    path = VAULT / f"{datetime.now():%Y-%m-%d}.md"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now():%H:%M} {text}\n")
    return "noted"


@registry.tool(
    name="search_notes", pack=PACK,
    description=("Find saved notes matching a query. Returns the matching lines so they can "
                 "be read back LOCALLY - note bodies are Class L and must not be sent to a "
                 "hosted model."),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "max results, default 5"},
        },
        "required": ["query"],
    },
)
def search_notes(query: str, limit: int = 5) -> list[dict]:
    if not VAULT.exists():
        return []
    q, out = query.lower(), []
    for path in sorted(VAULT.glob("*.md"), reverse=True):
        for line in path.read_text(encoding="utf-8").splitlines():
            if q in line.lower():
                out.append({"date": path.stem, "line": line.lstrip("- ")})
                if len(out) >= max(1, min(int(limit), 25)):
                    return out
    return out


@registry.tool(
    name="delete_note_day", pack=PACK, risk=Risk.CONFIRM,
    description="Delete an entire day's notes. Requires spoken confirmation.",
    parameters={
        "type": "object",
        "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
        "required": ["date"],
    },
    confirm_template="You want me to delete all notes from {args}. Say confirm.",
)
def delete_note_day(date: str) -> str:
    path = VAULT / f"{date}.md"
    if not path.is_file():
        raise FileNotFoundError(date)
    path.unlink()
    return f"deleted notes for {date}"
