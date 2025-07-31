"""Persistent memory — the difference between software and someone.

The agent already remembers within a conversation. Close the process and it forgets you
exist, which is the single largest gap between Sonara and something that feels like a
relationship: every session starts as a stranger.

Three ideas, all local, all free:

  FACTS      Durable things about the person. "I'm vegetarian." "My bank is HDFC."
             "Call me Vaish." Stored once, recalled forever.
  EPISODES   What happened, with a date. "Asked about GPU pricing (2026-08-01)."
  RECALL     Before each turn, the utterance is searched against memory and anything
             relevant is put in front of the model. This is what the research means by
             "notices patterns and adapts without manual prompting" - the person never
             types "remember that", it simply knows.

SQLite with FTS5, no embeddings. The design doc's decision stands: full-text first,
vectors only if recall measurably fails (>=15% miss over 100 labelled attempts). FTS5
costs nothing, needs no model, and survives a laptop closing mid-sentence.

PRIVACY: this is Class L data. It never leaves the machine and never enters a hosted
prompt wholesale - only the few lines relevant to the current utterance, which is both
safer and cheaper than shipping a life story every turn.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    kind     TEXT NOT NULL,          -- fact | episode | preference
    text     TEXT NOT NULL,
    created  REAL NOT NULL,
    last_hit REAL,
    hits     INTEGER NOT NULL DEFAULT 0,
    source   TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(text, content='memories', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

-- Every recall that returned nothing useful, so the FTS5-vs-embeddings decision is
-- settled by data rather than by feel (design doc, decision #6).
CREATE TABLE IF NOT EXISTS recall_log (
    ts REAL NOT NULL, query TEXT, hits INTEGER
);
"""

# Patterns that mean "this is worth keeping". Deliberately rules, not an LLM call:
# extracting memories with a model would double the cost of every single turn, and
# these catch the explicit cases, which are the ones people expect to be remembered.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("fact", re.compile(r"\b(?:remember|don'?t forget|keep in mind)\s+(?:that\s+)?(.{4,160})", re.I)),
    ("fact", re.compile(r"\bmy\s+(\w[\w\s]{2,40}?)\s+is\s+(.{2,80})", re.I)),
    ("fact", re.compile(r"\bcall me\s+(\w{2,30})", re.I)),
    ("fact", re.compile(r"\bi(?:'m| am)\s+(?:a\s+|an\s+)?((?:vegetarian|vegan|allergic|diabetic|left-handed)\b.{0,60})", re.I)),
    ("preference", re.compile(r"\bi\s+(?:prefer|like|love|hate|can'?t stand|always|never)\s+(.{3,100})", re.I)),
]

_STOP = re.compile(r"[^\w\s]")


@dataclass
class Memory_:
    id: int
    kind: str
    text: str
    created: float
    hits: int


class Memory:
    def __init__(self, path: Path | str = DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---------- writing ----------

    def remember(self, text: str, *, kind: str = "fact", source: str = "user") -> int | None:
        text = " ".join((text or "").split())
        if len(text) < 3:
            return None
        # Cheap dedup: an assistant that says "I'll remember that" three times about the
        # same fact sounds broken, not attentive.
        existing = self.db.execute(
            "SELECT id FROM memories WHERE lower(text)=lower(?)", (text,)
        ).fetchone()
        if existing:
            return existing[0]
        cur = self.db.execute(
            "INSERT INTO memories (kind, text, created, source) VALUES (?,?,?,?)",
            (kind, text, time.time(), source),
        )
        self.db.commit()
        return cur.lastrowid

    def observe(self, utterance: str) -> list[str]:
        """Extract anything worth keeping from something the person just said."""
        kept: list[str] = []
        for kind, pat in _PATTERNS:
            for m in pat.finditer(utterance or ""):
                groups = [g.strip(" .,!?") for g in m.groups() if g]
                if not groups:
                    continue
                # Store a READABLE SENTENCE, not a fragment. The first version joined
                # capture groups, so "my bank is HDFC" became "their bank HDFC" - and the
                # model could not read that as a statement, answering "I cannot find the
                # bank you use" while the fact sat right there in its context.
                if pat.pattern.startswith(r"\bmy\s"):
                    piece = f"their {groups[0]} is {groups[1]}" if len(groups) > 1 else f"their {groups[0]}"
                elif pat.pattern.startswith(r"\bcall me"):
                    piece = f"their name is {groups[0]}"
                elif pat.pattern.startswith(r"\bi(?:'m| am)"):
                    piece = f"they are {groups[0]}"
                elif kind == "preference":
                    piece = "they " + m.group(0).strip().lstrip("iI").strip()
                else:
                    piece = groups[0]
                piece = " ".join(piece.split())
                if self.remember(piece, kind=kind):
                    kept.append(piece)
        return kept

    def add_episode(self, summary: str) -> None:
        self.remember(f"{datetime.now():%Y-%m-%d}: {summary}", kind="episode", source="session")

    # ---------- reading ----------

    def recall(self, query: str, limit: int = 4) -> list[Memory_]:
        terms = [w for w in _STOP.sub(" ", query or "").split() if len(w) > 2]
        if not terms:
            return []
        # OR the terms: a spoken sentence rarely matches memory word-for-word, and an
        # AND query on "what's my bank again" finds nothing at all.
        match = " OR ".join(terms[:8])
        try:
            rows = self.db.execute(
                "SELECT m.id, m.kind, m.text, m.created, m.hits FROM memories_fts f "
                "JOIN memories m ON m.id = f.rowid WHERE memories_fts MATCH ? "
                "ORDER BY bm25(memories_fts) LIMIT ?", (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        self.db.execute("INSERT INTO recall_log (ts, query, hits) VALUES (?,?,?)",
                        (time.time(), query[:120], len(rows)))
        for r in rows:
            self.db.execute(
                "UPDATE memories SET hits = hits + 1, last_hit = ? WHERE id = ?",
                (time.time(), r[0]))
        self.db.commit()
        return [Memory_(*r) for r in rows]

    def core_facts(self, limit: int = 20) -> list[Memory_]:
        """Durable identity: name, diet, allergies, standing preferences.

        These are ALWAYS injected, never searched. Full-text search matches words, not
        meaning - "suggest dinner" shares no word with "they are vegetarian", so search
        misses exactly the recall that matters most. A person has perhaps twenty such
        facts; carrying all of them costs ~100 tokens and beats any retrieval.
        """
        return [Memory_(*r) for r in self.db.execute(
            "SELECT id, kind, text, created, hits FROM memories "
            "WHERE kind IN ('fact','preference') ORDER BY hits DESC, created DESC LIMIT ?",
            (limit,)).fetchall()]

    def context_for(self, utterance: str, limit: int = 4) -> str:
        """What to put in front of the model for THIS utterance.

        Two tiers, because they fail differently:
          core facts  always present, no query needed  (solves the semantic gap)
          episodes    searched, since there are many and most are irrelevant
        """
        core = self.core_facts()
        hits = [m for m in self.recall(utterance, limit=limit)
                if m.id not in {c.id for c in core}]
        if not core and not hits:
            return ""

        parts = []
        if core:
            parts.append("ESTABLISHED FACTS about the person you are speaking to. These "
                         "are true and you already know them - use them to answer "
                         "directly. Never say you do not know something listed here, and "
                         "never announce that you are recalling:\n"
                         + "\n".join(f"- {m.text}" for m in core))
        if hits:
            parts.append("Possibly relevant from earlier:\n"
                         + "\n".join(f"- {m.text}" for m in hits))
        return "\n\n".join(parts)

    # ---------- maintenance ----------

    def forget(self, memory_id: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()
        return cur.rowcount > 0

    def all(self, kind: str | None = None, limit: int = 50) -> list[Memory_]:
        sql = "SELECT id, kind, text, created, hits FROM memories"
        args: tuple = ()
        if kind:
            sql += " WHERE kind = ?"
            args = (kind,)
        sql += " ORDER BY created DESC LIMIT ?"
        return [Memory_(*r) for r in self.db.execute(sql, args + (limit,)).fetchall()]

    def miss_rate(self) -> tuple[int, int]:
        """(misses, total) recalls. Feeds the FTS5-vs-embeddings trip-wire: revisit
        embeddings only if >=15% of recalls return nothing over 100 attempts."""
        row = self.db.execute(
            "SELECT SUM(CASE WHEN hits = 0 THEN 1 ELSE 0 END), COUNT(*) FROM recall_log"
        ).fetchone()
        return (row[0] or 0, row[1] or 0)
