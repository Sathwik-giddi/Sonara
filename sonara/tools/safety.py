"""The leash. Premise 4: power needs one.

Everything that executes a tool goes through Executor. Nothing else may call
registry.call() - the gate lives outside the registry precisely so it cannot be
bypassed by reaching past it.

Three protections, in order of how often they save you:

  1. DRY RUN - a tool's first week speaks and logs what it WOULD do. New skills are
     wrong in ways nobody predicts; find out with the audit log, not your files.
  2. CONFIRMATION - destructive/outward tools repeat the action back and require an
     EXACT confirm phrase. No fuzzy matching: "yeah sure whatever" is not consent,
     and STT mishears (measured: "what's the" -> "watch the").
  3. AUDIT LOG - append-only JSONL of every attempt, including the blocked ones.
     Blocked attempts are the interesting ones.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import Risk, ToolRegistry, registry as default_registry

ROOT = Path(__file__).resolve().parents[2]
AUDIT_LOG = ROOT / "data" / "audit.jsonl"

# Exact-match only. A confirm phrase that STT can mangle into a near-miss is not a
# gate, so this is short, phonetically distinct, and never a word used in passing.
CONFIRM_PHRASE = "confirm"


class ConfirmationRequired(Exception):
    """Raised instead of executing. Carries what the assistant should SAY."""

    def __init__(self, tool_name: str, args: dict, prompt: str) -> None:
        super().__init__(prompt)
        self.tool_name = tool_name
        self.args = args
        self.prompt = prompt


class Blocked(Exception):
    """Denylisted, or confirmation was attempted and failed."""


@dataclass
class Outcome:
    ok: bool
    result: Any = None
    error: str | None = None
    dry_run: bool = False
    blocked: bool = False


class Executor:
    def __init__(self, reg: ToolRegistry | None = None, *,
                 dry_run_packs: set[str] | None = None,
                 allowlist: set[str] | None = None,
                 denylist: set[str] | None = None,
                 audit_path: Path = AUDIT_LOG) -> None:
        self.reg = reg or default_registry
        self.dry_run_packs = dry_run_packs or set()
        self.allowlist = allowlist          # None = everything not denied
        self.denylist = denylist or set()
        self.audit_path = audit_path
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: tuple[str, dict] | None = None

    # ---------- audit ----------

    def _audit(self, **row: Any) -> None:
        row["ts"] = time.time()
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    # ---------- gating ----------

    def _permitted(self, name: str) -> bool:
        if name in self.denylist:
            return False
        return self.allowlist is None or name in self.allowlist

    def execute(self, name: str, args: dict[str, Any], *,
                confirmed: bool = False) -> Outcome:
        tool = self.reg.get(name)
        if tool is None:
            self._audit(tool=name, args=args, outcome="unknown_tool")
            return Outcome(ok=False, error=f"unknown tool: {name}")

        if not self._permitted(name):
            self._audit(tool=name, args=args, risk=tool.risk, outcome="denied")
            return Outcome(ok=False, blocked=True, error=f"{name} is not permitted")

        # Confirmation gate. Note this fires BEFORE dry-run: a destructive tool in
        # dry-run still rehearses the confirmation, so the UX is tested too.
        if tool.risk is Risk.CONFIRM and not confirmed:
            prompt = tool.confirm_template.format(
                name=name.replace("_", " "),
                args=", ".join(f"{k}={v}" for k, v in args.items()) or "no arguments",
            )
            self._audit(tool=name, args=args, risk=tool.risk, outcome="awaiting_confirmation")
            self._pending = (name, args)
            raise ConfirmationRequired(name, args, prompt)

        if tool.pack in self.dry_run_packs:
            self._audit(tool=name, args=args, risk=tool.risk, outcome="dry_run",
                        confirmed=confirmed)
            return Outcome(ok=True, dry_run=True,
                           result=f"[dry run] would call {name}({args})")

        try:
            result = self.reg.call(name, args)
            self._audit(tool=name, args=args, risk=tool.risk, outcome="ok",
                        confirmed=confirmed)
            return Outcome(ok=True, result=result)
        except Exception as e:  # noqa: BLE001 - a failing tool must never kill the loop
            self._audit(tool=name, args=args, risk=tool.risk, outcome="error",
                        error=f"{type(e).__name__}: {e}")
            return Outcome(ok=False, error=f"{type(e).__name__}: {e}")

    # ---------- spoken confirmation ----------

    def resolve_confirmation(self, spoken: str) -> Outcome:
        """Called with whatever the user said after a confirmation prompt.

        Exact match on the confirm phrase. Anything else - silence, "ok", a
        near-miss from STT - cancels. Refusing an ambiguous yes is the entire point.
        """
        if self._pending is None:
            return Outcome(ok=False, error="nothing awaiting confirmation")
        name, args = self._pending
        self._pending = None

        if (spoken or "").strip().strip(".!?").lower() != CONFIRM_PHRASE:
            self._audit(tool=name, args=args, outcome="confirmation_refused",
                        heard=spoken)
            return Outcome(ok=False, blocked=True,
                           error=f"cancelled - say '{CONFIRM_PHRASE}' to go ahead")
        return self.execute(name, args, confirmed=True)

    @property
    def awaiting_confirmation(self) -> bool:
        return self._pending is not None
