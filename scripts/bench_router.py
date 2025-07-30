"""Show the router thinking: what each utterance is classified as, where it goes.

Two modes:
  --dry   classification and routing decisions only, no API calls, no quota burned
  (default) actually calls the chosen model and reports latency per task

Run:  uv run scripts/bench_router.py --dry
      uv run scripts/bench_router.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sonara import NoProviderAvailable, Router, classify  # noqa: E402

console = Console()

# One per task class, plus the ambiguous ones that should fall through to chat.
UTTERANCES = [
    "Hey Sonara, how are you?",
    "What's the weather today?",
    "Open Spotify and play something.",
    "Remind me to call the bank at 6 PM.",
    "Why is my laptop fan so loud, and what should I do about it?",
    "Compare renting versus buying a GPU for local inference.",
    "Write a Python function that debounces keyboard input.",
    "Summarise the key points of that article I pasted earlier.",
]

SYSTEM = ("You are Sonara, a voice assistant. Reply in natural spoken language, "
          "brief but complete. You have no live data yet, so say when you cannot "
          "look something up rather than inventing an answer.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="routing decisions only, no API calls")
    args = ap.parse_args()

    router = Router()

    have = [p for p in router.caps["providers"] if router.available(p)]
    missing = [p for p in router.caps["providers"] if not router.available(p)]
    console.print(f"[green]keys present:[/green] {', '.join(have) or 'none'}")
    if missing:
        console.print(f"[yellow]no key:[/yellow] {', '.join(missing)}")
    console.print()

    tbl = Table(title="Routing decisions" + (" (dry run)" if args.dry else ""))
    tbl.add_column("utterance", max_width=42)
    tbl.add_column("task")
    tbl.add_column("routed to")
    if not args.dry:
        tbl.add_column("1st tok", justify="right")
        tbl.add_column("reply", max_width=40)

    for text in UTTERANCES:
        task = classify(text)
        try:
            if args.dry:
                c = router.choose(text, task=task)
                tbl.add_row(text, task.value, f"{c.provider}/{c.model}")
            else:
                a = router.ask(text, system=SYSTEM, task=task, max_tokens=120)
                tbl.add_row(text, task.value, f"{a.choice.provider}/{a.choice.model}",
                            f"{a.first_token_ms:,.0f}ms", a.text)
        except NoProviderAvailable as e:
            tbl.add_row(text, task.value, f"[red]{e}[/red]", *([""] * (2 if not args.dry else 0)))
    console.print(tbl)

    if not args.dry:
        rows = router.ledger.burn_table()
        if rows:
            bt = Table(title="Ledger burn (last 7 days)")
            for c in ("provider", "model", "calls", "ok", "avg ms"):
                bt.add_column(c, justify="right" if c != "provider" else "left")
            for r in rows:
                bt.add_row(*[str(x) for x in r])
            console.print(bt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
