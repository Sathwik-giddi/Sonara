"""Hear the personas side by side, on identical questions.

Choosing a voice from a written description is how products end up sounding like every
other product. This runs the same three exchanges through each persona so the difference
is audible rather than theoretical, and --speak plays them so you judge with your ears -
which is the only sense that matters for a voice assistant.

Run:  uv run scripts/persona_demo.py            # text
      uv run scripts/persona_demo.py --speak    # text + spoken
      uv run scripts/persona_demo.py -p dry
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sonara import Agent  # noqa: E402
from sonara.agent import load_persona  # noqa: E402

console = Console()

# Chosen to expose character where it actually shows: a plain fact, a confirmation,
# and something it cannot do. The third is where personas differ most.
SCRIPT = [
    "what's the weather in Bengaluru",
    "remind me to call the bank at six pm",
    "book me a flight to Tokyo",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--persona", action="append", help="limit to these personas")
    ap.add_argument("--speak", action="store_true", help="also play each reply")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)

    cfg = yaml.safe_load((ROOT / "config/personas.yaml").read_text())
    names = args.persona or list(cfg["personas"])

    if args.speak:
        import smoke_test as st

    for key in names:
        p = cfg["personas"].get(key)
        if not p:
            console.print(f"[red]no persona {key!r}[/red]")
            continue
        prompt, _voice = load_persona(key)
        console.print(f"\n[bold cyan]{'=' * 62}[/bold cyan]")
        console.print(f"[bold cyan]{p['name'].upper()}[/bold cyan]  "
                      f"[dim]{' '.join(p['summary'].split())[:110]}[/dim]")
        console.print(f"[bold cyan]{'=' * 62}[/bold cyan]")

        agent = Agent(system=prompt)
        for utt in SCRIPT:
            r = agent.turn(utt)
            tools = " -> ".join(s.tool for s in r.steps) or "-"
            console.print(f"\n  [dim]you:[/dim]    {utt}")
            console.print(f"  [dim]tools:[/dim]  {tools}")
            console.print(f"  [bold]sonara:[/bold] {r.text}")
            if args.speak:
                st.speak(r.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
