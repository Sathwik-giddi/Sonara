"""Discover what each provider's key can ACTUALLY reach, and tool-call test it.

config/models.yaml carries candidate model ids marked `verify: true`. NVIDIA NIM
alone hosts 100+ models and the catalog moves, so guessing ids from documentation
is how you ship a router that 404s in month two. This asks each provider directly.

Run:  uv run scripts/probe_models.py            # list catalogs, check candidates
      uv run scripts/probe_models.py --tools    # also verify tool calling works
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sonara.router import KEY_VARS, Router  # noqa: E402

console = Console()

TOOL = {
    "type": "function",
    "function": {
        "name": "set_reminder",
        "description": "Create a reminder at a given time",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "what to be reminded of"},
                "when": {"type": "string", "description": "ISO time or natural language"},
            },
            "required": ["text", "when"],
        },
    },
}


def catalog(base_url: str, key: str) -> list[str] | str:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models",
                      headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        return sorted(m["id"] for m in r.json().get("data", []))
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}"


def tool_test(router: Router, provider: str, model: str) -> str:
    """Does this model emit a well-formed tool call? The whole product depends on it."""
    try:
        r = router._client(provider).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Remind me to call the bank at 6pm today."}],
            tools=[TOOL],
            max_tokens=200,
        )
        calls = r.choices[0].message.tool_calls or []
        if not calls:
            return "no tool call"
        fn = calls[0].function
        if fn.name != "set_reminder":
            return f"wrong tool: {fn.name}"
        import json

        args = json.loads(fn.arguments)
        missing = {"text", "when"} - set(args)
        return "OK" if not missing else f"missing {missing}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[0][:50]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", action="store_true", help="verify tool calling on each candidate")
    ap.add_argument("--list", action="store_true", help="print each provider's full catalog")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    router = Router()

    catalogs: dict[str, list[str] | str] = {}
    tbl = Table(title="Providers")
    for c in ("provider", "key", "catalog"):
        tbl.add_column(c)
    for provider, cfg in router.caps["providers"].items():
        var = KEY_VARS.get(provider)
        key = router.key_for(provider)
        if key is None:
            tbl.add_row(provider, f"[red]missing {var}[/red]", "-")
            continue
        if provider == "ollama_local":
            tbl.add_row(provider, "n/a (local)", "run `ollama list`")
            continue
        got = catalog(cfg["base_url"], key)
        catalogs[provider] = got
        tbl.add_row(provider, "[green]set[/green]",
                    f"{len(got)} models" if isinstance(got, list) else f"[red]{got}[/red]")
    console.print(tbl)

    # Are the ids we route to actually real?
    console.print("\n[bold]Candidates in config/models.yaml[/bold]")
    checked: set[tuple[str, str]] = set()
    ct = Table()
    for c in ("task", "provider", "model", "exists", "tool call" if args.tools else ""):
        if c:
            ct.add_column(c)
    for task, cands in router.models["routes"].items():
        for cand in cands:
            provider, model = cand["provider"], cand["model"]
            cat = catalogs.get(provider)
            if provider == "ollama_local":
                exists = "[dim]local[/dim]"
            elif not isinstance(cat, list):
                exists = "[dim]?[/dim]"
            else:
                exists = "[green]yes[/green]" if model in cat else "[red]NOT FOUND[/red]"

            cells = [task, provider, model, exists]
            if args.tools:
                if (provider, model) in checked or exists.endswith("NOT FOUND[/red]") \
                        or provider == "ollama_local" or not isinstance(cat, list):
                    cells.append("[dim]-[/dim]")
                else:
                    checked.add((provider, model))
                    res = tool_test(router, provider, model)
                    cells.append(f"[green]{res}[/green]" if res == "OK" else f"[yellow]{res}[/yellow]")
            ct.add_row(*cells)
    console.print(ct)

    if args.list:
        for provider, got in catalogs.items():
            if isinstance(got, list):
                console.print(f"\n[bold]{provider}[/bold] ({len(got)}):")
                for m in got:
                    console.print(f"  {m}")

    console.print("\n[dim]Fix any NOT FOUND in config/models.yaml, then drop its `verify: true`.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
