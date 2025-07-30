"""Measure which models on a provider are actually USABLE, not merely listed.

Discovered the hard way: NVIDIA NIM lists 102 models, but availability in the
catalog does not mean the free tier will serve you. meta/llama-3.3-70b-instruct
read-timed out after 180s while meta/llama-3.1-8b-instruct answered in 429ms.
A voice assistant cannot route to a model that might hang - so candidates are
picked on measured latency, not on a catalog listing.

Run:  uv run scripts/probe_latency.py                    # NIM shortlist
      uv run scripts/probe_latency.py --provider groq
      uv run scripts/probe_latency.py --timeout 30 --models a/b,c/d
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sonara.router import Router  # noqa: E402

console = Console()

# The powerful ones worth routing to, from the real NIM catalog.
NIM_SHORTLIST = [
    # flagship reasoning
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
    "moonshotai/kimi-k2.6",
    "minimaxai/minimax-m3",
    "openai/gpt-oss-120b",
    # tool calling / general
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/nemotron-3-nano-30b-a3b",
    "openai/gpt-oss-20b",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    # code
    "mistralai/codestral-22b-instruct-v0.1",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "meta/codellama-70b",
]

PROMPT = "In one sentence, why is the sky blue?"


def probe(base: str, key: str, model: str, timeout: float) -> tuple[str, float, str]:
    t = time.perf_counter()
    try:
        r = httpx.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": PROMPT}],
                  "max_tokens": 40},
            timeout=timeout,
        )
        ms = (time.perf_counter() - t) * 1000
        if r.status_code != 200:
            body = r.text[:60].replace("\n", " ")
            return (f"HTTP {r.status_code}", ms, body)
        txt = (r.json()["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")
        return ("ok", ms, txt[:52])
    except httpx.ReadTimeout:
        return ("TIMEOUT", timeout * 1000, "no response")
    except Exception as e:  # noqa: BLE001
        return (type(e).__name__, (time.perf_counter() - t) * 1000, str(e)[:52])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="nvidia_nim")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--models", default=None, help="comma-separated override")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    router = Router()
    key = router.key_for(args.provider)
    if not key:
        console.print(f"[red]no key for {args.provider}[/red]")
        return 1
    base = router.caps["providers"][args.provider]["base_url"]

    models = args.models.split(",") if args.models else (
        NIM_SHORTLIST if args.provider == "nvidia_nim" else
        [router.caps["providers"][args.provider]["pinned_model"]])

    tbl = Table(title=f"{args.provider} — usable? ({args.timeout:.0f}s timeout)")
    for c, j in (("model", "left"), ("status", "left"), ("ms", "right"), ("reply", "left")):
        tbl.add_column(c, justify=j, max_width=54 if c == "reply" else None)

    usable: list[tuple[str, float]] = []
    for m in models:
        status, ms, note = probe(base, key, m, args.timeout)
        if status == "ok":
            usable.append((m, ms))
            tbl.add_row(m, "[green]ok[/green]", f"{ms:,.0f}", note)
        else:
            colour = "red" if status == "TIMEOUT" else "yellow"
            tbl.add_row(m, f"[{colour}]{status}[/{colour}]", f"{ms:,.0f}", f"[dim]{note}[/dim]")
    console.print(tbl)

    if usable:
        console.print("\n[bold]Fastest usable, in order:[/bold]")
        for m, ms in sorted(usable, key=lambda x: x[1]):
            console.print(f"  {ms:>7,.0f} ms  {m}")
    console.print("\n[dim]Route only to models that answered. A catalog listing is not a promise.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
