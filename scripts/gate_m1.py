"""GATE-M1: the gate that unblocks everything else.

Two halves, from the design doc:

  LATENCY   p50 <= 2.0s end-of-speech -> first audio, over 50 REAL microphone
            exchanges. Scored per serving tier, because a 550B reasoning answer
            is allowed 2.5s and judging it against the chat budget would fail a
            system behaving exactly as designed.
  ECHO      zero self-interruptions across 20 playback exchanges. Sonara must not
            hear itself. This has been an unmeasured risk since the design doc
            named it, and it is the one that decides whether barge-in is possible
            on laptop speakers at all.

RESUMABLE. 50 exchanges is a lot of talking; results append to data/gate_m1.jsonl
and the run picks up where it stopped. Do it in batches over a few days if you
like - that is closer to real usage anyway.

Run:  uv run scripts/gate_m1.py              # continue the latency run
      uv run scripts/gate_m1.py --echo-test  # the AEC half (no talking needed)
      uv run scripts/gate_m1.py --status     # where am I
      uv run scripts/gate_m1.py --reset      # start over
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as st  # noqa: E402  reuses the measured pipeline, no duplication

console = Console()
RESULTS = ROOT / "data" / "gate_m1.jsonl"
TARGET_EXCHANGES = 50
ECHO_EXCHANGES = 20

# Prompts to speak. Deliberately spread across task classes so the p50 reflects
# real routing, not 50 easy chat turns on the fast tier.
PROMPTS = [
    "What time is it?", "Pause the music.", "Turn the volume up.",
    "Open Spotify.", "Remind me to stretch in ten minutes.",
    "Make a note that the gate is running.", "What reminders do I have?",
    "How are you doing today?", "What can you do?", "Tell me a short joke.",
]


def load() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]


def append(row: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def report(rows: list[dict]) -> bool:
    latency = [r for r in rows if r.get("kind") == "exchange" and r.get("total_s")]
    if not latency:
        console.print("[yellow]No exchanges recorded yet.[/yellow]")
        return False

    tbl = Table(title=f"Latency — {len(latency)}/{TARGET_EXCHANGES} exchanges")
    for c in ("tier", "n", "p50 s", "p95 s", "target", "verdict"):
        tbl.add_column(c, justify="right" if c != "tier" else "left")

    all_pass = True
    for tier in sorted({r["tier"] for r in latency}):
        vals = sorted(r["total_s"] for r in latency if r["tier"] == tier)
        target = latency[[r["tier"] for r in latency].index(tier)]["target_s"]
        p50 = statistics.median(vals)
        p95 = vals[max(0, int(len(vals) * 0.95) - 1)]
        ok = p50 <= target
        all_pass &= ok
        tbl.add_row(tier, str(len(vals)), f"{p50:.2f}", f"{p95:.2f}", f"{target:.1f}",
                    "[green]PASS[/green]" if ok else "[red]OVER[/red]")
    console.print(tbl)

    echo = [r for r in rows if r.get("kind") == "echo"]
    if echo:
        heard = sum(1 for r in echo if r["self_heard"])
        e_ok = heard == 0
        console.print(f"\n[bold]Echo test:[/bold] {len(echo)} playback exchanges, "
                      f"self-heard {heard} — "
                      f"{'[green]PASS[/green]' if e_ok else '[red]FAIL (AEC needed)[/red]'}")
        all_pass &= e_ok
    else:
        console.print("\n[yellow]Echo test not run yet (--echo-test)[/yellow]")
        all_pass = False

    complete = len(latency) >= TARGET_EXCHANGES and len(echo) >= ECHO_EXCHANGES
    if not complete:
        console.print(f"[yellow]Incomplete: need {TARGET_EXCHANGES} exchanges "
                      f"and {ECHO_EXCHANGES} echo trials.[/yellow]")
    verdict = all_pass and complete
    console.print(f"\n[bold]{'[green]GATE-M1 PASS[/green]' if verdict else '[red]GATE-M1 not yet passed[/red]'}"
                  f"[/bold]  (hard pause on pillars M5-M8 until this clears)")
    return verdict


def run_echo_test(n: int) -> None:
    """Does the microphone hear Sonara's own voice through the speakers?

    No talking required. Sonara speaks; the mic records at the same time; the same
    STT then reads that recording. If Whisper transcribes Sonara's own words, then
    with barge-in enabled Sonara would interrupt itself - and acoustic echo
    cancellation moves from 'named risk' to 'blocking requirement'.
    """
    from piper import PiperVoice

    voice_path = ROOT / "models" / "en_US-lessac-medium.onnx"
    voice = PiperVoice.load(str(voice_path))
    sr = voice.config.sample_rate
    line = "This is Sonara speaking a test sentence for the echo check."

    console.print(f"[bold]Echo test:[/bold] {n} playbacks. Speakers ON, do not speak.\n")
    for i in range(n):
        pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(line))
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        dur = len(audio) / sr

        rec = sd.rec(int((dur + 0.3) * st.SAMPLE_RATE), samplerate=st.SAMPLE_RATE,
                     channels=1, dtype="float32")
        sd.play(audio, sr)
        sd.wait()
        mic = rec[:, 0]

        peak = float(np.abs(mic).max())
        text = st.transcribe(mic) or ""
        # Two independent signals: did the mic pick up speech-level energy, and did
        # STT actually decode Sonara's words back out of it?
        self_heard = peak > 0.05 and len(text.split()) >= 3
        append({"kind": "echo", "i": i, "peak": round(peak, 4),
                "heard": text[:80], "self_heard": bool(self_heard),
                "ts": datetime.now().isoformat(timespec="seconds")})
        mark = "[red]HEARD ITSELF[/red]" if self_heard else "[green]clean[/green]"
        console.print(f"  {i+1:>2}/{n}  peak {peak:.3f}  {mark}  {text[:50]!r}")


def run_exchanges(rows: list[dict]) -> None:
    done = sum(1 for r in rows if r.get("kind") == "exchange")
    console.print(f"[bold]Latency run:[/bold] {done}/{TARGET_EXCHANGES} done. "
                  f"Ctrl-C any time — progress is saved.\n")
    st.warm_connection()

    i = done
    while i < TARGET_EXCHANGES:
        prompt = PROMPTS[i % len(PROMPTS)]
        console.print(f"\n[bold cyan]{i+1}/{TARGET_EXCHANGES}[/bold cyan]  say: [bold]{prompt}[/bold]")
        st.timings.clear()
        audio = st.record_utterance()
        text = st.transcribe(audio)
        if not text:
            console.print("[yellow]nothing heard — retrying this one[/yellow]")
            continue
        reply = st.ask_llm(text)
        if reply:
            st.speak(reply)

        s, f, t = (st.timings.get("stt_transcribe"), st.timings.get("llm_first_token"),
                   st.timings.get("tts_first_sentence"))
        if None in (s, f, t):
            console.print("[yellow]stage missing — not scored[/yellow]")
            continue
        total = (s + f + t) / 1000
        tier = str(st.last_route["tier"])
        target = float(st.last_route["target_s"])
        append({"kind": "exchange", "i": i, "prompt": prompt, "heard": text,
                "stt_ms": s, "first_token_ms": f, "tts_ms": t,
                "total_s": round(total, 3), "tier": tier, "target_s": target,
                "ts": datetime.now().isoformat(timespec="seconds")})
        ok = total <= target
        console.print(f"  [{'green' if ok else 'red'}]{total:.2f}s[/] on {tier} tier "
                      f"(target {target:.1f}s)")
        i += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--echo-test", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("-n", type=int, default=ECHO_EXCHANGES, help="echo trials")
    args = ap.parse_args()

    if args.reset:
        RESULTS.unlink(missing_ok=True)
        console.print("[yellow]reset — all GATE-M1 results cleared[/yellow]")
        return 0

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)

    if args.status:
        return 0 if report(load()) else 1
    if args.echo_test:
        run_echo_test(args.n)
    else:
        try:
            run_exchanges(load())
        except KeyboardInterrupt:
            console.print("\n[yellow]stopped — progress saved, rerun to continue[/yellow]")

    return 0 if report(load()) else 1


if __name__ == "__main__":
    sys.exit(main())
