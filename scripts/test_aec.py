"""Measure whether the numpy AEC actually stops Sonara hearing itself.

The echo test proved the problem: 19 of 20 playbacks were transcribed back out of
the microphone. This measures the fix on the same hardware, with two numbers that
matter and one that decides it:

  ERLE      how much echo energy the canceller removed (dB, higher is better)
  RESIDUAL  what is left, versus the room's noise floor
  DECIDES   does Whisper still transcribe Sonara out of the residual?

The last one is the real gate. ERLE is a nice number; "Whisper can no longer read
our own voice" is the product requirement.

Run:  uv run scripts/test_aec.py [-n 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as st  # noqa: E402
from sonara.audio.echo import nlms_cancel  # noqa: E402

console = Console()
LINE = "This is Sonara speaking a test sentence for the echo check."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--taps", type=int, default=512)
    ap.add_argument("--mu", type=float, default=0.35)
    args = ap.parse_args()

    from piper import PiperVoice

    voice = PiperVoice.load(str(ROOT / "models" / "en_US-lessac-medium.onnx"))
    sr_tts = voice.config.sample_rate
    SR = st.SAMPLE_RATE

    pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(LINE))
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    idx = np.linspace(0, len(a) - 1, int(len(a) * SR / sr_tts))
    ref = np.interp(idx, np.arange(len(a)), a).astype(np.float32)
    out = np.concatenate([ref, np.zeros(int(0.3 * SR), np.float32)]).reshape(-1, 1)

    console.print(f"[bold]AEC test:[/bold] {args.n} playbacks, {args.taps} taps, mu={args.mu}. "
                  f"Speakers ON, stay quiet.\n")

    rows = []
    for i in range(args.n):
        mic = np.asarray(sd.playrec(out, samplerate=SR, channels=1, dtype="float32"))
        sd.wait()
        mic = mic[:, 0]
        if not np.all(np.isfinite(mic)):
            console.print(f"  {i+1}: invalid capture, skipped")
            continue

        t0 = time.perf_counter()
        r = nlms_cancel(mic, ref, taps=args.taps, mu=args.mu)
        cpu_ms = (time.perf_counter() - t0) * 1000
        realtime_ratio = (len(mic) / SR * 1000) / max(cpu_ms, 1e-6)

        before = st.transcribe(mic) or ""
        after = st.transcribe(r.residual) or ""
        # The product test: did cancelling actually stop Whisper reading us back?
        fixed = len(before.split()) >= 3 and len(after.split()) < 3
        rows.append(dict(erle=r.erle_db, delay=r.delay_samples, mic=r.mic_rms,
                         res=r.residual_rms, before=before, after=after,
                         fixed=fixed, rt=realtime_ratio))
        mark = "[green]silenced[/green]" if fixed else (
            "[red]still heard[/red]" if len(after.split()) >= 3 else "[dim]n/a[/dim]")
        console.print(f"  {i+1:>2}/{args.n}  ERLE {r.erle_db:5.1f} dB  "
                      f"delay {r.delay_samples:>4}  {mark}  after={after[:34]!r}")

    if not rows:
        console.print("[red]no valid captures[/red]")
        return 1

    t = Table(title="AEC result")
    for c in ("metric", "value"):
        t.add_column(c)
    erle = float(np.median([r["erle"] for r in rows]))
    silenced = sum(r["fixed"] for r in rows)
    still = sum(1 for r in rows if len(r["after"].split()) >= 3)
    t.add_row("median ERLE", f"{erle:.1f} dB")
    t.add_row("median delay", f"{int(np.median([r['delay'] for r in rows]))} samples "
                              f"({np.median([r['delay'] for r in rows]) / SR * 1000:.0f} ms)")
    t.add_row("mic RMS -> residual RMS",
              f"{np.median([r['mic'] for r in rows]):.4f} -> "
              f"{np.median([r['res'] for r in rows]):.4f}")
    t.add_row("still transcribable", f"[{'red' if still else 'green'}]{still}/{len(rows)}[/]")
    t.add_row("speed vs realtime", f"{np.median([r['rt'] for r in rows]):.1f}x")
    console.print(t)

    ok = still == 0
    console.print(f"\n[bold]{'[green]AEC WORKS[/green]' if ok else '[red]AEC INSUFFICIENT[/red]'}"
                  f"[/bold] — barge-in is "
                  f"{'viable on laptop speakers' if ok else 'not safe yet; headset or half-duplex'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
