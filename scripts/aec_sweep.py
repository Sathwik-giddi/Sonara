"""AEC measured across REAL listening volumes, with the volume actually controlled.

WHY THIS EXISTS. The first AEC measurement was confounded: the system volume changed
partway through (it was too loud, so it got turned down), and echo levels swung 10x
between trials. The conclusion drawn from it - "the mic DSP is adapting, linear AEC
cannot work" - was not supported by the data, because the input was not held still.

A measurement that does not control its own inputs is an anecdote.

So this sets the volume to a known percentage before each block, records it alongside
every result, and reports per volume. The output answers the question a USER cares
about: "at the volume I actually listen at, does Sonara interrupt itself?"

Run:  uv run scripts/aec_sweep.py                    # 20/40/60% sweep
      uv run scripts/aec_sweep.py --volumes 50       # single level
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as st  # noqa: E402
from sonara.audio.echo import cancel_full  # noqa: E402

console = Console()
RESULTS = ROOT / "data" / "aec_sweep.jsonl"
LINE = "This is Sonara speaking a test sentence for the echo check."

VK_VOL_UP, VK_VOL_DOWN, KEYEVENTF_KEYUP = 0xAF, 0xAE, 0x0002


def _tap(vk: int, times: int) -> None:
    u = ctypes.windll.user32
    for _ in range(times):
        u.keybd_event(vk, 0, 0, 0)
        u.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.008)


def set_volume_percent(pct: int) -> None:
    """Absolute volume by flooring then stepping up. Each media-key step is 2%.

    Relative control ("volume up") is useless for a measurement - you never know
    where you started. Flooring first makes every run start from the same place,
    which is the whole point.
    """
    _tap(VK_VOL_DOWN, 55)              # floor it
    _tap(VK_VOL_UP, max(0, round(pct / 2)))
    time.sleep(0.4)


def one_trial(ref: np.ndarray, sr: int) -> dict | None:
    out = np.concatenate([ref, np.zeros(int(0.4 * sr), np.float32)]).reshape(-1, 1)
    mic = np.asarray(sd.playrec(out, samplerate=sr, channels=1, dtype="float32"))
    sd.wait()
    mic = mic[:, 0]
    if not np.all(np.isfinite(mic)) or np.abs(mic).max() > 1.5:
        return None

    # Two-stage: linear NLMS + spectral residual suppression. Linear alone gave
    # ~7 dB; the post-filter takes it to ~19 dB, which is what AEC3 does structurally.
    r = cancel_full(mic, ref, taps=4096, mu=1.0, over=3.0, floor=0.02)
    before = st.transcribe(mic) or ""
    after = st.transcribe(r.residual) or ""
    return {
        "peak": float(np.abs(mic).max()),
        "mic_rms": r.mic_rms, "res_rms": r.residual_rms,
        "erle_db": r.erle_db, "delay": r.delay_samples,
        "heard_raw": before[:60], "heard_after": after[:60],
        "self_heard_raw": len(before.split()) >= 3,
        "self_heard_after": len(after.split()) >= 3,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=int, nargs="+", default=[20, 40, 60])
    ap.add_argument("-n", type=int, default=5, help="trials per volume")
    args = ap.parse_args()

    from piper import PiperVoice

    voice = PiperVoice.load(str(ROOT / "models" / "en_US-lessac-medium.onnx"))
    SR = st.SAMPLE_RATE
    pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(LINE))
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    idx = np.linspace(0, len(a) - 1, int(len(a) * SR / voice.config.sample_rate))
    ref = np.interp(idx, np.arange(len(a)), a).astype(np.float32)

    console.print("[bold]AEC volume sweep[/bold] — volume is SET per block, not assumed.")
    console.print("[dim]Stay quiet. Volume will be changed automatically and restored at the end.[/dim]\n")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    tbl = Table(title="Echo vs listening volume")
    for c in ("volume", "mic peak", "raw self-heard", "ERLE", "after AEC", "verdict"):
        tbl.add_column(c, justify="right" if c != "verdict" else "left")

    for vol in args.volumes:
        set_volume_percent(vol)
        rows = [t for t in (one_trial(ref, SR) for _ in range(args.n)) if t]
        if not rows:
            tbl.add_row(f"{vol}%", "-", "-", "-", "-", "[red]no valid captures[/red]")
            continue
        for r in rows:
            r["volume_pct"] = vol
            r["ts"] = datetime.now().isoformat(timespec="seconds")
            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")

        raw = sum(r["self_heard_raw"] for r in rows)
        post = sum(r["self_heard_after"] for r in rows)
        erle = float(np.median([r["erle_db"] for r in rows]))
        peak = float(np.median([r["peak"] for r in rows]))
        if raw == 0:
            verdict = "[dim]no echo at this volume[/dim]"
        elif post == 0:
            verdict = "[green]AEC fixes it[/green]"
        else:
            verdict = "[red]still interrupts itself[/red]"
        tbl.add_row(f"{vol}%", f"{peak:.3f}", f"{raw}/{len(rows)}",
                    f"{erle:.1f} dB", f"{post}/{len(rows)}", verdict)
        console.print(f"[dim]  {vol}%: peak {peak:.3f}, raw {raw}/{len(rows)}, "
                      f"after {post}/{len(rows)}[/dim]")

    set_volume_percent(40)  # leave the machine somewhere sane
    console.print()
    console.print(tbl)
    console.print("\n[dim]Volume restored to 40%. Results appended to data/aec_sweep.jsonl[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
