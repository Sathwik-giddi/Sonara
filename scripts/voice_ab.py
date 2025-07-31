"""Compare every available voice on the same lines, then pick by ear.

Kokoro was REJECTED on 2026-07-30 after measuring 2,369-7,680ms per sentence on CPU,
with a note that it crashed under DirectML. Both facts were true. Both are now obsolete:
installing the CUDA toolchain for Whisper also gave onnxruntime a working
CUDAExecutionProvider, and Kokoro runs at 548-868ms - roughly 3x realtime.

That is worth stating plainly, because it is the second time in this project a
"measured, settled" decision was overturned by a change somewhere else entirely. A
measurement is true about a configuration, not about a model.

Run:  uv run scripts/voice_ab.py            # measure + save wavs
      uv run scripts/voice_ab.py --play     # and play each one
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

console = Console()
OUT = ROOT / "data" / "voices"

# Lines that expose different weaknesses: numbers, a natural confirmation, and a longer
# sentence where prosody either holds together or falls apart.
LINES = [
    "It's twenty six degrees and drizzling in Bengaluru.",
    "Done. Reminder set for six PM.",
    "I don't have a way to book flights yet, but I could add it if you want.",
]


def save(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def bench_piper(model: Path) -> tuple[list[float], np.ndarray, int]:
    from piper import PiperVoice

    v = PiperVoice.load(str(model))
    sr = v.config.sample_rate
    v.synthesize("warm up")  # exclude first-call cost from the comparison
    times, last = [], np.zeros(1, np.float32)
    for line in LINES:
        t0 = time.perf_counter()
        pcm = b"".join(c.audio_int16_bytes for c in v.synthesize(line))
        times.append((time.perf_counter() - t0) * 1000)
        last = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return times, last, sr


def bench_kokoro(voice: str) -> tuple[list[float], np.ndarray, int]:
    import cuda_dlls

    cuda_dlls.enable()
    from kokoro_onnx import Kokoro

    k = Kokoro(str(ROOT / "models/kokoro-v1.0.onnx"), str(ROOT / "models/voices-v1.0.bin"))
    k.create("warm up", voice=voice)
    times, last, sr = [], np.zeros(1, np.float32), 24000
    for line in LINES:
        t0 = time.perf_counter()
        last, sr = k.create(line, voice=voice)
        times.append((time.perf_counter() - t0) * 1000)
    return times, np.asarray(last, dtype=np.float32), sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()

    candidates: list[tuple[str, str, str]] = []
    for p in sorted((ROOT / "models").glob("en_*.onnx")):
        candidates.append((p.stem, "piper", str(p)))
    if (ROOT / "models/kokoro-v1.0.onnx").exists():
        # Kokoro ships many speakers in one file; these three are the most natural.
        for v in ("af_heart", "af_bella", "am_michael"):
            candidates.append((f"kokoro:{v}", "kokoro", v))

    tbl = Table(title="Voice comparison — same three lines, warm")
    for c in ("voice", "engine", "median ms", "x realtime", "file"):
        tbl.add_column(c, justify="right" if c not in ("voice", "engine", "file") else "left")

    for name, engine, ref in candidates:
        try:
            if engine == "piper":
                times, audio, sr = bench_piper(Path(ref))
            else:
                times, audio, sr = bench_kokoro(ref)
        except Exception as e:  # noqa: BLE001
            tbl.add_row(name, engine, "-", "-", f"[red]{type(e).__name__}[/red]")
            continue

        med = float(np.median(times))
        dur = len(audio) / sr
        out = OUT / f"{name.replace(':', '_')}.wav"
        save(out, audio, sr)
        tbl.add_row(name, engine, f"{med:,.0f}", f"{dur / (med / 1000):.1f}x",
                    str(out.relative_to(ROOT)))
        if args.play:
            import sounddevice as sd

            console.print(f"[dim]playing {name}...[/dim]")
            sd.play(audio, sr)
            sd.wait()

    console.print(tbl)
    console.print(f"\n[dim]Wavs in {OUT.relative_to(ROOT)} — listen and pick. "
                  f"Anything under ~1000ms fits the latency budget.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
