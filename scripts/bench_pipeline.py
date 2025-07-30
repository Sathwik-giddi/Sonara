"""Headless pipeline benchmark — the M0 number without a microphone.

Piper speaks the prompt, faster-whisper hears it, Groq answers, Piper replies.
Same four stages as scripts/smoke_test.py, but fully automated and repeatable,
so the latency budget can be re-measured after any change (model swap, driver
update, power plan) without a human in the loop.

The mic test is still the ground truth (real acoustics, real VAD endpointing).
This is the instrument for everything in between.

Run:  uv run scripts/bench_pipeline.py [--turns 3]
"""

from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
STT_RATE = 16_000
GATE_S = 2.0  # M1 interim gate

# Utterances Sonara will actually hear, spanning the four action packs.
PROMPTS = [
    "What is the weather like today?",
    "Remind me to call the bank at six PM.",
    "Pause the music and tell me the time.",
]

console = Console()


def synth(voice, text: str) -> tuple[bytes, int]:
    """Piper -> raw int16 PCM."""
    pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(text))
    return pcm, voice.config.sample_rate


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def first_sentence(text: str) -> str:
    for end in (". ", "! ", "? "):
        i = text.find(end)
        if i != -1:
            return text[: i + 1]
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=len(PROMPTS))
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        console.print("[red]GROQ_API_KEY not set in .env[/red]")
        return 1

    voice_path = MODELS / os.environ.get("SMOKE_TTS_VOICE", "en_US-lessac-medium.onnx")
    if not voice_path.exists():
        console.print(f"[red]Piper voice missing: {voice_path} — run ./setup.ps1[/red]")
        return 1

    # ---- load models (excluded from the latency budget) ----
    import cuda_dlls  # must run before faster_whisper imports ctranslate2

    cuda_dirs = cuda_dlls.enable()

    from faster_whisper import WhisperModel

    try:
        from faster_whisper.audio import decode_audio
    except ImportError:
        from faster_whisper import decode_audio  # type: ignore
    from openai import OpenAI
    from piper import PiperVoice

    t = time.perf_counter()
    voice = PiperVoice.load(str(voice_path))
    tts_load = (time.perf_counter() - t) * 1000

    # ctranslate2 fails LAZILY on Windows: the constructor succeeds even when the
    # CUDA runtime DLLs (cublas64_12.dll, cudnn*) are missing, and it only blows up
    # on the first encode. So warm up inside the try and fall back on that.
    warm = np.zeros(STT_RATE, dtype=np.float32)
    device = "cuda"
    t = time.perf_counter()
    try:
        stt = WhisperModel("distil-small.en", device="cuda", compute_type="int8")
        list(stt.transcribe(warm, language="en", beam_size=1)[0])
    except Exception as e:
        device = "cpu"
        console.print(
            f"[yellow]CUDA STT unavailable ({type(e).__name__}: "
            f"{str(e).splitlines()[0][:80]}) — falling back to CPU[/yellow]"
        )
        stt = WhisperModel("distil-small.en", device="cpu", compute_type="int8")
        list(stt.transcribe(warm, language="en", beam_size=1)[0])
    stt_load = (time.perf_counter() - t) * 1000

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    model = os.environ.get("SMOKE_LLM_MODEL", "llama-3.3-70b-versatile")

    console.print(
        f"[dim]STT distil-small.en on {device} ({stt_load:,.0f}ms load) · "
        f"Piper {voice_path.name} ({tts_load:,.0f}ms load) · LLM {model}[/dim]\n"
    )

    rows: list[dict] = []
    for i, prompt in enumerate(PROMPTS[: args.turns]):
        # 0. Piper speaks the prompt (stands in for the user's voice; not timed)
        pcm, rate = synth(voice, prompt)
        wav = pcm_to_wav(pcm, rate)
        wav_path = ROOT / f".bench_turn{i}.wav"
        wav_path.write_bytes(wav)

        try:
            audio = decode_audio(str(wav_path), sampling_rate=STT_RATE)

            # 1. STT — the clock starts here (t0 = end of user speech)
            t = time.perf_counter()
            segments, _ = stt.transcribe(audio, language="en", beam_size=1)
            heard = " ".join(s.text.strip() for s in segments).strip()
            stt_ms = (time.perf_counter() - t) * 1000

            # 2. LLM, streamed — first token is what matters
            first_tok = None
            parts: list[str] = []
            t = time.perf_counter()
            for chunk in client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are Sonara. Reply in one short spoken sentence."},
                    {"role": "user", "content": heard or prompt},
                ],
                stream=True,
                max_tokens=80,
            ):
                delta = chunk.choices[0].delta.content or ""
                if delta and first_tok is None:
                    first_tok = (time.perf_counter() - t) * 1000
                parts.append(delta)
            llm_full = (time.perf_counter() - t) * 1000
            reply = "".join(parts).strip()

            # 3. TTS — first sentence is what the user actually waits for
            t = time.perf_counter()
            synth(voice, first_sentence(reply))
            tts_first = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            synth(voice, reply)
            tts_full = (time.perf_counter() - t) * 1000
        finally:
            wav_path.unlink(missing_ok=True)

        streamed = (stt_ms + (first_tok or llm_full) + tts_first) / 1000
        serial = (stt_ms + llm_full + tts_full) / 1000
        rows.append(
            dict(stt=stt_ms, ftok=first_tok or 0.0, llm=llm_full, ttsf=tts_first,
                 streamed=streamed, serial=serial)
        )

        console.print(f'[bold]{i+1}.[/bold] heard: [dim]{heard}[/dim]')
        console.print(f'   Sonara: {reply}')
        console.print(
            f"   [cyan]streamed {streamed:.2f}s[/cyan]  "
            f"[dim](stt {stt_ms:,.0f} + first-token {first_tok or 0:,.0f} + tts₁ {tts_first:,.0f} ms)[/dim]  "
            f"serial {serial:.2f}s\n"
        )

    tbl = Table(title=f"Pipeline latency over {len(rows)} turns (ms unless noted)")
    for c in ("stage", "min", "median", "max"):
        tbl.add_column(c, justify="right" if c != "stage" else "left")
    for label, k in (("STT", "stt"), ("LLM first token", "ftok"),
                     ("LLM full reply", "llm"), ("TTS first sentence", "ttsf")):
        v = [r[k] for r in rows]
        tbl.add_row(label, f"{min(v):,.0f}", f"{statistics.median(v):,.0f}", f"{max(v):,.0f}")
    for label, k in (("→ STREAMED total (s)", "streamed"), ("→ serial total (s)", "serial")):
        v = [r[k] for r in rows]
        tbl.add_row(label, f"{min(v):.2f}", f"{statistics.median(v):.2f}", f"{max(v):.2f}")
    console.print(tbl)

    p50 = statistics.median(r["streamed"] for r in rows)
    verdict = "PASS" if p50 <= GATE_S else "OVER"
    colour = "green" if p50 <= GATE_S else "red"
    console.print(
        f"\n[bold {colour}]p50 streamed = {p50:.2f}s  →  {verdict} vs the {GATE_S:.1f}s M1 gate[/bold {colour}]"
    )
    console.print(
        "[dim]Headless: no mic, no VAD endpointing (~300-500ms in a real exchange), "
        "no acoustic noise. Treat as the floor, not the field result.[/dim]"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
