"""M0 smoke test: measure every stage of the voice loop on this laptop.

The number that decides the project: end-of-speech -> first audio.
M1 gate is p50 <= 2.0s; final target 1.5s (fast tier).

Run:  uv run scripts/smoke_test.py
Requires GROQ_API_KEY in .env for the LLM stage (others skip gracefully).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SAMPLE_RATE = 16_000
RECORD_SECONDS = float(os.environ.get("SMOKE_RECORD_SECONDS", "6"))
LAST_TAKE = ROOT / ".last_recording.wav"
STT_MODEL = os.environ.get("SMOKE_STT_MODEL", "distil-small.en")
STT_BEAM = int(os.environ.get("SMOKE_STT_BEAM", "5"))

console = Console()
timings: dict[str, float | None] = {}


def stage(name: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            timings[name] = (time.perf_counter() - self.t0) * 1000

    return _Timer()


def record_utterance() -> np.ndarray:
    console.print("\n[bold]Audio devices:[/bold]")
    console.print(sd.query_devices())
    console.input(
        f"\n[bold green]Press Enter, then speak one sentence "
        f"({RECORD_SECONDS:.0f}s recording)...[/bold green]"
    )
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    console.print("[dim]Recording done. The latency clock starts NOW.[/dim]")
    mono = audio[:, 0]

    # Keep the take. Lets STT settings be A/B'd against a real voice without
    # re-recording, and makes a mishearing reproducible instead of anecdotal.
    with wave.open(str(LAST_TAKE), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(mono, -1, 1) * 32767).astype(np.int16).tobytes())

    peak = float(np.abs(mono).max())
    if peak < 0.05:
        console.print(f"[yellow]Mic level very low (peak {peak:.3f}) — check input gain[/yellow]")
    return mono


def transcribe(audio: np.ndarray) -> str | None:
    try:
        import cuda_dlls  # must run before faster_whisper imports ctranslate2

        cuda_dlls.enable()
        from faster_whisper import WhisperModel
    except Exception as e:
        console.print(f"[red]STT skipped: faster-whisper not importable ({e})[/red]")
        return None

    # ctranslate2 fails LAZILY: the constructor succeeds without the CUDA DLLs and
    # only dies on the first encode. So warm up inside the try, or the fallback
    # never fires and the crash lands mid-conversation instead.
    warm = np.zeros(SAMPLE_RATE, dtype=np.float32)
    with stage("stt_model_load (excluded from budget)"):
        try:
            model = WhisperModel(STT_MODEL, device="cuda", compute_type="int8")
            list(model.transcribe(warm, language="en", beam_size=1)[0])
            console.print(f"[dim]STT {STT_MODEL} on CUDA (beam {STT_BEAM})[/dim]")
        except Exception as e:
            console.print(
                f"[yellow]CUDA unavailable ({type(e).__name__}: "
                f"{str(e).splitlines()[0][:70]}) — STT on CPU[/yellow]"
            )
            model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
            list(model.transcribe(warm, language="en", beam_size=1)[0])

    with stage("stt_transcribe"):
        # vad_filter drops the silence around the utterance; a fixed 6s buffer is
        # mostly silence and Whisper is known to degrade on that. NOTE: this is
        # principled, not yet validated as the fix for the real mishearing
        # ("what's the weather today" -> "A worst weather today"). A 4-way sweep
        # (beam 1/5 x vad on/off) on synthetic speech showed no difference at all,
        # so config is probably not the lever - model size and acoustics are.
        # Settle it against a real take: smoke_test.py --replay
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=STT_BEAM,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    console.print(f"[bold]Heard:[/bold] {text!r}")
    return text or None


def ask_llm(text: str) -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        console.print("[yellow]LLM skipped: GROQ_API_KEY not set in .env[/yellow]")
        return None
    from openai import OpenAI

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    first_token_ms = None
    reply_parts: list[str] = []
    t0 = time.perf_counter()
    with stage("llm_full_reply"):
        streamed = client.chat.completions.create(
            model=os.environ.get("SMOKE_LLM_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": "You are Sonara. Reply in one short spoken sentence."},
                {"role": "user", "content": text},
            ],
            stream=True,
            max_tokens=80,
        )
        for chunk in streamed:
            delta = chunk.choices[0].delta.content or ""
            if delta and first_token_ms is None:
                first_token_ms = (time.perf_counter() - t0) * 1000
            reply_parts.append(delta)
    timings["llm_first_token"] = first_token_ms
    reply = "".join(reply_parts).strip()
    console.print(f"[bold]Sonara:[/bold] {reply}")
    return reply or None


def speak(text: str) -> None:
    # Piper won the 2026-07-30 TTS bake-off on this CPU: ~1s/sentence warm vs
    # Kokoro's 2.4-7.7s (and Kokoro's ConvTranspose op crashes under DirectML).
    voice_path = MODELS / os.environ.get("SMOKE_TTS_VOICE", "en_US-lessac-medium.onnx")
    if not voice_path.exists():
        console.print("[yellow]TTS skipped: run setup.ps1 to download the Piper voice[/yellow]")
        return
    from piper import PiperVoice

    with stage("tts_model_load (excluded from budget)"):
        voice = PiperVoice.load(str(voice_path))
    sr = voice.config.sample_rate
    with stage("tts_synthesize"):
        pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(text))
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(samples, sr)
    sd.wait()


def warm_connection() -> None:
    """Pay the DNS + TLS + HTTP/2 setup cost before the user speaks.

    Measured: turn 1 of a session costs ~1.7-2.1s extra to first token,
    reproducibly, while every later turn is 160-290ms. Without this the very
    first thing a user ever says to Sonara is also the slowest.
    """
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return
    from openai import OpenAI

    try:
        t = time.perf_counter()
        OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key).chat.completions.create(
            model=os.environ.get("SMOKE_LLM_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        console.print(f"[dim]Groq connection warmed ({(time.perf_counter()-t)*1000:,.0f}ms, off the clock)[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warm-up failed ({type(e).__name__}) — first turn will be slow[/yellow]")


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", metavar="WAV", default=None,
                    help=f"re-run on a saved take instead of the mic (default {LAST_TAKE.name})")
    ap.add_argument("--replay", action="store_true", help=f"shorthand for --from-file {LAST_TAKE.name}")
    args = ap.parse_args()

    # override=True: .env wins over any stale OS-level key (see bench_pipeline.py)
    load_dotenv(ROOT / ".env", override=True)

    src = args.from_file or (str(LAST_TAKE) if args.replay else None)
    if src:
        p = Path(src)
        if not p.exists():
            console.print(f"[red]No such recording: {p}[/red]")
            return
        console.print(f"[dim]Replaying {p.name} (no mic, latency not comparable)[/dim]")
        audio = load_wav(p)
    else:
        warm_connection()
        audio = record_utterance()

    text = transcribe(audio)
    reply = ask_llm(text) if text else None
    if reply:
        speak(reply)

    table = Table(title="Smoke test timings (ms)")
    table.add_column("Stage")
    table.add_column("ms", justify="right")
    budget_keys = ("stt_transcribe", "llm_first_token", "llm_full_reply", "tts_synthesize")
    for k, v in timings.items():
        table.add_row(k, "-" if v is None else f"{v:,.0f}")
    console.print(table)

    # End-of-speech -> first audio, approximated as stt + llm full + tts synth
    # (streamed sentence-level TTS in M1 will start audio at llm_first_token + first-sentence synth)
    parts = [timings.get("stt_transcribe"), timings.get("llm_full_reply"), timings.get("tts_synthesize")]
    if all(p is not None for p in parts):
        total = sum(parts) / 1000
        verdict = "PASS" if total <= 2.0 else "OVER"
        console.print(
            f"\n[bold]End-of-speech -> first audio (worst-case, unstreamed): "
            f"{total:.2f}s  [{verdict} vs 2.0s M1 gate][/bold]"
        )
        ft = timings.get("llm_first_token")
        st = timings.get("stt_transcribe")
        ts = timings.get("tts_synthesize")
        if ft is not None and st is not None and ts is not None:
            streamed_est = (st + ft + ts) / 1000
            console.print(
                f"[bold]Streamed-pipeline estimate (what M1 will actually feel like): "
                f"~{streamed_est:.2f}s[/bold]"
            )
    else:
        console.print("\n[yellow]Total not computable: one or more stages skipped (see above).[/yellow]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
