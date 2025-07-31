"""M0 smoke test: measure every stage of the voice loop on this laptop.

The number that decides the project: end-of-speech -> first audio.
M1 gate is p50 <= 2.0s; final target 1.5s (fast tier).

Run:  uv run scripts/smoke_test.py
Requires GROQ_API_KEY in .env for the LLM stage (others skip gracefully).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `from sonara import ...` works when run as a script
MODELS = ROOT / "models"
SAMPLE_RATE = 16_000
RECORD_SECONDS = float(os.environ.get("SMOKE_RECORD_SECONDS", "6"))
LAST_TAKE = ROOT / ".last_recording.wav"
CORPUS = ROOT / "recordings"
PRIME_SECONDS = float(os.environ.get("SMOKE_PRIME_SECONDS", "0.4"))
# medium.en, chosen by A/B on real recordings of THIS voice (2026-08-01). On the same
# three takes distil-small.en produced "what the airiness" where medium.en produced
# "what the heck". It costs ~440ms more (273 -> 710ms) and the budget has it: measured
# exchanges ran 0.58-0.81s against a 2.0s gate, so this lands ~1.0-1.25s.
STT_MODEL = os.environ.get("SMOKE_STT_MODEL", "medium.en")
STT_BEAM = int(os.environ.get("SMOKE_STT_BEAM", "5"))

console = Console()
timings: dict[str, float | None] = {}
# Which tier served the last utterance, and the budget that tier is judged against.
last_route: dict[str, object] = {"tier": "fast", "target_s": 2.0}


def for_speech(text: str) -> str:
    """Strip markdown before it reaches TTS.

    Chat models answer in markdown. Piper reads it literally, so "**Buying**
    makes sense" is spoken as "asterisk asterisk Buying asterisk asterisk". The
    fix belongs here, not in the prompt: no instruction reliably stops every
    model from ever emitting a bullet.
    """
    t = re.sub(r"```.*?```", " ", text, flags=re.S)          # code fences
    t = re.sub(r"`([^`]*)`", r"\1", t)                        # inline code
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)                  # bold
    t = re.sub(r"(?<!\w)[*_]([^*_]+)[*_](?!\w)", r"\1", t)    # italic
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)       # headings
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)            # bullets
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.M)            # numbered lists
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)            # links -> label
    return re.sub(r"\s+", " ", t).strip()


def stage(name: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            timings[name] = (time.perf_counter() - self.t0) * 1000

    return _Timer()


_devices_shown = False


def record_utterance() -> np.ndarray:
    # Once per process, not once per exchange. Printing 24 device lines before all 50
    # gate exchanges buries the only lines that matter.
    global _devices_shown
    if not _devices_shown:
        console.print("\n[bold]Audio devices:[/bold]")
        console.print(sd.query_devices())
        _devices_shown = True
    console.input("\n[bold green]Press Enter to arm the mic...[/bold green]")

    # Open the stream and throw away the first PRIME_SECONDS. WASAPI needs a moment
    # to spin up, and sd.rec() starts the clock before the device is really ready -
    # so speaking immediately loses the leading phoneme. That is the prime suspect
    # for "what's the" being heard as "Watch the" / "A worst" on some takes but not
    # others: the model was never the variable, the onset was.
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    stream.start()
    try:
        stream.read(int(PRIME_SECONDS * SAMPLE_RATE))  # discarded
        console.print(f"[bold cyan]>>> SPEAK NOW ({RECORD_SECONDS:.0f}s) <<<[/bold cyan]")
        frames, _ = stream.read(int(RECORD_SECONDS * SAMPLE_RATE))
    finally:
        stream.stop()
        stream.close()
    console.print("[dim]Recording done. The latency clock starts NOW.[/dim]")
    mono = np.asarray(frames, dtype=np.float32)[:, 0]

    # If speech is already loud in the first 150ms, the take probably started
    # mid-word and any mistranscription is the capture's fault, not the model's.
    lead = mono[: int(0.15 * SAMPLE_RATE)]
    if lead.size and float(np.abs(lead).max()) > 0.15:
        console.print("[yellow]Speech detected at the very start — you may have clipped the first word. "
                      "Wait for the SPEAK NOW cue.[/yellow]")

    # Keep every take. Two purposes: A/B STT settings against a real voice without
    # re-recording, and accumulate the corpus the M1 gate needs (50 real exchanges)
    # so any future config change can be re-scored against all of it at once.
    pcm16 = (np.clip(mono, -1, 1) * 32767).astype(np.int16).tobytes()

    def _write(path: Path) -> None:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm16)

    _write(LAST_TAKE)
    CORPUS.mkdir(exist_ok=True)
    archived = CORPUS / f"{datetime.now():%Y%m%d-%H%M%S}.wav"
    _write(archived)
    console.print(f"[dim]saved {archived.relative_to(ROOT)}[/dim]")

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


# Two failures this replaces: one-sentence replies could not answer a two-part
# question, and with no tools yet it confidently invented a weather forecast.
# Until the M3 skill layer lands, "I can't know that yet" is the honest answer
# and the one that keeps trust.
SYSTEM_PROMPT = (
    "You are Sonara, a voice assistant. Reply in natural spoken language, brief "
    "but complete: if the user asks two things, answer both. You have no live "
    "data yet - no weather, news, web or calendar access - so when asked for "
    "real-time facts, say you cannot look that up yet instead of inventing an answer."
)


def ask_llm(text: str) -> str | None:
    """Route the utterance to whichever brain suits it, then stream the reply."""
    router = get_router()
    if router is None:
        return None

    from sonara import NoProviderAvailable, classify

    task = classify(text)
    try:
        with stage("llm_full_reply"):
            answer = router.ask(text, system=SYSTEM_PROMPT, task=task, max_tokens=200)
    except NoProviderAvailable as e:
        console.print(f"[red]No brain available: {e}[/red]")
        return None

    timings["llm_first_token"] = answer.first_token_ms
    last_route["tier"] = router.tier_of(answer.choice.provider)
    last_route["target_s"] = float(
        router.models.get("latency_targets_s", {}).get(last_route["tier"], 2.0)
    )
    # Show the routing decision - the whole point of the router is that this
    # line changes with the question, not with the config.
    console.print(
        f"[dim]routed {task.value} -> {answer.choice.provider}/{answer.choice.model}[/dim]"
    )
    if len(answer.attempts) > 1:
        console.print(f"[yellow]failover: {' | '.join(answer.attempts)}[/yellow]")
    console.print(f"[bold]Sonara:[/bold] {answer.text}")
    return answer.text or None


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

    # Sentence-level streaming: synthesize the FIRST sentence, start playing it, then
    # synthesize the rest while it plays. Time-to-first-audio must not grow with the
    # length of the answer - synthesizing the whole reply first made a fuller answer
    # cost 1,872ms before a single sample was heard. This is the M1 requirement,
    # previewed here so the measurement reflects what M1 will feel like.
    text = for_speech(text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()] or [text]

    def render(s: str) -> np.ndarray:
        pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(s))
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    with stage("tts_first_sentence"):
        first = render(sentences[0])
    t_all = time.perf_counter()
    sd.play(first, sr)
    for s in sentences[1:]:
        chunk = render(s)  # synthesized while the previous sentence is still playing
        sd.wait()
        sd.play(chunk, sr)
    sd.wait()
    timings["tts_all_sentences"] = (time.perf_counter() - t_all) * 1000
    if len(sentences) > 1:
        console.print(f"[dim]streamed {len(sentences)} sentences[/dim]")


_router = None


def get_router():
    """The router owns provider clients, task classification and failover.

    smoke_test used to hold one hardcoded Groq client. Now the same voice loop
    reaches whichever brain the utterance deserves - "remind me at 6" to a proven
    tool-caller, "compare these GPUs" to a 550B model - without this script
    knowing anything about providers.
    """
    global _router
    if _router is None:
        try:
            from sonara import Router

            _router = Router()
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]router unavailable ({type(e).__name__}: {e})[/red]")
            return None
    return _router


def warm_connection() -> None:
    """Pay the DNS + TLS + HTTP/2 setup cost before the user speaks.

    Measured: turn 1 of a session costs ~1.7-2.1s extra to first token,
    reproducibly, while every later turn is 160-290ms. Without this the very
    first thing a user ever says to Sonara is also the slowest.

    Warms the FAST tier only: it serves chat and command, which is most traffic,
    and warming every provider would spend scarce deep-tier requests on "hi".
    """
    router = get_router()
    if router is None:
        return
    for provider in router.models.get("tiers", {}).get("fast", []):
        if not router.available(provider):
            continue
        try:
            t = time.perf_counter()
            router._client(provider).chat.completions.create(
                model=router.caps["providers"][provider]["pinned_model"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            console.print(
                f"[dim]{provider} connection warmed "
                f"({(time.perf_counter()-t)*1000:,.0f}ms, off the clock)[/dim]"
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Warm-up failed for {provider} ({type(e).__name__}) — "
                          f"first turn will be slow[/yellow]")


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
        warm_connection()  # replay must warm too, or it reports a false OVER
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
    for k, v in timings.items():
        table.add_row(k, "-" if v is None else f"{v:,.0f}")
    console.print(table)

    # THE number: end-of-speech -> first audio. With sentence-level streaming this is
    # stt + llm-first-token + first-sentence synth, and it must NOT grow with the
    # length of the answer.
    st = timings.get("stt_transcribe")
    ft = timings.get("llm_first_token")
    ts = timings.get("tts_first_sentence")
    if st is not None and ft is not None and ts is not None:
        total = (st + ft + ts) / 1000
        # Judged against the tier that actually served it. A 550B reasoning answer
        # is allowed 2.5s; holding it to the chat budget would fail a system doing
        # exactly what it was designed to do.
        tier = last_route["tier"]
        target = float(last_route["target_s"])
        verdict = "PASS" if total <= target else "OVER"
        colour = "green" if total <= target else "red"
        console.print(
            f"\n[bold {colour}]End-of-speech -> first audio: {total:.2f}s  "
            f"[{verdict} vs {target:.1f}s {tier}-tier gate][/bold {colour}]"
        )
        console.print(
            f"[dim]  stt {st:,.0f} + first-token {ft:,.0f} + first-sentence tts {ts:,.0f} ms[/dim]"
        )
        full = timings.get("llm_full_reply")
        allt = timings.get("tts_all_sentences")
        if full is not None and allt is not None:
            console.print(
                f"[dim]  (whole reply: llm {full:,.0f} ms, all speech rendered+played {allt:,.0f} ms — "
                f"these grow with answer length; the number above must not)[/dim]"
            )
    else:
        console.print("\n[yellow]Total not computable: one or more stages skipped (see above).[/yellow]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
