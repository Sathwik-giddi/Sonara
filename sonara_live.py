"""Sonara, resident. The always-on assistant rather than a test script.

Everything intelligent already existed - routing, tools, memory, personality,
proactivity - but it lived in scripts you ran once. That is a demo, not an assistant.
Sonara is not smarter than what we built; it is PRESENT. It listens without being
launched, answers without a key press, and is still there an hour later.

This is that: one process, running until you stop it.

  - listens continuously and decides for itself when you started and stopped talking
  - never listens while it is speaking, so it cannot interrupt itself
  - remembers across restarts, and opens by telling you where you left off
  - volunteers reminders when they come due
  - notices repeated actions and offers to learn them

Run:  uv run sonara_live.py
      uv run sonara_live.py --persona apprentice
      uv run sonara_live.py --push-to-talk        (if the room is noisy)
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

console = Console()

SR = 16_000
FRAME_MS = 30                      # webrtcvad accepts 10, 20 or 30ms frames
FRAME = SR * FRAME_MS // 1000
START_FRAMES = 3                   # ~90ms of speech before we believe you started
SILENCE_FRAMES = 25                # ~750ms of quiet before we believe you stopped
MAX_UTTERANCE_S = 15
PREROLL_FRAMES = 10                # keep 300ms BEFORE detection, or the first word is lost


class Ear:
    """Continuous listening with voice activity detection.

    The preroll buffer matters more than it looks: VAD only fires once speech is already
    underway, so without keeping the frames from just before the trigger you clip the
    leading phoneme - which is exactly the "what's the" -> "watch the" failure measured
    earlier in this project.
    """

    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad

        self.vad = webrtcvad.Vad(aggressiveness)
        self.q: queue.Queue[bytes] = queue.Queue()
        self.muted = threading.Event()
        self.stream = sd.RawInputStream(
            samplerate=SR, blocksize=FRAME, dtype="int16", channels=1,
            callback=self._cb,
        )

    def _cb(self, indata, frames, t, status) -> None:  # noqa: ANN001
        if not self.muted.is_set():
            self.q.put(bytes(indata))

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()

    def mute(self, on: bool) -> None:
        """Deaf while speaking. Half-duplex is a deliberate choice: measured AEC gives
        22-33 dB, enough below ~50% volume but not above it, and an assistant that
        interrupts itself is worse than one you cannot interrupt."""
        if on:
            self.muted.set()
            with self.q.mutex:
                self.q.queue.clear()
        else:
            self.muted.clear()

    def listen(self) -> np.ndarray | None:
        """Block until a complete utterance has been spoken, then return it."""
        preroll: list[bytes] = []
        voiced: list[bytes] = []
        speaking = False
        run_start = run_silence = 0
        t0 = time.time()

        while True:
            try:
                frame = self.q.get(timeout=0.5)
            except queue.Empty:
                if speaking:
                    break
                continue

            is_speech = self.vad.is_speech(frame, SR)

            if not speaking:
                preroll.append(frame)
                if len(preroll) > PREROLL_FRAMES:
                    preroll.pop(0)
                run_start = run_start + 1 if is_speech else 0
                if run_start >= START_FRAMES:
                    speaking = True
                    voiced = preroll[:] + [frame]
                    t0 = time.time()
                    console.print("[bold cyan]  listening...[/bold cyan]", end="\r")
                continue

            voiced.append(frame)
            run_silence = 0 if is_speech else run_silence + 1
            if run_silence >= SILENCE_FRAMES or (time.time() - t0) > MAX_UTTERANCE_S:
                break

        if not voiced:
            return None
        pcm = b"".join(voiced)
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class Mouth:
    """Piper, streamed sentence by sentence, with the mic deaf throughout."""

    def __init__(self, ear: Ear, voice: str = "en_US-lessac-medium") -> None:
        from piper import PiperVoice

        self.voice = PiperVoice.load(str(ROOT / "models" / f"{voice}.onnx"))
        self.sr = self.voice.config.sample_rate
        self.ear = ear

    def say(self, text: str) -> None:
        import re

        import smoke_test as st

        text = st.for_speech(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()] or [text]
        self.ear.mute(True)
        try:
            for s in sentences:
                pcm = b"".join(c.audio_int16_bytes for c in self.voice.synthesize(s))
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                sd.play(audio, self.sr)
                sd.wait()
        finally:
            time.sleep(0.25)          # let the room settle before trusting the mic again
            self.ear.mute(False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default=None)
    ap.add_argument("--push-to-talk", action="store_true",
                    help="press Enter before each turn instead of listening continuously")
    ap.add_argument("--aggressiveness", type=int, default=2,
                    help="VAD strictness 0-3; raise it in a noisy room")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)

    import smoke_test as st
    from sonara import Agent
    from sonara.agent import load_persona

    console.print("[dim]waking up...[/dim]")
    prompt, voice = load_persona(args.persona)
    agent = Agent(system=prompt)

    ear = Ear(args.aggressiveness)
    mouth = Mouth(ear, voice)
    st.transcribe(np.zeros(SR, dtype=np.float32))     # load STT before the first word
    ear.start()

    console.print(f"\n[bold]Sonara[/bold] is listening. "
                  f"[dim]persona={args.persona or 'default'} · Ctrl-C to stop[/dim]\n")

    hello = agent.greeting() or "I'm here."
    console.print(f"[bold green]sonara:[/bold green] {hello}")
    mouth.say(hello)

    try:
        while True:
            if args.push_to_talk:
                ear.mute(True)
                console.input("[dim]press Enter to speak...[/dim]")
                ear.mute(False)

            audio = ear.listen()
            if audio is None or len(audio) < SR // 3:
                continue

            t0 = time.perf_counter()
            heard = st.transcribe(audio)
            if not heard or len(heard.split()) < 2:
                continue
            console.print(f"[bold]you:[/bold]    {heard}")

            result = agent.turn(heard)
            took = time.perf_counter() - t0
            console.print(f"[bold green]sonara:[/bold green] {result.text}"
                          f"   [dim]{took:.2f}s"
                          + (f" · {', '.join(s.tool for s in result.steps)}" if result.steps else "")
                          + "[/dim]")
            mouth.say(result.text)

    except KeyboardInterrupt:
        console.print("\n[dim]saving...[/dim]")
        agent.close()
        mouth.say("Goodbye.")
    finally:
        ear.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
