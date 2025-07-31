"""Does Sonara generalise, or is it tuned to one person's phrasing?

Every heuristic in this project so far was written after watching ONE user's utterances
fail: the _WANTS_TOOL keyword list, the "jot down"/"write this down" hints bolted onto
tool descriptions, the identity patterns. That is fitting to n=1. With a hundred users
saying "can you put on some music" or "stick a note somewhere", a hand-written keyword
list is guaranteed to miss.

So this stops guessing. 120 utterances across many phrasings and moods, each labelled
with what SHOULD happen, scored as a confusion matrix:

  needs a tool, used one         correct action
  needs a tool, just chatted     MISSED - the thing never got done
  no tool needed, chatted        correct conversation
  no tool needed, used a tool    INTRUSIVE - this is the failure that made
                                 "you are hallucinating too much" file a note

Then it compares gating strategies head to head, so the keyword list either earns its
place on evidence or gets deleted:

  gate    only attach tools when a hand-written regex matches   (current)
  always  attach all tools every turn, trust the prompt
  never   never attach tools                                    (control)

Run:  uv run scripts/eval_broad.py --strategy gate always
      uv run scripts/eval_broad.py --limit 40        # quick pass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sonara import Agent  # noqa: E402
from sonara.agent import load_persona  # noqa: E402
from sonara.tools import registry  # noqa: E402

console = Console()

# (utterance, needs_tool). Deliberately NOT the phrasings used while building this -
# these are other ways real people ask for the same things, plus the conversational
# traffic that a tool-happy assistant ruins.
CORPUS: list[tuple[str, bool]] = [
    # --- actions, phrased many different ways -----------------------------
    ("put on some music", True), ("fire up spotify", True), ("launch chrome", True),
    ("can you open my email", True), ("stick the calculator on screen", True),
    ("shut the music off", True), ("kill the audio", True), ("hit pause", True),
    ("next song please", True), ("go back a track", True),
    ("crank it up", True), ("turn it down, it's loud", True), ("silence everything", True),
    ("grab a picture of my screen", True), ("capture the screen", True),
    ("where did I put my tax file", True), ("dig up that invoice", True),
    ("hunt down my resume", True),
    # --- memory / notes, many phrasings -----------------------------------
    ("stick a note somewhere that the router works", True),
    ("scribble down buy milk", True), ("keep a note about the meeting", True),
    ("don't let me forget to email priya", True),
    ("nudge me at half six", True), ("ping me in twenty minutes", True),
    ("wake me in an hour", True), ("set something for 9pm", True),
    ("what have I got coming up", True), ("read me my reminders", True),
    ("did I write anything about the router", True),
    # --- volatile facts ----------------------------------------------------
    ("is it raining out", True), ("how cold is it", True),
    ("do I need a jacket today", True), ("what's it like outside", True),
    ("what's the time", True), ("what day is it", True),
    ("anything big happen in tech today", True), ("what's in the news", True),
    ("how's bitcoin doing", True),
    # --- knowledge it already has: a tool here is SLOWER AND WORSE ---------
    ("who was Ada Lovelace", False), ("what is a black hole", False),
    ("explain how a transistor works", False), ("why is the sky blue", False),
    ("what does GDP mean", False), ("how does a fridge work", False),
    ("what's the difference between RAM and storage", False),
    ("tell me about the Roman empire", False),
    ("what language is spoken in Brazil", False),
    ("how many continents are there", False),
    ("what is photosynthesis", False), ("who wrote Hamlet", False),
    # --- conversation: the traffic that a tool-happy assistant ruins -------
    ("yes, I was asking about that", False), ("no I meant the other one", False),
    ("hmm, not quite", False), ("that's not what I said", False),
    ("you are hallucinating too much", False), ("you're not very good at this", False),
    ("that was actually helpful, thanks", False), ("you sound like a robot", False),
    ("I want you to be smarter", False), ("can you be less formal", False),
    ("do you actually understand me", False), ("are you learning from this", False),
    ("what do you think about all this", False), ("do you get bored", False),
    ("I'm frustrated with you right now", False), ("okay let's start over", False),
    ("forget what I just said", False), ("never mind", False),
    ("I'm tired today", False), ("this has been a long week", False),
    ("what should I do with my life", False), ("do you have opinions", False),
    ("tell me a joke", False), ("say something interesting", False),
    ("I disagree with that", False), ("go on", False),
    ("wait, back up a second", False), ("that makes sense", False),
    ("how do you work internally", False), ("are you always listening", False),
    ("who made you", False), ("do you like being an assistant", False),
    ("I've been thinking about building something", False),
    ("my week has been rough honestly", False),
    ("what's your favourite colour", False),
    ("can we talk about something else", False),
    ("that's a weird answer", False), ("you misunderstood me", False),
    ("let me rephrase", False), ("I don't think that's right", False),
    ("sorry, ignore that", False), ("carry on", False),
    ("do you remember what we discussed", False),
    ("you're doing better now", False),
    ("I like talking to you", False), ("this is quite fun", False),
    # --- STT damage: real mishearings from this project's own recordings ---
    ("what the airiness", False), ("a worst weather today", True),
    ("watch the weather today", True), ("I said ai not a so", False),
]


def score(strategy: str, limit: int | None) -> dict:
    prompt, _ = load_persona("cloud")
    agent = Agent(system=prompt, use_memory=False)

    if strategy == "always":
        agent._tools_for = lambda _t: registry.openai_schemas()
    elif strategy == "never":
        agent._tools_for = lambda _t: []

    rows = CORPUS[:limit] if limit else CORPUS
    tp = fp = tn = fn = 0
    mistakes: list[str] = []

    for utterance, needs in rows:
        agent.history.clear()
        try:
            r = agent.turn(utterance)
        except Exception:  # noqa: BLE001
            continue
        used = bool(r.steps)
        if needs and used:
            tp += 1
        elif needs and not used:
            fn += 1
            mistakes.append(f"MISSED    {utterance!r}")
        elif not needs and not used:
            tn += 1
        else:
            fp += 1
            mistakes.append(f"INTRUSIVE {utterance!r} -> {r.steps[0].tool}")

    total = tp + fp + tn + fn
    return {
        "strategy": strategy, "n": total,
        "accuracy": 100.0 * (tp + tn) / max(total, 1),
        "recall": 100.0 * tp / max(tp + fn, 1),      # actions that actually happened
        "intrusive": fp, "missed": fn, "mistakes": mistakes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", nargs="+", default=["gate", "always"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="list every mistake")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)

    console.print(f"[bold]Generalisation eval[/bold] — {len(CORPUS)} utterances, "
                  f"phrasings deliberately unlike the ones used while building.\n")

    results = [score(s, args.limit) for s in args.strategy]

    t = Table()
    for c in ("strategy", "n", "accuracy", "action recall", "intrusive", "missed"):
        t.add_column(c, justify="right" if c != "strategy" else "left")
    for r in results:
        t.add_row(r["strategy"], str(r["n"]), f"{r['accuracy']:.1f}%",
                  f"{r['recall']:.1f}%", str(r["intrusive"]), str(r["missed"]))
    console.print(t)

    for r in results:
        if args.show and r["mistakes"]:
            console.print(f"\n[bold]{r['strategy']}[/bold]")
            for m in r["mistakes"][:25]:
                console.print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
