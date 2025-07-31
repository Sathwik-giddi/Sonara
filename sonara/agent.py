"""The agent loop — what turns a command parser into an assistant.

Two things were missing, and together they are most of the gap to "feels intelligent":

  MEMORY      Every utterance was standalone. Ask "what's the weather in Bengaluru?"
              then "what about tomorrow?" and the second question meant nothing, because
              nothing carried over. Conversation history fixes referring back - "it",
              "that one", "do it again" - which is most of how people actually speak.

  MULTI-STEP  It called one tool and stopped. Real requests chain: "if it's raining
              tomorrow, remind me to take an umbrella" needs weather THEN a reminder,
              with the second decision depending on the first answer.

Bounded on purpose. `max_steps` caps the chain so a confused model cannot spin, and
history is trimmed every turn so cost per exchange stays FLAT - the property the whole
$0 thesis rests on (P5). An assistant that gets more expensive the longer you talk to it
is not an assistant you keep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .compress import count_tokens, trim_history
from .router import Choice, NoProviderAvailable, Router
from .tasks import Task, classify
from .tools import ConfirmationRequired, Executor, registry

# Tool results that are Class L (design doc, component 6): note bodies, reminders, file
# paths. These never re-enter a hosted prompt; the agent stops and answers locally.
LOCAL_ONLY = {"search_notes", "list_reminders", "find_file"}

SYSTEM = (
    "You are Sonara, a voice assistant on Windows. Reply in natural spoken language, "
    "brief but complete.\n"
    "You HAVE tools - use them rather than guessing or apologising:\n"
    "- weather -> get_weather; news, prices, anything current -> web_search\n"
    "- facts about a person, place or thing -> look_up\n"
    "- time or date -> get_time; apps, media, volume, files -> the pc tools\n"
    "- reminders and notes -> set_reminder, add_note, search_notes\n"
    "You may use SEVERAL tools in a row when a request needs it: check something, then "
    "act on what you found.\n"
    "SAY THE ACTUAL VALUES a tool returned - numbers, names, times. 'This is the current "
    "weather' tells the user nothing; '27 degrees and drizzling' is the answer.\n"
    "USE WHAT IS ALREADY IN THE CONVERSATION before calling a tool again. If you looked "
    "something up two turns ago, compare against it rather than searching afresh.\n"
    "Never claim you cannot look something up when a tool can. Never invent a fact a "
    "tool could have given you. Speak plainly, no markdown, no lists."
)


@dataclass
class Step:
    tool: str
    args: dict
    ok: bool
    result: Any = None
    error: str | None = None


@dataclass
class TurnResult:
    text: str
    steps: list[Step] = field(default_factory=list)
    choice: Choice | None = None
    first_token_ms: float = 0.0
    awaiting_confirmation: bool = False
    tokens_in: int = 0


class Agent:
    """A conversation with hands. Owns history, the tool loop, and the safety gate."""

    def __init__(self, *, router: Router | None = None, executor: Executor | None = None,
                 system: str = SYSTEM, max_steps: int = 4,
                 max_turns: int = 6, max_history_tokens: int = 1200) -> None:
        self.router = router or Router()
        self.executor = executor or Executor()
        self.system = system
        self.max_steps = max_steps
        self.max_turns = max_turns
        self.max_history_tokens = max_history_tokens
        self.history: list[dict] = []

    # ---------- helpers ----------

    def _messages(self, extra: list[dict] | None = None) -> list[dict]:
        return ([{"role": "system", "content": self.system}] + self.history
                + (extra or []))

    @staticmethod
    def _args_of(call) -> dict:
        try:
            a = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        # Models emit "null" rather than "{}" for no-argument tools.
        return a if isinstance(a, dict) else {}

    def _speak_local(self, name: str, result: Any) -> str:
        """Template a Class L result into speech without a model seeing it."""
        rows = result or []
        if name == "search_notes":
            if not rows:
                return "I couldn't find any notes about that."
            return f"I found {len(rows)}: " + "; ".join(r["line"] for r in rows[:3])
        if name == "list_reminders":
            if not rows:
                return "You have no reminders set."
            return "You have " + ", and ".join(f"{r['text']} at {r['due']}" for r in rows[:3])
        if name == "find_file":
            if not rows:
                return "I couldn't find a file matching that."
            from pathlib import Path
            return f"I found {len(rows)}. The first is {Path(rows[0]).name}"
        return str(result)

    # ---------- the loop ----------

    def turn(self, text: str) -> TurnResult:
        # A pending confirmation owns the next utterance completely. It must never be
        # routed to a model and answered conversationally - that is how "confirm"
        # becomes a chat message and a delete quietly never happens (or worse, does).
        if self.executor.awaiting_confirmation:
            out = self.executor.resolve_confirmation(text)
            said = str(out.result) if out.ok else (out.error or "Cancelled.")
            self.history += [{"role": "user", "content": text},
                             {"role": "assistant", "content": said}]
            return TurnResult(said)

        task = classify(text)
        self.history.append({"role": "user", "content": text})
        scratch: list[dict] = []
        steps: list[Step] = []
        choice = None
        first_ms = 0.0
        tools = registry.openai_schemas()

        for _ in range(self.max_steps):
            msgs = self._messages(scratch)
            try:
                calls, said, choice, first_ms = self.router.chat_with_tools(
                    msgs, tools, task=task,
                )
            except NoProviderAvailable as e:
                said = f"I couldn't reach a model just now. {e}"
                break

            if not calls:
                break  # the model is done and wants to speak

            call = calls[0]
            name, args = call.function.name, self._args_of(call)
            try:
                out = self.executor.execute(name, args)
            except ConfirmationRequired as c:
                self.history.append({"role": "assistant", "content": c.prompt})
                return TurnResult(c.prompt, steps, choice, first_ms,
                                  awaiting_confirmation=True)

            steps.append(Step(name, args, out.ok, out.result, out.error))

            if not out.ok:
                # Hand the failure BACK to the model instead of giving up. Models invent
                # tools that sound plausible - asked to compare two known values it
                # called a non-existent "compare" - and telling it so lets it recover
                # on the next step. Ending the turn on the first error makes the
                # assistant brittle in exactly the moments it should be resourceful.
                scratch += [
                    {"role": "assistant", "content": None,
                     "tool_calls": [{"id": call.id, "type": "function",
                                     "function": {"name": name,
                                                  "arguments": call.function.arguments or "{}"}}]},
                    {"role": "tool", "tool_call_id": call.id,
                     "content": f"ERROR: {out.error}. That tool is not available - "
                                f"answer from what you already know, or use a real tool."},
                ]
                said = f"I couldn't do that. {out.error}"
                continue
            if name in LOCAL_ONLY:
                # Stop here: the answer is Class L and must not re-enter a hosted prompt.
                said = self._speak_local(name, out.result)
                break

            # Feed the result back using the REAL tool-calling protocol: an assistant
            # message carrying the tool_calls, then a `tool` message keyed by
            # tool_call_id. The first version faked this with a plain user message and
            # the model never registered that its call had been answered - it called
            # get_weather four times in a row and hit the step cap. The id is the part
            # that closes the loop.
            scratch += [
                {
                    "role": "assistant",
                    "content": said or None,
                    "tool_calls": [{
                        "id": call.id,
                        "type": "function",
                        "function": {"name": name, "arguments": call.function.arguments or "{}"},
                    }],
                },
                {"role": "tool", "tool_call_id": call.id, "content": str(out.result)[:2000]},
            ]
        else:
            said = "That turned into more steps than I could finish. Ask me a smaller piece?"

        said = (said or "").strip() or "I'm not sure how to answer that."
        self.history.append({"role": "assistant", "content": said})
        # Trim AFTER appending so the reply is kept and the oldest turns fall off.
        # This is what keeps cost per exchange flat instead of creeping upward.
        self.history = trim_history(self.history, max_turns=self.max_turns,
                                    max_tokens=self.max_history_tokens)
        return TurnResult(said, steps, choice, first_ms,
                          tokens_in=sum(count_tokens(m["content"]) for m in self._messages()))

    def reset(self) -> None:
        self.history.clear()
