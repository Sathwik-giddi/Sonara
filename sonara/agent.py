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
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

from .compress import count_tokens, trim_history
from .router import Choice, NoProviderAvailable, Router
from .tasks import Task, classify
from .tools import ConfirmationRequired, Executor, registry

# Tool results that are Class L (design doc, component 6): note bodies, reminders, file
# paths. These never re-enter a hosted prompt; the agent stops and answers locally.
LOCAL_ONLY = {"search_notes", "list_reminders", "find_file"}

def load_persona(name: str | None = None) -> tuple[str, str]:
    """Build the system prompt from config/personas.yaml. Returns (prompt, tts_voice).

    Personality lives in config, not in code, so it can be A/B'd by ear rather than
    argued about in the abstract - which is the only way anyone has ever chosen a voice.
    """
    import os
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/personas.yaml").read_text())
    key = name or os.environ.get("SONARA_PERSONA") or cfg.get("default", "cloud")
    p = cfg["personas"].get(key) or cfg["personas"][cfg["default"]]
    return (f"{p['prompt'].strip()}\n\n{cfg.get('behaviour', '').strip()}\n\n"
            f"{cfg['universal'].strip()}\n\n{TOOLS_BLOCK}",
            p.get("voice", "en_US-lessac-medium"))


# The capability half of the prompt: identical across personas, because what it CAN do
# is not a matter of character.
TOOLS_BLOCK = (
    "You have tools - use them rather than guessing:\n"
    "- weather -> get_weather; news, prices, anything current -> web_search\n"
    "- facts about a person, place or thing -> look_up\n"
    "- time or date -> get_time; apps, media, volume, files -> the pc tools\n"
    "- reminders and notes -> set_reminder, add_note, search_notes\n"
    "You may use several tools in a row when a request needs it: check something, then "
    "act on what you found. If a tool fails, recover - answer from what you know or try "
    "a real tool."
)

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
                 max_turns: int = 6, max_history_tokens: int = 1200,
                 memory: "Memory | None" = None, use_memory: bool = True) -> None:
        from .memory import Memory

        self.router = router or Router()
        self.executor = executor or Executor()
        self.system = system
        self.max_steps = max_steps
        self.max_turns = max_turns
        self.max_history_tokens = max_history_tokens
        self.history: list[dict] = []
        self._memory_context = ""
        self.learned_this_turn: list[str] = []
        self.suggest_skill: str | None = None
        self._steps_this_turn: list[Step] = []
        # Persistent across sessions. Without this every conversation starts as a
        # stranger, which is the largest single gap between an assistant and someone.
        self.memory = memory or (Memory() if use_memory else None)
        self.recalled: list[str] = []

    # ---------- helpers ----------

    def _messages(self, extra: list[dict] | None = None) -> list[dict]:
        system = self.system
        # Only the lines relevant to THIS utterance are injected - never the whole
        # store. Cheaper, and a model ignores one true fact buried in fifty irrelevant
        # ones. Memory text is Class L, so this is the only place it may appear.
        if self._memory_context:
            system = f"{system}\n\n{self._memory_context}"
        return ([{"role": "system", "content": system}] + self.history + (extra or []))

    @staticmethod
    def _calls_in_text(said: str) -> list:
        """Recover tool calls a model wrote as TEXT instead of using the API field.

        Observed live: llama-3.3-70b answered "what time is it and what's your name"
        with the literal string
            {"name": "get_time", "parameters": {}}; {"name": "look_up", ...}
        as its message content. Nothing was in tool_calls, so the agent took that JSON
        to be the answer and SPOKE IT ALOUD.

        This is a known failure mode that worsens with more tools and longer prompts, and
        every serious agent framework carries a parser like this. Recovering the intent
        is strictly better than reading punctuation to someone.
        """
        found = []
        for m in re.finditer(
            r'\{\s*"name"\s*:\s*"([A-Za-z_]\w*)"\s*,\s*'
            r'"(?:parameters|arguments|args)"\s*:\s*(\{.*?\})\s*\}', said or "", re.S,
        ):
            name, raw = m.group(1), m.group(2)
            if registry.get(name) is None:
                continue

            class _Fn:
                pass

            class _Call:
                pass

            c, f = _Call(), _Fn()
            f.name, f.arguments = name, raw
            c.id, c.function, c.type = f"recovered_{len(found)}", f, "function"
            found.append(c)
        return found

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

    # Identity questions have KNOWN answers, so no model should be involved. Four
    # separate prompt fixes failed to keep "your name" and "my name" apart, and the
    # answers flipped between identical runs - a coin flip, not a tuning problem.
    # Deterministic rules cost zero tokens, zero latency, and cannot be wrong. Same
    # principle as the intent shortcut that already handles "pause the music".
    _WHO_ARE_YOU = re.compile(
        r"\b(what(?:'s| is)? your name|who are you|what are you called|"
        r"what should i call you)\b", re.I)
    _WHO_AM_I = re.compile(
        r"\b(what(?:'s| is)? my name|who am i|what do you call me|"
        r"do you (?:know|remember) my name)\b", re.I)

    def _deterministic_answer(self, text: str) -> str | None:
        if self._WHO_ARE_YOU.search(text):
            return "Sonara."
        if self._WHO_AM_I.search(text):
            name = self.memory.name() if self.memory else None
            return f"{name}." if name else "You haven't told me your name yet."
        return None

    def _memory_shortcircuit(self, name: str, args: dict) -> str | None:
        """Refuse to search the world for something we already know about this person.

        THREE prompt fixes failed to stop this. Asked "what is your name" the model
        called look_up("Vaish") and read back a Wikipedia article on the Vaishya caste;
        asked "what is my name" it searched again rather than reading the fact in its
        own context. Prompts are advisory. Code is binding.

        This intercepts encyclopaedia and web lookups whose subject is the assistant or
        the user, and answers from memory instead - deterministically, every time.
        """
        if name not in ("look_up", "web_search"):
            return None
        q = str(args.get("topic") or args.get("query") or "").strip().lower()
        if not q:
            return None

        if "sonara" in q or "your name" in q or q in ("you", "yourself"):
            return "You are Sonara. That is your own name - no lookup required."

        if self.memory:
            user_name = (self.memory.name() or "").lower()
            facts = self.memory.core_facts()
            personal = any(kw in q for kw in ("my name", "i am", "me", "my "))
            if (user_name and user_name in q) or personal:
                known = "; ".join(f.text for f in facts) or "nothing recorded yet"
                return (f"That subject is the person you are talking to, not an "
                        f"encyclopaedia topic. What you know: {known}")
        return None

    def _wrap_up(self, scratch: list[dict]) -> str:
        """Force a spoken answer with NO tools attached.

        Used whenever the loop ends without the model having said anything - the step
        cap, or the repeat guard. Removing the tools is the point: with them available a
        stuck model simply calls another one, and the user hears an internal message
        instead of an answer.
        """
        # State plainly what WAS done. Without this the model confabulated success:
        # asked to book a flight it answered "I've booked a flight to Tokyo for you,
        # the details are in your reminders" having booked nothing at all. Claiming a
        # completed action that never happened is the worst failure this system can
        # have - worse than refusing, worse than being slow.
        done = [s.tool for s in self._steps_this_turn if s.ok]
        # Phrase this as a CONSTRAINT, never as a sentence. The first version read
        # "You successfully used: web_search" and the model repeated it back to the user
        # verbatim - "I successfully used web_search." Anything quotable in an internal
        # note will eventually be quoted.
        ledger = (f"(internal note, never mention tool names to the user: actions that "
                  f"actually succeeded this turn = {done or 'NONE'})")
        try:
            _c, said, _choice, _ms = self.router.chat_with_tools(
                self._messages(scratch + [{
                    "role": "user",
                    "content": f"{ledger} Reply to the person now in one or two spoken "
                               f"sentences, using only what those actions returned. If "
                               f"nothing succeeded, tell them plainly that you cannot do "
                               f"it - never claim an action you did not complete, and "
                               f"never name a tool.",
                }]),
                [], task=Task.CHAT,
            )
            return said
        except NoProviderAvailable:
            return "I can't do that one."

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

        # Answer identity questions without a model: known answer, zero tokens, and it
        # cannot get the pronouns backwards the way four prompt revisions did.
        fixed = self._deterministic_answer(text)
        if fixed is not None:
            self.history += [{"role": "user", "content": text},
                             {"role": "assistant", "content": fixed}]
            return TurnResult(fixed)

        task = classify(text)

        # Recall BEFORE answering, remember AFTER. Both happen without the person ever
        # asking - that automaticity is what makes it feel like being known rather than
        # like operating a database.
        self._memory_context = ""
        self.recalled = []
        if self.memory:
            self._memory_context = self.memory.context_for(text)
            self.recalled = [line[2:] for line in self._memory_context.splitlines()
                             if line.startswith("- ")]
            learned = self.memory.observe(text)
            if learned:
                self.learned_this_turn = learned

        self.history.append({"role": "user", "content": text})
        scratch: list[dict] = []
        seen: set[tuple] = set()
        steps: list[Step] = []
        self._steps_this_turn = steps
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
                # Before believing it wants to speak, check whether it wrote tool calls
                # into the text instead of the API field.
                calls = self._calls_in_text(said)
                if calls:
                    said = ""
                else:
                    break  # genuinely done and wants to speak

            call = calls[0]
            name, args = call.function.name, self._args_of(call)
            recovered = str(getattr(call, "id", "")).startswith("recovered_")

            # HARD REPEAT GUARD. A model that re-requests the same call with the same
            # arguments is stuck, and every extra lap costs a request and a second of
            # the user's life. Seen live: get_time four times in one turn.
            sig = (name, repr(sorted(args.items())))
            if sig in seen:
                said = self._wrap_up(scratch)
                break
            seen.add(sig)
            guard = self._memory_shortcircuit(name, args)
            if guard is not None:
                steps.append(Step(name, args, True, guard))
                scratch.append({"role": "user", "content":
                                f"Do not search for that. {guard} Answer in plain speech now."})
                continue

            try:
                out = self.executor.execute(name, args)
            except ConfirmationRequired as c:
                self.history.append({"role": "assistant", "content": c.prompt})
                return TurnResult(c.prompt, steps, choice, first_ms,
                                  awaiting_confirmation=True)

            steps.append(Step(name, args, out.ok, out.result, out.error))

            # Notice repetition. Three of the same action is the point at which a person
            # would say "can you just do this automatically?" - so Sonara says it first.
            if out.ok and self.memory and name not in LOCAL_ONLY:
                n = self.memory.note_action(name, args)
                if n == 3:
                    self.suggest_skill = (
                        f"You've asked me to do that three times now. "
                        f"Want me to learn it as a shortcut?")

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
            if recovered:
                # The model wrote this call as prose rather than using the API, so
                # replying with a `tool` message keyed to an id it never issued means
                # nothing to it - it just re-emits the same call. Answer in the register
                # it actually used.
                scratch.append({"role": "user",
                                "content": f"Result of {name}: {str(out.result)[:1200]}. "
                                           f"Now answer the question in plain speech."})
            else:
                scratch += [
                    {
                        "role": "assistant",
                        "content": said or None,
                        "tool_calls": [{
                            "id": call.id,
                            "type": "function",
                            "function": {"name": name,
                                         "arguments": call.function.arguments or "{}"},
                        }],
                    },
                    {"role": "tool", "tool_call_id": call.id,
                     "content": str(out.result)[:2000]},
                ]
        else:
            # Steps exhausted. Asked for something it has no tool for, a model will keep
            # trying plausible-sounding tools until the cap - and the user should never
            # see the cap. One final call with NO tools attached forces it to say what
            # it actually can and cannot do, in its own voice.
            try:
                _c, said, choice, first_ms = self.router.chat_with_tools(
                    self._messages(scratch + [{
                        "role": "user",
                        "content": "You have no tool for this. Tell the user plainly what "
                                   "you cannot do, in one sentence, in your own voice.",
                    }]),
                    [], task=Task.CHAT,
                )
            except NoProviderAvailable:
                said = "I can't do that one."

        said = (said or "").strip() or "I'm not sure how to answer that."

        # Volunteer things without being asked. A due reminder that waits for you to ask
        # "any reminders?" is not a reminder.
        extras = [f"By the way, {d}." for d in self._due_reminders()]
        if self.suggest_skill:
            extras.append(self.suggest_skill)
            self.suggest_skill = None
        if extras:
            said = said + " " + " ".join(extras)

        self.history.append({"role": "assistant", "content": said})
        # Trim AFTER appending so the reply is kept and the oldest turns fall off.
        # This is what keeps cost per exchange flat instead of creeping upward.
        self.history = trim_history(self.history, max_turns=self.max_turns,
                                    max_tokens=self.max_history_tokens)
        return TurnResult(said, steps, choice, first_ms,
                          tokens_in=sum(count_tokens(m["content"]) for m in self._messages()))

    # ---------- proactivity ----------

    def greeting(self) -> str | None:
        """What to say when a session opens, before being spoken to.

        This is the moment continuity becomes audible. An assistant that opens with
        "How can I help you today?" has told you it does not know who you are. One that
        opens with what you were doing last time has.

        Costs nothing: no model call, just memory and the reminder table.
        """
        if not self.memory:
            return None
        bits: list[str] = []
        name = self.memory.name()
        last = self.memory.last_episode()

        if last:
            when, _, what = last.text.partition(": ")
            today = datetime.now().strftime("%Y-%m-%d")
            lead = "Earlier" if when == today else "Last time"
            bits.append(f"{lead} you were {what.strip()}.")
        elif name:
            bits.append(f"Hello {name}.")

        for due in self._due_reminders():
            bits.append(f"Reminder: {due}.")

        return " ".join(bits) or None

    def _due_reminders(self) -> list[str]:
        """Reminders that have come due. Checked every turn AND at session open.

        Real proactivity does not need a background process to start being useful: the
        next time the person speaks is a perfectly good moment to say "that reminder you
        set has come due". The tray (M4) makes it timely; this makes it exist.
        """
        try:
            out = self.executor.execute("due_reminders", {})
            return list(out.result or []) if out.ok else []
        except Exception:  # noqa: BLE001 - never let a reminder check break a turn
            return []

    def close(self, summary: str | None = None) -> None:
        """Store what this session was about, so the next one can open knowing.

        Summarised from the conversation without a model call: the first thing the
        person actually asked for is a better session title than anything generated,
        and it costs nothing.
        """
        if not self.memory:
            return
        if summary is None:
            firsts = [m["content"] for m in self.history if m["role"] == "user"]
            if not firsts:
                return
            # QUOTE it rather than paraphrase. Stripping the question word produced
            # "asking about time is it" from "what time is it" - grammar that no
            # rewriting rule survives for long. A quote is always well-formed.
            summary = 'asking "' + firsts[0].strip().rstrip("?.")[:80] + '"'
        self.memory.add_episode(summary)

    def reset(self) -> None:
        self.history.clear()
