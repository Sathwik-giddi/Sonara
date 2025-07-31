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

# Explicit separator between the conversational half of the prompt and the tool half.
# The first version split on the literal string "You have tools"; rewording TOOLS_BLOCK
# silently broke the split and leaked tool vocabulary straight back into conversation
# mode - the exact bug the split existed to prevent.
TOOL_MARKER = "<<<TOOLS>>>"

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


# CONVERSATION FIRST. The previous version commanded "use tools rather than guessing"
# and attached all 20 schemas to every single turn. The result was an assistant that
# never tried to understand anything: "yes, I was asking about that" fetched the
# weather; "you are hallucinating too much" filed a note; a remark about self-directed
# learning became a web search for study tips.
#
# It is a 70B model. It can hold a conversation and it knows who Alan Turing was. The
# scaffolding was making it stupider than it is - exactly the failure Cherny describes,
# and measurable here: the minimal-prompt run scored 92.5% on tool calling against
# 97.5%, and I took the five points at the cost of comprehension.
# This block is attached ONLY when the gate decided the person wants something done.
# It must therefore be ACTION-POSITIVE. The previous wording ("most turns need none")
# was written for every turn and made the assistant tool-shy: measured on a 100-utterance
# corpus, action recall fell to 25% - ask it to do something and it worked one time in
# four. Conversation mode is handled by a separate prompt that never mentions tools, so
# this one no longer has to hedge.
TOOLS_BLOCK = (
    "The person wants something DONE, or wants a fact that changes. Call the right tool - "
    "do not describe it, do not ask permission, just call it.\n"
    "- open / play / pause / skip / volume / screenshot / find a file -> the pc tools\n"
    "- remind, note, remember, 'don't let me forget', 'nudge me' -> set_reminder, add_note\n"
    "- what's coming up, my reminders, my notes -> list_reminders, search_notes\n"
    "- weather, temperature, raining, 'need a jacket', 'what's it like outside' "
    "-> get_weather\n"
    "- the time, the date, what day it is -> get_time\n"
    "- news, prices, anything that happened recently -> web_search\n"
    "- do it later / check later / tell me when -> check_later, watch_for\n"
    "If the request is phrased unusually, map it to the closest tool anyway: 'crank it up' "
    "is volume, 'scribble that down' is a note, 'dig up my resume' is a file search.\n"
    "Only skip the tool if it is genuinely something you already know - history, science, "
    "definitions - or if they are simply talking to you.\n"
    "Say the values a tool returns. Never describe the shape of a result, never add a "
    "caveat, never name the tool."
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
        # Two prompts from one string, split on an explicit marker.
        self.system = system.replace(TOOL_MARKER + "\n", "").replace(TOOL_MARKER, "")
        self.system_conversation = system.split(TOOL_MARKER)[0].rstrip() + (
            "\n\nThis is a conversation. Reply in your own words, in plain speech. "
            "Never output JSON, function names, or anything resembling a tool call."
        )
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

    def _messages(self, extra: list[dict] | None = None, *, tools: bool = True) -> list[dict]:
        # TWO PROMPTS, not one. Handing the model a description of its tools teaches it
        # to answer IN TOOL SYNTAX - with zero schemas attached it still replied
        # {"name": "understand", "parameters": {...}}, inventing functions that do not
        # exist. The tool vocabulary has to leave the prompt entirely for a conversation,
        # not just leave the request.
        system = self.system if tools else self.system_conversation
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

    # Attaching tools is itself a nudge: a model handed 20 schemas tends to use one.
    # So they are only offered when the utterance plausibly wants an ACTION or a fact
    # that genuinely changes. Everything else is a conversation, and a conversation
    # with no tools on the table cannot be answered with a weather report.
    # Broadened from measurement, not intuition. The first version was a list of words
    # ONE user happened to say, and the corpus caught what it missed: "put on some
    # music", "crank it up", "scribble that down", "dig up my resume", "nudge me at
    # half six", "what have I got coming up", "is it raining out". A keyword list will
    # never be complete - but paired with a conversation prompt that has no tool
    # vocabulary at all, its job is only to be GENEROUS about actions while keeping
    # pure conversation out. Measured intrusive rate: 1 in 100 versus 22 when always on.
    _WANTS_TOOL = re.compile(
        r"\b(open|close|launch|start|fire up|boot|run|play|put on|pause|stop|resume|"
        r"skip|next|previous|back a track|mute|unmute|silence|volume|loud|quiet|turn "
        r"(it|the)|crank|screenshot|capture|picture of (my|the) screen|"
        r"remind|reminder|remember|nudge|ping|wake me|note|jot|scribble|write .{0,12}down|"
        r"don'?t (let me )?forget|set (something|a|an)|"
        r"delete|remove|search|look up|google|find|locate|dig up|hunt|where (is|did)|"
        r"check|watch for|tell me when|later|"
        r"weather|temperat|rain|snow|sunny|cold|hot|jacket|umbrella|outside|forecast|"
        r"news|headline|happen(ed|ing)|price|stock|bitcoin|"
        r"time|clock|date|what day|today|tonight|tomorrow|"
        # NO trailing \b. Stems must match inflections - with it, "rain" failed to match
        # "raining" and "happen" failed "happening", so "is it raining out" got no tools
        # at all. Every stem in this list was silently broken by one character.
        r"coming up|my (notes|reminders|jobs|schedule)|working on)", re.I)

    def _tools_for(self, text: str) -> list[dict]:
        """Decide whether this utterance wants an action.

        The regex below is a FAST PATH, not the decision. A keyword list written by
        watching one person is guaranteed to miss "put on some music", "crank it up",
        "dig up my resume" - measured, it did - and no amount of adding words fixes the
        next hundred users' phrasings.

        So anything the regex does not obviously catch goes to a MODEL. The local
        3B answers in ~700ms and costs nothing, which is exactly what the free local
        tier was built for. Generalising to phrasings nobody wrote down is what models
        are for; hand-maintaining a vocabulary is what they replace.
        """
        if self._WANTS_TOOL.search(text or "") or self._classify_wants_action(text):
            return registry.openai_schemas(self._packs_for(text))
        return []

    # Sending all 20 schemas costs 1,699 tokens on EVERY action turn - the single
    # largest line in the budget, and most of it describes tools that could not
    # possibly apply. "What's the time" does not need the notes pack. Narrowing to the
    # plausible packs cuts that by roughly three quarters; when nothing matches
    # confidently we send everything, so a narrowing miss costs tokens, never capability.
    _PACK_HINTS: list[tuple[str, str]] = [
        ("pc_control", r"open|launch|close|fire up|run|play|put on|pause|stop|skip|next|"
                       r"previous|track|mute|volume|loud|quiet|crank|turn (it|the)|"
                       r"screenshot|capture|screen|file|folder|resume|find|locate|dig up|"
                       r"hunt|where (is|did)|time|clock|date|what day|today|tonight"),
        ("notes", r"remind|reminder|nudge|ping|wake me|note|jot|scribble|write|"
                  r"don'?t (let me )?forget|remember|coming up|my (notes|reminders)|"
                  r"schedule|later|tomorrow|tonight|at \d|in \d"),
        ("web", r"weather|temperat|rain|snow|sunny|cold|hot|jacket|umbrella|outside|"
                r"forecast|news|headline|happen|price|stock|bitcoin|search|google|"
                r"look up|who (is|was)|what is|tell me about"),
        ("autonomy", r"later|check .*(at|in) |tell me when|watch|keep an eye|"
                     r"what are you (working|doing)|my jobs|cancel"),
    ]

    def _packs_for(self, text: str) -> list[str] | None:
        """Narrowing is OFF by default, and the measurement says why.

        Sending one pack instead of twenty saves 1,159 tokens per action turn
        (1,699 -> ~500). It also cost 7.5 points of action recall on the corpus
        (47.5% -> 40.0%), because when the pack guess is wrong the right tool is not
        merely deprioritised, it is ABSENT.

        Recall is the weaker number right now, so capability wins. Turn it on with
        SONARA_NARROW_PACKS=1 once recall is comfortably high and tokens are the
        binding constraint - that is a real future trade, just not today's.
        """
        import os

        if os.environ.get("SONARA_NARROW_PACKS") != "1":
            return None
        hits = [p for p, pat in self._PACK_HINTS if re.search(pat, text or "", re.I)]
        return hits or None      # None == every pack

    _CLASSIFIER_SYSTEM = (
        "Decide if the user wants the assistant to DO something or to look up "
        "something that changes (weather, time, news, prices), or whether they are "
        "just talking.\n"
        "Answer with exactly one word: ACTION or TALK.\n"
        "ACTION: open an app, play or pause music, change volume, take a screenshot, "
        "find a file, set a reminder, save or read a note, check the weather or time or "
        "news.\n"
        "TALK: questions you can answer from knowledge, opinions, feelings, criticism, "
        "small talk, corrections, or anything about the assistant itself."
    )

    def _classify_wants_action(self, text: str) -> bool:
        for provider in self.router.models.get("tiers", {}).get("local", []):
            if not self.router.available(provider):
                continue
            try:
                model = self.router.caps["providers"][provider]["pinned_model"]
                r = self.router._client(provider).chat.completions.create(
                    model=model, max_tokens=4,
                    messages=[{"role": "system", "content": self._CLASSIFIER_SYSTEM},
                              {"role": "user", "content": text}],
                )
                return "ACTION" in (r.choices[0].message.content or "").upper()
            except Exception:  # noqa: BLE001 - a classifier outage must not block a turn
                return False
        return False

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

    _JSONISH = re.compile(r'^\s*[\{\[].*"(?:name|function|parameters|arguments)"', re.S)

    def _never_speak_json(self, said: str, user_text: str) -> str:
        """Last line of defence: a person must never hear punctuation read aloud.

        Prompting got this from "always" down to a single case - a contentless fragment
        like "yes, I was asking about that" with no history to attach it to, where the
        model has nothing to say and falls back on structure. No amount of further
        prompt-wrestling makes that guarantee; a check does.

        One retry with an explicit instruction, then a plain human fallback.
        """
        said = (said or "").strip()
        if said and not self._JSONISH.match(said):
            return said

        try:
            _c, retry, _ch, _ms = self.router.chat_with_tools(
                [{"role": "system", "content": self.system_conversation},
                 {"role": "user", "content": user_text},
                 {"role": "user", "content": "Reply in one short spoken sentence. Plain "
                                             "words only - no JSON, no braces, no function "
                                             "names. If they were vague, ask what they mean."}],
                [], task=Task.CHAT,
            )
            retry = (retry or "").strip()
            if retry and not self._JSONISH.match(retry):
                return retry
        except NoProviderAvailable:
            pass
        return "Sorry, say that again?"

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
        tools = self._tools_for(text)

        for _ in range(self.max_steps):
            msgs = self._messages(scratch, tools=bool(tools))
            try:
                # Force the call only on the FIRST step: the gate said this is an
                # action, so something should happen. Later steps stay free, or the
                # model can never stop and answer.
                calls, said, choice, first_ms = self.router.chat_with_tools(
                    msgs, tools, task=task,
                    force_tool=bool(tools) and not steps,
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

        said = self._never_speak_json(said, text)

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
