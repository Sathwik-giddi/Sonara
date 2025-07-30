"""Task classification — decides what KIND of thinking an utterance needs.

Deterministic on purpose. Premise 3 says latency is life-or-death, so the hot
path must not spend an LLM call deciding which LLM to call. These are regex and
keyword rules: microseconds, no tokens, no quota burned.

The design doc calls this the `depth` hint: a skill or intent declares what it
needs, and the router honours it even when a cheaper tier has headroom. That is
what makes model switching *task-aware* rather than purely quota-driven.
"""

from __future__ import annotations

import re
from enum import Enum


class Task(str, Enum):
    """What kind of intelligence the utterance actually needs."""

    COMMAND = "command"      # "open spotify", "remind me at 6" - needs tool calling
    CHAT = "chat"            # greetings, banter, quick facts - needs speed
    REASON = "reason"        # planning, comparison, multi-step - needs depth
    CODE = "code"            # write/explain/debug code - needs a coder model
    SUMMARIZE = "summarize"  # long input in, short output - needs context, not genius


# Ordered: the first pattern that matches wins, so put the narrow ones first.
# Every rule here earns its place by being cheap and unambiguous. Anything
# genuinely ambiguous should fall through to CHAT (fast) rather than guess DEEP
# and spend a scarce deep-tier request on "hello".
_RULES: list[tuple[Task, re.Pattern[str]]] = [
    (Task.CODE, re.compile(
        r"\b(code|function|bug|stack ?trace|regex|refactor|compile|"
        r"python|javascript|typescript|rust|sql|git|api|repo)\b", re.I)),
    (Task.COMMAND, re.compile(
        r"^\s*(open|close|launch|start|stop|play|pause|resume|skip|next|"
        r"mute|unmute|volume|turn (up|down|off|on)|set|remind|remember|"
        r"note|add|delete|create|screenshot|search for|find)\b", re.I)),
    (Task.SUMMARIZE, re.compile(
        r"\b(summari[sz]e|tl;?dr|recap|digest|key points|gist of)\b", re.I)),
    (Task.REASON, re.compile(
        r"\b(why|how (do|does|would|should|can)|plan|strategy|compare|"
        r"trade-?offs?|pros and cons|analy[sz]e|explain|decide|"
        r"figure out|work out|step by step|plan out)\b", re.I)),
]

# Long input is summarization regardless of phrasing - a 400-word paste is not chat.
_LONG_INPUT_WORDS = 120


def classify(text: str) -> Task:
    """Map an utterance to the kind of model it deserves.

    Cheap, deterministic, and biased toward CHAT when unsure: an unnecessary
    deep-tier call wastes a scarce free-tier request, while an unnecessary
    fast-tier call just answers slightly less well.
    """
    if not text or not text.strip():
        return Task.CHAT

    if len(text.split()) >= _LONG_INPUT_WORDS:
        return Task.SUMMARIZE

    for task, pattern in _RULES:
        if pattern.search(text):
            return task

    return Task.CHAT


def wants_tools(task: Task) -> bool:
    """Whether this task will attach tool definitions to the request.

    Matters for routing: a model that is merely fast is useless here if it emits
    malformed tool calls, which is exactly the failure the design doc's fallback
    ladder exists for.
    """
    return task is Task.COMMAND
