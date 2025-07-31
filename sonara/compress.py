"""Prompt compression — send fewer tokens without losing the meaning.

HONEST SCOPE, up front. Published compression results (LLMLingua: 20x, one production
case $42k/month -> $2.1k) are measured on LONG prompts: documents, RAG context, chat
histories. A spoken utterance is about 6 tokens. Compressing "pause the music" saves
nothing and risks breaking it.

So this compresses where the tokens actually are, and deliberately does nothing where
they are not:

  utterance      ~6 tokens      -> untouched. The cheapest token is the one never sent,
                                   and the intent shortcut already sends zero for these.
  system prompt  ~60 tokens     -> sent every single turn, so it is the largest fixed
                                   cost in the whole system. Trimmed once, at import.
  history        grows forever  -> the real enemy. Unbounded history is what turns a
                                   cheap assistant into an expensive one by Thursday.
  pasted text    100s-1000s     -> where real compression belongs (summarise task).

The measurement to care about is tokens-per-exchange over a real session, not
compression ratio on a benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ~4 characters per token for English. Good enough to route on and to budget with;
# exact tokenisation differs per model and is not worth a dependency here.
CHARS_PER_TOKEN = 4

# Below this, compression cannot win: the risk of mangling meaning exceeds the saving.
MIN_COMPRESS_TOKENS = 120

_FILLER = re.compile(
    r"\b(um|uh|erm|like|you know|i mean|sort of|kind of|basically|actually|"
    r"literally|just|really|very|quite|simply|obviously|essentially)\b",
    re.I,
)
_WS = re.compile(r"\s+")
_REPEAT = re.compile(r"\b(\w+)(\s+\1\b)+", re.I)


def count_tokens(text: str) -> int:
    return max(0, len(text or "") // CHARS_PER_TOKEN)


@dataclass
class Compressed:
    text: str
    tokens_before: int
    tokens_after: int

    @property
    def saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def ratio(self) -> float:
        return self.tokens_before / max(self.tokens_after, 1)


def compress_text(text: str, *, aggressive: bool = False) -> Compressed:
    """Lossy-but-safe shortening for LONG inputs only.

    Short inputs are returned untouched: below MIN_COMPRESS_TOKENS the saving is a
    rounding error and the risk of changing what the user meant is not.
    """
    before = count_tokens(text)
    if before < MIN_COMPRESS_TOKENS:
        return Compressed(text, before, before)

    out = _WS.sub(" ", text).strip()
    out = _REPEAT.sub(r"\1", out)                    # "the the" -> "the"
    out = _FILLER.sub("", out)                        # spoken filler carries no meaning
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = _WS.sub(" ", out).strip()

    if aggressive:
        # Keep the head and tail, drop the middle. Models attend most strongly to the
        # start and end of a long context, so this is where a blunt cut does least harm.
        toks = out.split()
        if len(toks) > 400:
            out = " ".join(toks[:200] + ["[...]"] + toks[-200:])

    return Compressed(out, before, count_tokens(out))


def trim_history(history: list[dict], *, max_turns: int = 6,
                 max_tokens: int = 1200) -> list[dict]:
    """Bound the conversation so cost per exchange stays flat instead of creeping.

    This is the highest-value function in the file. Routing and compression save a
    percentage; unbounded history makes EVERY future turn more expensive than the last,
    so an assistant that is cheap on Monday is expensive by Thursday. Keeping the most
    recent turns is both the cheapest and the most useful window - the last thing said
    matters more than the first.
    """
    if not history:
        return []
    kept = history[-max_turns * 2:]
    while kept and sum(count_tokens(m.get("content", "")) for m in kept) > max_tokens:
        kept = kept[2:] if len(kept) > 2 else kept[1:]
    return kept


def compress_messages(messages: list[dict], *, aggressive: bool = False
                      ) -> tuple[list[dict], Compressed]:
    """Compress a full message list, reporting what it actually saved."""
    before = sum(count_tokens(m.get("content", "")) for m in messages)
    out = []
    for m in messages:
        content = m.get("content", "")
        # Never touch the system prompt at runtime: it carries the safety and honesty
        # instructions, and quietly shortening those is how an assistant starts
        # inventing weather forecasts again.
        if m.get("role") == "system" or count_tokens(content) < MIN_COMPRESS_TOKENS:
            out.append(m)
            continue
        out.append({**m, "content": compress_text(content, aggressive=aggressive).text})
    after = sum(count_tokens(m.get("content", "")) for m in out)
    return out, Compressed("", before, after)
