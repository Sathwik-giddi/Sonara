"""The router — picks which brain answers, then survives it saying no.

Two jobs, in this order:

  1. TASK-AWARE. What kind of thinking does this need? A reminder needs reliable
     tool calling; "plan my week" needs depth; "hey" needs speed. The route table
     in config/models.yaml encodes that, and a `deep` task is sent to the deep
     tier even when the fast tier has headroom - otherwise "switches models based
     on the task" is just a quota fallback wearing a costume.

  2. QUOTA-AWARE. Every candidate is checked against the ledger, and a live
     429/402 marks that provider+model exhausted immediately and moves on. With
     $0-forever as a hard constraint, running dry is a designed mode, not an error.

Every provider here is OpenAI-compatible, which is why one client shape reaches
Groq, NVIDIA NIM, OpenRouter, Mistral and local Ollama alike.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv

from .ledger import Ledger
from .tasks import Task, classify

ROOT = Path(__file__).resolve().parents[1]

# Which .env var holds each provider's key. Keys live in .env (gitignored) or
# Windows Credential Manager - never in config, never in the repo.
KEY_VARS = {
    "groq": "GROQ_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_KEY",
    "google_ai_studio": "GOOGLE_AI_STUDIO_KEY",
    "openrouter": "OPENROUTER_KEY",
    "mistral": "MISTRAL_API_KEY",
    "ollama_local": None,  # local server, no key
}


class NoProviderAvailable(RuntimeError):
    """Every candidate for this task is keyless, exhausted, or failing."""


@dataclass(frozen=True)
class Choice:
    provider: str
    model: str
    task: Task
    base_url: str
    reason: str


@dataclass
class Answer:
    text: str
    choice: Choice
    first_token_ms: float
    total_ms: float
    attempts: list[str]


class Router:
    def __init__(self, *, caps_path: Path | None = None, models_path: Path | None = None,
                 ledger: Ledger | None = None) -> None:
        # override=True: .env is the declared source of truth. A stale OS-level
        # key silently shadowing it cost an afternoon of false 401s once already.
        load_dotenv(ROOT / ".env", override=True)
        self.caps = yaml.safe_load((caps_path or ROOT / "config/caps.yaml").read_text())
        self.models = yaml.safe_load((models_path or ROOT / "config/models.yaml").read_text())
        self.ledger = ledger or Ledger()
        self._clients: dict[str, Any] = {}

    # ---------- provider plumbing ----------

    def key_for(self, provider: str) -> str | None:
        var = KEY_VARS.get(provider)
        if var is None:
            return "local"  # ollama needs no key; sentinel keeps the checks uniform
        return os.environ.get(var) or None

    def available(self, provider: str) -> bool:
        return self.key_for(provider) is not None

    def _forbidden(self, model: str) -> bool:
        """Block proprietary models proxied through a free provider. NIM's catalog
        includes ~vendor/model-*, ~openai/gpt-*, ~x-ai/grok-* - the likeliest
        things to burn paid credits, and $0-forever is a hard constraint."""
        return any(model.startswith(p) for p in self.models.get("forbidden_prefixes", []))

    def tier_of(self, provider: str) -> str:
        for tier, members in self.models.get("tiers", {}).items():
            if provider in members:
                return tier
        return "deep"

    def timeout_for(self, provider: str) -> float:
        """A voice assistant must never hang. meta/llama-3.3-70b on NIM read-timed
        out at 180s and the OpenAI client's default is 600s - either would mean
        Sonara silently freezing mid-conversation. Past the ceiling we treat it as
        a failure and move to the next candidate, which is always better than silence."""
        return float(self.models.get("timeouts_s", {}).get(self.tier_of(provider), 15))

    def _client(self, provider: str):
        """One client per provider, reused. A fresh client means a fresh TLS
        handshake, which measured ~1.7s on the first call - the entire latency
        budget spent on connection setup."""
        if provider not in self._clients:
            from openai import OpenAI

            base = self.caps["providers"][provider]["base_url"]
            key = self.key_for(provider)
            self._clients[provider] = OpenAI(
                base_url=base, api_key=key or "none",
                timeout=self.timeout_for(provider), max_retries=0,
            )
        return self._clients[provider]

    def _caps_for(self, provider: str) -> dict | None:
        caps = self.caps["providers"].get(provider, {}).get("caps")
        return caps if isinstance(caps, dict) else None

    # ---------- routing ----------

    def candidates(self, task: Task) -> Iterable[dict]:
        return self.models["routes"].get(task.value, [])

    def choose(self, text: str, *, task: Task | None = None) -> Choice:
        """Pick the best eligible brain for this utterance. Never calls an LLM."""
        task = task or classify(text)
        skipped: list[str] = []

        for cand in self.candidates(task):
            provider, model = cand["provider"], cand["model"]
            if self._forbidden(model):
                skipped.append(f"{model}(forbidden)")
                continue
            if not self.available(provider):
                skipped.append(f"{provider}(no key)")
                continue
            if not self.ledger.has_headroom(provider, model, self._caps_for(provider)):
                skipped.append(f"{provider}(no quota)")
                continue
            why = f"task={task.value}"
            if skipped:
                why += f"; skipped {', '.join(skipped)}"
            return Choice(provider, model, task, self.caps["providers"][provider]["base_url"], why)

        raise NoProviderAvailable(
            f"no provider for task={task.value}; tried: {', '.join(skipped) or 'nothing configured'}"
        )

    # ---------- calling ----------

    def ask(self, text: str, *, system: str | None = None, task: Task | None = None,
            max_tokens: int = 200, stream: bool = True) -> Answer:
        """Route, call, and fail over. Returns the first answer that works."""
        task = task or classify(text)
        attempts: list[str] = []
        last_error: Exception | None = None

        for cand in self.candidates(task):
            provider, model = cand["provider"], cand["model"]
            if not self.available(provider):
                attempts.append(f"{provider}: no key")
                continue
            if not self.ledger.has_headroom(provider, model, self._caps_for(provider)):
                attempts.append(f"{provider}/{model}: no headroom")
                continue

            choice = Choice(provider, model, task,
                            self.caps["providers"][provider]["base_url"], f"task={task.value}")
            try:
                return self._call(choice, text, system, max_tokens, stream, attempts)
            except Exception as e:  # noqa: BLE001 - any provider failure means try the next one
                last_error = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                self.ledger.record(provider, model, task=task.value, ok=False, status=status)
                if status in (429, 402):
                    retry = _retry_after(e)
                    self.ledger.mark_exhausted(provider, model, retry_after_s=retry,
                                               reason=f"HTTP {status}")
                    attempts.append(f"{provider}/{model}: {status} exhausted")
                else:
                    attempts.append(f"{provider}/{model}: {type(e).__name__}")

        raise NoProviderAvailable(
            f"all candidates failed for task={task.value}: {'; '.join(attempts)}"
        ) from last_error

    def ask_with_tools(self, text: str, tools: list[dict], *, system: str | None = None,
                       task: Task | None = None, max_tokens: int = 300) -> tuple[list, str, Choice]:
        """Route, attach tool definitions, and return (tool_calls, text, choice).

        Not streamed: a tool call is only useful once it is complete, so streaming
        buys nothing here and complicates parsing. The spoken reply that FOLLOWS the
        tool result is what gets streamed, on the fast tier.
        """
        task = task or classify(text)
        attempts: list[str] = []
        last_error: Exception | None = None

        for cand in self.candidates(task):
            provider, model = cand["provider"], cand["model"]
            if self._forbidden(model) or not self.available(provider):
                continue
            if not self.ledger.has_headroom(provider, model, self._caps_for(provider)):
                continue
            choice = Choice(provider, model, task,
                            self.caps["providers"][provider]["base_url"], f"task={task.value}")
            try:
                t0 = time.perf_counter()
                r = self._client(provider).chat.completions.create(
                    model=model,
                    messages=([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": text}],
                    tools=tools, max_tokens=max_tokens,
                )
                msg = r.choices[0].message
                self.ledger.record(provider, model, task=task.value, ok=True, status=200,
                                   latency_ms=(time.perf_counter() - t0) * 1000)
                return (msg.tool_calls or [], (msg.content or "").strip(), choice)
            except Exception as e:  # noqa: BLE001
                last_error = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                self.ledger.record(provider, model, task=task.value, ok=False, status=status)
                if status in (429, 402):
                    self.ledger.mark_exhausted(provider, model, retry_after_s=_retry_after(e),
                                               reason=f"HTTP {status}")
                attempts.append(f"{provider}/{model}: {type(e).__name__}")

        raise NoProviderAvailable(
            f"no provider served tools for task={task.value}: {'; '.join(attempts)}"
        ) from last_error

    def _call(self, choice: Choice, text: str, system: str | None,
              max_tokens: int, stream: bool, attempts: list[str]) -> Answer:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": text}]
        client = self._client(choice.provider)

        t0 = time.perf_counter()
        first_ms = 0.0
        parts: list[str] = []

        if stream:
            for chunk in client.chat.completions.create(
                model=choice.model, messages=messages, max_tokens=max_tokens, stream=True
            ):
                delta = chunk.choices[0].delta.content or ""
                if delta and not first_ms:
                    first_ms = (time.perf_counter() - t0) * 1000
                parts.append(delta)
            reply = "".join(parts).strip()
        else:
            r = client.chat.completions.create(
                model=choice.model, messages=messages, max_tokens=max_tokens
            )
            reply = (r.choices[0].message.content or "").strip()
            first_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        self.ledger.record(choice.provider, choice.model, task=choice.task.value,
                           ok=True, status=200, latency_ms=total_ms)
        attempts.append(f"{choice.provider}/{choice.model}: ok")
        return Answer(reply, choice, first_ms, total_ms, attempts)


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-requests"):
        raw = headers.get(key)
        if raw:
            try:
                return float(str(raw).rstrip("s"))
            except ValueError:
                continue
    return None
