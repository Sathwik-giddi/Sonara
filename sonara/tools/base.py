"""The skill layer — how "everything a human can do" stays an architecture, not a backlog.

Every ability is a Tool with an MCP-shaped manifest (name / description / inputSchema),
so it can be exposed over a real MCP server later without rewriting a thing. Tools
execute IN-PROCESS for now, deliberately: an out-of-process MCP round trip adds tens of
milliseconds to a hot path whose entire budget is 2 seconds. The schema is the contract;
the transport is an implementation detail we can change when a second consumer exists.

Risk is declared per tool, not inferred. A tool that deletes, sends, buys or posts is
CONFIRM class and cannot execute without an explicit spoken confirmation - premise 4,
and the thing GATE-M3 actually tests.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Risk(str, Enum):
    """How much damage a wrong tool call does.

    SAFE     - reversible, local, invisible to the outside world (read a note, get time)
    CONFIRM  - destructive or outward-facing (delete, send, buy, post). Requires the
               spoken confirm phrase. No fuzzy matching, no "sounds like yes".
    """

    SAFE = "safe"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]      # JSON Schema, MCP `inputSchema` shape
    handler: Callable[..., Any]
    risk: Risk = Risk.SAFE
    pack: str = "misc"
    # Spoken back to the user before a CONFIRM tool runs. "{args}" is filled in.
    confirm_template: str = "You want me to {name} with {args}, correct?"

    def openai_schema(self) -> dict[str, Any]:
        """The shape Groq / NIM / Ollama all expect for function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def mcp_manifest(self) -> dict[str, Any]:
        """The shape an MCP server advertises. Same content, different key names -
        which is exactly why the layer was built against this schema from day one."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }


class ToolRegistry:
    """Drop-in registration, G-Assist style: define a tool, register it, done."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(self, *, name: str, description: str, parameters: dict,
             risk: Risk = Risk.SAFE, pack: str = "misc",
             confirm_template: str | None = None) -> Callable:
        """Decorator form. Keeps the schema next to the code that implements it, so
        they cannot drift - the commonest way tool layers rot."""

        def deco(fn: Callable) -> Callable:
            self.register(Tool(
                name=name, description=description, parameters=parameters,
                handler=fn, risk=risk, pack=pack,
                confirm_template=confirm_template or Tool.confirm_template,
            ))
            return fn

        return deco

    # ---------- lookup ----------

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def by_pack(self, pack: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.pack == pack]

    def packs(self) -> list[str]:
        return sorted({t.pack for t in self._tools.values()})

    # ---------- export ----------

    def openai_schemas(self, packs: list[str] | None = None) -> list[dict]:
        tools = self.all() if packs is None else [t for t in self.all() if t.pack in packs]
        return [t.openai_schema() for t in tools]

    def mcp_manifests(self) -> list[dict]:
        return [t.mcp_manifest() for t in self.all()]

    # ---------- execution ----------

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """Execute. Safety is NOT checked here on purpose - see tools/safety.py.

        Keeping the gate out of the registry means it can never be bypassed by
        calling the registry directly: every caller must go through the executor.
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        sig = inspect.signature(tool.handler)
        accepted = {k: v for k, v in args.items() if k in sig.parameters}
        return tool.handler(**accepted)


# One registry per process. Packs import this and decorate themselves onto it.
registry = ToolRegistry()
