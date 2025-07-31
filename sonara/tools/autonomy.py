"""Pack: autonomy. Tools that ask Sonara to work while you are not talking to it.

A reminder makes YOU do the work at the right time. These make SONARA do it, so the
answer already exists by the time you hear about it - which is the actual difference
between a notification and an assistant.

  check_later   do something at a time, then report the RESULT
                "check the weather at seven and tell me"
  watch_for     poll until something becomes true, then say so once
                "tell me when it stops raining"
  my_jobs       what is it working on
  cancel_job    stop one

The scheduler is injected at startup by the live loop; without it these degrade to a
clear refusal rather than silently doing nothing, because a background task that was
never scheduled is the worst kind of failure - you find out by it never happening.
"""

from __future__ import annotations

import time

from .base import registry

PACK = "autonomy"

_scheduler = None       # set by sonara_live at startup


def attach(scheduler) -> None:
    global _scheduler
    _scheduler = scheduler


def _need():
    if _scheduler is None:
        raise RuntimeError("background scheduling is only available while Sonara is running live")
    return _scheduler


def _when_to_ts(when: str) -> float:
    from .notes import parse_when

    return parse_when(when)


@registry.tool(
    name="check_later", pack=PACK,
    description=("Do something at a later time and report the result then. Use for "
                 "'check the weather at 7 and tell me', 'look that up in an hour', "
                 "'find out later and let me know'."),
    parameters={
        "type": "object",
        "properties": {
            "what": {"type": "string",
                     "description": "one of: weather, news, time. What to check later"},
            "subject": {"type": "string", "description": "place for weather, topic for news"},
            "when": {"type": "string", "description": "'at 7 pm', 'in 2 hours', 'tomorrow'"},
        },
        "required": ["what", "when"],
    },
)
def check_later(what: str, when: str, subject: str = "") -> str:
    s = _need()
    what = what.lower().strip()
    if "weather" in what:
        tool, args = "get_weather", {"place": subject or "here"}
    elif "news" in what or "search" in what:
        tool, args = "web_search", {"query": subject or what, "limit": 3}
    else:
        tool, args = "get_time", {}
    ts = _when_to_ts(when)
    s.follow_up(tool, args, ts, say=f"the {what} {('in ' + subject) if subject else ''}".strip())
    from datetime import datetime
    return f"I'll check and tell you at {datetime.fromtimestamp(ts):%I:%M %p}".replace(" 0", " ")


@registry.tool(
    name="watch_for", pack=PACK,
    description=("Keep watching something and tell me when it changes. Use for 'tell me "
                 "when it stops raining', 'let me know when it drops below 20 degrees', "
                 "'watch for news about X'."),
    parameters={
        "type": "object",
        "properties": {
            "what": {"type": "string", "description": "'weather' or 'news'"},
            "subject": {"type": "string", "description": "place, or news topic"},
            "condition": {"type": "string",
                          "description": "text that must appear in the result, e.g. 'clear'"},
            "every_minutes": {"type": "integer", "description": "how often to check, default 15"},
        },
        "required": ["what", "condition"],
    },
)
def watch_for(what: str, condition: str, subject: str = "", every_minutes: int = 15) -> str:
    s = _need()
    if "weather" in what.lower():
        tool, args = "get_weather", {"place": subject or "here"}
    else:
        tool, args = "web_search", {"query": subject or what, "limit": 3}
    s.watch(tool, args, condition,
            every_s=max(60, int(every_minutes) * 60),
            say=f"You asked me to watch for {condition}. It's happened.")
    return f"I'll keep an eye on it and tell you when it's {condition}"


@registry.tool(
    name="my_jobs", pack=PACK,
    description="List what Sonara is currently working on in the background.",
    parameters={"type": "object", "properties": {}},
)
def my_jobs() -> list[dict]:
    rows = _need().pending()
    from datetime import datetime
    out = []
    for jid, kind, tool, due, cond in rows:
        out.append({
            "id": jid, "kind": kind, "what": tool,
            "when": datetime.fromtimestamp(due).strftime("%a %I:%M %p") if due else "ongoing",
            "waiting_for": cond or "",
        })
    return out


@registry.tool(
    name="cancel_job", pack=PACK,
    description="Stop a background job by its number, from my_jobs.",
    parameters={
        "type": "object",
        "properties": {"job_id": {"type": "integer"}},
        "required": ["job_id"],
    },
)
def cancel_job(job_id: int) -> str:
    return "Cancelled." if _need().cancel(int(job_id)) else "I couldn't find that job."
