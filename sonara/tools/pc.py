"""Pack: PC control. Windows-first, dependency-free.

Media and volume go through the OS media keys rather than any player's API. That is
a deliberate call from the design doc: media keys work with Spotify, YouTube in a
browser, VLC and everything else on day one, where a Spotify integration works with
Spotify and costs maintenance forever.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .base import Risk, registry

PACK = "pc_control"

# Windows virtual key codes for the media/volume keys.
VK = {
    "mute": 0xAD, "vol_down": 0xAE, "vol_up": 0xAF,
    "next": 0xB0, "prev": 0xB1, "stop": 0xB2, "play_pause": 0xB3,
}
KEYEVENTF_KEYUP = 0x0002


def _tap(vk: int, times: int = 1) -> None:
    user32 = ctypes.windll.user32
    for _ in range(times):
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)


@registry.tool(
    name="get_time", pack=PACK,
    description="Get the current date and time. Use this instead of guessing.",
    parameters={"type": "object", "properties": {}},
)
def get_time() -> str:
    return datetime.now().strftime("%A %d %B %Y, %I:%M %p").lstrip("0")


@registry.tool(
    name="open_app", pack=PACK,
    description="Open an application or a website by name, e.g. 'spotify', 'notepad', 'github.com'.",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "app or site to open"}},
        "required": ["name"],
    },
)
def open_app(name: str) -> str:
    target = name.strip()
    if "." in target and " " not in target:  # looks like a domain
        target = target if target.startswith(("http://", "https://")) else f"https://{target}"
    subprocess.Popen(["cmd", "/c", "start", "", target], shell=False,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"opened {name}"


@registry.tool(
    name="media_control", pack=PACK,
    description=("Control whatever is currently playing audio or video. Use for 'pause', "
                 "'play it again', 'resume', 'skip this track', 'next', 'go back to the "
                 "previous song', 'shut the music off'."),
    parameters={
        "type": "object",
        "properties": {"action": {"type": "string",
                                  "enum": ["play", "pause", "play_pause", "next", "previous", "stop"]}},
        "required": ["action"],
    },
)
def media_control(action: str) -> str:
    key = {"play": "play_pause", "pause": "play_pause", "play_pause": "play_pause",
           "next": "next", "previous": "prev", "stop": "stop"}.get(action)
    if key is None:
        raise ValueError(f"unknown media action: {action}")
    _tap(VK[key])
    return f"media {action}"


@registry.tool(
    name="set_volume", pack=PACK,
    description="Turn the system volume up, down, or mute it.",
    parameters={
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down", "mute"]},
            "steps": {"type": "integer", "description": "how many 2% steps, default 5"},
        },
        "required": ["direction"],
    },
)
def set_volume(direction: str, steps: int = 5) -> str:
    if direction == "mute":
        _tap(VK["mute"])
        return "muted"
    _tap(VK["vol_up"] if direction == "up" else VK["vol_down"], max(1, min(int(steps), 25)))
    return f"volume {direction}"


@registry.tool(
    name="find_file", pack=PACK,
    description=("Find files by name under the user's home folder. Use for 'find my X file', "
                 "'where is X', 'look for files called X'. Returns paths only, never contents."),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "part of the filename"},
            "limit": {"type": "integer", "description": "max results, default 5"},
        },
        "required": ["query"],
    },
)
def find_file(query: str, limit: int = 5) -> list[str]:
    # Class C data: file NAMES may reach a hosted model for the exchange that asked.
    # File CONTENTS are Class L and are never returned here.
    q = query.lower()
    home = Path.home()
    skip = {"AppData", "node_modules", ".git", ".venv", "__pycache__", "$Recycle.Bin"}
    hits: list[str] = []
    for root, dirs, files in os.walk(home):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for f in files:
            if q in f.lower():
                hits.append(str(Path(root) / f))
                if len(hits) >= max(1, min(int(limit), 25)):
                    return hits
    return hits


@registry.tool(
    name="take_screenshot", pack=PACK,
    description="Capture the screen to a file. Does NOT send or analyse the image.",
    parameters={"type": "object", "properties": {}},
)
def take_screenshot() -> str:
    # Screenshots are Class L: saved locally, never uploaded, never described to a
    # hosted model unless the vision toggle is explicitly on for that exchange.
    out = Path.home() / "Pictures" / f"sonara-{datetime.now():%Y%m%d-%H%M%S}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
        "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
        "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height; "
        "$g=[System.Drawing.Graphics]::FromImage($bmp); "
        "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size); "
        f"$bmp.Save('{out}')"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    return str(out)


@registry.tool(
    name="delete_file", pack=PACK, risk=Risk.CONFIRM,
    description="Permanently delete a file. Requires spoken confirmation.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "full path to delete"}},
        "required": ["path"],
    },
    confirm_template="You want me to permanently delete {args}. That cannot be undone. Say confirm.",
)
def delete_file(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    p.unlink()
    return f"deleted {path}"
