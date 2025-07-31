"""Pack: real-world knowledge. The answer to "I can't look that up yet."

Since the first mic test Sonara has been refusing weather and news questions, which was
the honest thing to do with no tools - but refusing forever is not a product. These are
the tools that let it actually know things.

EVERY SOURCE HERE IS KEYLESS AND FREE, which is not a coincidence. Premise 2 is $0
forever, and a weather tool that needs a paid API key would quietly break the whole
economic argument for one convenience.

  Open-Meteo   weather + geocoding, no key, no account, generous limits
  DuckDuckGo   search via ddgs, no key
  Wikipedia    REST summary endpoint, no key

Privacy note: a search query is Class C data - it goes out because the user asked it to,
for that exchange only. Location is resolved from a place NAME the user speaks, never
from device GPS or IP, so Sonara never silently learns where you are.
"""

from __future__ import annotations

import httpx

from .base import registry

PACK = "web"
TIMEOUT = 12.0

# WMO weather codes -> spoken English. Sonara SAYS these, so they are phrased for the
# ear: "light drizzle", not "Drizzle: Light intensity".
WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers", 95: "a thunderstorm",
    96: "a thunderstorm with hail", 99: "a severe thunderstorm with hail",
}


@registry.tool(
    name="get_weather", pack=PACK,
    description=("Get the current weather for a place. Use for 'what's the weather', "
                 "'is it raining', 'how hot is it', 'weather in <city>'."),
    parameters={
        "type": "object",
        "properties": {"place": {"type": "string",
                                 "description": "city or town name, e.g. 'Bengaluru'"}},
        "required": ["place"],
    },
)
def get_weather(place: str) -> dict:
    geo = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": place, "count": 1}, timeout=TIMEOUT).json()
    hits = geo.get("results") or []
    if not hits:
        raise ValueError(f"I couldn't find a place called {place}")
    loc = hits[0]
    r = httpx.get("https://api.open-meteo.com/v1/forecast",
                  params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                          "current": "temperature_2m,apparent_temperature,weather_code",
                          "daily": "temperature_2m_max,temperature_2m_min",
                          "forecast_days": 1, "timezone": "auto"},
                  timeout=TIMEOUT).json()
    cur, daily = r["current"], r.get("daily", {})
    return {
        "place": f"{loc['name']}, {loc.get('country', '')}".strip(", "),
        "temperature_c": cur["temperature_2m"],
        "feels_like_c": cur.get("apparent_temperature"),
        "conditions": WMO.get(cur.get("weather_code"), "unclear"),
        "high_c": (daily.get("temperature_2m_max") or [None])[0],
        "low_c": (daily.get("temperature_2m_min") or [None])[0],
    }


@registry.tool(
    name="web_search", pack=PACK,
    description=("Search the web for current information: news, events, facts, prices, "
                 "anything that changed recently. Use whenever the answer depends on "
                 "something after your training data."),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "results, default 4"},
        },
        "required": ["query"],
    },
)
def web_search(query: str, limit: int = 4) -> list[dict]:
    from ddgs import DDGS

    n = max(1, min(int(limit), 8))
    out = []
    with DDGS() as d:
        for r in d.text(query, max_results=n):
            out.append({"title": r.get("title", ""),
                        "snippet": (r.get("body") or "")[:280],
                        "url": r.get("href", "")})
    return out


@registry.tool(
    name="look_up", pack=PACK,
    description=("Look up a factual summary of a person, place, thing or concept from "
                 "Wikipedia. Use for 'who is X', 'what is Y', 'tell me about Z' when the "
                 "answer is established fact rather than current news."),
    parameters={
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    },
)
def look_up(topic: str) -> dict:
    slug = topic.strip().replace(" ", "_")
    r = httpx.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
                  headers={"User-Agent": "Sonara/0.1 (personal assistant)"},
                  timeout=TIMEOUT, follow_redirects=True)
    if r.status_code != 200:
        # Fall back to search rather than failing: a wrong slug is the common case,
        # and "I don't know" when the web does know is the failure we are removing.
        hits = web_search(topic, limit=1)
        if hits:
            return {"title": hits[0]["title"], "summary": hits[0]["snippet"],
                    "source": hits[0]["url"]}
        raise ValueError(f"I couldn't find anything about {topic}")
    j = r.json()
    return {"title": j.get("title", topic),
            "summary": (j.get("extract") or "")[:600],
            "source": j.get("content_urls", {}).get("desktop", {}).get("page", "")}
