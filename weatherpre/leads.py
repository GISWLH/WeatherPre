"""Forecast horizon ("lead") parsing and the weather / S2S split.

    weather  (≤ 15 days)  -> instantaneous fields at lead hours (6 h steps to 48 h, 12 h to 7 d, 24 h to 15 d)
    s2s      (> 15 days)  -> weekly means, week k = days [7(k-1), 7k) after init

Accepted specs (case-insensitive):
    "48h" "7d" "15d"           horizon, leads at the default step for that horizon
    "6w" "45d" "s2s"           S2S horizon -> weeks 1..N
    "w3" "week3-4" "w3,w4"     explicit S2S weeks
    "24,48,72" "6-48/6"        explicit lead hours (weather)
    24  or  [24, 48]           explicit lead hours (weather)
    "hours" "week" "15days"    legacy presets
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from .catalog import WEATHER, S2S, WEATHER_MAX_H

PRESETS = {"hours": list(range(6, 49, 6)), "week": list(range(12, 169, 12)), "15days": list(range(24, 361, 24))}
S2S_DEFAULT_WEEKS = 6

@dataclass(frozen=True)
class Leads:
    scale: str                       # WEATHER | S2S
    hours: tuple[int, ...] = ()      # weather: lead hours
    weeks: tuple[int, ...] = ()      # s2s: 1-based week numbers

    @property
    def max_hours(self) -> int:
        return max(self.hours) if self.scale == WEATHER else 7 * 24 * max(self.weeks)

    def __str__(self):
        if self.scale == WEATHER:
            return f"weather {self.hours[0]}..{self.hours[-1]} h ({len(self.hours)} leads)"
        return f"s2s weeks {','.join(map(str, self.weeks))}"

def horizon_hours(h: int) -> list[int]:
    """Default lead list for a weather horizon of h hours."""
    step = 6 if h <= 48 else 12 if h <= 168 else 24
    return list(range(step, int(h) + 1, step)) or [int(h)]

def _weather(hours) -> Leads:
    hs = tuple(sorted({int(h) for h in hours}))
    if not hs or hs[0] < 0: raise ValueError(f"bad lead hours {hours}")
    if hs[-1] > WEATHER_MAX_H:
        raise ValueError(f"lead {hs[-1]} h is beyond the 15-day weather scale; ask for S2S weeks instead, e.g. lead='6w'")
    return Leads(WEATHER, hours=hs)

def _weeks(n) -> Leads:
    return Leads(S2S, weeks=tuple(range(1, int(n) + 1)))

def parse(lead=None, *, lead_hours=None, lead_days=None, preset=None) -> Leads:
    if lead_hours is not None: return _weather(lead_hours if hasattr(lead_hours, "__iter__") else [lead_hours])
    if preset is not None: return _weather(PRESETS[preset])
    if lead_days is not None: lead = f"{int(lead_days)}d"
    if lead is None: lead = "7d"
    if isinstance(lead, Leads): return lead
    if isinstance(lead, (int, float)): return _weather([lead])
    if not isinstance(lead, str): return _weather(lead)
    s = lead.lower().replace(" ", "")
    if s in PRESETS: return _weather(PRESETS[s])
    if s == "s2s": return _weeks(S2S_DEFAULT_WEEKS)
    if m := re.fullmatch(r"(\d+)h", s): return _weather(horizon_hours(int(m[1])))
    if m := re.fullmatch(r"(\d+)d", s):
        d = int(m[1])
        return _weather(horizon_hours(24 * d)) if 24 * d <= WEATHER_MAX_H else _weeks(d // 7)
    if m := re.fullmatch(r"(\d+)w", s): return _weeks(m[1])
    if m := re.fullmatch(r"(\d+)m(onths?)?", s): return _weeks(int(m[1]) * 30 // 7)
    if m := re.fullmatch(r"w(?:eek)?s?(\d+)(?:-(\d+))?", s):
        a, b = int(m[1]), int(m[2] or m[1])
        return Leads(S2S, weeks=tuple(range(a, b + 1)))
    if s.startswith("w") and "," in s:
        return Leads(S2S, weeks=tuple(sorted({int(x.lstrip("weks")) for x in s.split(",")})))
    if re.fullmatch(r"[\d,]+h?", s): return _weather(int(x) for x in s.rstrip("h").split(","))
    if m := re.fullmatch(r"(\d+)-(\d+)(?:/(\d+))?h?", s): return _weather(range(int(m[1]), int(m[2]) + 1, int(m[3] or 6)))
    raise ValueError(f"cannot parse lead {lead!r}; try '48h', '7d', '15d', '6w' or 'week3-4'")

def scale_of(lead=None, **kw) -> str:
    return parse(lead, **kw).scale
