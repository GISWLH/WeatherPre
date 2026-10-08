"""Forecast horizon ("lead") parsing, product classes and target periods.

Product classes (what a request asks for, not a model property):
    weather      (<= 15 days)  instantaneous fields at lead hours (6 h steps to 48 h, 12 h to 7 d, 24 h to 15 d)
    subseasonal  ("s2s")       weekly means, week k = days [7(k-1), 7k) after init
    seasonal                   CALENDAR-month means: month 1 = the first full calendar month starting at or after init
    scenario                   conditional runs (prescribed boundary conditions) -- set by the backend, never by a lead spec

Accepted specs (case-insensitive):
    "48h" "7d" "15d"           weather horizon
    "6w" "45d" "s2s"           subseasonal weeks 1..N ("45d" -> weeks 1..6; days 43-45 are reported as an incomplete week)
    "w3" "week3-4" "w3,w4"     explicit weeks
    "6m" "6mo" "6months"       seasonal: calendar months 1..6  (NOT 6*30 days)
    "m2-4" "months2-4"         explicit calendar months
    "24,48,72" "6-48/6", 24    explicit lead hours (weather)
    target_start="2027-01-01", target_end="2027-06-30"   explicit target window (inclusive end date), aggregated by
                               calendar month (default), week, or as one period
"""
from __future__ import annotations
import datetime as dt, re
from dataclasses import dataclass, field
import numpy as np
from .catalog import WEATHER, S2S, SEASONAL, WEATHER_MAX_H

PRESETS = {"hours": list(range(6, 49, 6)), "week": list(range(12, 169, 12)), "15days": list(range(24, 361, 24))}
S2S_DEFAULT_WEEKS = 6
PRODUCT = {WEATHER: "weather", S2S: "subseasonal", SEASONAL: "seasonal"}

@dataclass(frozen=True)
class Period:
    label: str
    start: np.datetime64           # inclusive
    end: np.datetime64             # exclusive

    @property
    def days(self) -> float:
        return float((self.end - self.start) / np.timedelta64(1, "h")) / 24.0

@dataclass(frozen=True)
class Leads:
    scale: str                       # WEATHER | S2S | SEASONAL
    hours: tuple[int, ...] = ()      # weather: lead hours
    weeks: tuple[int, ...] = ()      # s2s: 1-based week numbers
    months: tuple[int, ...] = ()     # seasonal: 1-based calendar months after init
    target: tuple | None = None      # explicit (start, end_exclusive) as datetime64[s]
    aggregation: str = ""            # "week" | "month" | "period" for an explicit target
    requested_days: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def product(self) -> str:
        return PRODUCT[self.scale]

    @property
    def max_hours(self) -> int:
        if self.scale == WEATHER: return max(self.hours)
        if self.scale == S2S and not self.target: return 7 * 24 * max(self.weeks)
        raise ValueError("calendar-month / target horizons depend on the init date: use horizon_days(init)")

    def periods(self, init) -> list[Period]:
        """Target periods for this request and init (weather: one point per lead hour, as zero-length periods)."""
        t0 = _t64(init)
        if self.target:
            a, b = self.target
            if b <= t0: raise ValueError(f"target window ends {str(b)[:10]} before init {str(t0)[:10]}")
            if a < t0: raise ValueError(f"target window starts {str(a)[:10]} before init {str(t0)[:10]}: that part is not a forecast")
            if self.aggregation == "period": return [Period(f"{str(a)[:10]}..{str(b - np.timedelta64(1, 's'))[:10]}", a, b)]
            if self.aggregation == "week":
                n = int((b - a) / np.timedelta64(7, "D"))
                return [Period(f"w{k + 1}", a + k * np.timedelta64(7, "D"), a + (k + 1) * np.timedelta64(7, "D")) for k in range(n)]
            return _calendar_months(a, b)
        if self.scale == WEATHER:
            return [Period(f"+{h}h", t0 + np.timedelta64(h, "h"), t0 + np.timedelta64(h, "h")) for h in self.hours]
        if self.scale == S2S:
            W = np.timedelta64(7, "D")
            return [Period(f"w{k}", t0 + (k - 1) * W, t0 + k * W) for k in self.weeks]
        first = month_start(t0, ceil=True)
        out = []
        for k in self.months:
            s, e = add_months(first, k - 1), add_months(first, k)
            out.append(Period(f"m{k} {str(s)[:7]}", s, e))
        return out

    def horizon_days(self, init, ic_offset_days: float = 0.0) -> float:
        """Days of forecast needed: from the end of the initial state (init + ``ic_offset_days``; daily-mean models like
        FuXi-S2S are initialised from the init day's mean, so their forecast starts one day later) to the end of the last period."""
        t0 = _t64(init) + np.timedelta64(int(round(ic_offset_days * 86400)), "s")
        p = self.periods(t0)
        return float((max(x.end for x in p) - t0) / np.timedelta64(1, "h")) / 24.0

    def __str__(self):
        if self.target: return f"target {str(self.target[0])[:10]}..{str(self.target[1])[:10]} ({self.aggregation})"
        if self.scale == WEATHER: return f"weather {self.hours[0]}..{self.hours[-1]} h ({len(self.hours)} leads)"
        if self.scale == S2S: return f"s2s weeks {','.join(map(str, self.weeks))}"
        return f"seasonal calendar months {','.join(map(str, self.months))}"

# ---------------------------------------------------------------------------------------------------- calendar helpers
def _t64(t) -> np.datetime64:
    if isinstance(t, str) and t == "latest": raise ValueError("resolve 'latest' to a date before computing periods")
    return np.datetime64(t, "s")

def month_start(t, ceil=False) -> np.datetime64:
    t = _t64(t)
    m = t.astype("datetime64[M]")
    s = m.astype("datetime64[s]")
    return s if (s == t or not ceil) else (m + 1).astype("datetime64[s]")

def add_months(t, n: int) -> np.datetime64:
    return (_t64(t).astype("datetime64[M]") + int(n)).astype("datetime64[s]")

def _calendar_months(a, b) -> list[Period]:
    if month_start(a) != a or month_start(b) != b:
        raise ValueError(f"monthly aggregation needs a target window on calendar-month boundaries, got {str(a)[:10]}..{str(b)[:10]} "
                         "(exclusive end); use aggregation='period' for an arbitrary window")
    out, s = [], a
    while s < b:
        e = add_months(s, 1)
        out.append(Period(f"{str(s)[:7]}", s, e)); s = e
    return out

def target(start, end, aggregation: str = "month") -> Leads:
    """Explicit target window; ``end`` is an inclusive DATE ('2027-06-30') or an exclusive timestamp ('2027-07-01T00')."""
    a = np.datetime64(str(start)[:19], "s")
    e = str(end)
    b = np.datetime64(e[:19], "s") if "T" in e else np.datetime64(e[:10], "D").astype("datetime64[s]") + np.timedelta64(1, "D")
    if b <= a: raise ValueError("target_end must be after target_start")
    if aggregation not in ("month", "week", "period"): raise ValueError("aggregation must be month, week or period")
    scale = SEASONAL if aggregation in ("month", "period") else S2S
    return Leads(scale, target=(a, b), aggregation=aggregation)

# ------------------------------------------------------------------------------------------------------------- parsing
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

def _weeks(n, days=None) -> Leads:
    notes = ()
    if days is not None and days % 7:
        notes = (f"{days} days requested: weeks 1..{days // 7} are complete; days {7 * (days // 7) + 1}..{days} form an "
                 f"incomplete week {days // 7 + 1} and are not returned",)
    return Leads(S2S, weeks=tuple(range(1, int(n) + 1)), requested_days=days, notes=notes)

def _months(a, b) -> Leads:
    if a < 1 or b < a: raise ValueError(f"bad month range {a}..{b}")
    return Leads(SEASONAL, months=tuple(range(a, b + 1)))

def parse(lead=None, *, lead_hours=None, lead_days=None, preset=None, target_start=None, target_end=None,
          aggregation="month") -> Leads:
    if target_start is not None or target_end is not None:
        if target_start is None or target_end is None: raise ValueError("give both target_start and target_end")
        return target(target_start, target_end, aggregation)
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
        if 24 * d <= WEATHER_MAX_H: return _weather(horizon_hours(24 * d))
        if d < 7: raise ValueError(f"{d} days")
        return _weeks(d // 7, d)
    if m := re.fullmatch(r"(\d+)w", s): return _weeks(m[1])
    if m := re.fullmatch(r"(\d+)(?:m|mo|mon|months?)", s): return _months(1, int(m[1]))
    if m := re.fullmatch(r"m(?:onths?)?(\d+)(?:-(\d+))?", s): return _months(int(m[1]), int(m[2] or m[1]))
    if m := re.fullmatch(r"w(?:eek)?s?(\d+)(?:-(\d+))?", s):
        a, b = int(m[1]), int(m[2] or m[1])
        return Leads(S2S, weeks=tuple(range(a, b + 1)))
    if s.startswith("w") and "," in s:
        return Leads(S2S, weeks=tuple(sorted({int(x.lstrip("weks")) for x in s.split(",")})))
    if re.fullmatch(r"[\d,]+h?", s): return _weather(int(x) for x in s.rstrip("h").split(","))
    if m := re.fullmatch(r"(\d+)-(\d+)(?:/(\d+))?h?", s): return _weather(range(int(m[1]), int(m[2]) + 1, int(m[3] or 6)))
    raise ValueError(f"cannot parse lead {lead!r}; try '48h', '7d', '15d', '6w', 'week3-4', '6m' or target_start/target_end")

def scale_of(lead=None, **kw) -> str:
    return parse(lead, **kw).scale

def as_datetime(t) -> dt.datetime:
    return np.datetime64(t, "s").astype(dt.datetime)
