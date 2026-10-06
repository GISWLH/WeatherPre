"""Accuracy comparison in one call, for both scales.

    cmp = weatherpre.compare(["graphcast", "pangu", "ifs"], "z500", "7d", init="2020-10-03")
    cmp.table()            # model x lead RMSE
    cmp.ranking()          # who wins, averaged over leads
    cmp.plot("scores.png"); cmp.plot_maps("maps.png")

    cmp = weatherpre.compare(["ifs-ext", "gefs", "cfsv2", "persistence", "climatology"], "t2m", "6w",
                             init="2020-10-01..2020-10-29")     # S2S, averaged over every Mon/Thu init

Scores are area-weighted RMSE and ACC from WeatherBench-X. Weather: vs ERA5 (WeatherBench 2) for 2018-2021, vs the IFS
analysis (proxy) for recent cycles. S2S: weekly means vs ERA5 weekly means, ACC against the 1990-2017 climatology."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np, pandas as pd
from . import api, catalog as C, leads as L

METRICS = ("rmse", "acc")
UNITS = {"z500": "m", "t850": "K", "t2m": "K", "msl": "hPa", "tp": "mm/day"}

@dataclass
class Comparison:
    scale: str
    variables: list
    leads: L.Leads
    forecasts: dict                       # model -> Dataset (first init)
    scores: pd.DataFrame                  # tidy: init, model, variable, lead (h or week), rmse, acc
    truth: str | None
    skipped: dict = field(default_factory=dict)
    consensus: pd.DataFrame | None = None

    @property
    def lead_col(self): return "week" if self.scale == C.S2S else "lead_h"

    @property
    def inits(self): return sorted(self.scores["init"].unique()) if len(self.scores) else []

    def summary(self) -> pd.DataFrame:
        """Scores averaged over inits: RMSE = sqrt(mean MSE), ACC = mean ACC."""
        if not len(self.scores): return self.scores
        g = self.scores.groupby(["model", "variable", self.lead_col])
        out = g["rmse"].apply(lambda x: float(np.sqrt(np.nanmean(np.square(x))))).to_frame()
        if "acc" in self.scores: out["acc"] = g["acc"].mean()
        out["n_inits"] = g.size()
        return out.reset_index()

    def table(self, metric="rmse", variable=None) -> pd.DataFrame:
        """model x lead table for one variable (default: the first requested one)."""
        s = self.summary()
        if not len(s): return s
        v = variable or next(x for x in self.variables if x in set(s.variable))
        t = s[s.variable == v].pivot(index="model", columns=self.lead_col, values=metric)
        t.columns = [f"w{c}" if self.scale == C.S2S else f"+{c}h" for c in t.columns]
        t.columns.name = f"{v} {metric.upper()}" + (f" [{UNITS[v]}]" if metric == "rmse" else "")
        return t.sort_values(t.columns[-1], ascending=(metric == "rmse"))

    def ranking(self, metric="rmse") -> pd.DataFrame:
        """Mean rank of each model over every (variable, lead) it was scored on; 1 = best."""
        s = self.summary()
        if not len(s): return s
        r = s.dropna(subset=[metric]).copy()
        r["rank"] = r.groupby(["variable", self.lead_col])[metric].rank(ascending=(metric == "rmse"))
        out = r.groupby("model").agg(mean_rank=("rank", "mean"), cases=("rank", "size")).sort_values("mean_rank")
        for v in self.variables:
            x = r[r.variable == v].groupby("model")[metric].mean()
            if len(x): out[f"{v}_{metric}"] = x
        return out

    def plot(self, out="scores.png", title=None):
        from .maps import plot_skill
        return plot_skill(self.summary(), out, self.scale, self.variables, title or self._title())

    def _first_init(self):
        """Scores of the init whose forecasts are kept (the maps show that init)."""
        if not len(self.scores): return self.summary()
        one = Comparison(self.scale, self.variables, self.leads, {}, self.scores[self.scores["init"] == self.inits[0]], self.truth)
        return one.summary()

    def plot_maps(self, out="maps.png", variable=None, lead=None):
        from . import maps
        v = variable or self.variables[0]
        if self.scale == C.S2S:
            wk = lead or int(max(self.leads.weeks))
            return maps.plot_s2s(self.forecasts, v, wk, out=out, scores=self._first_init())
        h = lead or int(max(self.leads.hours))
        tab = self._first_init()
        tab = tab[tab.variable == v].rename(columns={"rmse": f"{v}_rmse"}) if len(tab) else None
        return maps.plot_models(self.forecasts, v, h, None, out, title=f"Model comparison · +{h} h · {self._title()}", scores=tab)

    def to_csv(self, path):
        self.scores.to_csv(path, index=False); return path

    def _title(self):
        i = self.inits
        when = f"init {i[0]}" if len(i) == 1 else f"{len(i)} inits {i[0]} … {i[-1]}" if i else ""
        return f"{'S2S weekly means' if self.scale == C.S2S else 'Weather'} · {when}"

    def __repr__(self):
        lines = [f"Comparison({self.scale}, {', '.join(self.variables)}, {self.leads}, truth={self.truth})"]
        if len(self.scores):
            for v in self.variables:
                if v in set(self.scores.variable):
                    lines.append(self.table("rmse", v).round(2).to_string())
        if self.skipped: lines.append(f"skipped: {self.skipped}")
        return "\n".join(lines)

def _init_list(init, scale, models) -> list:
    if isinstance(init, (list, tuple)): return list(init)
    if not (isinstance(init, str) and ".." in init): return [init]
    a, b = [api._init(x) for x in init.split("..")]
    if scale == C.S2S and "ifs-ext" in models:                     # WB2 extended range: Monday + Thursday inits
        days = pd.date_range(a, b, freq="D"); return [d.to_pydatetime() for d in days if d.weekday() in (0, 3)]
    step = pd.Timedelta(days=7) if scale == C.S2S else pd.Timedelta(days=1)
    return [d.to_pydatetime() for d in pd.date_range(a, b, freq=step)]

def _tidy_weather(table: pd.DataFrame, init) -> pd.DataFrame:
    rows = []
    for v in api.VARIABLES:
        cols = {k: f"{v}_{k}" for k in METRICS if f"{v}_{k}" in table}
        if "rmse" not in cols: continue
        d = table[["model", "lead_h"] + list(cols.values())].rename(columns={c: k for k, c in cols.items()})
        d.insert(1, "variable", v); rows.append(d)
    out = pd.concat(rows) if rows else pd.DataFrame(columns=["model", "variable", "lead_h", "rmse", "acc"])
    out.insert(0, "init", str(init)[:13])
    return out

def _weather(models, vs, lv, init, truth, cache, verbose, extra):
    from . import horizon
    fc, table, info = horizon.compare(models, init, cache=cache, truth=truth, verbose=verbose, extra=extra, leads=list(lv.hours))
    fc = {m: d[[v for v in vs if v in d]] for m, d in fc.items() if any(v in d for v in vs)}
    t0 = str(next(iter(fc.values())).init_time.values)[:13] if fc else str(init)
    sc = _tidy_weather(table, t0) if len(table) else pd.DataFrame()
    if len(sc): sc = sc[sc.variable.isin(vs)]
    return fc, sc, info.get("truth"), info.get("skipped", {}), info.get("consensus")

def _s2s(models, vs, lv, init, truth, cache, verbose, extra):
    from . import s2s
    names = api.best_models(init, "s2s") if models == "auto" else list(models)
    fc, skipped, rows = {}, {}, []
    for m in names:
        try:
            fc[m] = api.forecast(m, vs, lv, init=init, cache=cache, verbose=verbose)
        except Exception as e:
            skipped[m] = f"{type(e).__name__}: {str(e)[:200]}"
    for name, d in (extra or {}).items(): fc[name] = d
    if not fc: raise api.BackendUnavailable(f"no model produced an S2S forecast: {skipped}")
    last = max(np.datetime64(d.valid_start.values.max()) for d in fc.values())
    scorable = truth != "none" and last <= s2s.ERA5_WEEKLY_LAST
    for m, d in fc.items():
        if not scorable: break
        try:
            r = s2s.score(d, vs)
        except Exception as e:
            skipped[m + " (scoring)"] = f"{type(e).__name__}: {str(e)[:160]}"; continue
        for v in vs:
            if f"{v}_rmse" not in r: continue
            x = pd.DataFrame({"model": m, "variable": v, "week": r.week, "rmse": r[f"{v}_rmse"],
                              "acc": np.nan if m == "climatology" else r.get(f"{v}_acc", np.nan)})
            rows.append(x)
    t0 = str(next(iter(fc.values())).init_time.values)[:13]
    sc = pd.concat(rows) if rows else pd.DataFrame()
    if len(sc): sc.insert(0, "init", t0)
    tr = "ERA5 weekly means (WeatherBench 2), ACC vs 1990-2017 climatology" if scorable else \
        "none (ERA5 weekly truth in WeatherBench 2 ends 2023-01; future weeks cannot be scored)"
    return fc, sc, tr, skipped, None

def compare(models="auto", variable=None, lead="7d", init="latest", *, truth="auto", cache=None, verbose=True,
            extra=None, **legacy) -> Comparison:
    """Run several models for the same init/horizon and score them. `models`: list, comma string or 'auto'.
    `init`: one date, 'latest', a list, or 'START..END' (several inits, scores averaged)."""
    if legacy.get("preset"): lead = legacy.pop("preset")
    if legacy: raise TypeError(f"unexpected arguments {list(legacy)}")
    if isinstance(models, str) and models != "auto": models = [m.strip() for m in models.split(",") if m.strip()]
    if models != "auto": models = [api._canon(m) for m in models]
    lv = L.parse(lead)
    vs = api.variables(variable)
    inits = _init_list(init, lv.scale, models if models != "auto" else [])
    run = _s2s if lv.scale == C.S2S else _weather
    fc0, scores, truths, skipped, cons = None, [], set(), {}, None
    for i, t in enumerate(inits):
        if verbose and len(inits) > 1: print(f"[weatherpre] init {i + 1}/{len(inits)}: {t}")
        try:
            fc, sc, tr, sk, cs = run(models, vs, lv, t, truth, cache, verbose and len(inits) == 1, extra if i == 0 else None)
        except api.BackendUnavailable as e:
            skipped[str(t)] = str(e); continue
        fc0 = fc0 or fc; cons = cons if cons is not None else cs
        if len(sc): scores.append(sc)
        if tr: truths.add(tr)
        skipped.update({(f"{k} @ {str(t)[:10]}" if len(inits) > 1 else k): v for k, v in sk.items()})
    if fc0 is None: raise api.BackendUnavailable(f"nothing ran: {skipped}")
    sc = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(columns=["init", "model", "variable", "rmse", "acc"])
    return Comparison(lv.scale, vs, lv, fc0, sc, "; ".join(sorted(truths)) or None, skipped, cons)
