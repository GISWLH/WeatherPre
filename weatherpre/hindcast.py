"""Hindcast evaluation on common initial dates, leads and an observation reference -- without future information.

    split = Split(train=("1950-01", "1990-12"), calibration=("1991-01", "2010-12"), test=("2011-01", "2022-12"))
    tab = evaluate_index(forecasts, obs, split, refs={"climatology": clim_fc, "persistence": pers_fc})

Rules enforced here
  * train / calibration / test periods must not overlap; only TEST inits are scored for skill; calibration statistics
    (climatology, bias, terciles, weights) are computed from CALIBRATION-period data only (``climatology``) -- and a model's
    own training period (from its card) may not overlap the test period (``Split.check_model``);
  * every input used for an init must be available at that init (``assert_available``): no future observations;
  * an experiment whose boundary conditions are observed future data (ACE2 forced by observed SST) is a *conditional
    hindcast*: ``evaluate_index`` refuses it unless ``conditional=True``, and the result is labelled as such.
One successful run is never reported as skill: tables carry ``n_inits`` and are empty for leads with fewer than ``min_inits``.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd

def _m(s) -> np.datetime64:
    return np.datetime64(s, "M")

@dataclass(frozen=True)
class Split:
    train: tuple[str, str] | None
    calibration: tuple[str, str]
    test: tuple[str, str]

    def __post_init__(self):
        spans = [(n, _m(x[0]), _m(x[1])) for n, x in (("train", self.train), ("calibration", self.calibration), ("test", self.test)) if x]
        for n, a, b in spans:
            if b < a: raise ValueError(f"{n}: end before start")
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                (n1, a1, b1), (n2, a2, b2) = spans[i], spans[j]
                if a1 <= b2 and a2 <= b1:
                    raise ValueError(f"{n1} {a1}..{b1} overlaps {n2} {a2}..{b2}")

    def role(self, t) -> str | None:
        t = np.datetime64(t, "M")
        for n, span in (("test", self.test), ("calibration", self.calibration), ("train", self.train)):
            if span and _m(span[0]) <= t <= _m(span[1]): return n
        return None

    def check_model(self, model_train_period: tuple[str, str] | None):
        """A model trained on (part of) the test period cannot be evaluated on it."""
        if model_train_period is None: return
        a, b = _m(model_train_period[0]), _m(model_train_period[1])
        if a <= _m(self.test[1]) and _m(self.test[0]) <= b:
            raise ValueError(f"model training period {a}..{b} overlaps the test period {self.test}: choose a later test period")

def assert_available(inputs_valid_until, issue_time):
    """Raise if any input used at ``issue_time`` refers to data after it (future leakage)."""
    t = np.datetime64(issue_time, "s")
    late = [x for x in np.atleast_1d(inputs_valid_until) if np.datetime64(x, "s") > t]
    if late: raise ValueError(f"future information: inputs valid until {str(late[0])[:19]} used for an issue at {str(t)[:19]}")

def climatology(obs: pd.Series, split: Split) -> pd.Series:
    """Monthly climatology (index 1..12) from the calibration period only."""
    s = obs[(obs.index >= pd.Timestamp(split.calibration[0])) & (obs.index < pd.Timestamp(split.calibration[1]) + pd.offsets.MonthEnd(1))]
    return s.groupby(s.index.month).mean()

def persistence_forecast(obs: pd.Series, clim: pd.Series, inits, leads) -> pd.DataFrame:
    """Anomaly persistence of the initial-condition month (the month before init) -- uses only data available at init."""
    rows = []
    for t in inits:
        ic = pd.Timestamp(t) - pd.offsets.MonthBegin(1)
        if ic not in obs.index: continue
        anom = obs[ic] - clim[ic.month]
        for L in leads:
            v = pd.Timestamp(t) + pd.DateOffset(months=L - 1)
            rows.append(dict(init=pd.Timestamp(t), lead=L, valid=v, member=0, value=clim[v.month] + anom))
    return pd.DataFrame(rows)

def climatology_forecast(clim: pd.Series, inits, leads) -> pd.DataFrame:
    return pd.DataFrame([dict(init=pd.Timestamp(t), lead=L, valid=pd.Timestamp(t) + pd.DateOffset(months=L - 1), member=0,
                              value=clim[(pd.Timestamp(t) + pd.DateOffset(months=L - 1)).month]) for t in inits for L in leads])

def evaluate_index(fc: pd.DataFrame, obs: pd.Series, split: Split, refs: dict[str, pd.DataFrame] | None = None,
                   clim: pd.Series | None = None, min_inits: int = 10, conditional: bool = False, experiment_type: str = "forecast"):
    """Scores of an index forecast (columns init, lead, valid, member, value; init = common init_time, lead 1 = first month)
    on TEST-period inits: RMSE, anomaly correlation, MSSS vs each reference, CRPS and spread/skill when members > 1."""
    if experiment_type == "conditional_hindcast" and not conditional:
        raise ValueError("conditional hindcast (observed future boundary conditions): pass conditional=True; it is not a forecast")
    clim = clim if clim is not None else climatology(obs, split)
    test = fc[[split.role(t) == "test" for t in fc.init]]
    out = []
    for L, g in test.groupby("lead"):
        ens = g.pivot_table(index="init", columns="member", values="value")
        valid = g.groupby("init").valid.first()
        o = obs.reindex(valid.values).values
        ok = np.isfinite(o) & np.isfinite(ens.values).all(1)
        if ok.sum() < min_inits:
            out.append(dict(lead=L, n_inits=int(ok.sum()), note=f"fewer than {min_inits} test inits: not scored")); continue
        e, o, inits = ens.values[ok], o[ok], ens.index[ok]
        c = np.array([clim[pd.Timestamp(v).month] for v in valid.values[ok]])
        mean = e.mean(1)
        row = dict(lead=L, n_inits=int(ok.sum()), members=int(e.shape[1]), rmse=float(np.sqrt(np.mean((mean - o) ** 2))),
                   acc=float(np.corrcoef(mean - c, o - c)[0, 1]))
        if e.shape[1] > 1:
            m = e.shape[1]
            t1 = np.abs(e - o[:, None]).mean(1)
            t2 = np.abs(e[:, :, None] - e[:, None, :]).sum((1, 2)) / (m * (m - 1))
            row["crps_fair"] = float(np.mean(t1 - 0.5 * t2))
            row["spread_skill"] = float(np.sqrt(e.var(1, ddof=1).mean() * (m + 1) / m) / row["rmse"])
        for name, r in (refs or {}).items():
            rr = r[(r.lead == L) & r.init.isin(inits)].groupby("init").value.mean().reindex(inits).values
            if np.isfinite(rr).all():
                row[f"msss_vs_{name}"] = float(1 - np.mean((mean - o) ** 2) / np.mean((rr - o) ** 2))
        out.append(row)
    df = pd.DataFrame(out)
    df.attrs.update(experiment_type=experiment_type, split=str(split), test_inits=int(test.init.nunique()))
    return df
