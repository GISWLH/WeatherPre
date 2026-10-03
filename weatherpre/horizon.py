"""Multi-model comparison per forecast horizon (hours / week / 15 days) on top of forecast()."""
from __future__ import annotations
import datetime as dt, warnings
import numpy as np, pandas as pd, xarray as xr
from . import api, registry as R
from .common import gcs_open
from .evaluate import evaluate, FIELDS

ERA5_LAST = np.datetime64("2021-12-31T18")        # last time in the WB2 6-hourly ERA5 stores used here

def _as_targets(d: xr.Dataset) -> xr.Dataset:
    """(time=init, prediction_timedelta) forecast-like Dataset -> targets indexed by valid time."""
    x = d.isel(time=0); valid = x.time.values + x.prediction_timedelta.values
    return x.drop_vars("time").assign_coords(time=("prediction_timedelta", valid)).swap_dims({"prediction_timedelta": "time"}).drop_vars("prediction_timedelta")

def _score(ds, init64, leads, truth, clim):
    w = api.to_wb2(ds)
    fields = tuple(f for f, (v, _) in FIELDS.items() if v in w and f != "msl")      # e.g. NeuralGCM has no 2 m temperature
    return evaluate(w, truth, np.array([init64], "datetime64[ns]"), leads, fields, clim)

def compare(models="auto", init="latest", preset="week", cache=None, truth="auto", verbose=True, extra=None):
    """Returns (forecasts: dict[name -> Dataset], table: DataFrame, info: dict).
    Historic init  -> scored against ERA5 (WeatherBench-X RMSE/ACC; 1.5 deg models vs 1.5 deg ERA5, 0.25 deg vs 0.25 deg).
    Latest init    -> leads whose valid time already has an IFS analysis are scored against it (proxy truth);
                      every lead also gets the RMS difference to the multi-model mean ("consensus", NOT skill).
    Future valid times cannot be scored."""
    names = api.best_models(init, preset) if models == "auto" else list(models)
    fc, skipped = {}, {}
    for m in names:
        try:
            fc[m] = api.forecast(m, init, preset=preset, cache=cache, verbose=verbose)
        except Exception as e:                                       # keep going, report below
            skipped[m] = f"{type(e).__name__}: {str(e)[:160]}"
    for name, d in (extra or {}).items():            # forecasts produced elsewhere (e.g. HF/Colab Aurora), cut to this preset
        L = [h for h in api.PRESETS[preset] if h in d.lead.values]
        if L: fc[name] = d.sel(lead=L)
    if not fc: raise api.BackendUnavailable(f"no model produced a forecast: {skipped}")
    t0 = np.datetime64(next(iter(fc.values())).init_time.values, "ns")
    rows, info = [], {"skipped": skipped, "truth": None}
    maxlead = max(int(d.lead.max()) for d in fc.values())
    now = np.datetime64(dt.datetime.utcnow(), "ns")
    historic = t0 + np.timedelta64(maxlead, "h") <= min(now, ERA5_LAST) if truth == "auto" else truth == "era5"
    if historic and truth in ("auto", "era5"):
        info["truth"] = "ERA5 (WeatherBench 2)"
        clim = {"1.5": gcs_open(R.WB2_CLIM).sortby("latitude"),
                "0.25": gcs_open(R.B + "era5-hourly-climatology/1990-2019_6h_1440x721.zarr").sortby("latitude")}
        tg = {"1.5": gcs_open(R.WB2_TRUTH), "0.25": gcs_open(R.B + "era5/1959-2022-6h-1440x721.zarr")}
        for m, d in fc.items():
            g = "0.25" if d.sizes["latitude"] > 400 else "1.5"
            try:
                r = _score(d, t0, [int(h) for h in d.lead.values], tg[g], clim[g])
            except Exception as e:
                info["skipped"][m + " (scoring)"] = f"{type(e).__name__}: {str(e)[:120]}"; continue
            df = r.to_dataframe().reset_index(); df.insert(0, "model", m); df.insert(1, "grid", g + "deg"); rows.append(df)
    else:
        # recent: score the leads that already have an analysis; consensus for all leads
        from .adapters import ecmwf_opendata
        from .truth import ifs_analysis
        from pathlib import Path
        last_an = np.datetime64(ecmwf_opendata.latest_init("ifs-hres", 0), "ns")
        scorable = sorted({int(h) for d in fc.values() for h in d.lead.values if t0 + np.timedelta64(int(h), "h") <= last_an})
        if scorable:
            info["truth"] = f"IFS analysis (proxy), valid <= {str(last_an)[:13]}Z"
            valids = [t0.astype("datetime64[h]").astype(dt.datetime) + dt.timedelta(hours=h) for h in scorable]
            tgt = ifs_analysis(valids, Path(cache or api.DATA) / "_truth")
            for m, d in fc.items():
                L = [h for h in scorable if h in d.lead.values]
                if not L: continue
                r = _score(d, t0, L, tgt, None)
                df = r.to_dataframe().reset_index(); df.insert(0, "model", m); df.insert(1, "grid", "0.25deg"); rows.append(df)
        else:
            info["truth"] = "none yet (all valid times are in the future)"
        # consensus (spread) among same-grid models
        grids = {m: d.sizes["latitude"] for m, d in fc.items()}
        big = [m for m in fc if grids[m] == max(grids.values())]
        common = sorted(set.intersection(*[set(int(h) for h in fc[m].lead.values) for m in big])) if len(big) > 1 else []
        if len(big) > 1 and common:
            wb = {m: api.to_wb2(fc[m].sel(lead=common)) for m in big}
            mean = xr.concat(list(wb.values()), "model").mean("model")
            tgt = _as_targets(mean)
            cons = []
            for m in big:
                r = evaluate(wb[m], tgt, np.array([t0], "datetime64[ns]"), common, tuple(f for f, (v, _) in FIELDS.items() if v in wb[m] and f != "msl"), None)
                df = r.to_dataframe().reset_index().rename(columns=lambda c: c.replace("_rmse", "_spread") if c.endswith("_rmse") else c)
                df.insert(0, "model", m); cons.append(df)
            info["consensus"] = pd.concat(cons)
    table = pd.concat(rows) if rows else pd.DataFrame()
    if len(table):
        table["lead_h"] = (table["lead_time"] / pd.Timedelta(hours=1)).astype(int); table = table.drop(columns="lead_time")
    if "consensus" in info:
        c = info["consensus"]; c["lead_h"] = (c["lead_time"] / pd.Timedelta(hours=1)).astype(int); info["consensus"] = c.drop(columns="lead_time")
    for df in (table, info.get("consensus")):      # WB-X works in geopotential [m2 s-2]; report z500 error in metres
        if df is not None and len(df):
            for c in ("z500_rmse", "z500_spread"):
                if c in df: df[c] = df[c] / 9.80665
    return fc, table, info
