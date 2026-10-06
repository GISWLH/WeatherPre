"""Verification with WeatherBench-X (no metric code of our own): RMSE (+ACC if climatology given)
of gridded predictions vs gridded targets, area-weighted, per lead time.

Predictions/targets follow the WeatherBench 2 schema:
  predictions: dims (time=init, prediction_timedelta, [level], latitude, longitude)
  targets:     dims (time=valid, [level], latitude, longitude)
variable names: geopotential, temperature, 2m_temperature, mean_sea_level_pressure ...
"""
from __future__ import annotations
import numpy as np, xarray as xr
from .common import gcs_open
from weatherbenchX import aggregation, weighting
from weatherbenchX.data_loaders import xarray_loaders as xl
from weatherbenchX.metrics import deterministic

# fields we report: label -> (WB2 variable, selection)
FIELDS = {
    "z500": ("geopotential", {"level": 500}),
    "t850": ("temperature", {"level": 850}),
    "t2m": ("2m_temperature", {}),
    "msl": ("mean_sea_level_pressure", {}),
    "tp": ("total_precipitation_24hr", {}),          # S2S weekly means only (m/day)
}

def _loader(cls, src, var, sel, **kw):
    kws = dict(variables=[var], sel_kwargs=sel or None, **kw)
    if isinstance(src, str):
        src = gcs_open(src)   # anonymous GCS (WB2 buckets are public); avoids ADC lookups
    return cls(ds=src, **kws)

def evaluate(pred_src, target_src, init_times, lead_hours, fields=("z500", "t850", "t2m"),
             climatology: xr.Dataset | None = None) -> xr.Dataset:
    """Returns Dataset[(field)_rmse, (field)_acc] over prediction_timedelta."""
    init_times = np.asarray(init_times, dtype="datetime64[ns]")
    leads = np.asarray(lead_hours, dtype="timedelta64[h]").astype("timedelta64[ns]")
    agg = aggregation.Aggregator(reduce_dims=["init_time", "latitude", "longitude"],
                                 weigh_by=[weighting.GridAreaWeighting()])
    out = {}
    for f in fields:
        var, sel = FIELDS[f]
        p = _loader(xl.PredictionsFromXarray, pred_src, var, sel)
        t = _loader(xl.TargetsFromXarray, target_src, var, sel)
        pc, tc = p.load_chunk(init_times, leads), t.load_chunk(init_times, leads)
        metrics = {"rmse": deterministic.RMSE()}
        if climatology is not None:
            metrics["acc"] = deterministic.ACC(climatology=climatology[[var]].sel(**sel) if sel else climatology[[var]])
        res = aggregation.compute_metric_values_for_single_chunk(metrics, agg, pc, tc)
        for k in res.data_vars:
            name = str(k)
            kind = "acc" if "acc" in name.lower() else "rmse"
            da = res[k]
            da = da.squeeze([d for d in da.dims if d != "lead_time" and da.sizes[d] == 1], drop=True)
            out[f"{f}_{kind}"] = da.drop_vars("level", errors="ignore")
    return xr.Dataset(out)
