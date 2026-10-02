"""run / latest / compare implemented on top of the adapters + WeatherBench-X."""
from __future__ import annotations
import datetime as dt, json, os
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
from . import registry as R, grib, plot
from .common import outdir
from .evaluate import evaluate, gcs_open

DATA = Path(os.environ.get("WEATHERPRE_DATA", "data"))
FIELDS_LIVE = ("z500", "t850", "t2m")

def _live_fetch(model, init, leads, out):
    from .adapters import ecmwf_opendata, noaa_s3
    ad = ecmwf_opendata if R.LIVE[model]["adapter"] == "ecmwf" else noaa_s3
    return ad.fetch(model, init, leads, out)

def _live_latest(model):
    from .adapters import ecmwf_opendata, noaa_s3
    return (ecmwf_opendata.latest_init(model) if R.LIVE[model]["adapter"] == "ecmwf" else noaa_s3.latest_init(model))

def run(model: str, init: dt.datetime, leads: list[int], root=DATA, figs=True) -> Path:
    """Produce <root>/<model>/<init>/forecast.nc (WeatherBench2 schema). Route chosen from the registry."""
    out = outdir(root, model, init)
    nc = out / "forecast.nc"
    if model in R.WB2_HOSTED:
        ds = gcs_open(R.WB2_HOSTED[model]["path"])
        keep = [v for v in ("geopotential", "temperature", "2m_temperature", "mean_sea_level_pressure") if v in ds]
        ds = ds[keep].sel(time=[np.datetime64(init, "ns")],
                          prediction_timedelta=np.array(leads, "timedelta64[h]").astype("timedelta64[ns]"))
        if "level" in ds.dims: ds = ds.sel(level=[500, 850])
        ds.load().to_netcdf(nc)
    elif model in R.LIVE:
        files = _live_fetch(model, init, leads, out)
        grib.read(files, init).to_netcdf(nc)
        for f in files: f.unlink()          # keep only the small netCDF
    elif model in R.EARTH2STUDIO:
        raise SystemExit(f"{model}: needs a GPU. Use notebooks/colab_earth2studio.ipynb or the HF Space (see docs/GPU.md).")
    else:
        raise SystemExit(f"unknown model {model}; see `weatherpre models`")
    if figs:
        plot.maps(xr.open_dataset(nc), f"{model} init {init:%Y-%m-%d %HZ}", str(out / "maps.png"))
    return nc

def latest(model: str, leads: list[int], root=DATA) -> Path:
    init = _live_latest(model)
    print(f"[latest] {model}: newest complete cycle = {init:%Y-%m-%dT%HZ}")
    return run(model, init, leads, root)

def compare_historic(models, inits, leads, out: Path, fields=("z500", "t850", "t2m")):
    clim = gcs_open(R.WB2_CLIM).sortby("latitude")
    tgt = gcs_open(R.WB2_TRUTH)
    res = {}
    for m in models:
        ds = gcs_open(R.WB2_HOSTED[m]["path"])
        have = [i for i in inits if np.datetime64(i, "ns") in ds.time.values]
        if not have: print(f"  {m}: no forecasts at requested inits, skipped"); continue
        L = [h for h in leads if np.timedelta64(h, "h").astype("timedelta64[ns]") in ds.prediction_timedelta.values]
        print(f"  {m}: {len(have)}/{len(inits)} inits, leads {L}", flush=True)
        res[m] = evaluate(ds, tgt, np.array(have, "datetime64[ns]"), L, fields, clim)
    return _save(res, out, fields, f"WB2 ERA5 truth, {len(inits)} inits")

def compare_live(models, init, leads, out: Path, root=DATA, fields=FIELDS_LIVE):
    """Models fetched from live hosted products, truth = IFS analysis (open-data step 0, a proxy)."""
    from .truth import ifs_analysis
    valids = sorted({init + dt.timedelta(hours=h) for h in leads})
    tgt = ifs_analysis(valids, Path(root) / "_truth")
    res = {}
    for m in models:
        nc = run(m, init, sorted(set(leads)), root, figs=False)
        ds = xr.open_dataset(nc)
        res[m] = evaluate(ds, tgt, np.array([init], "datetime64[ns]"), leads, fields, None)
    return _save(res, out, fields, f"truth: IFS analysis (proxy), init {init:%Y-%m-%d %HZ}")

def _save(res, out: Path, fields, title):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m, r in res.items():
        df = r.to_dataframe().reset_index()
        df.insert(0, "model", m); rows.append(df)
    df = pd.concat(rows); df["lead_h"] = (df["lead_time"] / pd.Timedelta(hours=1)).astype(int)
    df = df.drop(columns="lead_time")
    df.round(4).to_csv(out / "metrics.csv", index=False)
    plot.skill_curves(res, str(out / "skill.png"), fields, title)
    return df
