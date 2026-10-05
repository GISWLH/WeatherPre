"""Sub-seasonal to seasonal (S2S, > 15 days): weekly-mean forecasts, baselines and the truth they are scored against.

User schema (what `weatherpre.forecast(..., lead="6w")` returns):
    dims (week, latitude, longitude), week = 1..N, week k = mean over days [7(k-1), 7k) after init
    coords valid_start(week), valid_end(week) (exclusive), scalar init_time
    variables z500 [m], t850 [K], t2m [K], msl [hPa], tp [mm/day]; attrs model / backend / source / scale="s2s"

Truth = ERA5 weekly means and the 1990-2017 weekly climatology, both hosted by WeatherBench 2 (label = first day of
the 7-day window, the same convention as the WB2 extended-range forecasts)."""
from __future__ import annotations
import datetime as dt, warnings
import numpy as np, xarray as xr
from .common import gcs_open
from .registry import B

G = "240x121_equiangular_with_poles_conservative"
EXT_PL = B + "ifs_extended_range/weekly/ifs-ext-pressure-levels-weekly_avg_ens_mean.zarr"
EXT_SL = B + "ifs_extended_range/weekly/ifs-ext-full-single-level-weekly_avg_ens_mean.zarr"
ERA5_WEEKLY = B + f"era5_weekly/1959-2023_01_10-1h-{G}.zarr"
CLIM_WEEKLY = B + f"era5-daily-climatology/1990-2017-weekly_clim_daily_mean_61_dw_{G}.zarr"
ERA5_WEEKLY_LAST = np.datetime64("2023-01-04")
WEEK = np.timedelta64(7, "D")
G_ = 9.80665

# user name -> (WB2 variable, level, to-user scale, units, long name)
WB2_VARS = {
    "z500": ("geopotential", 500, 1 / G_, "m", "500 hPa geopotential height, weekly mean"),
    "t850": ("temperature", 850, 1.0, "K", "850 hPa temperature, weekly mean"),
    "t2m": ("2m_temperature", None, 1.0, "K", "2 m temperature, weekly mean"),
    "msl": ("mean_sea_level_pressure", None, 0.01, "hPa", "mean sea-level pressure, weekly mean"),
    "tp": ("total_precipitation_24hr", None, 1000.0, "mm/day", "precipitation rate, weekly mean"),
}

def _t64(t) -> np.datetime64:
    return np.datetime64(t, "ns")

def _finish(out: dict, init, weeks, model, backend, source) -> xr.Dataset:
    u = xr.Dataset(out).assign_coords(week=list(weeks))
    it = _t64(init)
    starts = it + (np.array(weeks) - 1) * WEEK
    u = u.assign_coords(init_time=it, valid_start=("week", starts), valid_end=("week", starts + WEEK))
    u.attrs.update(model=model, backend=backend, source=source, scale="s2s", init=f"{init:%Y-%m-%dT%HZ}")
    return u.sortby("latitude").transpose("week", "latitude", "longitude")

def _from_wb2(ds: xr.Dataset, variables, sel_time=None) -> dict:
    out = {}
    for v in variables:
        name, lev, k, units, ln = WB2_VARS[v]
        if name not in ds: continue
        da = ds[name]
        if lev is not None and "level" in da.dims: da = da.sel(level=lev, drop=True)
        out[v] = (da * k).assign_attrs(units=units, long_name=ln)
    return out

# ---------------- backends ----------------
def ifs_ext(init: dt.datetime, weeks, variables) -> xr.Dataset:
    """ECMWF extended-range ensemble mean (WB2, 2016-2022, Monday/Thursday 00Z inits, 46 days)."""
    pl, sl = gcs_open(EXT_PL), gcs_open(EXT_SL)
    t = _t64(init)
    if t not in pl.time.values:
        near = pl.time.sel(time=t, method="nearest").values
        from .api import BackendUnavailable
        raise BackendUnavailable(f"ifs-ext: WeatherBench 2 holds extended-range inits on Mondays/Thursdays 2016-2022; "
                                 f"{init:%Y-%m-%d} is not one (nearest: {str(near)[:10]}).")
    td = (np.array(weeks) - 1) * WEEK
    have = set(sl.prediction_timedelta.values) & set(pl.prediction_timedelta.values)
    ok = [w for w, d in zip(weeks, td) if d.astype("timedelta64[ns]") in have]
    if len(ok) < len(weeks): warnings.warn(f"ifs-ext: weeks {sorted(set(weeks) - set(ok))} beyond 46 days - skipped")
    sel = dict(time=t, prediction_timedelta=((np.array(ok) - 1) * WEEK).astype("timedelta64[ns]"))
    want_pl = [WB2_VARS[v][0] for v in variables if WB2_VARS[v][0] in pl]
    want_sl = [WB2_VARS[v][0] for v in variables if WB2_VARS[v][0] in sl]
    parts = []
    if want_pl: parts.append(pl[want_pl].sel(**sel).sel(level=[500, 850]))
    if want_sl: parts.append(sl[want_sl].sel(**sel))
    d = xr.merge(parts, compat="override").load().rename(prediction_timedelta="week").assign_coords(week=ok)
    return _finish(_from_wb2(d, variables), init, ok, "ifs-ext", "wb2-ext", "ECMWF extended range ens. mean (WeatherBench 2, 1.5deg)")

def _clim_at(starts, variables) -> dict:
    c = gcs_open(CLIM_WEEKLY)
    doy = [int(str(np.datetime64(s, "D").astype(dt.date).timetuple().tm_yday)) for s in starts]
    names = [WB2_VARS[v][0] for v in variables if WB2_VARS[v][0] in c]
    d = c[names].sel(dayofyear=xr.DataArray(doy, dims="week"))
    if "level" in d.dims: d = d.sel(level=[500, 850])
    return _from_wb2(d.drop_vars("dayofyear").load(), variables)

def climatology(init: dt.datetime, weeks, variables) -> xr.Dataset:
    starts = _t64(init) + (np.array(weeks) - 1) * WEEK
    return _finish(_clim_at(starts, variables), init, weeks, "climatology", "baseline", "ERA5 1990-2017 weekly climatology (WeatherBench 2)")

def persistence(init: dt.datetime, weeks, variables) -> xr.Dataset:
    """Climatology + the ERA5 anomaly of the 7 days before init (a standard S2S reference forecast)."""
    t0 = _t64(init) - WEEK
    if t0 > ERA5_WEEKLY_LAST:
        from .api import BackendUnavailable
        raise BackendUnavailable("persistence: needs the ERA5 week before init; WeatherBench 2 ERA5 ends 2023-01.")
    era = gcs_open(ERA5_WEEKLY)
    names = [WB2_VARS[v][0] for v in variables if WB2_VARS[v][0] in era]
    last = era[names].sel(time=t0)
    if "level" in last.dims: last = last.sel(level=[500, 850])
    obs = _from_wb2(last.load(), variables)
    c0 = _clim_at([t0], variables)
    clim = _clim_at(_t64(init) + (np.array(weeks) - 1) * WEEK, variables)
    out = {v: (clim[v] + (obs[v] - c0[v].isel(week=0, drop=True))).assign_attrs(clim[v].attrs) for v in clim if v in obs}
    return _finish(out, init, weeks, "persistence", "baseline", "ERA5 anomaly of the week before init + climatology")

# ---------------- sub-daily forecasts -> weekly means ----------------
def weekly_from_leads(u: xr.Dataset, weeks, variables=None) -> xr.Dataset:
    """Weather-schema Dataset (lead [h], tp accumulated since init) -> S2S schema weekly means."""
    leads = u.lead.values.astype(int)
    step = int(np.min(np.diff(leads))) if len(leads) > 1 else 24
    per_week = 168 // step
    init = np.datetime64(u.init_time.values, "s").astype(dt.datetime)
    out, ok = {}, []
    for w in weeks:
        a, b = (w - 1) * 168, w * 168
        sel = [h for h in leads if a <= h < b]
        if len(sel) < per_week - 1:                             # tolerate a missing lead 0 (CFSv2 starts at +6 h)
            warnings.warn(f"{u.attrs.get('model')}: week {w} incomplete ({len(sel)}/{per_week} steps) - skipped"); continue
        ok.append(w)
        for v in u.data_vars:
            if variables and v not in variables: continue
            if v == "tp":
                if b in leads:
                    lo = u.tp.sel(lead=a) if a in leads else 0.0 * u.tp.isel(lead=0)
                    out.setdefault(v, []).append(((u.tp.sel(lead=b) - lo) / 7.0).drop_vars(["lead", "valid_time"], errors="ignore"))
                continue
            out.setdefault(v, []).append(u[v].sel(lead=sel).mean("lead").drop_vars(["valid_time"], errors="ignore"))
    if not ok:
        from .api import BackendUnavailable
        raise BackendUnavailable(f"{u.attrs.get('model')}: no complete week in leads {leads[0]}..{leads[-1]} h")
    res = {}
    for v, parts in out.items():
        if len(parts) != len(ok): continue                     # tp missing some week -> drop it rather than misalign
        da = xr.concat(parts, "week").drop_vars("init_time", errors="ignore")
        ln = (u[v].attrs.get("long_name", v) if v != "tp" else "precipitation rate") + ", weekly mean"
        res[v] = da.assign_attrs(units="mm/day" if v == "tp" else u[v].attrs.get("units", ""), long_name=ln)
    s = _finish(res, init, ok, u.attrs.get("model", ""), u.attrs.get("backend", ""), u.attrs.get("source", ""))
    for k in ("member", "note"):
        if k in u.attrs: s.attrs[k] = u.attrs[k]
    return s

def s2s_leads(weeks, step=6) -> list[int]:
    """Lead hours needed to form the requested weeks (0 .. 168*max(week), every `step` h)."""
    return list(range(0, 168 * max(weeks) + 1, step))

# ---------------- regridding to the WeatherBench 2 1.5 deg grid ----------------
LAT15 = np.arange(-90, 90.01, 1.5)
LON15 = np.arange(0, 360, 1.5)

def to_1p5(u: xr.Dataset) -> xr.Dataset:
    """Box-smooth to ~1.5 deg then sample the WB2 240x121 grid (exact points for 0.25 / 0.5 / 1 deg sources)."""
    if u.sizes.get("latitude") == 121 and u.sizes.get("longitude") == 240: return u
    dx = float(abs(u.longitude[1] - u.longitude[0]))
    n = max(1, int(round(1.5 / dx)))
    if n > 1:
        n += (n + 1) % 2                                                    # odd, centred window
        p = n // 2
        x = u.pad(longitude=p, mode="wrap")
        x = x.rolling(longitude=n, center=True, min_periods=1).mean().isel(longitude=slice(p, -p))
        x = x.rolling(latitude=n, center=True, min_periods=1).mean()
        u = x.assign_coords(longitude=u.longitude)
    u = u.sortby("latitude")
    ext = xr.concat([u, u.isel(longitude=0).assign_coords(longitude=u.longitude[0] + 360)], "longitude")
    out = ext.interp(latitude=LAT15, longitude=LON15, kwargs={"fill_value": None})
    out.attrs = u.attrs
    for v in out.data_vars: out[v].attrs = u[v].attrs
    return out

# ---------------- scoring ----------------
def to_wb2(u: xr.Dataset) -> xr.Dataset:
    """S2S user schema -> WB2 forecast schema (time=init, prediction_timedelta=week start) for WeatherBench-X."""
    n = {}
    for v, (name, lev, k, _, _) in WB2_VARS.items():
        if v not in u: continue
        da = u[v] / k
        n[name] = da.expand_dims(level=[lev]) if lev is not None else da
    d = xr.Dataset(n).drop_vars(["init_time", "valid_start", "valid_end"], errors="ignore")
    d = d.assign_coords(week=((d.week.values - 1) * WEEK).astype("timedelta64[ns]")).rename(week="prediction_timedelta")
    return d.expand_dims(time=[_t64(u.init_time.values)])

def score(u: xr.Dataset, variables=None):
    """RMSE + ACC of weekly means vs ERA5 weekly means (WeatherBench-X, area-weighted, 1.5 deg)."""
    import pandas as pd
    from .evaluate import evaluate, FIELDS
    u = to_1p5(u)
    last = _t64(u.valid_start.values.max())
    if last > ERA5_WEEKLY_LAST:
        raise ValueError(f"ERA5 weekly truth in WeatherBench 2 ends {ERA5_WEEKLY_LAST}; week starting {str(last)[:10]} cannot be scored")
    w = to_wb2(u)
    fields = tuple(f for f in (variables or WB2_VARS) if f in u and f in FIELDS)
    r = evaluate(w, gcs_open(ERA5_WEEKLY), np.array([_t64(u.init_time.values)]),
                 [int(h) for h in (u.week.values - 1) * 168], fields, gcs_open(CLIM_WEEKLY))
    df = r.to_dataframe().reset_index()
    df["week"] = (df.pop("lead_time") / pd.Timedelta(days=7)).astype(int) + 1
    df = df[["week"] + [c for c in df if c.endswith(("_rmse", "_acc"))]]
    if "z500_rmse" in df: df["z500_rmse"] = df["z500_rmse"] / G_
    if "tp_rmse" in df: df["tp_rmse"] = df["tp_rmse"] * 1000.0
    return df

# ---------------- anomalies (what S2S maps show) ----------------
def anomaly(u: xr.Dataset) -> xr.Dataset:
    """Weekly-mean forecast minus the ERA5 1990-2017 weekly climatology, on the 1.5 deg grid."""
    u = to_1p5(u)
    vs = [v for v in u.data_vars if v in WB2_VARS]
    clim = _clim_at(u.valid_start.values, vs)
    out = u.copy()
    for v in vs:
        c = clim[v].assign_coords(week=u.week.values, latitude=u.latitude, longitude=u.longitude)
        out[v] = (u[v] - c).assign_attrs(units=u[v].attrs.get("units", ""), long_name=u[v].attrs.get("long_name", v) + " anomaly")
    out.attrs.update(u.attrs, anomaly="vs ERA5 1990-2017 weekly climatology")
    return out

def era5_anomaly(start, var: str) -> xr.DataArray:
    """Observed (ERA5) weekly-mean anomaly for the week starting `start`."""
    name, lev, k, units, _ = WB2_VARS[var]
    obs = gcs_open(ERA5_WEEKLY)[name].sel(time=_t64(start))
    if lev is not None: obs = obs.sel(level=lev, drop=True)
    c = _clim_at([_t64(start)], [var])[var].isel(week=0, drop=True)
    return ((obs * k).load() - c).transpose("latitude", "longitude").sortby("latitude").assign_attrs(units=units)
