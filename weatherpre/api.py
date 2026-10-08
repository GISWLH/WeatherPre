"""One-call API.

    ds = weatherpre.forecast("graphcast", "z500", "7d", init="2020-10-03")    # weather scale (<= 15 days)
    ds = weatherpre.forecast("ifs-ext", "t2m", "6w", init="2020-10-01")       # S2S scale (> 15 days): weekly means

The backend is chosen automatically from the catalog (hosted data first, GPU inference second); every backend wraps an
existing solution (WeatherBench 2 zarr, ECMWF Open Data, NOAA S3, upstream `microsoft-aurora` on a HF Space,
NVIDIA Earth2Studio).

Weather result: dims (lead [h], latitude, longitude); z500 [m], t850 [K], t2m [K], msl [hPa], tp [mm since init].
S2S result:     dims (week, latitude, longitude); same names, weekly means, tp [mm/day]. See weatherpre.s2s.
Model runs (backend "weatherai": ORCA-DL, FuXi-S2S, ACE2): common schema (weatherpre.schema) with dims (member, period, ...),
period = week / calendar month / target window, valid_start / valid_end / completeness per period, masks valid_<var>.
Ocean variables: sst [degC], tos [degC] (ORCA-DL near-surface proxy, never renamed to sst), thetao, so, zos, uo, vo."""
from __future__ import annotations
import datetime as dt, os, re, warnings
from pathlib import Path
import numpy as np, xarray as xr
from . import registry as R, grib, catalog as C, leads as L, variables as V
from .common import parse_time, gcs_open

DATA = Path(os.environ.get("WEATHERPRE_DATA", "data"))
PRESETS = L.PRESETS
ALIASES = C.ALIASES
VARIABLES = V.ATMOS                               # default set (atmosphere); ocean variables are requested by name
VAR_ALIASES = V.ALIASES
# best candidates per horizon (first = default). Hosted history = WeatherBench 2; latest = Open Data / NOAA.
BEST_HIST = {"hours": ["ifs-hres", "graphcast", "pangu", "fuxi", "gencast", "neuralgcm"],
             "week": ["ifs-hres", "graphcast", "pangu", "fuxi", "gencast", "neuralgcm", "ifs-ens"],
             "15days": ["fuxi", "gencast", "neuralgcm", "ifs-ens", "ifs-hres", "graphcast", "pangu"],
             }
BEST_LATEST = {"hours": ["aifs-single", "ifs-hres", "aigfs", "gfs", "gefs"], "week": ["aifs-single", "ifs-hres", "aigfs", "gfs", "gefs"],
               "15days": ["aifs-single", "ifs-hres", "aigfs", "gfs", "gefs"]}
# S2S candidates are derived from the catalog (availability.candidates); no second hand-kept list
RETENTION_H = 90      # ECMWF Open Data keeps ~4 days; be conservative
NOAA_ARCHIVE = {"aigfs": dt.datetime(2026, 4, 16), "gfs": dt.datetime(2022, 1, 1), "gefs": dt.datetime(2020, 9, 23),
                "cfsv2": dt.datetime(2020, 1, 1)}   # earliest dates actually seen

class BackendUnavailable(RuntimeError):
    """No hosted data / inference route exists for this model + date (message says what to use instead)."""

def leads_from(lead_hours=None, lead_days=None, preset=None):          # legacy helper
    return list(L.parse(None, lead_hours=lead_hours, lead_days=lead_days, preset=preset).hours)

def _init(init):
    if isinstance(init, str) and init.lower() == "latest": return "latest"
    if isinstance(init, (dt.datetime, np.datetime64)): return parse_time(str(np.datetime_as_string(np.datetime64(init, "h"))))
    if isinstance(init, dt.date): return dt.datetime(init.year, init.month, init.day)
    return parse_time(init) if "T" in init or len(init) > 10 else parse_time(init + "T00")

def _canon(model):
    m = model.strip()
    return m if m.lower().startswith("e2s:") else C.canonical(m)

def _recent(t):
    return t == "latest" or (dt.datetime.utcnow() - t) < dt.timedelta(hours=RETENTION_H)

def variables(v=None) -> list[str]:
    """None/'all' -> the atmosphere set; 'T2M', ['t2m','precip'], 'sst', 'tos' -> canonical names (weatherpre.variables)."""
    return V.parse(v)

def _looks_like_init(x) -> bool:
    return isinstance(x, (dt.datetime, dt.date, np.datetime64)) or (
        isinstance(x, str) and (x.lower() == "latest" or bool(re.match(r"^\d{4}-?\d{2}-?\d{2}", x))))

def _gpu_route(m: str, t, scale) -> tuple[str, str, object]:
    from .adapters import earth2studio_run as E
    if not E.available():
        raise BackendUnavailable(f"{m}: runs through NVIDIA Earth2Studio on a GPU. Install it with "
                                 f"`pip install 'weatherpre[gpu]' 'earth2studio[...]'` on a CUDA machine, or open "
                                 "notebooks/colab_earth2studio.ipynb in Colab.")
    return "e2s", m, t

def resolve(model: str, init, leads=None) -> tuple[str, str, object]:
    """-> (backend, canonical_model, init_datetime|'latest').
    backend in wb2 | opendata | noaa | gefs | cfs | wb2-ext | baseline | hf | e2s | weatherai.
    Requests longer than the model's technical horizon are refused here, whatever the backend."""
    m, t = _canon(model), _init(init)
    lv = leads if isinstance(leads, L.Leads) else L.parse(lead_hours=leads) if leads is not None else L.parse("7d")
    scale = lv.scale
    if m.startswith("e2s:"): return _gpu_route(m, t, scale)
    if m not in C.MODELS: raise BackendUnavailable(f"unknown model {model!r}; see weatherpre.models()")
    spec = C.MODELS[m]
    if scale not in spec.scales:
        alt = ", ".join(x.name for x in C.for_scale(scale, hosted_only=True)) or ", ".join(x.name for x in C.for_scale(scale))
        names = {"weather": "weather (<= 15 days)", "subseasonal": "S2S / subseasonal (> 15 days)", "seasonal": "seasonal "
                 "(calendar months)", "scenario": "scenario (prescribed boundary conditions)"}
        what = " + ".join(names[p] for p in spec.products)
        raise BackendUnavailable(f"{m} is a {what} model; for {scale} leads use one of: {alt}")
    if spec.blocked or not spec.routes:
        raise BackendUnavailable(f"{m}: [blocked] {spec.blocked or 'no open route'} ({spec.period}); see {spec.ref}")
    if spec.research_only:
        raise BackendUnavailable(f"{m}: [research_only] {spec.research_only}")
    if t != "latest" and spec.technical_max_days is not None and not spec.hosted:   # hosted products: partial coverage, reported
        try:
            need = lv.horizon_days(t, spec.ic_offset_days)
        except ValueError:
            need = None
        if need is not None and need > spec.technical_max_days + 1e-9:
            raise BackendUnavailable(f"{m}: [beyond_technical_horizon] request needs {need:g} days from {t:%Y-%m-%d}; {m} runs at most "
                                     f"{spec.technical_max_days:g} days ({spec.technical_note or spec.lead})")
    if "weatherai" in spec.routes and (spec.name not in R.WB2_HOSTED or scale != C.WEATHER or t == "latest"):
        if t == "latest": raise BackendUnavailable(f"{m}: give an explicit init date for a model run")
        return "weatherai", m, t
    recent = _recent(t)
    if scale == C.S2S:
        if m == "ifs-ext":
            if t == "latest": raise BackendUnavailable("ifs-ext: only the WeatherBench 2 archive (2016-2022) is open; for the latest cycle use gefs or cfsv2.")
            return "wb2-ext", m, t
        if m in ("climatology", "persistence"):
            if t == "latest": t = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            return "baseline", m, t
    if m == "aurora" and "e2s" in spec.routes and _e2s_local():
        return "e2s", m, t
    if m == "aurora": return "hf", m, t
    if m in ("gefs", "cfsv2"):
        if t != "latest" and t < NOAA_ARCHIVE[m]:
            raise BackendUnavailable(f"{m}: NOAA S3 archive starts {NOAA_ARCHIVE[m]:%Y-%m-%d}; no hosted forecast for {t:%Y-%m-%d}.")
        return ("gefs" if m == "gefs" else "cfs"), m, t
    if t != "latest" and (m == "ifs-hres" or m in R.WB2_HOSTED) and not recent:
        key = "hres" if m == "ifs-hres" else m
        if key in R.WB2_HOSTED: return "wb2", key, t
    if m in R.LIVE:
        ad = R.LIVE[m]["adapter"]
        if ad == "ecmwf":
            if recent: return "opendata", m, t
            if spec.e2s and _e2s_local(): return "e2s", m, t
            raise BackendUnavailable(f"{m}: ECMWF Open Data keeps only ~4 days (no archive), and there is no hosted {t:%Y-%m-%d} forecast. "
                "Use init='latest', or a WeatherBench 2 model for 2018-22 (hres, graphcast, pangu, fuxi, gencast, neuralgcm), "
                "or run the model yourself with Earth2Studio on a GPU (notebooks/colab_earth2studio.ipynb).")
        if t == "latest" or t >= NOAA_ARCHIVE[m]: return "noaa", m, t
        raise BackendUnavailable(f"{m}: NOAA archive starts {NOAA_ARCHIVE[m]:%Y-%m-%d} (seen); no hosted forecast for {t:%Y-%m-%d}.")
    if m in R.WB2_HOSTED and t != "latest":
        return "wb2", m, t
    if "e2s" in spec.routes: return _gpu_route(m, t, scale)
    if m in R.WB2_HOSTED:
        raise BackendUnavailable(f"{m}: only archived WeatherBench 2 forecasts exist (no live feed). Latest-capable: aifs, ifs, aigfs, gfs, gefs, cfsv2, aurora(HF).")
    raise BackendUnavailable(f"{m}: no route for {init}")

def _e2s_local() -> bool:
    try:
        from .adapters import earth2studio_run as E
        return E.available() and E.device() is not None
    except Exception:
        return False

def models(scale: str | None = None):
    """Catalog as a DataFrame (optionally only 'weather' or 's2s')."""
    return C.table(scale)

# ---------- schema helpers ----------
def to_user(ds: xr.Dataset, init: dt.datetime, model: str, backend: str, source: str) -> xr.Dataset:
    """WeatherBench-2 schema (time, prediction_timedelta, level, ...) -> friendly names, lead in hours."""
    d = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
    out = {}
    if "geopotential" in d: out["z500"] = (d.geopotential.sel(level=500) / 9.80665).drop_vars("level").assign_attrs(units="m", long_name="500 hPa geopotential height")
    if "temperature" in d: out["t850"] = d.temperature.sel(level=850).drop_vars("level").assign_attrs(units="K", long_name="850 hPa temperature")
    if "2m_temperature" in d: out["t2m"] = d["2m_temperature"].assign_attrs(units="K", long_name="2 m temperature")
    if "mean_sea_level_pressure" in d: out["msl"] = (d.mean_sea_level_pressure / 100).assign_attrs(units="hPa", long_name="mean sea-level pressure")
    if "total_precipitation" in d: out["tp"] = d.total_precipitation.assign_attrs(units="mm", long_name="precipitation accumulated since init")
    u = xr.Dataset(out)
    lead = (u.prediction_timedelta.values / np.timedelta64(1, "h")).astype(int)
    u = u.assign_coords(prediction_timedelta=lead).rename(prediction_timedelta="lead")
    u["lead"].attrs["units"] = "hours"
    it = np.datetime64(init, "ns")
    u = u.assign_coords(init_time=it, valid_time=("lead", it + lead.astype("timedelta64[h]").astype("timedelta64[ns]")))
    u.attrs.update(model=model, backend=backend, source=source, init=f"{init:%Y-%m-%dT%HZ}", scale="weather")
    return u.sortby("latitude").transpose("lead", "latitude", "longitude")

def to_wb2(u: xr.Dataset) -> xr.Dataset:
    """Inverse of to_user (for WeatherBench-X scoring)."""
    n = {}
    lv = lambda a, l: a.expand_dims(level=[l])
    if "z500" in u: n["geopotential"] = lv(u.z500 * 9.80665, 500)
    if "t850" in u: n["temperature"] = lv(u.t850, 850)
    if "t2m" in u: n["2m_temperature"] = u.t2m
    if "msl" in u: n["mean_sea_level_pressure"] = u.msl * 100
    d = xr.Dataset(n).drop_vars(["init_time", "valid_time"], errors="ignore")
    d = d.assign_coords(lead=d.lead.values.astype("timedelta64[h]").astype("timedelta64[ns]")).rename(lead="prediction_timedelta")
    return d.expand_dims(time=[u.init_time.values])

# ---------- backends ----------
def _wb2(key, t, leads):
    spec = R.WB2_HOSTED[key]; ds = gcs_open(spec["path"])
    if np.datetime64(t, "ns") not in ds.time.values:
        a, b = str(ds.time.values[0])[:13], str(ds.time.values[-1])[:13]
        raise BackendUnavailable(f"{key}: WeatherBench 2 has forecasts for {a} .. {b} at 00/12Z only; {t:%Y-%m-%dT%H} not available.")
    avail = ds.prediction_timedelta.values / np.timedelta64(1, "h")
    use = [h for h in leads if h in avail]
    miss = [h for h in leads if h not in avail]
    if miss: warnings.warn(f"{key}: leads {miss} not in the hosted data (max {int(avail.max())} h; steps {int(np.diff(avail)[1])} h) - skipped")
    if not use: raise BackendUnavailable(f"{key}: none of the requested leads exist (max {int(avail.max())} h)")
    keep = [v for v in ("geopotential", "temperature", "2m_temperature", "mean_sea_level_pressure") if v in ds]
    d = ds[keep].sel(time=[np.datetime64(t, "ns")], prediction_timedelta=np.array(use, "timedelta64[h]").astype("timedelta64[ns]"))
    if "level" in d.dims: d = d.sel(level=[500, 850])
    return d.load(), spec["note"] + " (WeatherBench 2, 1.5deg)", miss

def _live(model, t, leads, cache: Path):
    from .adapters import ecmwf_opendata, noaa_s3
    ad = ecmwf_opendata if R.LIVE[model]["adapter"] == "ecmwf" else noaa_s3
    if t == "latest": t = ad.latest_init(model, max(leads))
    out = cache / model / f"{t:%Y%m%dT%H}"
    files = ad.fetch(model, t, sorted(set(leads)), out)
    d = grib.read(files, t)
    for f in files: f.unlink(missing_ok=True)
    src = {"ecmwf": "ECMWF Open Data 0.25deg", "noaa": "NOAA S3 0.25deg"}[R.LIVE[model]["adapter"]]
    return d, src, t

def _gefs(t, hours, cache: Path, precip: bool):
    from .adapters import gefs
    if max(hours) > gefs.MAX_LEAD:
        warnings.warn(f"gefs: runs end at {gefs.MAX_LEAD} h (35 days); later leads skipped")
        hours = [h for h in hours if h <= gefs.MAX_LEAD]
    if t == "latest": t = gefs.latest_init(max(hours))
    member = gefs.pick_member(t, max(hours))
    out = cache / "gefs" / f"{t:%Y%m%dT%H}"
    files = gefs.fetch(t, hours, out, member=member, precip=precip)
    d = gefs.to_accumulated(grib.read(files, t))
    for f in files: f.unlink(missing_ok=True)
    if not precip: d = d.drop_vars("total_precipitation", errors="ignore")
    note = "ensemble mean" if member == "avg" else "control member (ens. mean not available for these leads)"
    u = to_user(d, t, "gefs", "gefs", f"NOAA GEFSv12 {note}, S3 0.5deg")
    u.attrs["member"] = member
    return u

def _cfs(t, hours, variables_, cache: Path):
    from .adapters import cfs
    if t == "latest": t = cfs.latest_init()
    u = cfs.forecast(t, max(hours), variables_, cache / "cfsv2")
    return u.sel(lead=[h for h in hours if h in u.lead.values])

def _hf(t, leads):
    from .adapters import hf_space
    return hf_space.aurora(t, leads)

def forecast(model: str, variable=None, lead=None, init="latest", *, backend="auto", cache=None, verbose=True,
             lead_hours=None, lead_days=None, preset=None, target_start=None, target_end=None, aggregation="month",
             source=None, checkpoint=None, members=None, seed=0, device=None, model_backend=None, forcing=None,
             forcing_provenance=None, out_dir=None, resume=True, require_complete=True, **kw) -> xr.Dataset:
    """Forecast `variable` (None = atmosphere set) from `model` for horizon `lead` ('48h', '7d', '15d', '6w', 'week3-4', '6m', ...)
    or an explicit window (`target_start`, `target_end` inclusive date, `aggregation` month | week | period), starting at `init`.

    Run settings passed through to the backend: `source` (initial conditions: Earth2Studio data-source class name for e2s;
    'godas' | 'orca-example' | 'dir:<path>' for ORCA-DL; 'wb2-era5' | 'file:<input.nc>' | 'official-sample:<dir>' for FuXi-S2S),
    `checkpoint` (local weights root), `members` (count or ids; ORCA-DL ids are seeds), `seed`, `device`,
    `model_backend` (FuXi-S2S: 'onnx' official (default) | 'torch'), `forcing` + `forcing_provenance` (ACE2), `out_dir`
    (chunked, resumable files + manifests). Periods that the model output does not fully cover are dropped and listed in
    attrs['incomplete_periods'] (`require_complete=False` keeps them with their `completeness`)."""
    if variable is not None and _looks_like_init(variable):          # legacy: forecast(model, init, lead_hours=...)
        init, variable = variable, None
    if "init_time" in kw: init = kw.pop("init_time")
    if kw: raise TypeError(f"unexpected arguments {list(kw)}")
    cache = Path(cache or DATA)
    lv = L.parse(lead, lead_hours=lead_hours, lead_days=lead_days, preset=preset, target_start=target_start, target_end=target_end,
                 aggregation=aggregation)
    vs = variables(variable)
    be, m, t = resolve(model, init, lv) if backend == "auto" else (backend, _canon(model), _init(init))
    for v in vs:
        if m in C.MODELS and v not in C.MODELS[m].variables:
            prox = [x for x in C.MODELS[m].variables if V.VARS[x].proxy_for == v]
            if prox and len(vs) == 1:
                raise BackendUnavailable(f"{m}: [proxy_variable] no '{v}'; '{prox[0]}' is a proxy ({V.VARS[prox[0]].long_name}) and is "
                                         f"returned only when requested by name")
    if members not in (None, 1) and be != "weatherai":
        raise BackendUnavailable(f"{m}: members are only wired for model runs (backend weatherai); {be} returns its hosted product")
    if be == "weatherai":
        u = _forecast_run(m, t, lv, vs, source=source, checkpoint=checkpoint, members=members, seed=seed, device=device,
                          model_backend=model_backend, forcing=forcing, forcing_provenance=forcing_provenance, out_dir=out_dir,
                          resume=resume, require_complete=require_complete)
    elif lv.scale == C.S2S and not lv.target:
        u = _forecast_s2s(be, m, t, lv, vs, cache, source=source)
    elif lv.scale == C.WEATHER:
        u = _forecast_weather(be, m, t, list(lv.hours), vs, cache, source=source)
    else:
        raise BackendUnavailable(f"{m}: backend {be} has no calendar-month / target-window route yet")
    keep = [v for v in vs if v in u]
    if not keep:
        raise BackendUnavailable(f"{m}: none of {vs} available (has {list(u.data_vars)})")
    miss = [v for v in vs if v not in u]
    if miss and variable is not None: warnings.warn(f"{m}: variables {miss} not provided by this model - skipped")
    u = u[keep + [f"valid_{v}" for v in keep if f"valid_{v}" in u]]
    if verbose:
        if "period" in u.dims: rng = f"periods {u.period.values[0]}..{u.period.values[-1]} ({u.sizes['period']}), members {u.sizes.get('member', 1)}"
        elif "week" in u.dims: rng = f"weeks {u.week.values[0]}..{u.week.values[-1]}"
        else: rng = f"leads {int(u.lead[0])}..{int(u.lead[-1])}h ({u.sizes['lead']})"
        print(f"[weatherpre] {m} init={u.attrs.get('init', str(t)[:13])} backend={u.attrs.get('backend')} {rng} vars={','.join(keep)}")
    return u

def _forecast_run(m, t, lv, vs, *, source, checkpoint, members, seed, device, model_backend, forcing, forcing_provenance, out_dir,
                  resume, require_complete):
    """Model run through a registered adapter (weatherpre.runners), then aggregation to the requested periods."""
    from . import runners, schema
    req = runners.RunRequest(m, vs, lv, t, members=members or 1, seed=seed, device=device or "cpu", source=source,
                             checkpoint=checkpoint, backend=model_backend, forcing=forcing, forcing_provenance=forcing_provenance,
                             out_dir=Path(out_dir) if out_dir else None, resume=resume)
    native = runners.get(m)(req)
    init_time = np.datetime64(native.init_time.values, "s")
    if lv.scale == C.WEATHER:
        return native
    out = schema.aggregate(native, lv.periods(init_time), require_complete=require_complete)
    out.attrs["product"] = native.attrs.get("product") if native.attrs.get("product") == "scenario" else lv.product
    out.attrs["init"] = str(init_time)[:13]
    if lv.notes: out.attrs["lead_notes"] = " | ".join(lv.notes)
    if out_dir:
        p = Path(out_dir) / f"{m}_{str(init_time)[:10]}_{lv.product}.nc"
        out.to_netcdf(p); out.attrs["file"] = str(p)
    return out

def _forecast_weather(be, m, t, leads, vs, cache, source=None):
    if be == "wb2":
        d, src, _ = _wb2(m, t, leads)
        return to_user(d, t, m, be, src)
    if be in ("opendata", "noaa"):
        d, src, t = _live(m, t, leads, cache)
        return to_user(d, t, m, be, src)
    if be == "gefs":                        # 6 h precipitation buckets can only be summed when every 6 h lead is fetched
        six = list(range(6, max(leads) + 1, 6))
        precip = "tp" in vs and len(six) <= 2 * len(leads)        # <= 2x the downloads; otherwise skip tp
        u = _gefs(t, six if precip else leads, cache, precip=precip)
        return u.sel(lead=[h for h in leads if h in u.lead.values])
    if be == "cfs":
        return _cfs(t, list(range(6, max(leads) + 1, 6)), vs, cache).sel(lead=[h for h in leads if h % 6 == 0])
    if be == "hf":
        u = _hf(t, leads); u.attrs.update(model="aurora", scale="weather"); return u
    if be == "e2s":
        from .adapters import earth2studio_run as E
        cls = m[4:] if m.startswith("e2s:") else C.get(m).e2s
        u = E.forecast(cls, t, leads, vs, source=source, model_name=m); u.attrs["scale"] = "weather"; return u
    raise ValueError(f"backend {be!r} cannot serve weather-scale leads")

def _forecast_s2s(be, m, t, lv: L.Leads, vs, cache, source=None):
    from . import s2s
    weeks = list(lv.weeks)
    if be == "wb2-ext": return s2s.ifs_ext(t, weeks, vs)
    if be == "baseline": return getattr(s2s, m)(t, weeks, vs)
    hours = s2s.s2s_leads(weeks, 6)
    if be == "gefs": u = _gefs(t, hours, cache, precip="tp" in vs)
    elif be == "cfs": u = _cfs(t, hours[1:], vs, cache)
    elif be == "e2s":
        from .adapters import earth2studio_run as E
        cls = m[4:] if m.startswith("e2s:") else C.get(m).e2s
        if cls in E.DAILY_MEAN_MODELS:
            raise BackendUnavailable(f"{m}: Earth2Studio {cls} returns daily means labelled at the start of each day; the 6-hourly weekly "
                                     "path would mix the input day into week 1. Use backend='weatherai' (daily-mean aware).")
        u = E.forecast(cls, t, hours, vs, source=source, model_name=m)
    elif be in ("hf",):
        u = _hf(t, hours[1:])
    else:
        raise BackendUnavailable(f"{m}: backend {be} has no S2S route")
    return s2s.weekly_from_leads(u, weeks, vs)

def best_models(init="latest", preset="week"):
    t = _init(init)
    if preset == "s2s":                     # catalog-driven: hosted S2S models that are available for this init
        from . import availability as A
        lv = L.parse("6w")
        return [a.model for a in A.candidates(None, lv, t) if a.ok and C.get(a.model).hosted]
    return (BEST_LATEST if _recent(t) else BEST_HIST)[preset]
