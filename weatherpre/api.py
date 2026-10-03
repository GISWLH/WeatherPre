"""One-call API:  ds = weatherpre.forecast("graphcast", init="2020-10-03", lead_days=7)

Backends are chosen automatically (hosted data first, then HF GPU inference); every backend wraps an existing
solution (WeatherBench 2 zarr, ECMWF Open Data, NOAA S3, upstream `microsoft-aurora` on a HF Space).
Returned Dataset: dims (lead [h], latitude, longitude); variables z500 [m], t850 [K], t2m [K], msl [hPa],
tp [mm since init, ECMWF models only]; scalar coord init_time, coord valid_time(lead); attrs model/backend/source."""
from __future__ import annotations
import datetime as dt, os, warnings
from pathlib import Path
import numpy as np, xarray as xr
from . import registry as R, grib
from .common import parse_time, gcs_open

DATA = Path(os.environ.get("WEATHERPRE_DATA", "data"))
PRESETS = {"hours": list(range(6, 49, 6)), "week": list(range(12, 169, 12)), "15days": list(range(24, 361, 24))}
ALIASES = {"aifs": "aifs-single", "ifs": "ifs-hres", "hres": "ifs-hres", "ecmwf": "ifs-hres", "gfs-ai": "aigfs",
           "wn": "gencast"}
# best candidates per horizon (first = default). Hosted history = WeatherBench 2; latest = Open Data / NOAA.
BEST_HIST = {"hours": ["ifs-hres", "graphcast", "pangu", "fuxi", "gencast", "neuralgcm"],
             "week": ["ifs-hres", "graphcast", "pangu", "fuxi", "gencast", "neuralgcm", "ifs-ens"],
             "15days": ["fuxi", "gencast", "neuralgcm", "ifs-ens", "ifs-hres", "graphcast", "pangu"]}
BEST_LATEST = {k: ["aifs-single", "ifs-hres", "aigfs", "gfs"] for k in PRESETS}
RETENTION_H = 90      # ECMWF Open Data keeps ~4 days; be conservative
NOAA_ARCHIVE = {"aigfs": dt.datetime(2026, 4, 16), "gfs": dt.datetime(2022, 1, 1)}   # earliest dates actually seen

class BackendUnavailable(RuntimeError):
    """No hosted data / inference route exists for this model + date (message says what to use instead)."""

def leads_from(lead_hours=None, lead_days=None, preset=None):
    if lead_hours is not None: return sorted({int(h) for h in np.atleast_1d(lead_hours)})
    if preset: return PRESETS[preset]
    if lead_days is not None: return list(range(24, int(lead_days) * 24 + 1, 24))
    return PRESETS["week"]

def _init(init):
    if isinstance(init, str) and init.lower() == "latest": return "latest"
    if isinstance(init, (dt.datetime, np.datetime64)): return parse_time(str(np.datetime_as_string(np.datetime64(init, "h"))))
    return parse_time(init) if "T" in init or len(init) > 10 else parse_time(init + "T00")

def _canon(model):
    m = model.lower(); return ALIASES.get(m, m)

def resolve(model: str, init, leads: list[int] | None = None) -> tuple[str, str, object]:
    """-> (backend, canonical_model, init_datetime|'latest').  backend in wb2 | opendata | noaa | hf."""
    m, t = _canon(model), _init(init)
    now = dt.datetime.utcnow()
    recent = t == "latest" or (now - t) < dt.timedelta(hours=RETENTION_H)
    if m == "aurora": return "hf", m, t
    if t != "latest" and (m == "ifs-hres" or m in R.WB2_HOSTED) and not recent:
        key = "hres" if m == "ifs-hres" else m
        if key in R.WB2_HOSTED: return "wb2", key, t
    if m in R.LIVE:
        ad = R.LIVE[m]["adapter"]
        if ad == "ecmwf":
            if recent: return "opendata", m, t
            raise BackendUnavailable(f"{m}: ECMWF Open Data keeps only ~4 days (no archive), and there is no hosted {t:%Y-%m-%d} forecast. "
                "Use init='latest', or a WeatherBench 2 model for 2018-22 (hres, graphcast, pangu, fuxi, gencast, neuralgcm), "
                "or run the model yourself with Earth2Studio on a GPU (notebooks/colab_earth2studio.ipynb).")
        if t == "latest" or t >= NOAA_ARCHIVE[m]: return "noaa", m, t
        raise BackendUnavailable(f"{m}: NOAA archive starts {NOAA_ARCHIVE[m]:%Y-%m-%d} (seen); no hosted forecast for {t:%Y-%m-%d}.")
    if m in R.WB2_HOSTED:
        if t == "latest": raise BackendUnavailable(f"{m}: only archived WeatherBench 2 forecasts exist (no live feed). Latest-capable: aifs, ifs, aigfs, gfs, aurora(HF).")
        return "wb2", m, t
    raise BackendUnavailable(f"unknown model {model!r}; see weatherpre.models()")

def models():
    return {"wb2 (historic, hosted)": list(R.WB2_HOSTED), "live (hosted)": list(R.LIVE), "hf (GPU inference)": ["aurora"],
            "aliases": ALIASES}

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
    u.attrs.update(model=model, backend=backend, source=source, init=f"{init:%Y-%m-%dT%HZ}")
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
    mx = max(leads)
    if t == "latest": t = ad.latest_init(model, mx)
    out = cache / model / f"{t:%Y%m%dT%H}"
    L = sorted(set(leads) | {0}) if False else sorted(set(leads))
    files = ad.fetch(model, t, L, out)
    d = grib.read(files, t)
    for f in files: f.unlink(missing_ok=True)
    src = {"ecmwf": "ECMWF Open Data 0.25deg", "noaa": "NOAA S3 0.25deg"}[R.LIVE[model]["adapter"]]
    return d, src, t

def _hf(t, leads):
    from .adapters import hf_space
    return hf_space.aurora(t, leads)

def forecast(model: str, init="latest", lead_hours=None, lead_days=None, preset=None, backend="auto", cache=None, verbose=True) -> xr.Dataset:
    """Forecast `model` from `init` ('YYYY-MM-DD[THH]' or 'latest') for the given leads. See module docstring."""
    cache = Path(cache or DATA)
    leads = leads_from(lead_hours, lead_days, preset)
    be, m, t = resolve(model, init, leads) if backend == "auto" else (backend, _canon(model), _init(init))
    if be == "wb2":
        d, src, miss = _wb2(m, t, leads)
    elif be in ("opendata", "noaa"):
        d, src, t = _live(m, t, leads, cache)
    elif be == "hf":
        u = _hf(t, leads); u.attrs["model"] = "aurora"; return u
    else:
        raise ValueError(be)
    if verbose: print(f"[weatherpre] {m} init={t:%Y-%m-%dT%HZ} backend={be} leads={leads[0]}..{max(leads)}h ({len(leads)})")
    return to_user(d, t, m, be, src)

def best_models(init="latest", preset="week"):
    t = _init(init)
    recent = t == "latest" or (dt.datetime.utcnow() - t) < dt.timedelta(hours=RETENTION_H)
    return (BEST_LATEST if recent else BEST_HIST)[preset]
