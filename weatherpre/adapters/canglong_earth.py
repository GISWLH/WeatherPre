"""CAS-Canglong six-week atmospheric forecast, as published on https://canglong.earth (latest live run only).

`data/release.json` names the current immutable release; its `manifest.json` lists one little-endian float32 binary per
variable / pressure layer / week (path `data/fields/...`, served under the release as `fields/...`) on the 1° grid (181 x 360, latitude 90 -> -90, longitude 0 -> 359). Each week is the
mean over days 7(k-1)..7k after the forecast start date, the same convention as the WeatherPre S2S schema."""
from __future__ import annotations
import datetime as dt, json, urllib.request, warnings
from pathlib import Path
import numpy as np, xarray as xr

BASE = "https://canglong.earth/"
G_ = 9.80665
NLAT, NLON = 181, 360
# user var -> (manifest variable ids, pressure layer label, scale to user units, units, long name)
VARS = {"z500": (("upper_geopotential",), "500 hPa", 1 / G_, "m", "500 hPa geopotential height, weekly mean"),
        "t850": (("upper_temperature",), "850 hPa", 1.0, "K", "850 hPa temperature, weekly mean"),
        "t2m": (("2m_temperature",), None, 1.0, "K", "2 m temperature, weekly mean"),
        "msl": (("mean_sea_level_pressure",), None, 0.01, "hPa", "mean sea-level pressure, weekly mean"),
        "tp": (("large_scale_rain_rate", "convective_rain_rate"), None, 86400.0, "mm/day", "precipitation rate, weekly mean")}

def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "weatherpre"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def release() -> dict:
    """Current release record: {'version', 'source', 'published', ...}."""
    return json.loads(_get(BASE + "data/release.json"))

def manifest(version: str) -> dict:
    return json.loads(_get(f"{BASE}forecast-releases/{version}/manifest.json"))

def init_of(man: dict) -> dt.datetime:
    return dt.datetime.strptime(man["attributes"]["forecast_start_date"], "%Y-%m-%d")

def _layer(var: dict, label: str | None) -> dict:
    layers = var.get("layers") or [{"label": "surface", "files": var.get("files", [])}]
    if label is None: return layers[0]
    for lay in layers:
        if lay.get("label") == label: return lay
    raise KeyError(f"{var['id']}: no layer {label!r} (has {[x.get('label') for x in layers]})")

def _field(version: str, path: str, cache: Path | None) -> np.ndarray:
    f = cache / version / path if cache else None
    if f is not None and f.exists() and f.stat().st_size == NLAT * NLON * 4:
        raw = f.read_bytes()
    else:
        raw = _get(f"{BASE}forecast-releases/{version}/{path.removeprefix('data/')}")
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(raw)
    if len(raw) != NLAT * NLON * 4:
        raise ValueError(f"canglong.earth: {path} has {len(raw)} bytes, expected {NLAT * NLON * 4}")
    return np.frombuffer(raw, dtype="<f4").reshape(NLAT, NLON)

def forecast(init, weeks, variables, cache: Path | None = None) -> xr.Dataset:
    """S2S-schema Dataset (week, latitude, longitude) for the latest canglong.earth run; `init` = 'latest' or its date."""
    from ..api import BackendUnavailable
    from ..s2s import _finish
    rel = release()
    man = manifest(rel["version"])
    t0 = init_of(man)
    if init != "latest" and init.date() != t0.date():
        raise BackendUnavailable(f"canglong: canglong.earth serves only its latest run (starts {t0:%Y-%m-%d}); "
                                 f"no archived forecast for {init:%Y-%m-%d}. Use init='latest'.")
    times = {int(t["week"]): int(t["index"]) for t in man["times"]}
    ok = [w for w in weeks if w in times]
    if len(ok) < len(weeks): warnings.warn(f"canglong: weeks {sorted(set(weeks) - set(ok))} beyond its {len(times)} weeks - skipped")
    if not ok: raise BackendUnavailable(f"canglong: the run covers weeks 1..{len(times)} only")
    byid = {v["id"]: v for v in man["variables"]}
    lat = np.linspace(90.0, -90.0, NLAT)
    lon = np.arange(NLON, dtype=float)
    out = {}
    for v in variables:
        if v not in VARS: continue
        ids, label, k, units, ln = VARS[v]
        if not all(i in byid for i in ids): continue
        weekly = []
        for w in ok:
            total = 0.0
            for i in ids:
                entry = next(e for e in _layer(byid[i], label)["files"] if int(e["time"]) == times[w])
                total = total + _field(rel["version"], entry["path"], cache).astype("float64")
            weekly.append(total * k)
        a = np.stack(weekly).astype("float32")
        if v == "tp": a = np.maximum(a, 0.0)                                   # the model emits small negative rain rates
        out[v] = xr.DataArray(a, dims=("week", "latitude", "longitude"), coords=dict(latitude=lat, longitude=lon),
                              attrs=dict(units=units, long_name=ln))
    version = man["attributes"].get("model_version") or "CAS-Canglong"
    u = _finish(out, t0, ok, "canglong", "canglong", f"{version} via canglong.earth (release {rel['version']}, 1deg)")
    u.attrs["release"] = rel["version"]
    return u
