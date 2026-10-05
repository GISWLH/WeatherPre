"""NOAA CFSv2 on public S3 (bucket noaa-cfs-pds): per-variable 6-hourly time-series GRIBs (`time_grib_01`, member 1),
9 months long. Record n of each file is lead 6n h, so one byte-range request fetches the first N days of a variable."""
from __future__ import annotations
import datetime as dt, urllib.error, urllib.request, warnings
from pathlib import Path
import numpy as np, xarray as xr

BUCKET = "https://noaa-cfs-pds.s3.amazonaws.com"
# user var -> (file stem, scale to user units, units, long name)
VARS = {"z500": ("z500", 1.0, "m", "500 hPa geopotential height"), "t850": ("t850", 1.0, "K", "850 hPa temperature"),
        "t2m": ("tmp2m", 1.0, "K", "2 m temperature"), "msl": ("prmsl", 0.01, "hPa", "mean sea-level pressure"),
        "tp": ("prate", None, "mm", "precipitation accumulated since init")}

def url(init: dt.datetime, stem: str, member="01") -> str:
    d, c = f"{init:%Y%m%d}", f"{init:%H}"
    return f"{BUCKET}/cfs.{d}/{c}/time_grib_{member}/{stem}.{member}.{init:%Y%m%d%H}.daily.grb2"

def _get(u, rng=None) -> bytes:
    req = urllib.request.Request(u, headers={"Range": f"bytes={rng}"} if rng else {})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()

def exists(init, stem="tmp2m") -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url(init, stem) + ".idx", method="HEAD"), timeout=30); return True
    except urllib.error.HTTPError:
        return False

def latest_init(now: dt.datetime | None = None) -> dt.datetime:
    now = (now or dt.datetime.utcnow()).replace(minute=0, second=0, microsecond=0)
    t = now - dt.timedelta(hours=now.hour % 6)
    for _ in range(16):
        if all(exists(t, VARS[v][0]) for v in ("t2m", "z500")): return t
        t -= dt.timedelta(hours=6)
    raise RuntimeError("no complete CFSv2 cycle in the last 4 days")

def _series(init, stem, max_h, out: Path) -> xr.DataArray:
    from ..common import cfgrib as _cg; _cg()
    n = max_h // 6
    f = out / f"cfs_{stem}_{init:%Y%m%d%H}_{max_h}h.grib2"
    if not (f.exists() and f.stat().st_size):
        u = url(init, stem)
        idx = _get(u + ".idx").decode().strip().splitlines()
        offs = [int(l.split(":")[1]) for l in idx]
        n = min(n, len(offs))
        f.write_bytes(_get(u, f"0-{offs[n] - 1 if n < len(offs) else ''}"))
    ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    if "isobaricInhPa" in da.coords: da = da.drop_vars("isobaricInhPa")
    lead = (da.step.values / np.timedelta64(1, "h")).astype(int)
    da = da.assign_coords(step=lead).rename(step="lead")
    return da.drop_vars([c for c in da.coords if c not in ("lead", "latitude", "longitude")]).load()

def forecast(init: dt.datetime, max_h: int, variables, out: Path) -> xr.Dataset:
    """Weather-schema Dataset (lead [h], latitude, longitude), 6-hourly to `max_h` (no lead 0)."""
    warnings.filterwarnings("ignore", module="cfgrib")
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for v in variables:
        if v not in VARS: continue
        stem, k, units, ln = VARS[v]
        da = _series(init, stem, max_h, out)
        if v == "tp":                                  # 6 h mean rate [kg m-2 s-1] -> mm accumulated since init
            da = (da * 6 * 3600).cumsum("lead")
        else:
            da = da * k
        res[v] = da.astype("float32").assign_attrs(units=units, long_name=ln)
    # pressure files are on a regular 1 deg grid, flux files (t2m, prmsl?, prate) on T126 Gaussian: put all on 1 deg
    ref = next((d for d in res.values() if d.sizes["latitude"] == 181), next(iter(res.values())))
    for v, da in res.items():
        if da.shape[1:] == ref.shape[1:]: res[v] = da.assign_coords(latitude=ref.latitude, longitude=ref.longitude)
        else: res[v] = da.sortby("latitude").interp(latitude=ref.latitude, longitude=ref.longitude,
                                                                    kwargs={"fill_value": "extrapolate"}).assign_attrs(da.attrs)
    u = xr.Dataset(res)
    u = u.assign_coords(longitude=u.longitude % 360).sortby("longitude").sortby("latitude")
    it = np.datetime64(init, "ns")
    u = u.assign_coords(init_time=it, valid_time=("lead", it + u.lead.values.astype("timedelta64[h]").astype("timedelta64[ns]")))
    u["lead"].attrs["units"] = "hours"
    u.attrs.update(model="cfsv2", backend="cfs", source="NOAA CFSv2 member 1 (S3), regridded to 1deg",
                   init=f"{init:%Y-%m-%dT%HZ}")
    return u.transpose("lead", "latitude", "longitude")
