"""NOAA OISST v2.1 (AVHRR) daily SST from the NOAA open-data S3 bucket -> monthly means and Nino indices.

Bucket: noaa-cdr-sea-surface-temp-optimum-interpolation-pds, data/v2.1/avhrr/YYYYMM/oisst-avhrr-v02r01.YYYYMMDD[_preliminary].nc
(0.25 deg, degC, ~1.6 MB per day; final files ~2 weeks behind, preliminary files ~1-3 days). This is an SST *product*
(bulk SST, optimally interpolated); ORCA-DL ``tos`` (GODAS 5 m proxy) must be evaluated against it only with that caveat.
"""
from __future__ import annotations
import os, urllib.error, urllib.request
import numpy as np, xarray as xr

BUCKET = "https://noaa-cdr-sea-surface-temp-optimum-interpolation-pds.s3.amazonaws.com"
NINO = {"nino34": (-5, 5, 190, 240), "nino3": (-5, 5, 210, 270), "nino4": (-5, 5, 160, 210), "nino12": (-10, 0, 270, 280)}

def day_url(day, preliminary=False) -> str:
    d = np.datetime64(day, "D").astype(object)
    return f"{BUCKET}/data/v2.1/avhrr/{d:%Y%m}/oisst-avhrr-v02r01.{d:%Y%m%d}{'_preliminary' if preliminary else ''}.nc"

def fetch_day(day, cache: str) -> str:
    os.makedirs(cache, exist_ok=True)
    d = np.datetime64(day, "D")
    for prelim in (False, True):
        p = os.path.join(cache, os.path.basename(day_url(d, prelim)))
        if os.path.exists(p): return p
        try:
            urllib.request.urlretrieve(day_url(d, prelim), p + ".part"); os.replace(p + ".part", p); return p
        except urllib.error.HTTPError:
            continue
    raise FileNotFoundError(f"OISST {d} not on S3 (neither final nor preliminary)")

def monthly_mean(month: str, cache: str = "data/_cache/oisst", box=None, min_days: int | None = None) -> xr.DataArray:
    """Mean SST [degC] of calendar month 'YYYY-MM' (lat ascending, lon 0..360); ``box`` = (lat0, lat1, lon0, lon1) crops before
    averaging. Requires every day of the month unless ``min_days`` is given (then the count is recorded in attrs)."""
    start = np.datetime64(month + "-01")
    days = np.arange(start, (start.astype("datetime64[M]") + 1).astype("datetime64[D]"))
    acc, n, prelim = None, 0, 0
    for d in days:
        try: p = fetch_day(d, cache)
        except FileNotFoundError:
            continue
        prelim += "_preliminary" in p
        x = xr.open_dataset(p).sst.isel(time=0, zlev=0, drop=True)
        if box: x = x.sel(lat=slice(box[0], box[1]), lon=slice(box[2], box[3]))
        x = x.load()
        acc = x if acc is None else acc + x
        n += 1
    need = len(days) if min_days is None else min_days
    if n < need: raise ValueError(f"OISST {month}: {n}/{len(days)} days available (need {need})")
    out = (acc / n).rename(latitude="lat") if "latitude" in acc.dims else acc / n
    return out.assign_attrs(units="degC", source="NOAA OISST v2.1 AVHRR (S3)", days=n, preliminary_days=prelim, definition="oisst_bulk")

def box_mean(da: xr.DataArray, box) -> float:
    lat0, lat1, lon0, lon1 = box
    x = da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    w = np.cos(np.deg2rad(x.lat))
    return float(x.weighted(w).mean(("lat", "lon")))

def index(da: xr.DataArray, name: str = "nino34") -> float:
    return box_mean(da, NINO[name])
