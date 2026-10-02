"""NOAA hosted products on public S3 (anonymous HTTPS, byte-range via the .idx sidecar):
   * AIGFS  (operational AI-GFS)  bucket noaa-nws-graphcastgfs-pds, prefix aigfs.YYYYMMDD/HH/model/atmos/grib2/
   * GFS    (NWP baseline/analysis) bucket noaa-gfs-bdp-pds,       prefix gfs.YYYYMMDD/HH/atmos/
"""
from __future__ import annotations
import datetime as dt, re, urllib.request, urllib.error
from pathlib import Path
import numpy as np, xarray as xr

AIGFS_B = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"
GFS_B = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# (grib idx regex, WB2-name, level)
WANT = [(r":HGT:500 mb:", "geopotential", 500), (r":HGT:850 mb:", "geopotential", 850), (r":TMP:500 mb:", "temperature", 500), (r":TMP:850 mb:", "temperature", 850),
        (r":TMP:2 m above ground:", "2m_temperature", None), (r":PRMSL:mean sea level:", "mean_sea_level_pressure", None)]

def _urls(model, init, h):
    d, c = f"{init:%Y%m%d}", f"{init:%H}"
    if model == "aigfs":
        p = f"{AIGFS_B}/aigfs.{d}/{c}/model/atmos/grib2/aigfs.t{c}z"
        return [f"{p}.pres.f{h:03d}.grib2", f"{p}.sfc.f{h:03d}.grib2"]
    if model == "gfs":
        return [f"{GFS_B}/gfs.{d}/{c}/atmos/gfs.t{c}z.pgrb2.0p25.f{h:03d}"]
    raise KeyError(model)

def _get(url, rng=None) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

def available(model, init, h) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(_urls(model, init, h)[0] + ".idx" if model == "gfs" else _urls(model, init, h)[0] + ".idx", method="HEAD"), timeout=30)
        return True
    except urllib.error.HTTPError:
        return False

def latest_init(model="aigfs", now: dt.datetime | None = None) -> dt.datetime:
    now = (now or dt.datetime.utcnow()).replace(minute=0, second=0, microsecond=0)
    t = now - dt.timedelta(hours=now.hour % 6)
    for _ in range(12):
        if available(model, t, 0) and available(model, t, 6):
            return t
        t -= dt.timedelta(hours=6)
    raise RuntimeError(f"no {model} cycle found in last 3 days")

def fetch(model: str, init: dt.datetime, leads: list[int], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True); files = []
    for h in leads:
        f = out / f"{model}_{init:%Y%m%dT%H}_f{h:03d}.grib2"
        if not f.exists():
            buf = b""
            for u in _urls(model, init, h):
                idx = _get(u + ".idx").decode().strip().splitlines()
                rows = [l.split(":") for l in idx]
                offs = [int(r[1]) for r in rows]
                for pat, _, _ in WANT:
                    for i, l in enumerate(idx):
                        if re.search(pat, l):
                            if "ACC" in l or "acc" in l: continue
                            a = offs[i]; b = offs[i + 1] - 1 if i + 1 < len(offs) else ""
                            buf += _get(u, f"{a}-{b}")
                            break
            f.write_bytes(buf)
        files.append(f)
    return files
