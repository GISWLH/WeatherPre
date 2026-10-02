"""Shared helpers: standard variable set + a tiny GRIB->xarray reader (cfgrib)."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import xarray as xr

# Standard comparison fields (name -> (grib shortName, typeOfLevel, level))
STD = {
    "z500": ("z", "isobaricInhPa", 500),
    "t850": ("t", "isobaricInhPa", 850),
    "t2m": ("2t", "surface", None),
    "msl": ("msl", "surface", None),
}

def parse_time(s: str) -> dt.datetime:
    s = s.replace("Z", "")
    for f in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y%m%d%H", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, f)
        except ValueError:
            pass
    raise ValueError(f"bad time {s!r}; use YYYY-MM-DDTHH")

def parse_leads(s: str) -> list[int]:
    """'0-72/24' or '24,48,72' -> hours."""
    if "," in s:
        return [int(x) for x in s.split(",")]
    if "-" in s:
        a, rest = s.split("-")
        b, _, st = rest.partition("/")
        return list(range(int(a), int(b) + 1, int(st or 6)))
    return [int(s)]

def outdir(root: str | Path, model: str, init: dt.datetime) -> Path:
    p = Path(root) / model / init.strftime("%Y%m%dT%H")
    p.mkdir(parents=True, exist_ok=True)
    return p


def gcs_open(path: str):
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    return xr.open_zarr(fs.get_mapper(path.replace("gs://", "")), chunks=None, decode_timedelta=True)


