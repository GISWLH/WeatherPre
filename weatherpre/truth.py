"""Truth: ERA5 (WB2) for historic dates; IFS analysis (open-data step 0) as a *proxy* for recent dates."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import xarray as xr
from . import grib
from .adapters import ecmwf_opendata

def ifs_analysis(valid_times: list[dt.datetime], cache: Path) -> xr.Dataset:
    """WB2-schema targets (dims time=valid, level, lat, lon) from IFS HRES step 0 of each cycle."""
    parts = []
    for v in valid_times:
        files = ecmwf_opendata.fetch("ifs-hres", v, [0], cache)
        ds = grib.read(files, v).isel(prediction_timedelta=0, drop=True)
        parts.append(ds)
    return xr.concat(parts, dim="time")
