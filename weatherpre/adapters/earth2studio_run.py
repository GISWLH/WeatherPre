"""Thin wrapper over NVIDIA Earth2Studio (`pip install 'earth2studio[<model>]'`): upstream model wrapper + upstream
data source + upstream `run.deterministic`; we only pick the combination and convert the output to our netCDF layout.
STATUS: written from Earth2Studio's documented API; NOT executed yet (needs a GPU runtime and the per-model extra)."""
from __future__ import annotations
import datetime as dt
from collections import OrderedDict
import numpy as np

SOURCES = {"gfs": "GFS", "ifs": "IFS", "arco": "ARCO", "wb2": "WB2ERA5"}   # earth2studio.data classes
KEEP = ("z500", "t850", "t2m", "msl")

def run_e2s(model: str, init: str, nsteps: int, source: str, out: str):
    import earth2studio.models.px as px, earth2studio.data as D
    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic
    from .. import registry as R
    from ..common import parse_time
    cls = getattr(px, R.EARTH2STUDIO.get(f"e2s-{model}", model))
    m = cls.load_model(cls.load_default_package())
    data = getattr(D, SOURCES[source])()
    if init == "latest":
        from . import noaa_s3
        init_t = noaa_s3.latest_init("gfs") - dt.timedelta(hours=6)
    else:
        init_t = parse_time(init)
    have = list(m.output_coords(m.input_coords())["variable"])
    sel = [v for v in KEEP if v in have]
    io = deterministic([np.datetime64(init_t)], nsteps, m, data, ZarrBackend(),
                       output_coords=OrderedDict({"variable": np.array(sel)}))
    import xarray as xr
    ds = xr.Dataset({v: (("time", "lead_time", "lat", "lon"), io[v][:]) for v in sel},
                    coords=dict(time=[np.datetime64(init_t)], lead_time=io["lead_time"][:], lat=io["lat"][:], lon=io["lon"][:]))
    ds.to_netcdf(out)
    return out
