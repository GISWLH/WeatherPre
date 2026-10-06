"""NVIDIA Earth2Studio route (`pip install earth2studio[<model>]`, Python >= 3.11, a CUDA GPU for real models).

We only choose the combination of upstream pieces -- model wrapper (`earth2studio.models.px.<Class>`), initial-condition
source (`earth2studio.data`) and the upstream `earth2studio.run.deterministic` loop -- and convert the output to the
WeatherPre schema. Any Earth2Studio prognostic model can be used directly as ``"e2s:<ClassName>"``.

Tested end-to-end on CPU with the upstream `Persistence` model + `WB2ERA5_121x240` initial conditions (see tests);
the large models need their extra (e.g. ``earth2studio[fcn3]``) and a GPU."""
from __future__ import annotations
import datetime as dt, math, os
from collections import OrderedDict
import numpy as np, xarray as xr

G = 9.80665
# WeatherPre variable -> (Earth2Studio variable ids to try, scale to user units, units, long name)
OUT = {"z500": (("z500",), 1 / G, "m", "500 hPa geopotential height"),
       "t850": (("t850",), 1.0, "K", "850 hPa temperature"),
       "t2m": (("t2m",), 1.0, "K", "2 m temperature"),
       "msl": (("msl",), 0.01, "hPa", "mean sea-level pressure"),
       "tp": (("tp", "tp06", "tp12", "tp24"), 1000.0, "mm", "precipitation accumulated since init")}
ECMWF_IC = {"AIFS", "AIFS2", "AIFSENS", "AIFS2ENS"}          # upstream recommends IFS initial conditions for these

def available() -> bool:
    try:
        import earth2studio  # noqa: F401
        return True
    except ImportError:
        return False

def device():
    import torch
    if torch.cuda.is_available(): return torch.device("cuda")
    if os.environ.get("WEATHERPRE_E2S_CPU"): return torch.device("cpu")
    return None

def _ic_source(cls_name: str, init, source: str | None):
    import earth2studio.data as D
    if source: return getattr(D, source)()
    if init == "latest" or (dt.datetime.utcnow() - init) < dt.timedelta(days=5):
        return D.IFS() if cls_name in ECMWF_IC else D.GFS()
    if cls_name in ("Persistence",): return D.WB2ERA5_121x240()
    return D.ARCO()                                           # ERA5 0.25 deg, anonymous GCS, 1940 -> ~3 months ago

def _build(cls_name: str, variables):
    import earth2studio.models.px as px
    if cls_name == "Persistence":                             # tiny upstream model: lets the route run on CPU / in CI
        lat, lon = np.linspace(90, -90, 121), np.linspace(0, 358.5, 240)
        return px.Persistence(variable=[OUT[v][0][0] for v in variables if v != "tp"],
                              domain_coords=OrderedDict(lat=lat, lon=lon))
    cls = getattr(px, cls_name)
    return cls.load_model(cls.load_default_package())

def forecast(cls_name: str, init, lead_hours, variables=("z500", "t850", "t2m", "msl", "tp"), source: str | None = None,
             model_name: str | None = None) -> xr.Dataset:
    """Run `earth2studio.models.px.<cls_name>` and return the WeatherPre weather schema (lead [h], latitude, longitude)."""
    from earth2studio.io import XarrayBackend
    from earth2studio.run import deterministic
    dev = device()
    if dev is None and cls_name != "Persistence":
        from ..api import BackendUnavailable
        raise BackendUnavailable(f"{model_name or cls_name}: Earth2Studio needs a CUDA GPU (set WEATHERPRE_E2S_CPU=1 to force "
                                 "CPU, very slow). Use Colab (notebooks/colab_earth2studio.ipynb) or any GPU box.")
    model = _build(cls_name, variables)
    if init == "latest":
        from . import noaa_s3
        init = noaa_s3.latest_init("gfs", 0)
    have = list(model.output_coords(model.input_coords())["variable"])
    step_h = int(model.output_coords(model.input_coords())["lead_time"][-1] / np.timedelta64(1, "h"))
    pick = {v: next((e for e in OUT[v][0] if e in have), None) for v in variables if v in OUT}
    pick = {k: e for k, e in pick.items() if e}
    nsteps = math.ceil(max(lead_hours) / step_h)
    io = XarrayBackend()
    deterministic([np.datetime64(init)], nsteps, model, _ic_source(cls_name, init, source), io,
                  output_coords=OrderedDict(variable=np.array(list(pick.values()))), device=dev)
    raw = io.root.isel(time=0)
    lead = (raw.lead_time.values / np.timedelta64(1, "h")).astype(int)
    raw = raw.assign_coords(lead_time=lead).rename(lead_time="lead", lat="latitude", lon="longitude")
    out = {}
    for v, e in pick.items():
        _, k, units, ln = OUT[v]
        da = raw[e] * k
        if v == "tp": da = da.fillna(0).cumsum("lead")               # per-step accumulation -> since init
        out[v] = da.astype("float32").assign_attrs(units=units, long_name=ln)
    u = xr.Dataset(out).drop_vars("time", errors="ignore")
    keep = [h for h in lead_hours if h in u.lead.values]
    u = u.sel(lead=keep) if keep else u
    it = np.datetime64(init, "ns")
    u = u.assign_coords(init_time=it, valid_time=("lead", it + u.lead.values.astype("timedelta64[h]").astype("timedelta64[ns]")))
    u["lead"].attrs["units"] = "hours"
    u.attrs.update(model=model_name or f"e2s:{cls_name}", backend="e2s", init=f"{init:%Y-%m-%dT%HZ}",
                   source=f"Earth2Studio {cls_name}, ICs {type(_ic_source(cls_name, init, source)).__name__}")
    return u.sortby("latitude").transpose("lead", "latitude", "longitude")

def run_e2s(model: str, init: str, nsteps: int, source: str, out: str):
    """Kept for the Colab notebook: run `model` for nsteps*6 h and write netCDF."""
    from ..catalog import get
    from ..common import parse_time
    srcs = {"gfs": "GFS", "ifs": "IFS", "arco": "ARCO", "wb2": "WB2ERA5"}
    try: cls = get(model).e2s or model
    except KeyError: cls = model
    t = "latest" if init == "latest" else parse_time(init)
    forecast(cls, t, list(range(6, 6 * nsteps + 1, 6)), source=srcs.get(source, source), model_name=model).to_netcdf(out)
    return out
