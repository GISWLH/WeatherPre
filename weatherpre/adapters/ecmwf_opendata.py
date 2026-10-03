"""ECMWF Open Data (IFS HRES/ENS, AIFS-single, AIFS-ENS) via the official `ecmwf-opendata` client."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
from ecmwf.opendata import Client

# our model name -> opendata `model`, `stream`
MODELS = {"ifs-hres": ("ifs", "oper"), "aifs-single": ("aifs-single", "oper")}

def _levels(model):  # 500 hPa z, 850 hPa t
    return dict(pl=["gh" if model == "ifs" else "z", "t"], levs=[500, 850])

def latest_init(model="aifs-single", max_lead=0) -> dt.datetime:
    """Newest cycle whose file for `max_lead` already exists (06/18Z runs only reach ~144 h)."""
    m, stream = MODELS[model]
    return Client(source="ecmwf", model=m).latest(type="fc", stream=stream, step=int(max_lead),
                                                  param="2t", levtype="sfc")

def fetch(model: str, init: dt.datetime, leads: list[int], out: Path) -> list[Path]:
    """Download z500,t850,2t,msl for given leads (byte-range via .index; ~few MB each)."""
    m, stream = MODELS[model]
    cl = Client(source="ecmwf", model=m)
    out.mkdir(parents=True, exist_ok=True); files = []
    for h in leads:
        for tag, kw in (("sfc", dict(param=["2t", "msl"] + (["tp"] if h > 0 else []))),
                        ("pl", dict(param=["gh" if m == "ifs" else "z", "t"], levelist=[500, 850]))):
            f = out / f"{model}_{init:%Y%m%dT%H}_f{h:03d}_{tag}.grib2"
            if not f.exists():
                cl.retrieve(date=init.strftime("%Y%m%d"), time=init.hour, step=h, stream=stream,
                            type="fc", target=str(f), **kw)
            files.append(f)
    return files
