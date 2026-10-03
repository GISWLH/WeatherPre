"""Run Aurora on the HF ZeroGPU Space via gradio_client. Token: env HF_TOKEN, else the cached huggingface login. Never printed."""
from __future__ import annotations
import datetime as dt, os, tempfile, time
import numpy as np, xarray as xr

SPACE = os.environ.get("WEATHERPRE_SPACE", "LonghaoWang/weatherai-graphcast-smoke")
ERA5_LAST = dt.datetime(2021, 12, 31)       # last init with 6 h ERA5 in the WB2 0.25deg store used for ICs (verified for 1.5deg store; 0.25deg assumed)

def aurora(t, leads):
    from gradio_client import Client
    from huggingface_hub import get_token
    if t == "latest":
        from . import ecmwf_opendata
        t = ecmwf_opendata.latest_init("ifs-hres", 0)
    source, variant = ("era5", "pretrained") if t <= ERA5_LAST else ("ifs", "finetuned")
    c = Client(SPACE, token=os.environ.get("HF_TOKEN") or get_token())
    parts, first = [], True
    while True:                      # one GPU chunk per call (ZeroGPU proxy token is per request); state is kept on the Space
        for attempt in range(4):     # a ZeroGPU task can be aborted: the Space keeps the rollout state, so just continue
            try:
                f, msg = c.predict(f"{t:%Y-%m-%dT%H}", source, ",".join(str(h) for h in leads), variant, first, api_name="/weatherpre_aurora")
                if f is None and not first and attempt < 3: raise RuntimeError(msg.strip().splitlines()[-1])
                break
            except Exception as e:
                if first or attempt == 3: raise
                print(f"[weatherpre] HF chunk failed ({str(e)[:80]}), retrying", flush=True); time.sleep(30)
        if f is None: raise RuntimeError(msg)
        print("[weatherpre] HF:", msg[:200]); first = False
        d = xr.load_dataset(f)
        if d.sizes.get("lead"): parts.append(d)
        if msg.startswith("OK"): break
    u = xr.concat(parts, "lead")
    it = np.datetime64(t, "ns")
    u = u.assign_coords(init_time=it, valid_time=("lead", it + u.lead.values.astype("timedelta64[h]").astype("timedelta64[ns]")))
    u.attrs.update(model="aurora", backend="hf", init=f"{t:%Y-%m-%dT%HZ}",
                   source=f"microsoft-aurora {variant} on HF ZeroGPU, {source} initial conditions, 0.25deg")
    for k, (un, ln) in {"z500": ("m", "500 hPa geopotential height"), "t850": ("K", "850 hPa temperature"), "t2m": ("K", "2 m temperature"), "msl": ("hPa", "mean sea-level pressure")}.items():
        u[k].attrs.update(units=un, long_name=ln)
    return u
