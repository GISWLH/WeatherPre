"""Aurora (microsoft-aurora, MIT weights) fed with real initial conditions.
   historic init -> ERA5 from WeatherBench 2 (anonymous GCS, 0.25 deg)   -> aurora-0.25-pretrained
   recent   init -> IFS HRES analysis (ECMWF open data step 0)           -> aurora-0.25-finetuned (HRES-T0 trained)
The model code, rollout and checkpoints are the upstream package's; this file only builds the input Batch."""
from __future__ import annotations
import datetime as dt, pickle
from pathlib import Path
import numpy as np, torch, xarray as xr

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
WB2_ERA5_025 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
SURF = {"2t": "2m_temperature", "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure"}
ATMOS = {"z": "geopotential", "u": "u_component_of_wind", "v": "v_component_of_wind", "t": "temperature", "q": "specific_humidity"}

def _static():
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download("microsoft/aurora", "aurora-0.25-static.pickle"), "rb") as f:
        return pickle.load(f)

def _batch(surf, atmos, lat, lon, t_last):
    from aurora import Batch, Metadata
    st = _static()
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    return Batch(
        surf_vars={k: T(v)[None] for k, v in surf.items()},          # (1, 2, H, W)
        static_vars={k: T(v) for k, v in st.items()},
        atmos_vars={k: T(v)[None] for k, v in atmos.items()},        # (1, 2, C, H, W)
        metadata=Metadata(lat=T(lat), lon=T(lon), time=(t_last,), atmos_levels=LEVELS))

def batch_from_era5(init: dt.datetime):
    from ..common import gcs_open
    ds = gcs_open(WB2_ERA5_025)
    t = [np.datetime64(init - dt.timedelta(hours=6), "ns"), np.datetime64(init, "ns")]
    ds = ds.sel(time=t, level=list(LEVELS))
    surf = {k: ds[v].transpose("time", "latitude", "longitude").values for k, v in SURF.items()}
    atmos = {k: ds[v].transpose("time", "level", "latitude", "longitude").values for k, v in ATMOS.items()}
    lat = ds.latitude.values; lon = ds.longitude.values
    if lat[0] < lat[-1]:     # Aurora wants 90 -> -90
        surf = {k: v[..., ::-1, :] for k, v in surf.items()}; atmos = {k: v[..., ::-1, :] for k, v in atmos.items()}; lat = lat[::-1]
    return _batch(surf, atmos, lat, lon, init)

def batch_from_ifs(init: dt.datetime, cache: Path):
    """Two consecutive IFS HRES analyses (step 0 of cycles init-6h, init) via ecmwf-opendata."""
    from ecmwf.opendata import Client
    cl = Client(source="ecmwf", model="ifs"); cache.mkdir(parents=True, exist_ok=True)
    import cfgrib
    surf, atmos = {k: [] for k in SURF}, {k: [] for k in ATMOS}
    for t in (init - dt.timedelta(hours=6), init):
        f1, f2 = cache / f"ifs_{t:%Y%m%dT%H}_sfc.grib2", cache / f"ifs_{t:%Y%m%dT%H}_pl.grib2"
        if not f1.exists(): cl.retrieve(date=t.strftime("%Y%m%d"), time=t.hour, step=0, stream="oper", type="fc", param=["2t", "10u", "10v", "msl"], target=str(f1))
        if not f2.exists(): cl.retrieve(date=t.strftime("%Y%m%d"), time=t.hour, step=0, stream="oper", type="fc", param=["gh", "u", "v", "t", "q"], levelist=list(LEVELS), target=str(f2))
        for d in cfgrib.open_datasets(str(f1), backend_kwargs={"indexpath": ""}):
            for k, g in (("2t", "t2m"), ("10u", "u10"), ("10v", "v10"), ("msl", "msl")):
                if g in d: surf[k].append(d[g].values)
        for d in cfgrib.open_datasets(str(f2), backend_kwargs={"indexpath": ""}):
            lat, lon = d.latitude.values, d.longitude.values % 360
            for k, g in (("z", "gh"), ("u", "u"), ("v", "v"), ("t", "t"), ("q", "q")):
                if g in d:
                    a = d[g].sel(isobaricInhPa=list(LEVELS)).values
                    atmos[k].append(a * 9.80665 if k == "z" else a)
    order = np.argsort(lon)
    surf = {k: np.stack(v)[..., order] for k, v in surf.items()}
    atmos = {k: np.stack(v)[..., order] for k, v in atmos.items()}
    return _batch(surf, atmos, lat, lon[order], init)

def forecast(batch, steps: int, variant="pretrained", device="cpu") -> xr.Dataset:
    from aurora import Aurora, AuroraPretrained, AuroraSmallPretrained, rollout
    if variant == "small":
        model = AuroraSmallPretrained(); model.load_checkpoint()
    elif variant == "finetuned":
        model = Aurora(); model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
    else:
        model = AuroraPretrained(); model.load_checkpoint()
    model = model.to(device).eval(); batch = batch.to(device)
    outs, leads = [], []
    with torch.inference_mode():
        for i, p in enumerate(rollout(model, batch, steps=steps), 1):
            outs.append(p.to("cpu")); leads.append(6 * i)
    lat = outs[0].metadata.lat.numpy(); lon = outs[0].metadata.lon.numpy()
    lv = list(outs[0].metadata.atmos_levels); li = [lv.index(500), lv.index(850)]
    g = lambda f: xr.DataArray(np.stack([f(p) for p in outs])[None].astype("float32"))
    dims_s = ("time", "prediction_timedelta", "latitude", "longitude"); dims_a = ("time", "prediction_timedelta", "level", "latitude", "longitude")
    ds = xr.Dataset({
        "geopotential": (dims_a, np.stack([p.atmos_vars["z"][0, 0][li].numpy() for p in outs])[None]),
        "temperature": (dims_a, np.stack([p.atmos_vars["t"][0, 0][li].numpy() for p in outs])[None]),
        "2m_temperature": (dims_s, np.stack([p.surf_vars["2t"][0, 0].numpy() for p in outs])[None]),
        "mean_sea_level_pressure": (dims_s, np.stack([p.surf_vars["msl"][0, 0].numpy() for p in outs])[None]),
    }, coords=dict(time=[np.datetime64(batch.metadata.time[0], "ns")],
                   prediction_timedelta=np.array(leads, "timedelta64[h]").astype("timedelta64[ns]"),
                   level=[500, 850], latitude=lat, longitude=lon))
    return ds.sortby("latitude")

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--source", choices=["era5", "ifs"], required=True); ap.add_argument("--variant", default="pretrained", choices=["small", "pretrained", "finetuned"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ..common import parse_time
    init = parse_time(a.init)
    b = batch_from_era5(init) if a.source == "era5" else batch_from_ifs(init, Path(a.out).parent / "_ifs")
    ds = forecast(b, a.steps, a.variant, a.device); ds.to_netcdf(a.out); print("wrote", a.out, dict(ds.sizes))

if __name__ == "__main__":
    main()
