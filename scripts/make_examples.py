"""Regenerates every README image/table from real runs.   python scripts/make_examples.py <stage> ...
stages: aifs_latest | aurora | hours | week | days15 | latest_models"""
import sys, warnings, pathlib, numpy as np, xarray as xr
warnings.filterwarnings("ignore")
import weatherpre as wp
from weatherpre import maps, registry as R, horizon
from weatherpre.common import gcs_open
IMG = pathlib.Path("docs/img"); IMG.mkdir(parents=True, exist_ok=True)
RES = pathlib.Path("results/examples"); RES.mkdir(parents=True, exist_ok=True)
INIT = "2020-10-03"
AUR = pathlib.Path("data/forecasts/aurora_2020-10-03T00Z_360h.nc")
AUR_H = pathlib.Path("data/forecasts/aurora_2020-10-03T00Z_48h.nc")

def era5_z500(valid):
    ds = gcs_open(R.WB2_TRUTH); return (ds.geopotential.sel(time=np.datetime64(valid, "ns"), level=500) / 9.80665).sortby("latitude")

def aurora_extra(preset=None):
    f = AUR_H if preset == "hours" else AUR       # Aurora 0.25deg run on HF ZeroGPU (see run_aurora.py); optional
    if not f.exists(): return {}
    return {"aurora": xr.open_dataset(f).load()}

def horizon_stage(preset, lead_for_map, title):
    fc, table, info = horizon.compare("auto", INIT, preset, extra=aurora_extra(preset))
    print("truth:", info["truth"], "skipped:", info["skipped"])
    table.round(3).to_csv(RES / f"{INIT}_{preset}_scores.csv", index=False)
    truth = era5_z500(np.datetime64(INIT + "T00") + np.timedelta64(lead_for_map, "h"))
    maps.plot_models({m: d for m, d in fc.items() if d.sizes["latitude"] < 400 or m == "aurora"}, "z500", lead_for_map, truth, str(IMG / f"{INIT}_{preset}_z500_models.png"),
                     f"{title}: z500 +{lead_for_map} h, init {INIT} 00Z")
    maps.plot_scores(table, str(IMG / f"{INIT}_{preset}_scores.png"), f"RMSE vs {info['truth']}, init {INIT} 00Z")
    return fc, table

stage = sys.argv[1:]
if "aifs_latest" in stage:
    ds = xr.open_dataset(sorted(pathlib.Path("data/forecasts").glob("aifs-single_*_360h.nc"))[-1]).load()
    maps.plot_maps(ds, ("z500", "t2m", "tp"), [24, 120, 240, 360], str(IMG / "aifs_latest_maps.png"))
if "hours" in stage:
    fc, table = horizon_stage("hours", 24, "Next 48 h")
    maps.plot_maps(fc["ifs-hres"], ("z500", "t2m", "msl"), [6, 12, 24, 48], str(IMG / f"{INIT}_hours_hres_maps.png"))
if "week" in stage:   horizon_stage("week", 168, "Next week")
if "days15" in stage: horizon_stage("15days", 360, "Next 15 days")
if "latest_models" in stage:
    fc, table, info = horizon.compare("auto", "latest", "week")
    print(info["truth"], info["skipped"])
    if len(table): table.round(3).to_csv(RES / "latest_week_scores.csv", index=False)
    if "consensus" in info: info["consensus"].round(3).to_csv(RES / "latest_week_consensus.csv", index=False)
    init = next(iter(fc.values())).attrs["init"]
    maps.plot_models(fc, "z500", 120, None, str(IMG / "latest_week_z500_models.png"), f"Latest cycle {init}: z500 +120 h (no truth yet)")
    maps.plot_models(fc, "t2m", 120, None, str(IMG / "latest_week_t2m_models.png"), f"Latest cycle {init}: t2m +120 h (no truth yet)")
