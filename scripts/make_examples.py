"""Regenerates every README image/table from real runs.   python scripts/make_examples.py <stage> ...
stages: aifs_latest | aurora | hours | week | days15 | latest_models"""
import sys, warnings, pathlib, pickle, numpy as np, xarray as xr
warnings.filterwarnings("ignore")
import weatherpre as wp
from weatherpre import maps, registry as R, horizon
from weatherpre.common import gcs_open
IMG = pathlib.Path("docs/img"); IMG.mkdir(parents=True, exist_ok=True)
RES = pathlib.Path("results/examples"); RES.mkdir(parents=True, exist_ok=True)
INIT = "2020-10-03"
AUR = pathlib.Path("data/forecasts/aurora_2020-10-03T00Z_360h.nc")
AUR_H = pathlib.Path("data/forecasts/aurora_2020-10-03T00Z_48h.nc")
CACHE = pathlib.Path("data/cache"); CACHE.mkdir(parents=True, exist_ok=True)   # re-style plots without re-downloading

def cached(key, fn):
    f = CACHE / f"{key}.pkl"
    if f.exists(): return pickle.load(open(f, "rb"))
    out = fn(); pickle.dump(out, open(f, "wb")); return out

def era5_z500(valid):
    ds = gcs_open(R.WB2_TRUTH); return (ds.geopotential.sel(time=np.datetime64(valid, "ns"), level=500) / 9.80665).sortby("latitude").load()

def aurora_extra(preset=None):
    f = AUR_H if preset == "hours" else AUR       # Aurora 0.25deg run on HF ZeroGPU (see run_aurora.py); optional
    if not f.exists(): return {}
    return {"aurora": xr.open_dataset(f).load()}

def horizon_stage(preset, lead_for_map, title):
    def run():
        fc, table, info = horizon.compare("auto", INIT, preset, extra=aurora_extra(preset))
        return {m: d.load() for m, d in fc.items()}, table, info
    fc, table, info = cached(preset, run)
    print("truth:", info["truth"], "skipped:", info["skipped"])
    table.round(3).to_csv(RES / f"{INIT}_{preset}_scores.csv", index=False)
    valid = np.datetime64(INIT + "T00") + np.timedelta64(lead_for_map, "h")
    truth = cached(f"era5_z500_{str(valid)[:13]}", lambda: era5_z500(valid))
    maps.plot_models({m: d for m, d in fc.items() if d.sizes["latitude"] < 400 or m == "aurora"}, "z500", lead_for_map, truth,
                     str(IMG / f"{INIT}_{preset}_z500_models.png"), f"{title}: +{lead_for_map} h from {INIT} 00Z",
                     scores=table, subtitle=f"500 hPa geopotential height [dam], valid {str(valid)[:13].replace('T', ' ')}Z  ·  "
                     f"ERA5 = verifying analysis  ·  bold line = 588 dam (subtropical high edge)")
    maps.plot_scores(table, str(IMG / f"{INIT}_{preset}_scores.png"), f"{title}: forecast error vs lead time, init {INIT} 00Z",
                     subtitle=f"area-weighted RMSE vs {info['truth'].replace(' (WeatherBench 2)', '')}, scored with WeatherBench-X  ·  solid = AI, dashed = NWP  ·  lower is better")
    return fc, table

stage = sys.argv[1:]
if "aifs_latest" in stage:
    ds = cached("aifs", lambda: wp.forecast("aifs", init="latest", lead_days=15).load())
    maps.plot_maps(ds, ("z500", "t2m", "tp"), [24, 120, 240, 360], str(IMG / "aifs_latest_maps.png"))
if "hours" in stage:
    fc, table = horizon_stage("hours", 24, "Next 48 h")
    maps.plot_maps(fc["ifs-hres"], ("z500", "t2m", "msl"), [6, 12, 24, 48], str(IMG / f"{INIT}_hours_hres_maps.png"))
if "week" in stage:   horizon_stage("week", 168, "Next week")
if "days15" in stage: horizon_stage("15days", 360, "Next 15 days")
if "latest_models" in stage:
    fc, table, info = cached("latest", lambda: horizon.compare("auto", "latest", "week"))
    print(info["truth"], info["skipped"])
    if len(table): table.round(3).to_csv(RES / "latest_week_scores.csv", index=False)
    if "consensus" in info: info["consensus"].round(3).to_csv(RES / "latest_week_consensus.csv", index=False)
    init = next(iter(fc.values())).attrs["init"]
    for v in ("z500", "t2m"):
        maps.plot_models(fc, v, 120, None, str(IMG / f"latest_week_{v}_models.png"), f"Latest cycle: +120 h from {str(init)[:13].replace('T', ' ')}Z",
                         subtitle=f"{maps.STYLE[v]['label']}  ·  valid times are in the future: no verification yet, compare the models by eye")
if "quickstart" in stage:     # README hero: one call, four models, scored against ERA5
    c = cached("quickstart", lambda: wp.compare(["graphcast", "pangu", "gefs", "ifs-hres"], "z500,t2m", "7d", init=INIT))
    c.summary().round(3).to_csv(RES / f"{INIT}_quickstart_scores.csv", index=False)
    c.plot(str(IMG / "quickstart_scores.png"), title=f"wp.compare(['graphcast', 'pangu', 'gefs', 'ifs-hres'], 'z500,t2m', '7d', init='{INIT}')")
    c.plot_maps(str(IMG / "quickstart_z500_maps.png"), "z500", 120)
    print(c); print(c.ranking())
if "s2s" in stage:            # S2S: every Monday/Thursday init of October 2020, weeks 1-6, vs ERA5 weekly means
    S2S_MODELS = ["ifs-ext", "gefs", "cfsv2", "persistence", "climatology"]
    def run_s2s():
        c = wp.compare(S2S_MODELS, "t2m,z500,tp", "6w", init="2020-10-01..2020-10-29")
        c.forecasts = {m: d.load() for m, d in c.forecasts.items()}; return c
    c = cached("s2s_oct2020", run_s2s)
    c.scores.round(4).to_csv(RES / "s2s_2020-10_scores_per_init.csv", index=False)
    c.summary().round(3).to_csv(RES / "s2s_2020-10_scores.csv", index=False)
    c.plot(str(IMG / "s2s_2020-10_scores.png"), title=f"S2S weekly means: {len(c.inits)} inits, Oct 2020 (Mon/Thu), scored vs ERA5")
    for wk in (1, 3):
        c.plot_maps(str(IMG / f"s2s_2020-10-01_week{wk}_t2m.png"), "t2m", wk)
    print(c); print(c.ranking("acc")); print("skipped:", c.skipped)
