"""Baseline hindcast of monthly Nino3.4 SST: anomaly persistence vs calibration-period climatology (no AI model involved).

    python scripts/nino34_baseline_hindcast.py results/validation/nino34_baseline

Observation reference: ERA5 sea_surface_temperature, WeatherBench2 6-hourly 1.5 deg store; each monthly mean is estimated from
the 24 six-hourly samples in the 2-day chunks containing days 1, 11 and 21 (SST varies slowly; documented approximation).
Split: calibration 1991-01..2010-12 (climatology), test 2011-01..2022-12 (scored). Init = first day of a month (common
init_time = end of the initial-condition month); lead 1 = the init month."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weatherpre import hindcast as H
from weatherpre.common import gcs_open

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
csv = os.path.join(out, "era5_nino34_monthly.csv")
if os.path.exists(csv):
    obs = pd.read_csv(csv, index_col=0, parse_dates=True)["nino34"]
else:
    ds = gcs_open("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
    sst = ds.sea_surface_temperature.sel(latitude=slice(-5, 5), longitude=slice(190, 240))
    t0 = np.datetime64("1959-01-01T00", "h")
    w = np.cos(np.deg2rad(sst.latitude))
    vals = {}
    for m in pd.date_range("1991-01-01", "2022-12-01", freq="MS"):
        idx = []
        for d in (1, 11, 21):
            k = int((np.datetime64(m.replace(day=d), "h") - t0) / np.timedelta64(6, "h")) // 8 * 8
            idx += list(range(k, k + 8))
        x = sst.isel(time=idx).load()
        vals[m] = float(x.weighted(w).mean(("latitude", "longitude")).mean("time")) - 273.15
        if m.month == 12: print(m.year, round(vals[m], 2), flush=True)
    obs = pd.Series(vals, name="nino34"); obs.to_csv(csv)
split = H.Split(train=None, calibration=("1991-01", "2010-12"), test=("2011-01", "2022-12"))
clim = H.climatology(obs, split)
inits = pd.date_range("2011-02-01", "2022-12-01", freq="MS")
leads = list(range(1, 7))
pers = H.persistence_forecast(obs, clim, inits, leads)
cfc = H.climatology_forecast(clim, inits, leads)
tab_p = H.evaluate_index(pers, obs, split, refs={"climatology": cfc}, clim=clim)
tab_c = H.evaluate_index(cfc, obs, split, refs={"persistence": pers}, clim=clim)
anom = obs - np.array([clim[t.month] for t in obs.index])
res = {"observation": "ERA5 SST (WB2 6-hourly 1.5 deg), monthly means from 24 samples/month", "split": str(split),
       "anomaly_checks_vs_1991_2010": {k: round(float(anom[k]), 2) for k in ("2015-12-01", "2010-12-01", "1997-12-01", "2020-12-01")},
       "persistence": tab_p.to_dict("records"), "climatology": tab_c.to_dict("records"),
       "note": "baseline skill only: the bar an AI SST model must clear on the same split; not a model result"}
print(tab_p.round(3).to_string(index=False)); print(json.dumps(res["anomaly_checks_vs_1991_2010"]))
json.dump(res, open(os.path.join(out, "nino34_baseline_hindcast.json"), "w"), indent=1, default=str)
