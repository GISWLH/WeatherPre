"""Validate weatherpre.sources.era5_daily against the official FuXi-S2S sample (ERA5 2020-06-01/02), all 76 channels.

    python scripts/validate_fuxi_inputs.py /path/to/FuXi-S2S/data results/validation/fuxi_inputs_2020-06-02.json
Metric per channel: RMSE / spatial std of the sample (NaN-aware), plus area-weighted global means."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weatherpre.sources import era5_daily as E
from weatherai.inference.fuxi_s2s import CHANNELS, FuXiS2SInputs, check_inputs

data, out = sys.argv[1], sys.argv[2]
ref = FuXiS2SInputs.from_official_files(os.path.join(data, "sample"), os.path.join(data, "mask.nc"))
got = E.fuxi_inputs("2020-06-02", cache_dir=os.path.join(os.path.dirname(out) or ".", "_cache"))
w = np.cos(np.deg2rad(E.LAT))[:, None] * np.ones((1, 240))
rows = {}
for t in range(2):
    for c, name in enumerate(CHANNELS):
        a, b = got.x[t, c].astype(np.float64), ref.x[t, c].astype(np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        rows[f"{str(ref.times[t])}|{name}"] = dict(
            rmse_over_std=float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)) / (np.std(b[ok]) + 1e-30)),
            gmean_built=float((a[ok] * w[ok]).sum() / w[ok].sum()), gmean_sample=float((b[ok] * w[ok]).sum() / w[ok].sum()),
            nan_built=int((~np.isfinite(a)).sum()), nan_sample=int((~np.isfinite(b)).sum()))
worst = sorted(rows.items(), key=lambda kv: -kv[1]["rmse_over_std"])[:8]
summary = {"checks_pass": bool(check_inputs(got) is not None), "max_rmse_over_std": worst[0][1]["rmse_over_std"],
           "worst": {k: round(v["rmse_over_std"], 4) for k, v in worst},
           "median_rmse_over_std": float(np.median([v["rmse_over_std"] for v in rows.values()]))}
print(json.dumps(summary, indent=1))
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
json.dump({"summary": summary, "channels": rows, "note": "sample = official FuXi-S2S ERA5 inputs (regridding method unknown); built = "
           "WB2 hourly 1.5 deg conservative + ARCO 100 m winds block-averaged"}, open(out, "w"), indent=1)
