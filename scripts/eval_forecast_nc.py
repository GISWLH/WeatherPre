"""Score a forecast.nc (WB2 schema, 0.25 deg) against WB2 ERA5 0.25 deg with WeatherBench-X (RMSE+ACC); optional reference HRES."""
import sys, warnings, numpy as np, xarray as xr, pandas as pd
warnings.filterwarnings("ignore")
from weatherpre.common import gcs_open
from weatherpre.evaluate import evaluate
B = "gs://weatherbench2/datasets/"
nc, out = sys.argv[1], sys.argv[2]
ds = xr.open_dataset(nc)
init = ds.time.values
leads = (ds.prediction_timedelta.values / np.timedelta64(1, "h")).astype(int).tolist()
tgt = gcs_open(B + "era5/1959-2022-6h-1440x721.zarr")
clim = gcs_open(B + "era5-hourly-climatology/1990-2019_6h_1440x721.zarr").sortby("latitude")
ref = gcs_open(B + "hres/2016-2022-0012-1440x721.zarr")
rows = {}
for name, p in (("aurora", ds), ("ifs-hres", ref)):
    L = [h for h in leads if h % 12 == 0] if name == "ifs-hres" else leads
    r = evaluate(p, tgt, init, L, ("z500", "t850", "t2m"), clim)
    d = r.to_dataframe().reset_index(); d.insert(0, "model", name); rows[name] = d
df = pd.concat(rows.values()); df["lead_h"] = (df.lead_time / pd.Timedelta(hours=1)).astype(int); df = df.drop(columns="lead_time")
df.round(4).to_csv(out, index=False); print(df.round(3).to_string(index=False))
