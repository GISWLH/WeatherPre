"""Aurora (0.25deg) for 2020-10-03 on HF ZeroGPU via the one-call API; saves the netCDF that make_examples.py picks up.
   HF_TOKEN=... python scripts/run_aurora.py days15|hours"""
import sys, warnings; warnings.filterwarnings("ignore")
import weatherpre as wp
if sys.argv[1] == "hours":
    ds = wp.forecast("aurora", init="2020-10-03", lead_hours=list(range(6, 49, 6))); out = "data/forecasts/aurora_2020-10-03T00Z_48h.nc"
else:
    ds = wp.forecast("aurora", init="2020-10-03", lead_days=15); out = "data/forecasts/aurora_2020-10-03T00Z_360h.nc"
ds.to_netcdf(out); print(ds)
