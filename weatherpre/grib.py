"""GRIB files (z500,t850,2t,msl in any mix) -> WeatherBench2-schema Dataset (time, prediction_timedelta, level?, latitude, longitude)."""
from __future__ import annotations
import warnings
import numpy as np, xarray as xr

def _std(da: xr.DataArray) -> xr.DataArray:
    da = da.rename({k: v for k, v in {"latitude": "latitude", "longitude": "longitude"}.items() if k in da.dims})
    lon = (da.longitude % 360)
    da = da.assign_coords(longitude=lon).sortby("longitude").sortby("latitude")
    return da

def read(files, init) -> xr.Dataset:
    import cfgrib
    warnings.filterwarnings("ignore")
    rows = {}   # (var) -> {lead_hours: DataArray}
    for f in files:
        for ds in cfgrib.open_datasets(str(f), backend_kwargs={"indexpath": ""}):
            for v in ds.data_vars:
                da = ds[v]
                lead = int(np.timedelta64(da.step.values, "ns") / np.timedelta64(1, "h"))
                if "isobaricInhPa" in da.dims or "isobaricInhPa" in da.coords:
                    if v in ("z", "gh", "t"):
                        name = "temperature" if v == "t" else "geopotential"
                        levs = np.atleast_1d(da.isobaricInhPa.values).astype(int)
                        for lev in levs:
                            if lev not in (500, 850): continue
                            x = da.sel(isobaricInhPa=lev) if "isobaricInhPa" in da.dims else da
                            x = x.drop_vars("isobaricInhPa", errors="ignore")
                            if v == "gh": x = x * 9.80665          # gpm -> m2 s-2
                            rows.setdefault(name, {}).setdefault(lead, {})[int(lev)] = x
                elif v == "t2m":
                    rows.setdefault("2m_temperature", {}).setdefault(lead, {})[None] = da
                elif v == "tp":     # accumulated since forecast start; mm (kg m-2) as delivered, metres converted
                    rows.setdefault("total_precipitation", {}).setdefault(lead, {})[None] = da * (1000.0 if str(da.attrs.get("GRIB_units", "")).strip() == "m" else 1.0)
                elif v in ("msl", "prmsl"):
                    rows.setdefault("mean_sea_level_pressure", {}).setdefault(lead, {})[None] = da
    out = {}
    for name, per in rows.items():
        leads = sorted(per)
        levs = sorted({l for h in leads for l in per[h] if l is not None})
        def clean(da):
            return _std(da.drop_vars([c for c in da.coords if c not in ("latitude", "longitude")], errors="ignore"))
        arrs = []
        for h in leads:
            if levs:
                arrs.append(xr.concat([clean(per[h][l]) for l in levs], dim=xr.DataArray(levs, dims="level")))
            else:
                arrs.append(clean(per[h][None]))
        a = xr.concat(arrs, dim=xr.DataArray(np.array(leads, "timedelta64[h]").astype("timedelta64[ns]"), dims="prediction_timedelta"))
        out[name] = a.expand_dims(time=[np.datetime64(init, "ns")]).astype("float32")
    return xr.Dataset(out).transpose("time", "prediction_timedelta", ..., "latitude", "longitude")
