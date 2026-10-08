"""GODAS monthly means (NOAA PSL) -> ORCA-DL initial state, following the official ORCA-DL preprocessing.

Official recipe (OpenEarthLab/ORCA-DL README): ``cdo remapbil`` to the 1 deg 128x360 grid (lat -63.5..63.5), ``cdo intlevel`` to
10..1000 m (16 levels), units: degC, g/kg, m/s, m, N/m^2; ``sst`` = GODAS potential temperature at 5 m (the official example's
``sst`` equals that level to 6e-6 degC). The conversions, depth interpolation and regridding are WeatherAI's model-side helpers
(``weatherai.inference.orca_dl``); this module only downloads and selects.

Files: https://downloads.psl.noaa.gov/Datasets/godas/<var>.<year>.nc (salt, pottmp, ucur, vcur: 40 levels; sshg, uflx, vflx).
"""
from __future__ import annotations
import os, urllib.request, warnings
import numpy as np, xarray as xr

URL = "https://downloads.psl.noaa.gov/Datasets/godas/{var}.{year}.nc"
GODAS = {"so": "salt", "thetao": "pottmp", "uo": "ucur", "vo": "vcur", "zos": "sshg", "tauu": "uflx", "tauv": "vflx"}
EXAMPLE_URL = "https://raw.githubusercontent.com/OpenEarthLab/ORCA-DL/main/example_data/{v}.nc"

def download(var: str, year: int, cache: str) -> str:
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"{var}.{year}.nc")
    if not os.path.exists(p):
        urllib.request.urlretrieve(URL.format(var=var, year=year), p + ".part"); os.replace(p + ".part", p)
    return p

def download_example(cache: str) -> str:
    d = os.path.join(cache, "orca_example")
    os.makedirs(d, exist_ok=True)
    for v in ["salt", "pottmp", "sst", "ucur", "vcur", "sshg", "uflx", "vflx"]:
        p = os.path.join(d, f"{v}.nc")
        if not os.path.exists(p): urllib.request.urlretrieve(EXAMPLE_URL.format(v=v), p)
    return d

def orca_inputs(ic_month: str, cache: str = "data/_cache/godas", files: dict | None = None):
    """``ORCADLInputs`` for the GODAS monthly mean of ``ic_month`` ('YYYY-MM'). ``files`` maps GODAS names to local paths
    (otherwise downloaded per year from PSL)."""
    from weatherai.inference import orca_dl as O
    year = int(ic_month[:4])
    t = np.datetime64(ic_month + "-01")
    fields, notes = {}, []
    for v, g in GODAS.items():
        p = (files or {}).get(g) or download(g, year, cache)
        ds = xr.open_dataset(p)
        da = ds[g].sel(time=t, method="nearest")
        if abs((da.time.values - t) / np.timedelta64(1, "D")) > 16:
            raise ValueError(f"{g}: no monthly mean for {ic_month} in {p}")
        units = da.attrs.get("units", "")
        da = da.copy(data=O.convert_units(v, da.values, units))
        if "level" in da.dims:
            top = da.isel(level=0)
            if v == "thetao":
                tos = O.to_model_grid(top).values                              # 5 m potential temperature = official 'sst'
                fields["tos"] = tos.astype(np.float32)
                notes.append(f"tos = GODAS pottmp at {float(da.level[0]):g} m (official ORCA-DL convention)")
            src = da.level.values.astype(float)
            col = O.interp_depth(da.values, src, O.ORCA_DEPTHS)
            da = xr.DataArray(col, dims=("level", "lat", "lon"), coords={"level": list(O.ORCA_DEPTHS), "lat": da.lat, "lon": da.lon})
        fields[v] = O.to_model_grid(da).values.astype(np.float32)
    # sanity: eastward-positive wind stress (official example: equatorial Pacific mean < 0 in January)
    la, lo = (O.ORCA_LAT >= -2) & (O.ORCA_LAT <= 2), (O.ORCA_LON >= 180) & (O.ORCA_LON <= 260)
    eq = float(np.nanmean(fields["tauu"][np.ix_(la, lo)]))
    if eq > 0:
        warnings.warn(f"GODAS uflx equatorial-Pacific mean {eq:+.3f} N/m^2 > 0: check the sign convention before trusting ORCA-DL")
    notes.append(f"tauu equatorial Pacific (2S-2N, 180-260E) mean {eq:+.4f} N/m^2")
    inp = O.ORCADLInputs(fields, ic_month, source=f"GODAS (NOAA PSL) monthly mean {ic_month}")
    inp.tos_definition = notes[0]
    return inp

def orca_example_inputs(cache: str = "data/_cache"):
    """The official ORCA-DL example (GODAS 1980-01, already preprocessed) -- the reproduction baseline."""
    from weatherai.inference.orca_dl import ORCADLInputs
    return ORCADLInputs.from_official_example(download_example(cache), "1980-01")
