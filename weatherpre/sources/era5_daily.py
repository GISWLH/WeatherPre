"""ERA5 daily means in the FuXi-S2S input layout (76 channels, 1.5 deg, lat 90 -> -90), from public cloud copies of ERA5.

Convention (official FuXi-S2S sample, verified in WeatherAI docs/results/fuxi_s2s_input_check.json and stated in
Earth2Studio's FuXiS2S docstring): instantaneous fields = mean of the hourly values 00..23 UTC; accumulated fields (tp, ttr)
= mean of the hourly accumulations ending 01..24 UTC. tp channel = clip(1000 * mean hourly accumulation [m], 0, 1000) (mm/h);
ttr channel = mean top net thermal flux [W m-2]; sst NaN over land from the official ``data/mask.nc``.

Sources
    WeatherBench2 hourly ERA5 1.5 deg (conservative), 1959-01-01 .. 2023-01-10: all channels except the 100 m winds
    ARCO-ERA5 0.25 deg hourly (Google public dataset): 100 m winds, block-averaged (conservative weights) to 1.5 deg
Later dates need ERA5/ERA5T from the CDS (5-day latency) prepared in the same layout (``source='file:<input.nc>'``).
"""
from __future__ import annotations
import os, urllib.request
import numpy as np, xarray as xr

WB2_1H = "weatherbench2/datasets/era5/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
ARCO = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
MASK_URL = "https://raw.githubusercontent.com/tpys/FuXi-S2S/main/data/mask.nc"
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PL = [("geopotential", "z"), ("temperature", "t"), ("u_component_of_wind", "u"), ("v_component_of_wind", "v"), ("specific_humidity", "q")]
SFC = [("2m_temperature", "t2m"), ("2m_dewpoint_temperature", "d2m"), ("sea_surface_temperature", "sst"),
       ("mean_top_net_long_wave_radiation_flux", "ttr"), ("10m_u_component_of_wind", "10u"), ("10m_v_component_of_wind", "10v"),
       ("100m_u_component_of_wind", "100u"), ("100m_v_component_of_wind", "100v"), ("mean_sea_level_pressure", "msl"),
       ("total_column_water_vapour", "tcwv"), ("total_precipitation", "tp")]
ACCUM = {"ttr", "tp"}
ARCO_ONLY = {"100u", "100v"}
LAT = np.linspace(90.0, -90.0, 121)

def _open(path):
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    return xr.open_zarr(fs.get_mapper(path), chunks=None)

def _hours(day, accum: bool):
    d = np.datetime64(day, "D").astype("datetime64[h]")
    return d + (np.arange(1, 25) if accum else np.arange(0, 24)).astype("timedelta64[h]")

def block_mean_to_1p5(a: np.ndarray) -> np.ndarray:
    """(..., 721, 1440) 0.25 deg (lat 90..-90, lon 0..359.75) -> (..., 121, 240) 1.5 deg, cell-centred conservative weights
    (each 1.5 deg cell spans centre +-0.75 deg: 7 x 7 points with half weights on the shared edges; polar cells are half cells)."""
    w1 = np.array([0.5, 1, 1, 1, 1, 1, 0.5])
    lat_q = np.linspace(90, -90, 721)
    coslat = np.cos(np.deg2rad(lat_q))
    coslat[[0, -1]] = np.cos(np.deg2rad(90 - 0.125))       # pole rows represent a 0.125 deg band
    out = np.empty(a.shape[:-2] + (121, 240), dtype=np.float64)
    pad = np.concatenate([a[..., -3:], a, a[..., :3]], -1)   # periodic longitude
    for j in range(121):
        c = 6 * j
        rows = np.arange(c - 3, c + 4)
        ok = (rows >= 0) & (rows < 721)
        wr = (w1 * coslat[np.clip(rows, 0, 720)])[ok]
        r = rows[ok]
        band = np.tensordot(pad[..., r, :], wr / wr.sum(), axes=([-2], [0]))          # (..., 1446)
        win = np.lib.stride_tricks.sliding_window_view(band, 7, axis=-1)[..., ::6, :]  # (..., 240, 7)
        out[..., j, :] = (win * (w1 / w1.sum())).sum(-1)
    return out

def mask(cache_dir: str) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, "fuxi_s2s_mask.nc")
    if not os.path.exists(p): urllib.request.urlretrieve(MASK_URL, p)
    return xr.open_dataarray(p).values.astype(bool)

def daily_fields(day, cache_dir: str = "data/_cache") -> np.ndarray:
    """(76, 121, 240) float32 FuXi-S2S channels for the daily mean of ``day`` (UTC)."""
    if np.datetime64(day, "D") >= np.datetime64("2023-01-10"):
        raise ValueError("WeatherBench2 hourly ERA5 ends 2023-01-10; build later inputs from the CDS (source='file:...')")
    wb, ar = _open(WB2_1H), None
    chans = []
    def to_grid(da):                                  # WB2 (.., longitude, latitude) ascending lat -> (.., lat desc, lon)
        dims = [d for d in da.dims if d not in ("longitude", "latitude")] + ["latitude", "longitude"]
        return da.transpose(*dims).sortby("latitude", ascending=False).values.astype(np.float64)
    for long, short in PL:
        da = wb[long].sel(time=_hours(day, False), level=LEVELS).mean("time")
        chans.append(to_grid(da))
    for long, short in SFC:
        if short in ARCO_ONLY:
            ar = ar or _open(ARCO)
            x = ar[long].sel(time=_hours(day, False)).mean("time").transpose("latitude", "longitude")
            if float(x.latitude[0]) < float(x.latitude[-1]): x = x.sortby("latitude", ascending=False)
            chans.append(block_mean_to_1p5(x.values.astype(np.float64))[None]); continue
        da = wb[long].sel(time=_hours(day, short in ACCUM)).mean("time")
        a = to_grid(da)[None]
        if short == "tp": a = np.clip(a * 1000.0, 0, 1000)
        chans.append(a)
    x = np.concatenate(chans, 0).astype(np.float32)
    assert x.shape == (76, 121, 240), x.shape
    m = mask(cache_dir)
    i = 13 * 5 + 2                                     # sst channel
    x[i] = np.where(m, x[i], np.nan)
    return x

def fuxi_inputs(init, cache_dir: str = "data/_cache"):
    """``weatherai.inference.fuxi_s2s.FuXiS2SInputs`` for initialisation day ``init`` (inputs: init-1 day, init)."""
    from weatherai.inference.fuxi_s2s import FuXiS2SInputs
    d = np.datetime64(init, "D")
    p = os.path.join(cache_dir, f"fuxi_inputs_{d}.npy")
    if os.path.exists(p):
        x = np.load(p)
    else:
        x = np.stack([daily_fields(d - np.timedelta64(1, "D"), cache_dir), daily_fields(d, cache_dir)])
        os.makedirs(cache_dir, exist_ok=True); np.save(p, x)
    return FuXiS2SInputs(x, [d - np.timedelta64(1, "D"), d], "ERA5: WeatherBench2 hourly 1.5 deg + ARCO-ERA5 100 m winds", mask(cache_dir))
