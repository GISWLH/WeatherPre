"""Verification metrics for gridded fields and indices (numpy / xarray; deterministic and ensemble).

    rmse, acc (centred anomaly correlation over space), corr (temporal), msss (vs a reference forecast), crps_ensemble (fair
    CRPS, members kept), spread_skill, tercile_probabilities, reliability, box_index (area-weighted box mean, e.g. Nino3.4)

These complement the WeatherBench-X based scores in evaluate.py (which stay the reference for the WB2 weather / weekly
products). Area weights are cos(latitude); NaN (land, outside a model domain) is excluded, never treated as 0.
"""
from __future__ import annotations
import numpy as np, xarray as xr

NINO = {"nino34": (-5, 5, 190, 240), "nino3": (-5, 5, 210, 270), "nino4": (-5, 5, 160, 210), "nino12": (-10, 0, 270, 280)}

def weights(lat) -> xr.DataArray:
    lat = lat if isinstance(lat, xr.DataArray) else xr.DataArray(np.asarray(lat), dims="latitude")
    return np.cos(np.deg2rad(lat))

def _sp(da):
    return [d for d in ("latitude", "longitude") if d in da.dims]

def wmean(da: xr.DataArray, dims=None) -> xr.DataArray:
    dims = dims or _sp(da)
    w = weights(da.latitude).broadcast_like(da).where(np.isfinite(da))
    return (da * w).sum(dims) / w.sum(dims)

def rmse(f: xr.DataArray, o: xr.DataArray, dims=None) -> xr.DataArray:
    return np.sqrt(wmean((f - o) ** 2, dims))

def acc(f: xr.DataArray, o: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """Centred spatial anomaly correlation (anomalies w.r.t. ``clim``; area weighted; NaN excluded)."""
    fa, oa = f - clim, o - clim
    ok = np.isfinite(fa) & np.isfinite(oa)
    fa, oa = fa.where(ok), oa.where(ok)
    fa, oa = fa - wmean(fa), oa - wmean(oa)
    return wmean(fa * oa) / np.sqrt(wmean(fa ** 2) * wmean(oa ** 2))

def corr(f, o, dim="init") -> xr.DataArray:
    return xr.corr(f, o, dim=dim)

def msss(f, o, ref, dim="init") -> xr.DataArray:
    """Mean-squared skill score 1 - MSE(f)/MSE(ref) along ``dim`` (ref = climatology or persistence forecast)."""
    return 1 - ((f - o) ** 2).mean(dim) / ((ref - o) ** 2).mean(dim)

def crps_ensemble(ens: xr.DataArray, obs: xr.DataArray, member_dim="member", fair: bool = True) -> xr.DataArray:
    """CRPS of an ensemble (empirical CDF). ``fair=True`` uses the m(m-1) estimator (unbiased for finite ensembles)."""
    m = ens.sizes[member_dim]
    t1 = abs(ens - obs).mean(member_dim)
    e2 = ens.rename({member_dim: "_m2"})
    t2 = abs(ens - e2).sum((member_dim, "_m2")) / (m * (m - 1) if fair and m > 1 else m * m)
    return t1 - 0.5 * t2

def spread_skill(ens: xr.DataArray, obs: xr.DataArray, member_dim="member", dims=("init",)) -> dict:
    mean = ens.mean(member_dim)
    m = ens.sizes[member_dim]
    spread = np.sqrt((ens.var(member_dim, ddof=1)).mean(list(dims)) * (m + 1) / m) if m > 1 else xr.zeros_like(mean.mean(list(dims)))
    skill = np.sqrt(((mean - obs) ** 2).mean(list(dims)))
    return {"spread": spread, "rmse_ens_mean": skill, "ratio": spread / skill}

def tercile_probabilities(ens: xr.DataArray, lower: xr.DataArray, upper: xr.DataArray, member_dim="member") -> xr.DataArray:
    """P(below), P(normal), P(above) from member counts against climatological terciles."""
    below = (ens < lower).mean(member_dim)
    above = (ens > upper).mean(member_dim)
    return xr.concat([below, 1 - below - above, above], dim=xr.DataArray(["below", "normal", "above"], dims="category"))

def reliability(prob: np.ndarray, event: np.ndarray, bins=np.linspace(0, 1, 11)) -> dict:
    """Reliability table: forecast-probability bin -> observed frequency and count (NaN entries ignored)."""
    p, e = np.asarray(prob, float).ravel(), np.asarray(event, float).ravel()
    ok = np.isfinite(p) & np.isfinite(e)
    p, e = p[ok], e[ok]
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    out = {"bin_lo": bins[:-1].tolist(), "bin_hi": bins[1:].tolist(), "mean_prob": [], "obs_freq": [], "count": []}
    for k in range(len(bins) - 1):
        s = idx == k
        out["count"].append(int(s.sum()))
        out["mean_prob"].append(float(p[s].mean()) if s.any() else float("nan"))
        out["obs_freq"].append(float(e[s].mean()) if s.any() else float("nan"))
    return out

def box_index(da: xr.DataArray, box) -> xr.DataArray:
    """Area-weighted mean over (lat0, lat1, lon0, lon1) [deg, lon 0..360]; NaN-aware (e.g. Nino3.4 = NINO['nino34'])."""
    if isinstance(box, str): box = NINO[box]
    lat0, lat1, lon0, lon1 = box
    lon = da.longitude % 360
    x = da.where((da.latitude >= lat0) & (da.latitude <= lat1) & (lon >= lon0) & (lon <= lon1))
    return wmean(x)
