"""Common output schema for model-run products (``conventions = "weatherpre-common-1"``) and period aggregation.

Layout
    dims        member, period (or lead at native resolution), [depth], latitude (ascending), longitude (0..360)
    coords      init_time (scalar), valid_start(period), valid_end(period) [exclusive], member, depth [m]
                completeness(period) after aggregation (fraction of the period covered by model output)
    data vars   canonical variable names (weatherpre.variables) in canonical units; ``valid_<var>`` int8 masks (lat, lon)
                where a model has a spatial domain / land mask -- values outside are NaN, never 0
    attrs       model, backend, checkpoint, source, product, time_semantics, aggregation, member_kind, independence_group,
                manifest (path), conventions

Individual members are always kept; ``ensemble_mean`` is a view for plots, not a replacement.
Aggregation (``aggregate``) weights each native interval by its overlap with the target period and reports
``completeness``; incomplete periods are dropped (default) and listed in ``attrs['incomplete_periods']``. Native intervals
longer than the target (monthly means -> weeks) are refused rather than interpolated.
"""
from __future__ import annotations
import json
import numpy as np, xarray as xr

CONVENTIONS = "weatherpre-common-1"
REQUIRED_ATTRS = ("model", "backend", "source", "product", "time_semantics", "member_kind", "independence_group", "conventions")
EPS = 1e-6

def finalize(ds: xr.Dataset, **attrs) -> xr.Dataset:
    ds = ds.assign_attrs(conventions=CONVENTIONS, **{k: (v if isinstance(v, (str, int, float)) else json.dumps(v, default=str))
                                                       for k, v in attrs.items()})
    if "latitude" in ds.dims: ds = ds.sortby("latitude")
    if "longitude" in ds.dims:
        ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude")
    return ds

def validate(ds: xr.Dataset) -> list[str]:
    """Problems with a common-schema dataset (empty list = valid)."""
    p = [f"missing attr {a}" for a in REQUIRED_ATTRS if a not in ds.attrs]
    time_dim = next((d for d in ("period", "lead") if d in ds.dims), None)
    if time_dim is None: p.append("no period/lead dimension")
    for c in ("init_time", "valid_start", "valid_end"):
        if c not in ds.coords: p.append(f"missing coord {c}")
    if "member" not in ds.dims: p.append("no member dimension (keep members; use size 1 for deterministic runs)")
    if "latitude" in ds.dims and not bool((np.diff(ds.latitude.values) > 0).all()): p.append("latitude must be ascending")
    for v in ds.data_vars:
        if v.startswith("valid_"):
            base = v[len("valid_"):]
            if base in ds:
                m = ds[v].astype(bool)
                inside = ds[base].where(~m)
                if bool(np.isfinite(inside).any()): p.append(f"{base}: finite values outside its valid mask")
            continue
        if "units" not in ds[v].attrs: p.append(f"{v}: no units")
    return p

def ensemble_mean(ds: xr.Dataset) -> xr.Dataset:
    """Member mean for plots / deterministic scoring; attrs record the member count."""
    if "member" not in ds.dims: return ds
    out = ds.mean("member", keep_attrs=True)
    for v in ds.data_vars:
        if v.startswith("valid_"): out[v] = ds[v]
    out.attrs["ensemble_mean_of"] = int(ds.sizes["member"])
    return out

def _intervals(ds, dim):
    s = ds["valid_start"].values.astype("datetime64[s]")
    e = ds["valid_end"].values.astype("datetime64[s]")
    if s.shape != (ds.sizes[dim],): raise ValueError("valid_start must be 1-D along the time dimension")
    if np.any(e <= s): raise ValueError("aggregation needs interval data (valid_end > valid_start); instantaneous fields go through "
                                       "s2s.weekly_from_leads")
    return s, e

def aggregate(ds: xr.Dataset, periods, dim: str = "lead", require_complete: bool = True, label: str = "period") -> xr.Dataset:
    """Time-weighted means of interval data (daily or monthly means) over target ``periods`` (leads.Period list)."""
    s, e = _intervals(ds, dim)
    dur = ((e - s) / np.timedelta64(1, "s")).astype(float)
    tvars = [v for v in ds.data_vars if dim in ds[v].dims]
    rows, keep, comp, dropped = [], [], [], []
    for p in periods:
        a, b = np.datetime64(p.start, "s"), np.datetime64(p.end, "s")
        plen = float((b - a) / np.timedelta64(1, "s"))
        ov = np.clip((np.minimum(e, b) - np.maximum(s, a)) / np.timedelta64(1, "s"), 0, None).astype(float)
        straddle = (ov > 0) & (ov < dur * (1 - EPS))
        if straddle.any():
            raise ValueError(f"native intervals straddle the boundary of target period {p.label}: cannot downscale "
                             f"(e.g. {dur[straddle][0] / 86400:g}-day means to {plen / 86400:g}-day periods)")
        c = float(ov.sum() / plen)
        if c < 1 - EPS and require_complete:
            dropped.append(f"{p.label} ({c:.0%} covered)"); continue
        if c <= 0:
            dropped.append(f"{p.label} (0% covered)"); continue
        w = xr.DataArray(ov / ov.sum(), dims=dim)
        part = {}
        for v in tvars:
            # NaN-aware weighting: a grid point missing at some native step is NaN for the period (no silent re-weighting)
            part[v] = (ds[v] * w).sum(dim, skipna=False).assign_attrs(ds[v].attrs)
        rows.append(xr.Dataset(part)); keep.append(p); comp.append(c)
    if not rows:
        from .api import BackendUnavailable
        raise BackendUnavailable(f"{ds.attrs.get('model', '?')}: no target period is covered ({'; '.join(dropped)})")
    out = xr.concat(rows, label, coords="minimal", compat="override")
    for v in ds.data_vars:
        if dim not in ds[v].dims: out[v] = ds[v]
    out = out.drop_vars([c for c in ("valid_start", "valid_end", dim) if c in out.coords])
    out = out.assign_coords({label: [p.label for p in keep],
                             "valid_start": (label, np.array([np.datetime64(p.start, "ns") for p in keep])),
                             "valid_end": (label, np.array([np.datetime64(p.end, "ns") for p in keep])),
                             "completeness": (label, np.array(comp))})
    out.attrs.update(ds.attrs)
    out.attrs["aggregation"] = f"time-weighted mean of {ds.attrs.get('time_semantics', 'native intervals')} over {label}s"
    if dropped: out.attrs["incomplete_periods"] = "; ".join(dropped)
    return out
