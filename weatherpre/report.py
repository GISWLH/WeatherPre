"""Hand-off to downstream climate reports (SST_S2S): common-schema files + a decision table per model, variable and period.

    bundle = export_bundle({"orca-dl": ds_orca, "fuxi-s2s": ds_fuxi}, "out/report_2027H1", region=(-60, 60, 0, 360),
                           hindcasts={"orca-dl": table})          # hindcast tables from weatherpre.hindcast.evaluate_index

Writes ``models.json`` / ``models.csv`` (one row per model x variable x period) and ``<model>.nc``. Status per row:
    executed         the run exists and passed schema validation
    comparison_only  shown next to the report's products, never weighted into them (default for every AI model today)
    eligible         may be weighted: needs an independent hindcast on the test split with positive skill vs climatology at
                     that lead (msss_vs_climatology > 0, >= min_inits) AND a calibration step recorded downstream
    excluded         not usable (reason given: schema problems, no coverage, conditional hindcast, duplicate, ...)
``independence_group``: seeds / implementations of one model share a group; only the first export per group counts as an
independent model (``duplicate_of`` marks the rest). ``coverage`` = valid-area fraction of ``region``; downstream effective
weights must be renormalised where a model is missing (e.g. ORCA-DL poleward of 63.5 deg).
"""
from __future__ import annotations
import json, os
from dataclasses import asdict, dataclass, field
import numpy as np, pandas as pd, xarray as xr
from . import catalog as C, schema as S

@dataclass
class Decision:
    model: str
    independence_group: str
    variable: str
    period: str
    valid_start: str
    valid_end: str
    status: str
    reasons: list[str] = field(default_factory=list)          # why the row is not eligible (blocking)
    notes: list[str] = field(default_factory=list)            # caveats that travel with the data (not blocking)
    members: int = 1
    coverage: float | None = None
    completeness: float | None = None
    duplicate_of: str | None = None
    experiment_type: str = "forecast"
    file: str | None = None

def coverage(ds: xr.Dataset, var: str, region=None) -> float:
    """Area fraction of ``region`` (lat0, lat1, lon0, lon1) where ``var`` has finite values (first member / period)."""
    da = ds[var]
    for d in [d for d in da.dims if d not in ("latitude", "longitude")]: da = da.isel({d: 0})
    lat, lon = da.latitude, da.longitude % 360
    if region is not None:
        lat0, lat1, lon0, lon1 = region
        sel = (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)
    else:
        sel = xr.ones_like(da, dtype=bool)
    w = np.cos(np.deg2rad(lat)).broadcast_like(da)
    inside = w.where(sel)
    full = float(inside.sum())
    if region is not None:                   # region parts outside the model's latitude range count as missing (spherical area)
        r0, r1 = max(region[0], -90.0), min(region[1], 90.0)
        dl = float(np.median(np.diff(lat.values))) / 2 if lat.size > 1 else 0.0
        g0, g1 = max(r0, float(lat.min()) - dl), min(r1, float(lat.max()) + dl)
        s = lambda x: np.sin(np.deg2rad(x))
        frac_lat = max(0.0, s(g1) - s(g0)) / (s(r1) - s(r0)) if r1 > r0 else 1.0
    else:
        frac_lat = 1.0
    return float(inside.where(np.isfinite(da)).sum()) / full * frac_lat if full > 0 else 0.0

def assess(name: str, ds: xr.Dataset, region=None, hindcast: pd.DataFrame | None = None, min_inits: int = 10,
           calibrated: bool = False) -> list[Decision]:
    m = C.MODELS.get(name)
    problems = S.validate(ds)
    exp = ds.attrs.get("experiment_type", "forecast")
    rows = []
    tdim = "period" if "period" in ds.dims else "week" if "week" in ds.dims else "lead"
    for v in [v for v in ds.data_vars if not v.startswith("valid_")]:
        cov = coverage(ds, v, region)
        for i in range(ds.sizes[tdim]):
            reasons, notes, status = [], [], "comparison_only"
            if problems:
                status, reasons = "excluded", [f"schema: {p}" for p in problems]
            elif exp == "conditional_hindcast":
                status, reasons = "excluded", ["conditional hindcast (observed future boundary conditions): not a forecast"]
            elif cov <= 0:
                status, reasons = "excluded", ["no valid data in the report region"]
            else:
                lead = i + 1
                if hindcast is None:
                    reasons.append("no independent hindcast supplied")
                else:
                    h = hindcast[hindcast.lead == lead]
                    ok = len(h) and h.n_inits.iloc[0] >= min_inits and float(h.get("msss_vs_climatology", pd.Series([np.nan])).iloc[0]) > 0
                    if not ok: reasons.append(f"hindcast at lead {lead}: no positive MSSS vs climatology on >= {min_inits} test inits")
                if not calibrated: reasons.append("no calibration step recorded")
                if v in ("tos",): notes.append("tos is a near-surface proxy of SST (ORCA-DL: GODAS 5 m): verify against a matching reference")
                if m is not None and m.state("report_eligible") != "yes" and reasons:
                    notes.append("catalog default: " + m.evidence.get("report_eligible", ("no", ""))[1])
                if not reasons:
                    status = "eligible"
            comp = float(ds.completeness.values[i]) if "completeness" in ds.coords else None
            rows.append(Decision(name, ds.attrs.get("independence_group", name), v, str(ds[tdim].values[i]),
                                 str(ds.valid_start.values[i])[:10] if "valid_start" in ds.coords else "",
                                 str(ds.valid_end.values[i])[:10] if "valid_end" in ds.coords else "", status, reasons, notes,
                                 int(ds.sizes.get("member", 1)), round(cov, 4), comp, None, exp))
    return rows

def export_bundle(results: dict, out_dir: str, region=None, hindcasts: dict | None = None, calibrated: dict | None = None):
    os.makedirs(out_dir, exist_ok=True)
    rows, groups = [], {}
    for name, ds in results.items():
        p = os.path.join(out_dir, f"{name}.nc")
        ds.to_netcdf(p)
        dec = assess(name, ds, region, (hindcasts or {}).get(name), calibrated=(calibrated or {}).get(name, False))
        g = ds.attrs.get("independence_group", name)
        for d in dec:
            d.file = os.path.basename(p)
            if g in groups and groups[g] != name:
                d.duplicate_of, d.status = groups[g], "excluded"
                d.reasons = [f"same independence group as {groups[g]} (seed / implementation of one model)"] + d.reasons
        groups.setdefault(g, name)
        rows += dec
    tab = pd.DataFrame([asdict(r) for r in rows])
    tab.to_csv(os.path.join(out_dir, "models.csv"), index=False)
    meta = {"conventions": S.CONVENTIONS, "region": region, "independent_models": sorted(set(groups.values())),
            "rows": [asdict(r) for r in rows],
            "notes": ["Members are stored individually in each file; the ensemble mean is not a substitute.",
                      "comparison_only rows must not enter weighted products; eligible requires hindcast + calibration.",
                      "Renormalise weights where coverage < 1 (models missing in part of the region)."]}
    with open(os.path.join(out_dir, "models.json"), "w") as f: json.dump(meta, f, indent=1, default=str)
    return tab
