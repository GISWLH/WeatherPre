from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, xarray as xr

def maps(ds: xr.Dataset, title: str, out: str, lead_index=-1):
    """Quick-look maps of z500 / t850 / t2m / msl at one lead time."""
    ds = ds.isel(time=0, prediction_timedelta=lead_index)
    lead = int(ds.prediction_timedelta.values / np.timedelta64(1, "h"))
    panels = []
    if "geopotential" in ds: panels.append(("z500 [m]", ds.geopotential.sel(level=500) / 9.80665, "viridis"))
    if "temperature" in ds: panels.append(("t850 [K]", ds.temperature.sel(level=850), "RdBu_r"))
    if "2m_temperature" in ds: panels.append(("t2m [K]", ds["2m_temperature"], "RdBu_r"))
    if "mean_sea_level_pressure" in ds: panels.append(("msl [hPa]", ds.mean_sea_level_pressure / 100, "viridis"))
    n = len(panels); fig, axs = plt.subplots((n + 1) // 2, 2, figsize=(12, 3.2 * ((n + 1) // 2)), squeeze=False)
    for ax, (t, da, cm) in zip(axs.ravel(), panels):
        im = ax.pcolormesh(da.longitude, da.latitude, da.values, cmap=cm, shading="auto")
        ax.set_title(t, fontsize=9); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{title}  +{lead} h"); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)

def skill_curves(res: dict[str, xr.Dataset], out: str, fields=("z500", "t850", "t2m"), title=""):
    """res: model -> Dataset[<field>_rmse/_acc](lead_time)."""
    kinds = [k for k in ("rmse", "acc") if any(f"{fields[0]}_{k}" in r for r in res.values())]
    fig, axs = plt.subplots(len(kinds), len(fields), figsize=(4.2 * len(fields), 3.4 * len(kinds)), squeeze=False)
    for i, k in enumerate(kinds):
        for j, f in enumerate(fields):
            ax = axs[i][j]
            for m, r in res.items():
                if f"{f}_{k}" in r:
                    x = r.lead_time.values / np.timedelta64(1, "h") / 24
                    ax.plot(x, r[f"{f}_{k}"].values, marker="o", ms=3, label=m)
            ax.set_title(f"{f} {k.upper()}", fontsize=9); ax.set_xlabel("lead [days]"); ax.grid(alpha=.3)
    axs[0][0].legend(fontsize=7); fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
