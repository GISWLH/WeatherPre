"""Global maps straight from the returned xarray Dataset (xarray .plot + cartopy)."""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, xarray as xr

STYLE = {"z500": dict(cmap="Spectral_r", label="z500 [m]"), "t850": dict(cmap="RdBu_r", label="t850 [K]"),
         "t2m": dict(cmap="RdBu_r", label="t2m [K]"), "msl": dict(cmap="viridis", label="msl [hPa]"),
         "tp": dict(cmap="YlGnBu", label="precip since init [mm]", vmin=0, vmax=None)}

def _wrap(x):
    x = x.transpose(..., "latitude", "longitude")
    """lon 0..360 -> -180..180 (cartopy/pcolormesh need a monotonic, non-wrapping axis)."""
    return x.assign_coords(longitude=(((x.longitude + 180) % 360) - 180)).sortby("longitude")

def _ax(fig, n, i, proj):
    import cartopy.crs as ccrs
    return fig.add_subplot(n[0], n[1], i, projection=proj)

def plot_maps(ds: xr.Dataset, variables=("z500", "t2m", "tp"), leads=None, out="maps.png", title=None, robust=True):
    """Rows = variables, columns = lead times:  ds[var].sel(lead=h).plot(...) on a Robinson map."""
    import cartopy.crs as ccrs
    ds = _wrap(ds); variables = [v for v in variables if v in ds]
    leads = list(leads) if leads is not None else [int(x) for x in ds.lead.values[:: max(1, len(ds.lead) // 4)]][:4]
    leads = [h for h in leads if h in ds.lead.values]
    proj = ccrs.Robinson()
    fig = plt.figure(figsize=(4.6 * len(leads), 2.9 * len(variables) + 0.6))
    for r, v in enumerate(variables):
        st = STYLE[v]; allv = ds[v].sel(lead=leads)
        vmin = st.get("vmin", float(allv.quantile(0.02))); vmax = st.get("vmax") or float(allv.quantile(0.98))
        for c, h in enumerate(leads):
            ax = _ax(fig, (len(variables), len(leads)), r * len(leads) + c + 1, proj)
            da = ds[v].sel(lead=h)
            im = da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=st["cmap"], vmin=vmin, vmax=vmax, add_colorbar=False, rasterized=True)
            ax.coastlines(linewidth=0.5); ax.set_global()
            ax.set_title(f"{v}  +{h} h\nvalid {str(ds.valid_time.sel(lead=h).values)[:13]}Z", fontsize=8)
            if c == len(leads) - 1:
                fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label=st["label"])
    fig.suptitle(title or f"{ds.attrs.get('model')}  init {ds.attrs.get('init')}  ({ds.attrs.get('backend')}; {ds.attrs.get('source','')})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=110); plt.close(fig)
    return out

def plot_models(dsets: dict, var="z500", lead=168, truth: xr.DataArray | None = None, out="models.png", title=""):
    """One panel per model at a fixed lead (plus optional truth panel)."""
    import cartopy.crs as ccrs
    panels = ([("ERA5 (truth)", _wrap(truth))] if truth is not None else []) + [(m, _wrap(d[var].sel(lead=lead))) for m, d in dsets.items() if lead in d.lead.values]
    n = len(panels); cols = 2 if n <= 4 else 3; rows = int(np.ceil(n / cols))
    st = STYLE[var]; vals = np.concatenate([p.values.ravel() for _, p in panels if p is not None])
    vmin, vmax = np.nanpercentile(vals, [2, 98])
    fig = plt.figure(figsize=(6.2 * cols, 3.3 * rows + 0.5))
    for i, (name, da) in enumerate(panels, 1):
        ax = fig.add_subplot(rows, cols, i, projection=ccrs.Robinson())
        im = da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=st["cmap"], vmin=vmin, vmax=vmax, add_colorbar=False, rasterized=True)
        ax.coastlines(linewidth=0.5); ax.set_global(); ax.set_title(name, fontsize=9)
    fig.colorbar(im, ax=fig.axes, shrink=0.6, pad=0.02, label=st["label"])
    fig.suptitle(title, fontsize=10); fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
    return out

def plot_scores(table, out, title="", fields=("z500", "t850", "t2m")):
    """Lines = models, x = lead [days]: RMSE of z500/t850/t2m (WeatherBench-X) from the compare() table."""
    fig, axs = plt.subplots(1, len(fields), figsize=(4.3 * len(fields), 3.4), squeeze=False)
    for ax, f in zip(axs[0], fields):
        for (m, g), d in table.groupby(["model", "grid"]):
            if f"{f}_rmse" in d and d[f"{f}_rmse"].notna().any():
                d = d.sort_values("lead_h"); ax.plot(d.lead_h / 24, d[f"{f}_rmse"], marker="o", ms=3, label=f"{m} ({g})")
        ax.set_title(f"{f} RMSE [{ {'z500': 'm', 't850': 'K', 't2m': 'K'}.get(f, '') }]", fontsize=9); ax.set_xlabel("lead [days]"); ax.grid(alpha=.3)
    axs[0][0].legend(fontsize=6); fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(out, dpi=105); plt.close(fig)
