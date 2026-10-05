"""Publication-style global maps and score charts straight from the returned xarray Dataset.

Maps use a Robinson globe cut at 60°S with `robinson_lat_clip` (GISWLH/cartopy-robinson-lat-clip): the oval keeps
its curved sides instead of the rectangular side cuts `set_extent(lat_min=-60)` would give."""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np, xarray as xr

MIN_LAT = -60.0                                   # southern cut of every map (Antarctica hidden)
INK, INK2, INK3 = "#1b1b1a", "#52514e", "#8a8984"  # text: primary / secondary / muted
RULE, LAND, OCEAN = "#d9d8d3", "#e9e8e4", "#ffffff"
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "savefig.facecolor": "white",
                     "figure.facecolor": "white", "axes.titlesize": 9})

# one fixed colour per model across every figure (validated 8-slot categorical order); NWP drawn dashed
MODEL_COLORS = {"graphcast": "#2a78d6", "pangu": "#eb6834", "fuxi": "#1baf7a", "gencast": "#eda100",
                "neuralgcm": "#e87ba4", "aurora": "#008300", "ifs-ens": "#4a3aa7", "ifs-hres": "#e34948",
                "aifs-single": "#2a78d6", "aigfs": "#1baf7a", "gfs": "#4a3aa7"}
NWP = {"ifs-hres", "ifs-ens", "gfs"}
LABELS = {"ifs-hres": "IFS HRES", "hres": "IFS HRES", "ifs-ens": "IFS ENS mean", "graphcast": "GraphCast", "pangu": "Pangu-Weather",
          "fuxi": "FuXi", "gencast": "GenCast mean", "neuralgcm": "NeuralGCM", "aurora": "Aurora",
          "aifs-single": "AIFS", "aigfs": "AIGFS", "gfs": "GFS", "era5": "ERA5"}

def _cmap(colors, name):
    return mcolors.LinearSegmentedColormap.from_list(name, colors)

# per variable: unit conversion, discrete levels, colormap, contour overlay (step, highlighted level)
Z500 = _cmap(["#2c1e5c", "#3b4fa0", "#3f8fc0", "#8cc7d4", "#e8eed2", "#f6d58c", "#ef9a4d", "#d4513a", "#8e1c2e"], "z500")
T2M = _cmap(["#1d2f6f", "#3164a8", "#6fa3cf", "#c5dbe9", "#f2f1ee", "#f4cfae", "#e3895d", "#bd3f32", "#6e1426"], "t2m")
MSL = _cmap(["#27407a", "#5b86bb", "#a9c6df", "#eef0ef", "#eccba5", "#cf8a57", "#8f4423"], "msl")
TP = _cmap(["#e6f2ee", "#a8d8c8", "#5fb3b4", "#3a84b0", "#3a529c", "#4b2c86", "#7a1c6e"], "tp")
STYLE = {
    "z500": dict(label="500 hPa geopotential height [dam]", scale=0.1, levels=np.arange(492, 597, 4), cmap=Z500,
                 contour=8, highlight=588, extend="both"),
    "t850": dict(label="850 hPa temperature [°C]", offset=-273.15, levels=np.arange(-36, 33, 3), cmap=T2M,
                 contour=None, extend="both"),
    "t2m": dict(label="2 m temperature [°C]", offset=-273.15, levels=np.arange(-40, 41, 4), cmap=T2M,
                contour=None, extend="both"),
    "msl": dict(label="mean sea-level pressure [hPa]", levels=np.arange(976, 1045, 4), cmap=MSL,
                contour=8, highlight=None, extend="both"),
    "tp": dict(label="precipitation since init [mm]", levels=[2, 5, 10, 25, 50, 100, 200, 400], cmap=TP,
               contour=None, extend="max", under="white"),
}
RMSE_UNIT = {"z500": "m", "t850": "K", "t2m": "K"}

def _wrap(x):
    """lon 0..360 -> -180..180 (cartopy needs a monotonic, non-wrapping axis); south of the cut is dropped."""
    x = x.transpose(..., "latitude", "longitude")
    x = x.assign_coords(longitude=(((x.longitude + 180) % 360) - 180)).sortby("longitude").sortby("latitude")
    return x.sel(latitude=slice(MIN_LAT - 3, None))

def _field(da, var):
    st = STYLE[var]
    v = da * st.get("scale", 1) + st.get("offset", 0)
    from cartopy.util import add_cyclic_point
    data, lon = add_cyclic_point(v.values, coord=v.longitude.values)
    return lon, v.latitude.values, data

def _norm(var):
    st = STYLE[var]; cm = st["cmap"]; lv = st["levels"]
    n = len(lv) - 1 + (st["extend"] in ("both", "min")) + (st["extend"] in ("both", "max"))
    colors = cm(np.linspace(0, 1, n))
    cmap, norm = mcolors.from_levels_and_colors(lv, colors, extend=st["extend"])
    if st.get("under"): cmap.set_under(st["under"])
    return cmap, norm

def _aspect():
    import cartopy.crs as ccrs
    from robinson_lat_clip import equatorial_x_limits, projected_parallel_y
    p = ccrs.Robinson(); x0, x1 = equatorial_x_limits(p)
    return (x1 - x0) / (p.y_limits[1] - projected_parallel_y(MIN_LAT, p))

def _layout(rows, cols, panel_w=3.6, head=0.78, title=0.36, gap_w=0.16, gap_h=0.12, foot=0.0, side=0.0):
    """Explicit panel rectangles (inches) -> fig + list of axes rects in figure fractions."""
    ph = panel_w / _aspect()
    W = 0.3 + cols * panel_w + (cols - 1) * gap_w + 0.3 + side
    H = head + rows * (title + ph) + (rows - 1) * gap_h + foot + 0.15
    fig = plt.figure(figsize=(W, H))
    rects = []
    for r in range(rows):
        for c in range(cols):
            x = 0.3 + c * (panel_w + gap_w); y = H - head - (r + 1) * (title + ph) - r * gap_h
            rects.append((x / W, y / H, panel_w / W, ph / H))
    return fig, rects, (W, H)

def _map(fig, rect, da, var, cmap, norm):
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    from robinson_lat_clip import clip_robinson_south_of
    ax = fig.add_axes(rect, projection=ccrs.Robinson()); ax.set_global()
    pc = ccrs.PlateCarree(); st = STYLE[var]
    lon, lat, z = _field(da, var)
    im = ax.contourf(lon, lat, z, levels=st["levels"], cmap=cmap, norm=norm, extend=st["extend"], transform=pc)
    if st.get("contour"):
        f = int(round(1.0 / abs(float(da.latitude[1] - da.latitude[0]))))      # isolines from a ~1 deg field: smooth on 0.25 deg
        if f > 1: lon, lat, z = _field(da.coarsen(latitude=f, longitude=f, boundary="trim").mean(), var)
        step = st["levels"][1] - st["levels"][0]
        lv = np.arange(st["levels"][0], st["levels"][-1] + step, st["contour"])
        ax.contour(lon, lat, z, levels=lv, colors="#1b1b1a", linewidths=0.35, alpha=0.45, transform=pc)
        if st.get("highlight"):
            ax.contour(lon, lat, z, levels=[st["highlight"]], colors="#1b1b1a", linewidths=1.0, transform=pc)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.45, edgecolor="#2d2d2b", alpha=0.85)
    ax.gridlines(xlocs=range(-180, 181, 60), ylocs=range(-60, 91, 30), linewidth=0.3, color="white", alpha=0.55)
    clip_robinson_south_of(ax, min_lat=MIN_LAT, keep_full_ylim=False)
    ax.spines["geo"].set_edgecolor("#9a9994"); ax.spines["geo"].set_linewidth(0.6)
    return ax, im

def _panel_title(fig, rect, left, right="", color=None, size=9.5):
    x, y, w, h = rect; yy = y + h + 0.012
    if color:
        fig.patches.append(matplotlib.patches.Rectangle((x, yy + 0.002), 0.006, 0.022, transform=fig.transFigure,
                                                        color=color, figure=fig, clip_on=False))
        x += 0.011
    fig.text(x, yy, left, fontsize=size, fontweight="bold", color=INK, va="bottom")
    if right: fig.text(rect[0] + w, yy, right, fontsize=size - 1, color=INK2, va="bottom", ha="right")

def _header(fig, title, subtitle, H):
    fig.text(0.3 / fig.get_figwidth(), 1 - 0.22 / H, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.3 / fig.get_figwidth(), 1 - 0.5 / H, subtitle, fontsize=9, color=INK2, va="top")

def _cbar(fig, cax_rect, im, var, orientation="horizontal"):
    cax = fig.add_axes(cax_rect)
    cb = fig.colorbar(im, cax=cax, orientation=orientation, spacing="uniform", drawedges=False)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor("#9a9994")
    cax.tick_params(labelsize=8, length=2, width=0.4, color="#9a9994")
    st = STYLE[var]; lv = st["levels"]
    if var == "tp": cb.set_ticks(lv)
    else: cb.set_ticks(lv[:: max(1, len(lv) // 8)])
    cb.set_label(st["label"], fontsize=8.5, color=INK2, labelpad=4)
    return cb

def _name(m): return LABELS.get(m, m)
def _valid(ds, h): return np.datetime_as_string(ds.valid_time.sel(lead=h).values, unit="h").replace("T", " ") + "Z"
def _init(ds):
    return np.datetime_as_string(np.datetime64(ds.init_time.values), unit="h").replace("T", " ") + "Z" if "init_time" in ds.coords else str(ds.attrs.get("init"))

def plot_maps(ds: xr.Dataset, variables=("z500", "t2m", "tp"), leads=None, out="maps.png", title=None, robust=True):
    """Rows = variables, columns = lead times, one model. Shared colour scale per row."""
    ds = _wrap(ds); variables = [v for v in variables if v in ds]
    leads = list(leads) if leads is not None else [int(x) for x in ds.lead.values[:: max(1, len(ds.lead) // 4)]][:4]
    leads = [h for h in leads if h in ds.lead.values]
    rows, cols = len(variables), len(leads)
    fig, rects, (W, H) = _layout(rows, cols, panel_w=3.7, side=1.05, gap_h=0.22)
    for r, v in enumerate(variables):
        cmap, norm = _norm(v)
        for c, h in enumerate(leads):
            rect = rects[r * cols + c]
            ax, im = _map(fig, rect, ds[v].sel(lead=h), v, cmap, norm)
            _panel_title(fig, rect, f"+{h} h", f"valid {_valid(ds, h)}")
        x, y, w, hgt = rects[r * cols + cols - 1]
        _cbar(fig, (x + w + 0.22 / W, y + 0.08 * hgt, 0.11 / W, 0.84 * hgt), im, v, "vertical")
    model = ds.attrs.get("model", "")
    _header(fig, title or f"{_name(model)} forecast · init {_init(ds)}",
            f"{ds.attrs.get('source', ds.attrs.get('backend', ''))}  ·  "
            + "  ·  ".join(STYLE[v]["label"] for v in variables)
            + ("  ·  bold line = 588 dam" if "z500" in variables else ""), H)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out

def plot_models(dsets: dict, var="z500", lead=168, truth: xr.DataArray | None = None, out="models.png", title="",
                scores=None, subtitle=None):
    """One panel per model at a fixed lead (+ optional truth panel first). `scores`: compare() table -> RMSE badge."""
    panels = ([("era5", _wrap(truth))] if truth is not None else []) + \
             [(m, _wrap(d[var].sel(lead=lead))) for m, d in dsets.items() if lead in d.lead.values]
    rmse = {}
    if scores is not None and len(scores) and f"{var}_rmse" in scores:
        s = scores[scores.lead_h == lead].set_index("model")[f"{var}_rmse"].dropna(); rmse = s.to_dict()
    if rmse:   # truth first, then best -> worst
        panels = panels[:1] + sorted(panels[1:], key=lambda p: rmse.get(p[0], np.inf))
    n = len(panels); cols = 2 if n <= 4 else 4 if n > 6 else 3; rows = int(np.ceil(n / cols))
    fig, rects, (W, H) = _layout(rows, cols, panel_w=3.9 if cols > 2 else 4.6, foot=0.62, gap_h=0.2)
    cmap, norm = _norm(var); best = min(rmse, key=rmse.get) if rmse else None
    for i, (name, da) in enumerate(panels):
        ax, im = _map(fig, rects[i], da, var, cmap, norm)
        if name == "era5": right = "verifying analysis"
        elif name in rmse: right = f"RMSE {rmse[name]:.1f} {RMSE_UNIT.get(var, '')}" + ("  ★ best" if name == best else "")
        else: right = "NWP" if name in NWP else "AI"
        _panel_title(fig, rects[i], _name(name), right, color=None if name == "era5" else MODEL_COLORS.get(name, INK3))
    for j in range(n, rows * cols):   # empty slots: legend note
        x, y, w, h = rects[j]
        fig.text(x + w / 2, y + h / 2, "panels ordered by RMSE\n(area-weighted, WeatherBench-X)" if rmse else "",
                 ha="center", va="center", fontsize=8.5, color=INK3, linespacing=1.5)
    x0 = rects[0][0]; x1 = rects[cols - 1][0] + rects[cols - 1][2]; cw = 0.5 * (x1 - x0)
    _cbar(fig, (x0 + (x1 - x0 - cw) / 2, 0.36 / H, cw, 0.11 / H), im, var)
    _header(fig, title, subtitle or (STYLE[var]["label"] + ("  ·  bold line = 588 dam" if var == "z500" else "")), H)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out

def plot_scores(table, out, title="", fields=("z500", "t850", "t2m"), subtitle=None):
    """Lines = models (fixed colours, NWP dashed), x = lead [days]: RMSE from the compare() table."""
    fields = [f for f in fields if f"{f}_rmse" in table and table[f"{f}_rmse"].notna().any()]
    fig, axs = plt.subplots(1, len(fields), figsize=(4.0 * len(fields) + 0.6, 3.9), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.66, bottom=0.15, wspace=0.22)
    order = [m for m in MODEL_COLORS if m in set(table.model)] + sorted(set(table.model) - set(MODEL_COLORS))
    handles = {}
    for ax, f in zip(axs[0], fields):
        for m in order:
            d = table[table.model == m].sort_values("lead_h")
            if not d[f"{f}_rmse"].notna().any(): continue
            ln, = ax.plot(d.lead_h / 24, d[f"{f}_rmse"], color=MODEL_COLORS.get(m, INK3), lw=1.8,
                          ls=(0, (4, 2)) if m in NWP else "-", marker="o", ms=3.2, mec="white", mew=0.6,
                          solid_capstyle="round", zorder=3)
            handles[m] = ln
        ax.set_title(f"{STYLE[f]['label'].split(' [')[0]}", loc="left", fontsize=10, fontweight="bold", color=INK, pad=18)
        ax.text(0, 1.03, f"RMSE [{RMSE_UNIT[f]}]", transform=ax.transAxes, fontsize=8, color=INK3, va="bottom", ha="left")
        ax.set_xlabel("lead time [days]", fontsize=8.5)
        ax.grid(axis="y", color=RULE, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(bottom=0)
        ax.tick_params(length=0, labelsize=8)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#9a9994"); ax.spines["bottom"].set_linewidth(0.6)
    fig.legend(handles.values(), [_name(m) for m in handles], loc="upper left", bbox_to_anchor=(0.055, 0.855),
               ncol=len(handles), frameon=False, fontsize=8.5, handlelength=2.2, columnspacing=1.4)
    fig.text(0.06, 0.97, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.905, subtitle or "area-weighted RMSE (WeatherBench-X) · solid = AI, dashed = NWP · lower is better",
             fontsize=8.5, color=INK2, va="top")
    fig.savefig(out, dpi=150); plt.close(fig)
    return out
