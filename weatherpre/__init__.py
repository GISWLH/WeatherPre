"""WeatherPre: one-call weather (<= 15 days) and S2S (> 15 days) forecasts from frontier AI and NWP models,
assembled from hosted data and upstream inference frameworks, with built-in accuracy comparison.

    import weatherpre as wp
    ds  = wp.forecast("graphcast", "z500", "7d", init="2020-10-03")
    cmp = wp.compare(["ifs-ext", "gefs", "climatology"], "t2m", "6w", init="2020-10-01")
    wp.availability.check("fuxi-s2s", "t2m,tp", "6w", "2022-12-01")       # reasons, no download
    ds  = wp.forecast("orca-dl", "tos", target_start="2027-01-01", target_end="2027-06-30", init="2026-12-01", members=[1, 2])
"""
__version__ = "0.3.0"

_API = ("forecast", "models", "best_models", "BackendUnavailable", "PRESETS", "variables", "resolve")

def __getattr__(name):            # lazy: keep `import weatherpre` light
    if name in _API:
        from . import api; return getattr(api, name)
    if name in ("compare", "Comparison"):
        from . import comparison as c; return getattr(c, name)
    if name in ("plot_maps", "plot_skill", "plot_s2s", "plot_models"):
        from . import maps; return getattr(maps, name)
    if name in ("anomaly",):
        from . import s2s; return s2s.anomaly
    if name in ("catalog", "leads", "availability", "planner", "schema", "hindcast", "metrics", "report", "variables", "runners"):
        import importlib; return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)

__all__ = list(_API) + ["compare", "Comparison", "plot_maps", "plot_skill", "plot_s2s", "plot_models", "anomaly"]
