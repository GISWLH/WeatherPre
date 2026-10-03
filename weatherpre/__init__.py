__version__ = "0.2.0"

def __getattr__(name):            # lazy: keep `import weatherpre` light
    if name in ("forecast", "compare", "models", "best_models", "BackendUnavailable", "PRESETS"):
        from . import api
        if name == "compare":
            from . import horizon; return horizon.compare
        return getattr(api, name)
    if name in ("plot_maps",):
        from . import maps; return maps.plot_maps
    raise AttributeError(name)
