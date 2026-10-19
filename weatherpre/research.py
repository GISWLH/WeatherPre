"""Explicit research routes into WeatherAI, separate from deterministic forecasts."""
from . import catalog


def load_research(name, *, lite=False, **kwargs):
    """Load pinned author implementations by default; lite requires opt-in.

    PBC requires work_dir (official historical-data fitting pipeline).
    GenFocal/Translocation download official pretrained parameters; install
    WeatherAI's documented Linux JAX runtime. Errors never trigger lite fallback.
    """
    name = catalog.canonical(name)
    if name not in ("pbc", "genfocal", "translocation"):
        raise ValueError("research methods: pbc, genfocal, translocation")
    try:
        import weatherai
    except ImportError as exc:
        raise ImportError("Install `pip install 'weatherpre[research]'` to use WeatherAI research methods") from exc
    classes = {"pbc": (weatherai.PBC, weatherai.PBC_lite),
               "genfocal": (weatherai.GenFocal, weatherai.GenFocal_lite),
               "translocation": (weatherai.Translocation, weatherai.Translocation_lite)}
    if lite:
        return classes[name][1](**kwargs)
    if name == "pbc":
        return classes[name][0](**kwargs)
    return classes[name][0].from_pretrained(**kwargs)


def calibrate(model, raw, climatology, lag1, lag2, *, probabilities=True):
    """Apply fitted PBC to aligned xarray CDFs with a 'threshold' dimension.

    Preserve grid/date coordinates, return category masses or threshold CDFs.
    Deterministic ensemble means cannot substitute for raw probabilities.
    """
    import numpy as np
    import torch
    import xarray as xr
    if getattr(model, "implementation", None) == "official":
        raise TypeError("Official PBC uses model.run(...) with author-format histories/NetCDF. This tensor helper requires explicit lite=True.")
    arrays = (raw, climatology, lag1, lag2)
    if any(not isinstance(a, xr.DataArray) or "threshold" not in a.dims for a in arrays):
        raise ValueError("pass four xarray.DataArrays with a 'threshold' dimension")
    if any(set(a.dims) != set(raw.dims) for a in arrays):
        raise ValueError("CDF arrays must have identical dimensions")
    aligned = xr.align(*arrays, join="exact")
    dims = tuple(d for d in raw.dims if d != "threshold") + ("threshold",)
    tensors = [torch.as_tensor(a.transpose(*dims).values.copy(), device=model.bias.device, dtype=model.bias.dtype)
               for a in aligned]
    with torch.no_grad():
        result = model.predict_proba(*tensors) if probabilities else model(*tensors)
    coords = {d: raw.coords[d] for d in dims[:-1] if d in raw.coords}
    last = "category" if probabilities else "threshold"
    coords[last] = np.arange(1, result.shape[-1] + 1) if probabilities else raw.threshold.values
    return xr.DataArray(result.cpu().numpy(), dims=dims[:-1] + (last,), coords=coords,
                        name="probability" if probabilities else "cdf",
                        attrs={"method": "PBC_lite", "units": "1", "source": "user-fitted WeatherAI lite approximation",
                               "paper": catalog.get("pbc").ref})


def downscale(model, x, cond, **kwargs):
    """Two-stage GenFocal normalized tensor route; see model.downscale docs."""
    if getattr(model, "implementation", None) == "official":
        raise TypeError("Official GenFocal uses model.sample_debiasing(...) or model.trajectory_sampler(...); downscale is the explicit lite tensor helper")
    return model.downscale(x, cond, **kwargs)


def diagnose(model, x, **kwargs):
    """Signed effective receptive field; model must be in eval mode."""
    return model.effective_receptive_field(x, **kwargs)
