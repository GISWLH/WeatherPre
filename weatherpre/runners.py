"""Model-run adapters: ``RunRequest`` -> common-schema Dataset at the model's native time resolution.

Adding a model = one catalog entry (catalog.py) + one function registered here::

    @register("my-model")
    def run_my_model(req: RunRequest) -> xr.Dataset: ...      # weatherpre-common-1, members kept, masks as valid_<var>

``api.forecast`` aggregates the native output to the requested weeks / calendar months / target window with
``schema.aggregate``; plotting and scoring code see the same schema for every model and need no change.
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

RUNNERS: dict[str, Callable] = {}

def register(model: str):
    def deco(fn):
        RUNNERS[model] = fn
        return fn
    return deco

@dataclass
class RunRequest:
    model: str
    variables: list[str]
    leads: object                         # leads.Leads
    init: dt.datetime
    members: int | list[int] = 1
    seed: int = 0
    device: str = "cpu"
    source: str | None = None             # initial-condition source (wb2-era5 | godas | orca-example | file:<path> | dir:<path>)
    checkpoint: str | None = None         # local checkpoint root (default: availability.weights_config())
    backend: str | None = None            # model-specific backend option (fuxi-s2s: onnx | torch)
    forcing: object = None                # ACE2 / NeuralGCM boundary conditions
    forcing_provenance: str | None = None
    out_dir: Path | None = None
    resume: bool = True
    options: dict = field(default_factory=dict)

def get(model: str):
    from .adapters import weatherai_run  # noqa: F401  (registers the WeatherAI-backed runners)
    if model not in RUNNERS:
        from .api import BackendUnavailable
        raise BackendUnavailable(f"{model}: no run adapter registered (have {sorted(RUNNERS)})")
    return RUNNERS[model]
