"""WeatherPre run manifest: the WeatherAI manifest (weights / normalisation / input / output hashes, code revision, timings,
memory) plus WeatherPre's own revision, the request, the source, the unit conversions applied and the output hash."""
from __future__ import annotations
import datetime as dt, hashlib, json, os, subprocess
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def revision() -> dict:
    from . import __version__
    out = {"package": "weatherpre", "version": __version__}
    try:
        r = subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            out["git_revision"] = r.stdout.strip()
            s = subprocess.run(["git", "-C", _ROOT, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10)
            out["git_dirty"] = bool(s.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out

def dataset_hash(ds) -> dict:
    out = {}
    for v in sorted(ds.data_vars):
        a = np.ascontiguousarray(ds[v].values)
        h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        out[v] = h.hexdigest()
    return out

def write_manifest(req, ds, weatherai_manifest=None, path=None, **extra) -> str:
    d = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), weatherpre=revision(),
             request=dict(model=req.model, variables=req.variables, leads=str(req.leads), init=str(req.init), members=req.members,
                          seed=req.seed, device=req.device, source=req.source, checkpoint=req.checkpoint, backend=req.backend,
                          forcing_provenance=req.forcing_provenance),
             output=dict(attrs={k: str(v) for k, v in ds.attrs.items()}, sha256=dataset_hash(ds),
                         dims={k: int(v) for k, v in ds.sizes.items()}),
             units={v: ds[v].attrs.get("units", "") for v in ds.data_vars}, **extra)
    if weatherai_manifest is not None:
        d["weatherai"] = weatherai_manifest.to_dict() if hasattr(weatherai_manifest, "to_dict") else weatherai_manifest
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f: json.dump(d, f, indent=1, default=str)
    return str(path)
