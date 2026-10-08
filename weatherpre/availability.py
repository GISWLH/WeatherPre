"""Can model X serve request R here and now?  ->  ``Availability`` with machine-readable reasons.

Reason codes (blocking unless marked *):
    blocked                   the model cannot run at all (e.g. UniCM: no published weights)
    research_only             never auto-selected (e.g. SamudrACE: CM4 piControl model years, not real initial conditions)
    wrong_product             the request's product class is not served (e.g. seasonal months from a weather model)
    variable_unsupported      none of the requested variables is produced (* when only some are missing)
    proxy_variable            the model offers only a proxy (ORCA-DL ``tos`` for ``sst``): request the proxy name explicitly
    beyond_technical_horizon  the request needs more lead than the code / checkpoint can run (FuXi-S2S > 42 days)
    partial_coverage*         a hosted product ends before the last requested period (those periods are reported as incomplete)
    beyond_validated_horizon* no independent skill evidence at this lead
    unverified*               route implemented but never checked on real data
    out_of_period             the hosted archive does not contain this init
    input_missing             no initial-condition source covers this init (or the source is not configured)
    input_not_published       the initial-condition data for this init are not published yet (latency)
    forcing_missing           prescribed boundary conditions (ACE2) not supplied / provenance not stated
    not_installed             required package missing (weatherai, onnxruntime, torch, jax/neuralgcm, earth2studio)
    weights_missing           checkpoint not found locally (WEATHERPRE_WEIGHTS_<MODEL> or ~/.config/weatherpre/weights.json)
    gpu_insufficient          no single GPU with enough memory (memory of several GPUs is never added up)
    too_many_members          more members than the model / plan allows
    licence*                  usage restrictions (non-commercial, no redistribution)

``status``: "available" (no blocking reason, verified), "available_unverified" (runs, but evidence is partial/none),
"unavailable" (at least one blocking reason).
"""
from __future__ import annotations
import datetime as dt, importlib.util, json, os, shutil, subprocess
from dataclasses import dataclass, field
import numpy as np
from . import catalog as C, leads as L, variables as V

NON_BLOCKING = {"beyond_validated_horizon", "unverified", "licence", "cpu_only", "partial_coverage"}
WB2_ERA5_HOURLY_END = np.datetime64("2023-01-10")            # gs://weatherbench2 ... 1959-2023_01_10-1h-240x121 store
GODAS_LATENCY_DAYS = 10                                       # estimate: monthly GODAS fields appear early in the next month
PKG = {"weatherai": "weatherai", "torch": "torch", "onnxruntime": "onnxruntime", "jax": "jax", "neuralgcm": "neuralgcm",
       "earth2studio": "earth2studio"}
WEIGHT_FILES = {"fuxi-s2s": "model-1.0/fuxi_s2s.onnx", "orca-dl": "model_weights/seed_1.bin", "ace2": "ace2_era5_ckpt.tar"}

@dataclass(frozen=True)
class Reason:
    code: str
    message: str

    @property
    def blocking(self) -> bool:
        return self.code not in NON_BLOCKING

@dataclass
class Availability:
    model: str
    backend: str | None
    reasons: list[Reason] = field(default_factory=list)
    required_days: float | None = None
    variables: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(r.blocking for r in self.reasons)

    @property
    def status(self) -> str:
        if not self.ok: return "unavailable"
        return "available_unverified" if any(r.code in ("unverified", "beyond_validated_horizon") for r in self.reasons) else "available"

    def codes(self) -> list[str]:
        return [r.code for r in self.reasons]

    def explain(self) -> str:
        return "; ".join(f"[{r.code}] {r.message}" for r in self.reasons) or "ok"

    def to_dict(self) -> dict:
        return dict(model=self.model, backend=self.backend, status=self.status, required_days=self.required_days,
                    variables=self.variables, reasons=[dict(code=r.code, blocking=r.blocking, message=r.message) for r in self.reasons])

# --------------------------------------------------------------------------------------------------------- environment
@dataclass
class Environment:
    packages: dict[str, bool]
    weights: dict[str, str | None]
    gpus: list[tuple[str, float]]                  # (name, memory GB) per device; never summed

    @classmethod
    def detect(cls) -> "Environment":
        pk = {k: importlib.util.find_spec(m) is not None for k, m in PKG.items()}
        return cls(pk, weights_config(), _gpus())

    @property
    def max_gpu_gb(self) -> float:
        return max((g[1] for g in self.gpus), default=0.0)

def _gpus() -> list[tuple[str, float]]:
    """nvidia-smi only: detection must not initialise CUDA or import torch."""
    exe = shutil.which("nvidia-smi")
    if not exe: return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=20).stdout
        return [(n.strip(), float(m) / 1024.0) for n, m in (l.rsplit(",", 1) for l in out.strip().splitlines() if "," in l)]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []

def weights_config() -> dict[str, str | None]:
    """Local checkpoint roots: env WEATHERPRE_WEIGHTS_<MODEL> (e.g. WEATHERPRE_WEIGHTS_FUXI_S2S=/ckpt/fuxi_s2s), then
    ~/.config/weatherpre/weights.json {"fuxi-s2s": "/ckpt/fuxi_s2s", ...}. Only roots whose expected file exists count."""
    cfg = {}
    p = os.path.expanduser(os.environ.get("WEATHERPRE_WEIGHTS_CONFIG", "~/.config/weatherpre/weights.json"))
    if os.path.exists(p):
        with open(p) as f: cfg.update(json.load(f))
    out = {}
    for k, rel in WEIGHT_FILES.items():
        root = os.environ.get("WEATHERPRE_WEIGHTS_" + k.upper().replace("-", "_"), cfg.get(k))
        out[k] = root if root and os.path.exists(os.path.join(root, rel)) else None
    return out

# ---------------------------------------------------------------------------------------------------------- inputs
def input_reason(model: str, source: str | None, init, now=None) -> Reason | None:
    """Is there an initial-condition source for this init (and is it published yet)?"""
    now = np.datetime64(now or dt.datetime.utcnow(), "s")
    t = np.datetime64(init, "s")
    if model == "fuxi-s2s":
        src = source or "wb2-era5"
        if src == "wb2-era5" and t > WB2_ERA5_HOURLY_END - np.timedelta64(1, "D"):
            return Reason("input_missing", f"FuXi-S2S inputs from WeatherBench2 hourly ERA5 exist up to {WB2_ERA5_HOURLY_END}; for "
                          f"{str(t)[:10]} pass source='file:<input.nc>' built from ERA5/ERA5T daily means (official layout)")
        if t + np.timedelta64(1, "D") > now:
            return Reason("input_not_published", f"the daily mean of {str(t)[:10]} is complete only after {str(t + np.timedelta64(1, 'D'))[:10]}")
    if model == "orca-dl":
        src = source or "godas"
        ic_month = (t.astype("datetime64[M]") - 1)          # init_time = end of the initial-condition month
        if src == "orca-example" and str(ic_month) != "1980-01":
            return Reason("input_missing", "the official ORCA-DL example holds only the 1980-01 GODAS state")
        if src == "godas":
            ready = (ic_month + 1).astype("datetime64[s]") + np.timedelta64(GODAS_LATENCY_DAYS, "D")
            if ready > now:
                return Reason("input_not_published", f"GODAS monthly mean of {ic_month} expected around {str(ready)[:10]} "
                              f"(latency estimate {GODAS_LATENCY_DAYS} d)")
            if ic_month < np.datetime64("1980-01"):
                return Reason("input_missing", "GODAS starts 1980-01")
    return None

# ------------------------------------------------------------------------------------------------------------ check
def _required_days(lv: L.Leads, init, offset: float = 0.0) -> float | None:
    if init in (None, "latest"):
        try: return lv.max_hours / 24.0
        except ValueError: return None
    try:
        return lv.horizon_days(init, offset)
    except ValueError:
        return None

def check(model: str, variables=None, lead="6w", init=None, *, env: Environment | None = None, backend: str | None = None,
          members: int = 1, source: str | None = None, forcing_provenance: str | None = None, now=None) -> Availability:
    """Availability of ``model`` for (variables, lead, init) in ``env`` (detected if None). Never runs anything."""
    m = C.get(model)
    lv = lead if isinstance(lead, L.Leads) else L.parse(lead)
    vs = V.parse(variables) if variables is not None else [v for v in m.variables][:1] or ["t2m"]
    if isinstance(init, dt.datetime): init = np.datetime64(init, "s")
    a = Availability(m.name, backend, variables=vs)
    add = lambda c, msg: a.reasons.append(Reason(c, msg))
    if m.blocked: add("blocked", m.blocked)
    if m.research_only: add("research_only", m.research_only)
    if not m.routes and not m.blocked: add("blocked", "no route")
    # product / scale
    if lv.scale not in m.scales:
        have = "+".join(m.products)
        add("wrong_product", f"{m.name} serves {have}; the request is {lv.product}")
    # variables
    unsupported = [v for v in vs if v not in m.variables]
    for v in unsupported:
        prox = [x for x in m.variables if V.VARS[x].proxy_for == v]
        if prox:
            add("proxy_variable", f"{m.name} has no '{v}'; it provides the proxy '{prox[0]}' ({V.VARS[prox[0]].long_name}). "
                                  f"Request '{prox[0]}' explicitly; it is never renamed to '{v}'")
    if unsupported and len(unsupported) == len(vs) and not any(r.code == "proxy_variable" for r in a.reasons):
        add("variable_unsupported", f"{m.name} provides {', '.join(m.variables)}; requested {', '.join(vs)}")
    # horizons
    need = _required_days(lv, init, m.ic_offset_days)
    a.required_days = need
    if need is not None and m.technical_max_days is not None and need > m.technical_max_days + 1e-9 and m.hosted:
        a.reasons.append(Reason("partial_coverage", f"hosted product ends at {m.technical_max_days:g} days; periods beyond are not "
                                                    f"returned and are listed as incomplete (request needs {need:g} days)"))
    elif need is not None and m.technical_max_days is not None and need > m.technical_max_days + 1e-9:
        add("beyond_technical_horizon", f"request needs {need:g} days from init; {m.name} runs at most {m.technical_max_days:g} days"
                                        + (f" ({m.technical_note})" if m.technical_note else ""))
    if m.kind != "baseline" and need is not None and (m.validated_max_days is None or need > m.validated_max_days):
        add("beyond_validated_horizon", m.validated_note)
    if m.kind == "AI" and m.state("real_data_verified") != "yes" and not m.blocked:
        add("unverified", f"real-data evidence: {m.evidence.get('real_data_verified', ('none', ''))[1] or 'none'}")
    if "NC" in m.licence:
        add("licence", m.licence)
    if m.max_members is not None and members > m.max_members:
        add("too_many_members", f"{members} members requested; {m.name} allows {m.max_members}")
    if m.product == "scenario" and forcing_provenance is None:
        add("forcing_missing", "boundary conditions (SST, sea ice, insolation, CO2) for every step and their provenance "
                               "('forecast', 'scenario', 'persistence' or 'observed' = conditional hindcast) must be given")
    # runtime routes
    rt = [r for r in m.routes if r in ("weatherai", "e2s", "hf")]
    if rt and not any(r not in ("weatherai", "e2s", "hf") for r in m.routes) or backend in ("weatherai", "e2s"):
        env = env or Environment.detect()
        route = backend or rt[0]
        a.backend = route
        if route == "weatherai":
            need_pk = [r.split(":")[1] for r in m.requires if r.startswith("pkg:")]
            miss = [p for p in need_pk if not env.packages.get(p, False)]
            if miss: add("not_installed", f"install {', '.join(miss)} (WeatherAI extras: weatherai[onnx] / [neuralgcm]); separate "
                                          "environments for torch / onnxruntime / jax are recommended")
            wk = [r.split(":")[1] for r in m.requires if r.startswith("weights:")]
            for k in wk:
                if not env.weights.get(k):
                    add("weights_missing", f"set WEATHERPRE_WEIGHTS_{k.upper().replace('-', '_')} to the local checkpoint root "
                                           f"(expects {WEIGHT_FILES.get(k, '?')}); see weatherai.inference.card('{k}').weights_source")
            if not env.gpus and m.cpu_ok:
                a.reasons.append(Reason("cpu_only", "no GPU detected: runs on CPU (slow)"))
        elif route == "e2s":
            if not env.packages.get("earth2studio"):
                add("not_installed", "pip install 'weatherpre[gpu]' 'earth2studio[...]' on a CUDA machine")
            gb = m.gpu_mem_gb or 16.0
            if env.max_gpu_gb < gb:
                add("gpu_insufficient", f"needs one GPU with >= {gb:g} GB (largest here: {env.max_gpu_gb:.0f} GB; memory of several GPUs "
                                        "is not added up)")
        if init not in (None, "latest"):
            r = input_reason(m.name, source, np.datetime64(init, "s"), now)
            if r: a.reasons.append(r)
    elif init is not None and not m.blocked and lv.scale in m.scales:
        try:
            from .api import resolve, BackendUnavailable
            try:
                a.backend = resolve(m.name, init if init == "latest" else np.datetime64(init, "s").astype(dt.datetime), lv)[0]
            except BackendUnavailable as e:
                add("out_of_period", str(e))
        except ImportError:
            pass
    return a

def candidates(variables, lead, init, *, env: Environment | None = None, include_gpu: bool = False, **kw) -> list[Availability]:
    """Every catalogued model with its availability for the request (catalog-driven: no separate hand-kept list).
    Order: available hosted, available GPU/AI (only when ``include_gpu``), then unavailable with reasons."""
    env = env or Environment.detect()
    out = [check(m.name, variables, lead, init, env=env, **kw) for m in C.MODELS.values()]
    kind = {"NWP": 0, "AI": 1, "baseline": 2}
    def key(a):
        m = C.get(a.model)
        runnable = a.ok and not m.research_only and (m.hosted or include_gpu)
        return (not runnable, a.status != "available", kind.get(m.kind, 3))
    return sorted(out, key=key)
