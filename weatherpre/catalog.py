"""Model catalog: the single source of truth for every model WeatherPre knows about.

Each entry says which scales / products it serves, how it is reached (hosted data first, then inference backends:
Earth2Studio ``e2s``, WeatherAI ``weatherai`` (native PyTorch, official ONNX, official JAX), a HF Space), its time step,
variables, **technical** horizon (what the code / checkpoint can run) and **validated** horizon (independently verified
skill; ``None`` = not validated), member semantics, requirements and evidence.

Four states are kept apart (they are NOT the same thing):
    catalogued          listed here
    runnable            a route is implemented and its requirements are met *in this environment* (see availability.py)
    real_data_verified  someone ran it from real initial conditions and checked the output (``evidence``)
    report_eligible     independent hindcast skill + calibration exist for the variable/period (no AI model qualifies yet)

`weatherpre models`, the README model table, automatic candidate lists and backend resolution are all driven from here."""
from __future__ import annotations
from dataclasses import dataclass, field

WEATHER, S2S, SEASONAL = "weather", "s2s", "seasonal"
WEATHER_MAX_H = 360                       # 15 days: the weather / S2S boundary
ATM = ("z500", "t850", "t2m", "msl", "tp")
EVIDENCE_LEVELS = ("real_data_verified", "report_eligible")

@dataclass(frozen=True)
class Model:
    name: str
    label: str
    kind: str                             # "AI" | "NWP" | "baseline"
    scales: tuple[str, ...]               # subset of (WEATHER, S2S, SEASONAL)
    routes: tuple[str, ...]               # wb2 | opendata | noaa | gefs | cfs | wb2-ext | baseline | hf | e2s | weatherai
    grid: str
    lead: str
    period: str                           # where hosted data exists / which inits are possible
    licence: str
    ref: str = ""                         # paper / model card
    e2s: str = ""                         # earth2studio.models.px class name (GPU route)
    aliases: tuple[str, ...] = field(default_factory=tuple)
    # ---- added for AI S2S / seasonal / ocean models -------------------------------------------------------------------
    time_step: str = ""                   # 6h | 12h | 1d | 1mo | weekly-mean | ...
    variables: tuple[str, ...] = ATM
    technical_max_days: float | None = None   # None = no code limit; what can RUN
    technical_note: str = ""
    validated_max_days: float | None = None   # independently verified skill horizon; None = not validated
    validated_note: str = "no independent hindcast evaluated in WeatherPre"
    members: str = "1"
    max_members: int | None = None
    product: str = ""                     # "" = from scales; "scenario" = conditional on prescribed boundary conditions
    weatherai: str = ""                   # WeatherAI card key (weatherai.inference.card)
    requires: tuple[str, ...] = ()        # pkg:<module> | weights:<key> | gpu:<GB> | input:<source>
    input_sources: tuple[str, ...] = ()
    gpu_mem_gb: float | None = None       # measured peak where known (single GPU)
    cpu_ok: bool = False
    independence_group: str = ""          # implementations / seeds of one model share a group (never double-counted)
    evidence: dict = field(default_factory=dict)   # level -> (state, detail)
    blocked: str = ""                     # non-empty -> cannot run at all, with the concrete reason
    research_only: str = ""               # non-empty -> never auto-selected, with the reason
    ic_offset_days: float = 0.0           # daily/monthly-mean models: the initial state ends this long after the nominal init

    @property
    def hosted(self) -> bool:             # runs on any laptop (no GPU)
        return any(r not in ("hf", "e2s", "weatherai") for r in self.routes)

    @property
    def status(self) -> str:
        if self.blocked or not self.routes: return "blocked"
        if self.hosted: return "hosted"
        return "run" if "weatherai" in self.routes else "gpu"

    @property
    def group(self) -> str:
        return self.independence_group or self.name

    @property
    def products(self) -> tuple[str, ...]:
        if self.product: return (self.product,)
        return tuple({WEATHER: "weather", S2S: "subseasonal", SEASONAL: "seasonal"}[s] for s in self.scales)

    def state(self, level: str) -> str:
        return self.evidence.get(level, ("no", ""))[0]

_W, _S, _WS = (WEATHER,), (S2S,), (WEATHER, S2S)
_NOT_ELIGIBLE = ("no", "no independent hindcast + calibration for the target variable/period in WeatherPre")
_HOSTED_OK = {"real_data_verified": ("yes", "hosted operational/archived forecasts"), "report_eligible": _NOT_ELIGIBLE}
MODELS: dict[str, Model] = {m.name: m for m in [
    # ---------------- weather scale, hosted (no GPU) ----------------
    Model("ifs-hres", "ECMWF IFS HRES", "NWP", _W, ("opendata", "wb2"), "0.25° / 1.5°", "≤15 d",
          "latest ~4 d (Open Data) · 2016–2022 (WB2)", "CC-BY-4.0", "https://www.ecmwf.int/en/forecasts",
          aliases=("ifs", "hres", "ecmwf"), time_step="6h", technical_max_days=15, evidence=_HOSTED_OK),
    Model("aifs-single", "ECMWF AIFS", "AI", _W, ("opendata",), "0.25°", "≤15 d", "latest ~4 d (Open Data)",
          "CC-BY-4.0", "https://arxiv.org/abs/2406.01465", e2s="AIFS", aliases=("aifs",), time_step="6h", technical_max_days=15,
          evidence=_HOSTED_OK),
    Model("aigfs", "NOAA AIGFS", "AI", _W, ("noaa",), "0.25°", "≤16 d", "2026-04-16 → now (NOAA S3)",
          "public domain", "https://registry.opendata.aws/noaa-nws-graphcastgfs-pds/", aliases=("gfs-ai",), time_step="6h",
          technical_max_days=16, evidence=_HOSTED_OK),
    Model("gfs", "NOAA GFS", "NWP", _W, ("noaa",), "0.25°", "≤16 d", "2022 → now (NOAA S3)", "public domain",
          "https://registry.opendata.aws/noaa-gfs-bdp-pds/", time_step="6h", technical_max_days=16, evidence=_HOSTED_OK),
    Model("gefs", "NOAA GEFS ens. mean", "NWP", _WS, ("gefs",), "0.5°", "≤35 d (00Z)", "2020-09-23 → now (NOAA S3, GEFSv12)",
          "public domain", "https://registry.opendata.aws/noaa-gefs/", time_step="6h", technical_max_days=35,
          members="ensemble mean (or control)", evidence=_HOSTED_OK),
    Model("graphcast", "GraphCast", "AI", _W, ("wb2", "e2s"), "1.5° (hosted)", "≤10 d", "2019-11 → 2021-01 (WB2)",
          "weights CC-BY-NC-SA-4.0", "https://doi.org/10.1126/science.adi2336", e2s="GraphCastOperational", time_step="6h",
          technical_max_days=10, evidence=_HOSTED_OK),
    Model("pangu", "Pangu-Weather", "AI", _W, ("wb2", "e2s"), "1.5° (hosted)", "≤10 d", "2018–2022 (WB2)",
          "weights CC-BY-NC-SA-4.0", "https://doi.org/10.1038/s41586-023-06185-3", e2s="Pangu6", time_step="6h",
          technical_max_days=10, evidence=_HOSTED_OK),
    Model("fuxi", "FuXi", "AI", _W, ("wb2", "e2s"), "1.5° (hosted)", "≤15 d", "2020 (WB2)", "see upstream",
          "https://doi.org/10.1038/s41612-023-00512-1", e2s="FuXi", time_step="6h", technical_max_days=15, evidence=_HOSTED_OK),
    Model("gencast", "GenCast (ens. mean)", "AI", _W, ("wb2",), "1.5°", "≤15 d, 12 h", "2020 (WB2)",
          "weights CC-BY-NC-SA-4.0", "https://doi.org/10.1038/s41586-024-08252-9", aliases=("wn",), time_step="12h",
          technical_max_days=15, evidence=_HOSTED_OK),
    Model("neuralgcm", "NeuralGCM", "AI", _W, ("wb2", "weatherai"), "1.5° (hosted) / 2.8–0.7° (JAX)", "≤15 d, 12 h (hosted)",
          "2020 (WB2) · any date via the official JAX model (inputs on the model grid)", "code Apache-2.0, weights CC-BY-SA-4.0",
          "https://doi.org/10.1038/s41586-024-07744-y", time_step="12h (hosted) / any (JAX)", technical_max_days=15,
          technical_note="hosted archive 15 d; the JAX model has no code limit but SST/sea ice are prescribed (persisted or given)",
          weatherai="neuralgcm", requires=("pkg:neuralgcm", "pkg:jax"), input_sources=("dataset",), cpu_ok=True,
          evidence={"real_data_verified": ("partial", "hosted WB2 forecasts; WeatherAI ran the official JAX model on the demo snapshot "
                                           "(CPU, 4x6 h) and the precip checkpoint 24 h vs ERA5"), "report_eligible": _NOT_ELIGIBLE}),
    Model("ifs-ens", "ECMWF IFS ENS mean", "NWP", _W, ("wb2",), "1.5°", "≤15 d", "2018–2022 (WB2)", "WB2 terms",
          "https://www.ecmwf.int/en/forecasts", time_step="12h", technical_max_days=15, evidence=_HOSTED_OK),
    # ---------------- S2S / seasonal, hosted (no GPU) ----------------
    Model("ifs-ext", "ECMWF extended range (ens. mean)", "NWP", _S, ("wb2-ext",), "1.5°", "46 d, weekly means",
          "2016–2022, Mon/Thu inits (WB2)", "WB2 terms", "https://www.ecmwf.int/en/forecasts/documentation-and-support/extended-range",
          aliases=("ecmwf-ext", "ifs-s2s"), time_step="weekly-mean", technical_max_days=46, members="ensemble mean",
          evidence=_HOSTED_OK),
    Model("cfsv2", "NOAA CFSv2", "NWP", _WS, ("cfs",), "1°", "≤9 months", "2020 → now (NOAA S3), member 1",
          "public domain", "https://registry.opendata.aws/noaa-cfs/", aliases=("cfs",), time_step="6h", technical_max_days=270,
          technical_note="WeatherPre fetches the 6-hourly time series; monthly aggregation from it is not wired yet",
          members="member 1", evidence=_HOSTED_OK),
    Model("climatology", "ERA5 climatology", "baseline", (S2S,), ("baseline",), "1.5°", "any", "any date (1990–2017 clim.)",
          "Copernicus", "https://weatherbench2.readthedocs.io", aliases=("clim",), time_step="weekly-mean",
          evidence={"real_data_verified": ("yes", "baseline"), "report_eligible": ("yes", "reference baseline")}),
    Model("persistence", "ERA5 anomaly persistence", "baseline", (S2S,), ("baseline",), "1.5°", "any",
          "1959 – 2023-01 (ERA5 in WB2)", "Copernicus", "https://weatherbench2.readthedocs.io", time_step="weekly-mean",
          evidence={"real_data_verified": ("yes", "baseline"), "report_eligible": ("yes", "reference baseline")}),
    # ---------------- GPU: HF ZeroGPU / Earth2Studio ----------------
    Model("aurora", "Aurora", "AI", _W, ("hf", "e2s"), "0.25°", "any (chunked)", "any date (ERA5 / IFS ICs)", "MIT",
          "https://doi.org/10.1038/s41586-025-09005-y", e2s="Aurora", time_step="6h", technical_max_days=15),
    Model("aurora-1.5", "Aurora 1.5", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "MIT",
          "https://github.com/microsoft/aurora", e2s="Aurora1p5_6h", time_step="6h", technical_max_days=15),
    Model("aifs2", "ECMWF AIFS v2", "AI", _W, ("e2s",), "0.25°", "6 h steps", "latest (IFS ICs)", "CC-BY-4.0",
          "https://arxiv.org/abs/2509.18994", e2s="AIFS2", time_step="6h", technical_max_days=15),
    Model("aifs2-ens", "ECMWF AIFS-ENS v2", "AI", _W, ("e2s",), "0.25°", "6 h steps", "latest (IFS ICs)", "CC-BY-4.0",
          "https://arxiv.org/abs/2506.10868", e2s="AIFS2ENS", time_step="6h", technical_max_days=15),
    Model("fcn3", "FourCastNet 3", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "see model card",
          "https://arxiv.org/abs/2507.12144", e2s="FCN3", aliases=("fourcastnet3",), time_step="6h", technical_max_days=15),
    Model("atlas", "NVIDIA Atlas", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "see model card",
          "https://huggingface.co/nvidia/atlas-era5", e2s="Atlas", time_step="6h", technical_max_days=15),
    Model("ucast", "U-CAST", "AI", _W, ("e2s",), "1.5°", "12 h steps", "any date", "see model card",
          "https://arxiv.org/abs/2604.09041", e2s="UCast", aliases=("u-cast",), time_step="12h", technical_max_days=None,
          technical_note="autoregressive 12 h steps, no code limit; Earth2Studio 0.19 labels it class:medium-range (gpu:40gb). "
                         "S2S use is not validated, so it is offered for the weather scale only",
          members="MC-dropout members (dropout active at inference; seed via torch)", gpu_mem_gb=40.0,
          requires=("pkg:earth2studio", "gpu:40")),
    Model("weathernext2", "WeatherNext 2 (Cyclones)", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date",
          "see model card", "https://github.com/google-deepmind/weathernext", e2s="WeatherNext2Cyclones", time_step="6h",
          technical_max_days=15),
    Model("gencast-mini", "GenCast mini", "AI", _W, ("e2s",), "1°", "12 h steps", "any date", "see model card",
          "https://github.com/google-deepmind/graphcast", e2s="GenCastMini", time_step="12h", technical_max_days=15),
    Model("sfno", "SFNO", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "see model card",
          "https://arxiv.org/abs/2306.03838", e2s="SFNO", time_step="6h", technical_max_days=15),
    Model("fengwu", "FengWu", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "unclear (keep private)",
          "https://arxiv.org/abs/2304.02948", e2s="FengWu", time_step="6h", technical_max_days=15),
    # ---------------- AI sub-seasonal / seasonal / ocean (WeatherAI and Earth2Studio) ----------------
    Model("fuxi-s2s", "FuXi-S2S", "AI", _S, ("weatherai", "e2s"), "1.5°", "≤42 d, daily means", "any date with ERA5-like daily inputs",
          "weights CC-BY-NC-ND-4.0 (non-commercial, no redistribution)", "https://doi.org/10.1038/s41467-024-50714-1",
          e2s="FuXiS2S", time_step="1d", variables=("z500", "t850", "t2m", "msl", "tp", "sst"), technical_max_days=42,
          technical_note="lead encoding trained for 42 daily steps; longer requests are refused (Earth2Studio does not stop)",
          members="stochastic latent draws, seeded per member (official example: 11)", max_members=51, weatherai="fuxi-s2s",
          requires=("pkg:weatherai", "pkg:onnxruntime", "weights:fuxi-s2s"), input_sources=("wb2-era5", "file"),
          gpu_mem_gb=5.4, cpu_ok=True, independence_group="fuxi-s2s", ic_offset_days=1.0,
          evidence={"real_data_verified": ("partial", "official ERA5 sample: native 1 step + 3-step chain vs ONNX (WeatherAI); "
                                           "11x42 ONNX run reported by SST_S2S (external)"), "report_eligible": _NOT_ELIGIBLE}),
    Model("orca-dl", "ORCA-DL (ocean)", "AI", (SEASONAL,), ("weatherai",), "1° (63.5S–63.5N)", "monthly means, autoregressive",
          "any month with GODAS-like monthly inputs", "code: no licence file; weights licence unstated (do not redistribute)",
          "https://doi.org/10.1126/sciadv.adu2488", time_step="1mo", variables=("tos", "thetao", "so", "zos", "uo", "vo"),
          technical_max_days=None, technical_note="lead experts 1..6, then feedback; official predict.sh 24 months",
          members="training seeds 1..8 of ONE model (not independent)", max_members=8, weatherai="orca-dl",
          requires=("pkg:weatherai", "pkg:torch", "weights:orca-dl"), input_sources=("godas", "orca-example", "dir"),
          gpu_mem_gb=2.4, cpu_ok=True, independence_group="orca-dl",
          evidence={"real_data_verified": ("partial", "one GODAS init (1980-01), seed 1, 11 leads (WeatherAI); monthly run reported "
                                           "by SST_S2S (external)"), "report_eligible": _NOT_ELIGIBLE}),
    Model("dlesym", "DLESyM (atmos.+ocean)", "AI", (S2S, SEASONAL), ("e2s",), "~1° HEALPix (lat/lon wrapper)", "6 h atmos / 48 h ocean",
          "any date", "see model card", "https://arxiv.org/abs/2409.16247", e2s="DLESyMLatLon", time_step="6h atmos, 48h ocean",
          variables=("z500", "t850", "t2m", "sst"), technical_max_days=None,
          technical_note="atmosphere outputs every 6 h, ocean (sst) every 48 h; only the valid entries of each output tensor are kept",
          requires=("pkg:earth2studio", "gpu:40"), gpu_mem_gb=40.0),
    Model("ace2", "ACE2-ERA5", "AI", (WEATHER, S2S, SEASONAL), ("weatherai", "e2s"), "1° Gauss–Legendre", "6 h → climate",
          "any date WITH forcing (SST, sea ice, insolation, CO2) for the whole period", "Apache-2.0",
          "https://doi.org/10.1038/s41612-025-01090-0", e2s="ACE2ERA5", time_step="6h", variables=("t2m", "msl", "tp", "z500", "t850"),
          technical_max_days=None, technical_note="no code limit; every step needs the forcing at t and t+1",
          product="scenario", weatherai="ace2", requires=("pkg:weatherai", "pkg:torch", "weights:ace2", "input:forcing"),
          input_sources=("ace2-files", "dataset"), gpu_mem_gb=3.1, cpu_ok=True,
          evidence={"real_data_verified": ("partial", "one 8-day case from 2020-06-01 (WeatherAI)"), "report_eligible": _NOT_ELIGIBLE}),
    Model("unicm", "UniCM (climate modes)", "AI", (SEASONAL,), (), "5° (12x72)", "24 months", "-", "code MIT; no weights",
          "https://github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes (Yuan et al., Nat. Mach. Intell. 8, 930-941, 2026)", time_step="1mo", variables=("sst_anom",), weatherai="unicm",
          blocked="no published checkpoint (official GitHub HEAD 67fe4c1 has no weights or link; Zenodo record is code-only); "
                  "random-weight outputs are never forecasts; retraining is a separate project"),
    Model("samudrace", "SamudrACE (CM4 piControl)", "AI", (SEASONAL,), ("e2s",), "1°", "coupled 6 h atmos / 5 d ocean",
          "CM4 model years only", "see model card", "https://huggingface.co/allenai/SamudrACE-CM4-piControl", e2s="SamudrACE", time_step="6h/5d",
          variables=("sst", "t2m"), research_only="trained on and initialised from GFDL CM4 pre-industrial control (model years, e.g. "
          "year 151); not applicable to real 2026 initial conditions without a separate validation"),
    # ---------------- blocked ----------------
    Model("weathernext", "Google WeatherNext 2/3 (hosted)", "AI", _W, (), "0.25°", "≤15 d", "allow-list only",
          "experimental ToS", "https://developers.google.com/weathernext", blocked="hosted product requires an allow-list"),
]}

ALIASES = {a: m.name for m in MODELS.values() for a in m.aliases}

def get(name: str) -> Model:
    key = canonical(name)
    if key not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    return MODELS[key]

def canonical(name: str) -> str:
    n = name.lower().strip()
    return ALIASES.get(n, n)

def for_scale(scale: str, hosted_only=False) -> list[Model]:
    return [m for m in MODELS.values() if scale in m.scales and (m.hosted or not hosted_only)]

def serves(m: Model, variable: str) -> bool:
    return variable in m.variables

def table(scale: str | None = None):
    """pandas DataFrame of the catalog (optionally one scale): every model, with horizons and evidence."""
    import pandas as pd
    fmt = lambda d: "-" if d is None else f"{d:g} d"
    rows = [dict(model=m.name, label=m.label, kind=m.kind, scales="+".join(m.scales), product="+".join(m.products), status=m.status,
                 route=",".join(m.routes) or "-", time_step=m.time_step, grid=m.grid, lead=m.lead,
                 technical_max=("unbounded" if m.technical_max_days is None and m.routes else fmt(m.technical_max_days)),
                 validated_max=fmt(m.validated_max_days), real_data=m.state("real_data_verified"),
                 report_eligible=m.state("report_eligible"), period=m.period, licence=m.licence,
                 note=m.blocked or m.research_only)
            for m in MODELS.values() if scale is None or scale in m.scales]
    return pd.DataFrame(rows)
