"""Model catalog: the single source of truth for every model WeatherPre knows about.

Each entry says which forecast scale it serves (``weather`` = up to 15 days, ``s2s`` = sub-seasonal to seasonal,
beyond 15 days), how it is reached (hosted data first, GPU inference second) and where its licence / paper live.
`weatherpre models`, the README model table and backend resolution are all driven from here."""
from __future__ import annotations
from dataclasses import dataclass, field

WEATHER, S2S = "weather", "s2s"
WEATHER_MAX_H = 360                       # 15 days: the weather / S2S boundary

@dataclass(frozen=True)
class Model:
    name: str
    label: str
    kind: str                             # "AI" | "NWP" | "baseline"
    scales: tuple[str, ...]               # subset of (WEATHER, S2S)
    routes: tuple[str, ...]               # wb2 | opendata | noaa | gefs | cfs | wb2-ext | baseline | hf | e2s
    grid: str
    lead: str
    period: str                           # where hosted data exists
    licence: str
    ref: str = ""                         # paper / model card
    e2s: str = ""                         # earth2studio.models.px class name (GPU route)
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def hosted(self) -> bool:             # runs on any laptop (no GPU)
        return any(r not in ("hf", "e2s") for r in self.routes)

    @property
    def status(self) -> str:
        if self.hosted: return "hosted"
        return "gpu" if self.routes else "blocked"

_W, _S, _WS = (WEATHER,), (S2S,), (WEATHER, S2S)
MODELS: dict[str, Model] = {m.name: m for m in [
    # ---------------- weather scale, hosted (no GPU) ----------------
    Model("ifs-hres", "ECMWF IFS HRES", "NWP", _W, ("opendata", "wb2"), "0.25° / 1.5°", "≤15 d",
          "latest ~4 d (Open Data) · 2016–2022 (WB2)", "CC-BY-4.0", "https://www.ecmwf.int/en/forecasts",
          aliases=("ifs", "hres", "ecmwf")),
    Model("aifs-single", "ECMWF AIFS", "AI", _W, ("opendata",), "0.25°", "≤15 d", "latest ~4 d (Open Data)",
          "CC-BY-4.0", "https://arxiv.org/abs/2406.01465", e2s="AIFS", aliases=("aifs",)),
    Model("aigfs", "NOAA AIGFS", "AI", _W, ("noaa",), "0.25°", "≤16 d", "2026-04-16 → now (NOAA S3)",
          "public domain", "https://registry.opendata.aws/noaa-nws-graphcastgfs-pds/", aliases=("gfs-ai",)),
    Model("gfs", "NOAA GFS", "NWP", _W, ("noaa",), "0.25°", "≤16 d", "2022 → now (NOAA S3)", "public domain",
          "https://registry.opendata.aws/noaa-gfs-bdp-pds/"),
    Model("gefs", "NOAA GEFS ens. mean", "NWP", _WS, ("gefs",), "0.5°", "≤35 d (00Z)", "2020-09-23 → now (NOAA S3, GEFSv12)",
          "public domain", "https://registry.opendata.aws/noaa-gefs/"),
    Model("graphcast", "GraphCast", "AI", _W, ("wb2", "e2s"), "1.5° (hosted)", "≤10 d", "2019-11 → 2021-01 (WB2)",
          "weights CC-BY-NC-SA-4.0", "https://doi.org/10.1126/science.adi2336", e2s="GraphCastOperational"),
    Model("pangu", "Pangu-Weather", "AI", _W, ("wb2", "e2s"), "1.5° (hosted)", "≤10 d", "2018–2022 (WB2)",
          "weights CC-BY-NC-SA-4.0", "https://doi.org/10.1038/s41586-023-06185-3", e2s="Pangu6"),
    Model("fuxi", "FuXi", "AI", _W, ("wb2", "e2s"), "1.5° (hosted)", "≤15 d", "2020 (WB2)", "see upstream",
          "https://doi.org/10.1038/s41612-023-00512-1", e2s="FuXi"),
    Model("gencast", "GenCast (ens. mean)", "AI", _W, ("wb2",), "1.5°", "≤15 d, 12 h", "2020 (WB2)",
          "weights CC-BY-NC-SA-4.0", "https://doi.org/10.1038/s41586-024-08252-9", aliases=("wn",)),
    Model("neuralgcm", "NeuralGCM", "AI", _W, ("wb2",), "1.5°", "≤15 d, 12 h", "2020 (WB2)", "see upstream",
          "https://doi.org/10.1038/s41586-024-07744-y"),
    Model("ifs-ens", "ECMWF IFS ENS mean", "NWP", _W, ("wb2",), "1.5°", "≤15 d", "2018–2022 (WB2)", "WB2 terms",
          "https://www.ecmwf.int/en/forecasts"),
    # ---------------- S2S scale, hosted (no GPU) ----------------
    Model("ifs-ext", "ECMWF extended range (ens. mean)", "NWP", _S, ("wb2-ext",), "1.5°", "46 d, weekly means",
          "2016–2022, Mon/Thu inits (WB2)", "WB2 terms", "https://www.ecmwf.int/en/forecasts/documentation-and-support/extended-range",
          aliases=("ecmwf-ext", "ifs-s2s")),
    Model("cfsv2", "NOAA CFSv2", "NWP", _WS, ("cfs",), "1°", "≤9 months", "2020 → now (NOAA S3), member 1",
          "public domain", "https://registry.opendata.aws/noaa-cfs/", aliases=("cfs",)),
    Model("climatology", "ERA5 climatology", "baseline", _S, ("baseline",), "1.5°", "any", "any date (1990–2017 clim.)",
          "Copernicus", "https://weatherbench2.readthedocs.io", aliases=("clim",)),
    Model("persistence", "ERA5 anomaly persistence", "baseline", _S, ("baseline",), "1.5°", "any",
          "1959 – 2023-01 (ERA5 in WB2)", "Copernicus", "https://weatherbench2.readthedocs.io"),
    # ---------------- GPU: HF ZeroGPU / Earth2Studio ----------------
    Model("aurora", "Aurora", "AI", _W, ("hf", "e2s"), "0.25°", "any (chunked)", "any date (ERA5 / IFS ICs)", "MIT",
          "https://doi.org/10.1038/s41586-025-09005-y", e2s="Aurora"),
    Model("aurora-1.5", "Aurora 1.5", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "MIT",
          "https://github.com/microsoft/aurora", e2s="Aurora1p5_6h"),
    Model("aifs2", "ECMWF AIFS v2", "AI", _W, ("e2s",), "0.25°", "6 h steps", "latest (IFS ICs)", "CC-BY-4.0",
          "https://arxiv.org/abs/2509.18994", e2s="AIFS2"),
    Model("aifs2-ens", "ECMWF AIFS-ENS v2", "AI", _W, ("e2s",), "0.25°", "6 h steps", "latest (IFS ICs)", "CC-BY-4.0",
          "https://arxiv.org/abs/2506.10868", e2s="AIFS2ENS"),
    Model("fcn3", "FourCastNet 3", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "see model card",
          "https://arxiv.org/abs/2507.12144", e2s="FCN3", aliases=("fourcastnet3",)),
    Model("atlas", "NVIDIA Atlas", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "see model card",
          "https://huggingface.co/nvidia/atlas-era5", e2s="Atlas"),
    Model("ucast", "U-CAST", "AI", _WS, ("e2s",), "1.5°", "12 h steps", "any date", "see model card",
          "https://arxiv.org/abs/2604.09041", e2s="UCast", aliases=("u-cast",)),
    Model("weathernext2", "WeatherNext 2 (Cyclones)", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date",
          "see model card", "https://github.com/google-deepmind/weathernext", e2s="WeatherNext2Cyclones"),
    Model("gencast-mini", "GenCast mini", "AI", _W, ("e2s",), "1°", "12 h steps", "any date", "see model card",
          "https://github.com/google-deepmind/graphcast", e2s="GenCastMini"),
    Model("sfno", "SFNO", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "see model card",
          "https://arxiv.org/abs/2306.03838", e2s="SFNO"),
    Model("fengwu", "FengWu", "AI", _W, ("e2s",), "0.25°", "6 h steps", "any date", "unclear (keep private)",
          "https://arxiv.org/abs/2304.02948", e2s="FengWu"),
    Model("fuxi-s2s", "FuXi-S2S", "AI", _S, ("e2s",), "1.5°", "42 d, daily", "any date", "CC-BY-NC-ND",
          "https://doi.org/10.1038/s41467-024-50714-1", e2s="FuXiS2S"),
    Model("dlesym", "DLESyM (atmos.+ocean)", "AI", _S, ("e2s",), "~1° HEALPix", "6 h → seasonal", "any date",
          "see model card", "https://arxiv.org/abs/2409.16247", e2s="DLESyMLatLon"),
    Model("ace2", "ACE2-ERA5", "AI", _S, ("e2s",), "1°", "6 h → climate", "any date (forcings)", "Apache-2.0",
          "https://arxiv.org/abs/2411.11268", e2s="ACE2ERA5"),
    # ---------------- blocked ----------------
    Model("weathernext", "Google WeatherNext 2/3 (hosted)", "AI", _W, (), "0.25°", "≤15 d", "allow-list only",
          "experimental ToS", "https://developers.google.com/weathernext"),
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

def table(scale: str | None = None):
    """pandas DataFrame of the catalog (optionally one scale)."""
    import pandas as pd
    rows = [dict(model=m.name, label=m.label, kind=m.kind, scales="+".join(m.scales), status=m.status,
                 route=",".join(m.routes) or "-", grid=m.grid, lead=m.lead, period=m.period, licence=m.licence)
            for m in MODELS.values() if scale is None or scale in m.scales]
    return pd.DataFrame(rows)
