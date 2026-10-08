"""Variable registry: canonical names, units, definitions and which names must never be substituted for each other.

Atmosphere (unchanged): z500 [m], t850 [K], t2m [K], msl [hPa], tp [mm since init] (weather) / [mm/day] (weekly, monthly).
Ocean (new): sst [degC], sst_anom [degC], tos [degC], thetao [degC, depth], so [g/kg, depth], zos [m], siconc [1].

``sst`` is a sea-surface temperature product (ERA5 sst, OISST, a model trained on them). ``tos`` is ORCA-DL's top-layer
temperature initialised from GODAS 5 m potential temperature: a near-surface PROXY that is never returned under the name
``sst``. ``t2m`` is never used for either.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Variable:
    name: str
    units: str
    long_name: str
    realm: str                          # atmos | ocean | ice
    depth: bool = False                 # has a depth coordinate
    proxy_for: str = ""                 # e.g. tos -> sst
    never_substitute: tuple = ()

VARS: dict[str, Variable] = {v.name: v for v in [
    Variable("z500", "m", "500 hPa geopotential height", "atmos"),
    Variable("t850", "K", "850 hPa temperature", "atmos"),
    Variable("t2m", "K", "2 m air temperature", "atmos", never_substitute=("sst", "tos")),
    Variable("msl", "hPa", "mean sea-level pressure", "atmos"),
    Variable("tp", "mm/day", "precipitation rate (period mean); weather scale: mm accumulated since init", "atmos"),
    Variable("sst", "degC", "sea surface temperature (product definition in attrs['definition'])", "ocean",
             never_substitute=("t2m", "tos")),
    Variable("sst_anom", "degC", "sea surface temperature anomaly (climatology in attrs)", "ocean", never_substitute=("t2m",)),
    Variable("tos", "degC", "ocean-model top-layer temperature; near-surface proxy of SST (ORCA-DL: GODAS 5 m)", "ocean",
             proxy_for="sst", never_substitute=("sst", "t2m")),
    Variable("thetao", "degC", "sea water potential temperature", "ocean", depth=True),
    Variable("so", "g/kg", "sea water salinity", "ocean", depth=True),
    Variable("zos", "m", "sea surface height", "ocean"),
    Variable("uo", "m/s", "eastward sea water velocity", "ocean", depth=True),
    Variable("vo", "m/s", "northward sea water velocity", "ocean", depth=True),
    Variable("siconc", "1", "sea-ice concentration", "ice"),
]}
ATMOS = ("z500", "t850", "t2m", "msl", "tp")
ALIASES = {"z": "z500", "gh": "z500", "gh500": "z500", "geopotential": "z500", "hgt500": "z500",
           "t": "t850", "temperature850": "t850", "t2": "t2m", "2t": "t2m", "temperature": "t2m", "tas": "t2m",
           "mslp": "msl", "slp": "msl", "pressure": "msl", "precip": "tp", "precipitation": "tp", "rain": "tp", "pr": "tp",
           "sea_surface_temperature": "sst", "ssta": "sst_anom", "sst_anomaly": "sst_anom", "pottmp": "thetao",
           "salt": "so", "salinity": "so", "ssh": "zos", "sshg": "zos", "sic": "siconc", "sea_ice": "siconc"}

def canonical(name: str) -> str:
    k = str(name).strip().lower()
    k = ALIASES.get(k, k)
    if k not in VARS:
        raise ValueError(f"unknown variable {name!r}; choose from {', '.join(VARS)}")
    return k

def parse(v=None, default=ATMOS) -> list[str]:
    """None/'all' -> the default atmosphere set; 'T2M,sst' / ['t2m','precip'] -> canonical names."""
    if v is None or (isinstance(v, str) and v.lower() == "all"): return list(default)
    vs = [v] if isinstance(v, str) else list(v)
    return list(dict.fromkeys(canonical(y) for x in vs for y in str(x).split(",") if y.strip()))
