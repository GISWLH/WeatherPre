"""WeatherAI-backed runners: ORCA-DL (monthly ocean), FuXi-S2S (official ONNX, daily), ACE2 (forced atmosphere), NeuralGCM (JAX).

WeatherAI owns the model side (weights, normalisation, model-specific inference, masks, run manifest); this module owns input
acquisition (weatherpre.sources), conversion to the common schema and canonical units, and the WeatherPre manifest.
Nothing here edits or imports research scripts.
"""
from __future__ import annotations
import math, os
import numpy as np, xarray as xr
from .. import availability as A, schema as S, leads as L
from ..manifest import write_manifest
from ..runners import RunRequest, register

G = 9.80665

def _unavailable(msg):
    from ..api import BackendUnavailable
    return BackendUnavailable(msg)

def _weights(req: RunRequest, key: str) -> str:
    root = req.checkpoint or A.weights_config().get(key)
    if not root:
        raise _unavailable(f"{req.model}: [weights_missing] set WEATHERPRE_WEIGHTS_{key.upper().replace('-', '_')} or pass checkpoint=")
    return root

def _members(req, default_ids):
    return list(req.members) if isinstance(req.members, (list, tuple)) else default_ids(int(req.members))

# ================================================================================================================ ORCA-DL
def orca_ic_month(init) -> str:
    """Initial-condition month of a monthly model. The common-schema init_time is the END of that month: init 2026-12-01T00 ->
    November 2026 monthly mean, lead 1 = December; any init inside December -> the December mean (published ~10 January)."""
    return str(L.month_start(np.datetime64(init, "s"), ceil=True).astype("datetime64[M]") - 1)


@register("orca-dl")
def run_orca_dl(req: RunRequest) -> xr.Dataset:
    from weatherai import inference as wi
    from ..sources import godas
    ic = orca_ic_month(req.init)
    init_time = (np.datetime64(ic, "M") + 1).astype("datetime64[ns]")
    src = req.source or "godas"
    r = A.input_reason("orca-dl", src, init_time, None)
    if r: raise _unavailable(f"orca-dl: [{r.code}] {r.message}")
    cache = str((req.out_dir or "data") / "_cache") if req.out_dir else "data/_cache"
    if src == "orca-example": inp = godas.orca_example_inputs(cache)
    elif src.startswith("dir:"):
        from weatherai.inference.orca_dl import ORCADLInputs
        inp = ORCADLInputs.from_official_example(src[4:], ic)
    elif src == "godas": inp = godas.orca_inputs(ic, os.path.join(cache, "godas"))
    else: raise _unavailable(f"orca-dl: unknown source {src!r} (godas | orca-example | dir:<path>)")
    periods = req.leads.periods(init_time)
    last = max(p.end for p in periods)
    n = int((np.datetime64(last, "M") - np.datetime64(ic, "M")).astype(int)) - 1
    if n < 1: raise _unavailable("orca-dl: the requested periods end before the first forecast month")
    seeds = _members(req, lambda k: list(range(1, k + 1)))
    f = req.options.get("forecaster") or wi.load("orca-dl", root=_weights(req, "orca-dl"), seeds=seeds, device=req.device)
    out_nc = None
    if req.out_dir:
        os.makedirs(req.out_dir, exist_ok=True)
        out_nc = str(req.out_dir / f"orca-dl_ic{ic}_{n}m_seeds{'-'.join(map(str, seeds))}.weatherai.nc")
    res = f.run(inp, lead_months=n, seeds=seeds, out_path=out_nc)
    keep = [v for v in req.variables if v in res.data] or [v for v in ("tos",) if v in res.data]
    c = res.coords
    ds = xr.Dataset(coords={"member": c["member"], "lead": c["lead"], "latitude": c["lat"], "longitude": c["lon"], "depth": c["depth"],
                            "init_time": np.datetime64(init_time, "ns"), "valid_start": ("lead", c["valid_start"][1]),
                            "valid_end": ("lead", c["valid_end"][1])})
    ds["depth"].attrs["units"] = "m"
    for v in keep:
        dims = tuple({"lat": "latitude", "lon": "longitude"}.get(d, d) for d in res.dims[v])
        ds[v] = (dims, res.data[v], dict(res.var_attrs.get(v, {})))
        ds[f"valid_{v}"] = (("latitude", "longitude"), res.masks[v].astype("int8"), {"long_name": f"valid-data mask of {v}"})
    if "tos" in ds: ds["tos"].attrs.update(proxy_for="sst", definition="ORCA-DL top layer; initial state " + inp.tos_definition)
    ds = S.finalize(ds, model="orca-dl", backend="weatherai:torch-native", checkpoint=f.checkpoint, source=inp.source,
                    product="seasonal", time_semantics="monthly mean of calendar month (init month + lead)",
                    member_kind="checkpoint_seed", independence_group="orca-dl", lead_units="months",
                    ic_period=f"{ic} monthly mean", domain="63.5S-63.5N; NaN outside the input-valid ocean mask",
                    weatherai_config_hash=res.manifest.config_hash())
    if req.out_dir:
        write_manifest(req, ds, res.manifest, req.out_dir / f"orca-dl_ic{ic}.weatherpre.json", weatherai_file=out_nc)
    return ds

# ================================================================================================================ FuXi-S2S
FUXI_MAP = {"z500": ("z500", 1 / G, "m"), "t850": ("t850", 1.0, "K"), "t2m": ("t2m", 1.0, "K"), "msl": ("msl", 0.01, "hPa"),
            "tp": ("tp", 24.0, "mm/day"), "sst": ("sst", 1.0, "degC")}

def fuxi_inputs(req: RunRequest, cache: str):
    from ..sources import era5_daily
    from weatherai.inference.fuxi_s2s import FuXiS2SInputs
    src = req.source or "wb2-era5"
    if src == "wb2-era5":
        r = A.input_reason("fuxi-s2s", src, req.init, None)
        if r: raise _unavailable(f"fuxi-s2s: [{r.code}] {r.message}")
        return era5_daily.fuxi_inputs(req.init, cache)
    if src.startswith("official-sample:"):
        d = src.split(":", 1)[1]
        return FuXiS2SInputs.from_official_files(os.path.join(d, "sample"), os.path.join(d, "mask.nc"), "official FuXi-S2S sample")
    if src.startswith("file:"):
        da = xr.open_dataarray(src[5:])
        inp = FuXiS2SInputs.from_dataarray(da, era5_daily.mask(cache), source=f"file {src[5:]}")
        if inp.init != np.datetime64(req.init, "D"):
            raise _unavailable(f"fuxi-s2s: {src[5:]} initialises {inp.init}, request is {np.datetime64(req.init, 'D')}")
        return inp
    raise _unavailable(f"fuxi-s2s: unknown source {src!r} (wb2-era5 | file:<input.nc> | official-sample:<dir>)")

@register("fuxi-s2s")
def run_fuxi_s2s(req: RunRequest) -> xr.Dataset:
    from weatherai import inference as wi
    from weatherai.inference.fuxi_s2s import MAX_LEAD_DAYS
    init = np.datetime64(req.init, "s")
    if init != init.astype("datetime64[D]").astype("datetime64[s]"):
        raise _unavailable("fuxi-s2s: init must be a date (00Z); the model is initialised from the daily means of init-1 and init")
    t0 = init + np.timedelta64(1, "D")              # end of the init day's mean = start of lead day 1 (common init_time)
    need = req.leads.horizon_days(t0)
    if need > MAX_LEAD_DAYS:
        raise _unavailable(f"fuxi-s2s: [beyond_technical_horizon] request needs {need:g} days; the model runs at most {MAX_LEAD_DAYS}")
    days = int(math.ceil(need))
    vs = [v for v in req.variables if v in FUXI_MAP]
    if not vs: raise _unavailable(f"fuxi-s2s: [variable_unsupported] provides {', '.join(FUXI_MAP)}")
    cache = str(req.out_dir / "_cache") if req.out_dir else "data/_cache"
    inp = fuxi_inputs(req, cache)
    ids = _members(req, lambda k: list(range(k)))
    f = req.options.get("forecaster")
    if f is None:
        root = _weights(req, "fuxi-s2s")
        f = wi.load("fuxi-s2s", backend=req.backend or "onnx", path=root, device=req.device)
    keep = [FUXI_MAP[v][0] for v in vs]
    if req.out_dir:
        run_dir = req.out_dir / f"fuxi-s2s_{str(init)[:10]}_seed{req.seed}"
        idx = f.run(inp, days, members=ids, seed=req.seed, keep=keep, out_dir=str(run_dir), resume=req.resume)
        raw = idx.open()
        fields, units = raw["fields"].values, raw["units"].values
        ocean = raw["ocean_mask"].isel(member=0).values.astype(bool) if "member" in raw["ocean_mask"].dims else raw["ocean_mask"].values.astype(bool)
        lat, lon, vstart, vend = raw.lat.values, raw.lon.values, raw.valid_start.values, raw.valid_end.values
        if "member" in raw["valid_start"].dims: vstart, vend = vstart[0], vend[0]
        from weatherai.inference.manifest import RunManifest
        man = RunManifest.read(idx.files[0] + ".manifest.json")
    else:
        res = f.run(inp, days, members=ids, seed=req.seed, keep=keep)
        fields, ocean = res.data["fields"], res.data["ocean_mask"].astype(bool)
        lat, lon, vstart, vend = res.coords["lat"], res.coords["lon"], res.coords["valid_start"][1], res.coords["valid_end"][1]
        man = res.manifest
    order = np.argsort(lat)
    ds = xr.Dataset(coords={"member": ids, "lead": np.arange(1, days + 1), "latitude": lat[order], "longitude": lon,
                            "init_time": np.datetime64(t0, "ns"), "valid_start": ("lead", vstart), "valid_end": ("lead", vend)})
    for v in vs:
        ch, k, units = FUXI_MAP[v]
        a = fields[:, :, keep.index(ch)][:, :, order].astype(np.float64) * k
        if v == "sst": a = a - 273.15
        attrs = {"units": units, "long_name": f"FuXi-S2S daily mean {v}"}
        if v == "tp": attrs.update(definition="24 x daily-mean hourly precipitation rate (model channel mm/h)")
        if v == "sst": attrs.update(definition="ERA5-like sea surface temperature (FuXi-S2S emulates ERA5 sst)")
        ds[v] = (("member", "lead", "latitude", "longitude"), a.astype(np.float32), attrs)
    if "sst" in ds:
        ds["valid_sst"] = (("latitude", "longitude"), ocean[order].astype("int8"), {"long_name": "official FuXi-S2S ocean mask"})
    ds = S.finalize(ds, model="fuxi-s2s", backend=f"weatherai:{man.backend}", checkpoint=man.checkpoint, source=inp.source,
                    product="subseasonal", time_semantics="daily mean of UTC day init+lead", member_kind="stochastic_latent_draw",
                    independence_group="fuxi-s2s", lead_units="days", seed=req.seed, licence="weights CC-BY-NC-ND-4.0",
                    ic_period=f"daily means of {str(init - np.timedelta64(1, 'D'))[:10]} and {str(init)[:10]} (official init {str(init)[:10]})",
                    max_lead_days=MAX_LEAD_DAYS, weatherai_config_hash=man.config_hash())
    if req.out_dir:
        write_manifest(req, ds, man, req.out_dir / f"fuxi-s2s_{str(init)[:10]}.weatherpre.json")
    return ds

# ================================================================================================================ ACE2
ACE2_MAP = {"t2m": ("TMP2m", 1.0, "K"), "t850": ("TMP850", 1.0, "K"), "z500": ("h500", 1.0, "m"), "msl": ("PRMSL", 0.01, "hPa"),
            "tp": ("PRATEsfc", 86400.0, "mm/day")}

@register("ace2")
def run_ace2(req: RunRequest) -> xr.Dataset:
    """``req.forcing`` = dict(ic={name: (H,W)}, forcing={name: (T,H,W) | (T,) | (H,W)}, times=datetime64[T]); provenance mandatory."""
    from weatherai.inference.ace2 import ACE2Forecaster, PROVENANCE
    if req.forcing_provenance not in PROVENANCE:
        raise _unavailable(f"ace2: [forcing_missing] forcing_provenance must be one of {sorted(PROVENANCE)}; with observed future SST the "
                           "run is a conditional hindcast, not a forecast")
    if not req.forcing or "forcing" not in req.forcing or "ic" not in req.forcing:
        raise _unavailable("ace2: [forcing_missing] pass forcing=dict(ic=..., forcing=..., times=...) covering every 6-h step")
    init = np.datetime64(req.init, "s")
    steps = int(math.ceil(req.leads.horizon_days(init) * 4))
    f = req.options.get("forecaster") or ACE2Forecaster.from_checkpoint(_weights(req, "ace2"), device=req.device)
    res = f.run(req.forcing["ic"], req.forcing["forcing"], steps, req.forcing_provenance, init=init,
                forcing_times=req.forcing.get("times"), forcing_source=str(req.forcing.get("source", "")),
                lat=req.forcing.get("lat"), lon=req.forcing.get("lon"))
    lat = np.asarray(res.coords["lat"]); order = np.argsort(lat)
    lead_h = np.asarray(res.coords["lead"])
    vt = np.datetime64(init, "ns") + lead_h.astype("timedelta64[h]").astype("timedelta64[ns]")
    ds = xr.Dataset(coords={"member": [0], "lead": lead_h, "latitude": lat[order], "longitude": res.coords["lon"],
                            "init_time": np.datetime64(init, "ns"),
                            "valid_start": ("lead", vt - np.timedelta64(6, "h").astype("timedelta64[ns]")), "valid_end": ("lead", vt)})
    for v in req.variables:
        if v not in ACE2_MAP or ACE2_MAP[v][0] not in res.data: continue
        name, k, units = ACE2_MAP[v]
        ds[v] = (("member", "lead", "latitude", "longitude"), (res.data[name][:, :, order] * k).astype(np.float32),
                 {"units": units, "definition": "6-h mean rate" if v == "tp" else "instantaneous at valid_end, used as the 6-h interval value"})
    return S.finalize(ds, model="ace2", backend="weatherai:torch-native", checkpoint=res.manifest.checkpoint,
                      source=str(req.forcing.get("source", "user arrays")), product="scenario",
                      time_semantics="6-hourly steps; instantaneous state fields approximated as 6-h interval values",
                      member_kind="deterministic", independence_group="ace2", experiment_type=res.attrs["experiment_type"],
                      forcing_provenance=req.forcing_provenance, sst_is_prescribed="yes")

# ================================================================================================================ NeuralGCM
NGCM_MAP = {"z500": ("geopotential", 500, 1 / G, "m"), "t850": ("temperature", 850, 1.0, "K")}

@register("neuralgcm")
def run_neuralgcm(req: RunRequest) -> xr.Dataset:
    """Official JAX NeuralGCM. ``forcing=dict(dataset=<xarray on the checkpoint grid/levels with input + forcing variables>,
    forcing_policy='persist_initial' | 'provided')``; the checkpoint name goes in ``checkpoint`` (default deterministic_2_8_deg).
    Preparing ERA5 on the model grid follows the official inference demo and is left to the caller."""
    fz = req.forcing or {}
    if fz.get("dataset") is None:
        raise _unavailable("neuralgcm: [input_missing] pass forcing=dict(dataset=..., forcing_policy=...) with ERA5 regridded to the "
                           "checkpoint's grid and 37 pressure levels (official neuralgcm inference demo)")
    if fz.get("forcing_policy") not in ("persist_initial", "provided"):
        raise _unavailable("neuralgcm: [forcing_missing] forcing_policy must be 'persist_initial' (SST/sea ice held at the initial "
                           "values) or 'provided' (forcing for every output time in the dataset)")
    from weatherai.inference.neuralgcm import NeuralGCMForecaster
    f = req.options.get("forecaster") or NeuralGCMForecaster.from_pretrained(req.checkpoint or "deterministic_2_8_deg")
    hours = list(req.leads.hours) if req.leads.scale == "weather" else [int(req.leads.horizon_days(req.init) * 24)]
    steps = max(hours) // 6 + 1
    ids = _members(req, lambda k: list(range(k)))
    res = f.run(fz["dataset"], steps, step_hours=6, forcing_policy=fz["forcing_policy"], members=len(ids), seed=req.seed,
                variables=sorted({NGCM_MAP[v][0] for v in req.variables if v in NGCM_MAP}))
    c = res.coords
    lead = np.asarray(c["lead"])
    sel = np.isin(lead, hours)                          # lead 0 is the decoded initial state, never returned
    lat = np.asarray(c["latitude"]); order = np.argsort(lat)
    ds = xr.Dataset(coords={"member": ids, "lead": lead[sel], "latitude": lat[order], "longitude": c["longitude"],
                            "init_time": np.datetime64(req.init, "ns")})
    for v in req.variables:
        if v not in NGCM_MAP: continue
        name, lev, k, units = NGCM_MAP[v]
        a = res.data[name]                              # (member, lead, level, longitude, latitude)
        li = list(np.asarray(c["level"])).index(lev)
        a = np.moveaxis(a[:, sel, li], -1, -2)[:, :, order] * k
        ds[v] = (("member", "lead", "latitude", "longitude"), a.astype(np.float32), {"units": units})
    vt = ds.init_time.values + ds.lead.values.astype("timedelta64[h]").astype("timedelta64[ns]")
    ds = ds.assign_coords(valid_start=("lead", vt), valid_end=("lead", vt))
    return S.finalize(ds, model="neuralgcm", backend="weatherai:jax-official", checkpoint=f.checkpoint, source="caller dataset",
                      product="weather", time_semantics="instantaneous", member_kind="stochastic_seed" if f.stochastic else "deterministic",
                      independence_group="neuralgcm", forcing_policy=fz["forcing_policy"])
