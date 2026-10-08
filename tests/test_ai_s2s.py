"""Regression tests for the AI S2S / seasonal / ocean path (offline; real-data inputs where downloaded)."""
import os
import numpy as np, pytest, xarray as xr
from weatherpre import api, availability as A, catalog as C, leads as L, schema as S, variables as V

ENV = A.Environment(packages={"weatherai": True, "torch": True, "onnxruntime": True, "earth2studio": True},
                    weights={"fuxi-s2s": "/w/fuxi", "orca-dl": "/w/orca", "ace2": "/w/ace2"}, gpus=[("RTX 4090", 24.0), ("RTX 4090", 24.0)])

# ------------------------------------------------------------------------------------------------- leads / products
def test_6m_is_calendar_months_not_175_days():
    lv = L.parse("6m")
    assert lv.scale == C.SEASONAL and lv.months == (1, 2, 3, 4, 5, 6)
    p = lv.periods("2026-12-01")
    assert [x.label[3:] for x in p] == ["2026-12", "2027-01", "2027-02", "2027-03", "2027-04", "2027-05"]
    assert lv.horizon_days("2026-12-01") == 182.0            # Dec..May, not 175
    q = L.parse("6m").periods("2026-12-15")                  # mid-month init: month 1 is the first FULL month
    assert str(q[0].start)[:10] == "2027-01-01" and str(q[-1].end)[:10] == "2027-07-01"
    assert L.parse("m2-3").months == (2, 3)

def test_explicit_target_window_and_incomplete_week_note():
    lv = L.parse(target_start="2027-01-01", target_end="2027-06-30")
    p = lv.periods("2026-12-01")
    assert len(p) == 6 and str(p[0].start)[:10] == "2027-01-01" and str(p[-1].end)[:10] == "2027-07-01"
    assert L.parse(target_start="2027-01-01", target_end="2027-03-31", aggregation="period").periods("2026-12-01")[0].days == 90
    with pytest.raises(ValueError, match="calendar-month boundaries"):
        L.parse(target_start="2027-01-05", target_end="2027-03-31").periods("2026-12-01")
    with pytest.raises(ValueError, match="before init"):
        L.parse(target_start="2026-11-01", target_end="2027-03-31").periods("2026-12-01")
    assert "incomplete week 7" in L.parse("45d").notes[0]

# ------------------------------------------------------------------------------------------------- variables
def test_sst_is_a_variable_and_never_t2m_or_proxy():
    assert api.variables("sst") == ["sst"] and api.variables("SST,t2m") == ["sst", "t2m"]
    assert V.VARS["tos"].proxy_for == "sst" and "sst" in V.VARS["t2m"].never_substitute
    a = A.check("orca-dl", "sst", "6m", "2022-12-01", env=ENV, now="2026-10-08")
    assert "proxy_variable" in a.codes() and not a.ok
    with pytest.raises(api.BackendUnavailable, match="proxy"):
        api.forecast("orca-dl", "sst", "6m", init="2022-12-01", verbose=False)

# ------------------------------------------------------------------------------------------------- catalog / availability
def test_listing_shows_all_s2s_models_and_states_are_separate():
    t = C.table("s2s")
    assert {"fuxi-s2s", "dlesym", "ace2", "ucast"} - set(t.model) == {"ucast"}       # ucast is weather-scale only now
    assert {"fuxi-s2s", "ace2"} <= set(t.model) and {"real_data", "report_eligible", "technical_max"} <= set(t.columns)
    assert set(C.table("seasonal").model) >= {"orca-dl", "unicm", "samudrace"}
    for m in C.MODELS.values():
        if m.kind == "AI": assert m.state("report_eligible") == "no", m.name

def test_fuxi_42_day_limit_in_resolver_and_availability():
    for lead in ("6m", "8w", "7w"):
        with pytest.raises(api.BackendUnavailable):
            api.resolve("fuxi-s2s", "2020-10-01", L.parse(lead))
    assert api.resolve("fuxi-s2s", "2020-10-01", L.parse("6w"))[0] == "weatherai"       # 6 weeks from the day after init = 42 d
    a = A.check("fuxi-s2s", "t2m", "8w", "2020-10-01", env=ENV)
    assert "beyond_technical_horizon" in a.codes()
    assert A.check("fuxi-s2s", "t2m", "6w", "2020-10-01", env=ENV, now="2026-10-08").ok

def test_reasons_are_distinct():
    none = A.Environment(packages={}, weights={}, gpus=[])
    r = A.check("fuxi-s2s", "t2m", "6w", "2020-10-01", env=none, now="2026-10-08")
    assert {"not_installed", "weights_missing"} <= set(r.codes()) and "cpu_only" in r.codes()
    r = A.check("fuxi-s2s", "t2m", "6w", "2026-12-01", env=ENV, now="2026-10-08")
    assert "input_missing" in r.codes()
    r = A.check("fuxi-s2s", "t2m", "6w", "2026-10-08", env=ENV, now="2026-10-08T12", source="file:x.nc")
    assert "input_not_published" in r.codes()
    r = A.check("orca-dl", "tos", "6m", "2026-12-01", env=ENV, now="2026-10-08")
    assert "input_not_published" in r.codes()
    assert A.check("orca-dl", "tos", "6m", "2026-12-01", env=ENV, now="2026-12-20").ok
    assert "blocked" in A.check("unicm", "sst_anom", "6m", "2026-12-01", env=ENV).codes()
    assert "research_only" in A.check("samudrace", "sst", "6m", "2026-12-01", env=ENV).codes()
    assert "gpu_insufficient" in A.check("ucast", "t2m", "7d", "2026-10-01", env=ENV).codes()   # 40 GB badge > one 24 GB card
    assert "forcing_missing" in A.check("ace2", "t2m", "6w", "2020-10-01", env=ENV).codes()
    assert "wrong_product" in A.check("graphcast", "t2m", "6m", "2020-10-01", env=ENV).codes()
    assert "too_many_members" in A.check("orca-dl", "tos", "6m", "2022-12-01", env=ENV, members=9).codes()

def test_auto_s2s_candidates_come_from_the_catalog():
    names = api.best_models("2020-10-01", "s2s")
    assert {"gefs", "ifs-ext", "cfsv2", "climatology", "persistence"} <= set(names)
    assert "partial_coverage" in A.check("gefs", "t2m", "6w", "2020-10-01").codes()      # 35-day product: week 6 reported incomplete
    assert not {"fuxi-s2s", "orca-dl", "ace2"} & set(names)                           # model runs need an explicit plan
    latest = api.best_models("latest", "s2s")
    assert "ifs-ext" not in latest

def test_plan_never_starts_gpu_work_without_permission():
    from weatherpre import planner
    p = planner.plan(["fuxi-s2s", "orca-dl", "ucast"], "t2m", "6w", "2020-10-01", env=ENV, members=11)
    t = p.table().set_index("model")
    assert t.loc["fuxi-s2s", "decision"] in ("run", "skip") and t.loc["fuxi-s2s", "device"] != "gpu"
    p2 = planner.plan(["fuxi-s2s"], "t2m", "6w", "2020-10-01", env=ENV, members=11, limits=planner.Limits(allow_gpu=True))
    assert p2.table().loc[0, "device"] == "gpu"
    assert set(p.execute(dry_run=True)) <= {"fuxi-s2s", "orca-dl"}

# ------------------------------------------------------------------------------------------------- Earth2Studio path
def test_e2s_source_is_passed_through(monkeypatch):
    import weatherpre.adapters.earth2studio_run as E
    seen = {}
    def fake(cls, t, leads, vs, source=None, model_name=None):
        seen.update(source=source, cls=cls)
        lead = np.array(leads); lat = np.linspace(-90, 90, 3); lon = np.array([0.0, 180.0])
        u = xr.Dataset({"z500": (("lead", "latitude", "longitude"), np.full((len(lead), 3, 2), 5500.0), {"units": "m"})},
                       coords=dict(lead=lead, latitude=lat, longitude=lon))
        return u.assign_attrs(model=model_name, backend="e2s", init="x")
    monkeypatch.setattr(E, "forecast", fake); monkeypatch.setattr(E, "available", lambda: True)
    api.forecast("e2s:Persistence", "z500", "12h", init="2020-10-03", source="WB2ERA5_121x240", verbose=False)
    assert seen == {"source": "WB2ERA5_121x240", "cls": "Persistence"}

def test_e2s_precip_semantics():
    import weatherpre.adapters.earth2studio_run as E
    da = xr.DataArray(np.full(4, 1e-3), dims="lead", coords={"lead": [0, 24, 48, 72]})
    np.testing.assert_allclose(E.precip_mm_per_step("FuXiS2S", "tp", 24, da).values, 24.0)     # m/h daily mean -> 24 mm per day
    np.testing.assert_allclose(E.precip_mm_per_step("SFNO", "tp06", 6, da).values, 1.0)       # 6-h accumulation in m
    with pytest.warns(UserWarning, match="12 h"):
        assert E.precip_mm_per_step("X", "tp06", 12, da) is None
    with pytest.warns(UserWarning, match="no verified meaning"):
        assert E.precip_mm_per_step("FCN3", "tp", 6, da) is None
    with pytest.raises(api.BackendUnavailable, match="daily means"):
        api._forecast_s2s("e2s", "fuxi-s2s", api._init("2020-10-01"), L.parse("6w"), ["t2m"], None)

# ------------------------------------------------------------------------------------------------- aggregation
def _daily(n=42, members=2, init="2027-01-01"):
    t0 = np.datetime64(init, "ns"); day = np.timedelta64(1, "D")
    a = np.arange(n, dtype=float)[None, :, None, None] + np.arange(members)[:, None, None, None] * 100 + np.zeros((members, n, 2, 3))
    ds = xr.Dataset({"t2m": (("member", "lead", "latitude", "longitude"), a, {"units": "K"})},
                    coords=dict(member=list(range(members)), lead=np.arange(1, n + 1), latitude=[-1.0, 1.0], longitude=[0.0, 1.0, 2.0],
                                init_time=t0, valid_start=("lead", t0 + np.arange(n) * day), valid_end=("lead", t0 + np.arange(1, n + 1) * day)))
    return S.finalize(ds, model="toy", backend="test", source="synthetic", product="subseasonal", time_semantics="daily mean",
                      member_kind="test", independence_group="toy")

def test_weekly_and_monthly_aggregation_with_completeness():
    ds = _daily()
    w = S.aggregate(ds, L.parse("6w").periods("2027-01-01"))
    assert w.sizes["period"] == 6 and w.sizes["member"] == 2                          # members kept
    np.testing.assert_allclose(w.t2m.isel(member=0, period=0, latitude=0, longitude=0), 3.0)   # mean of days 0..6
    np.testing.assert_allclose(w.t2m.isel(member=1, period=0, latitude=0, longitude=0), 103.0)
    m = S.aggregate(ds, L.parse("2m").periods("2027-01-01"))                          # Feb only has 11 of 28 days
    assert list(m.period.values) == ["m1 2027-01"] and "2027-02" in m.attrs["incomplete_periods"]
    m2 = S.aggregate(ds, L.parse("2m").periods("2027-01-01"), require_complete=False)
    np.testing.assert_allclose(m2.completeness.values, [1.0, 11 / 28])
    assert S.validate(w) == []

def test_monthly_means_are_not_downscaled_to_weeks():
    t0 = np.datetime64("2027-01-01", "ns")
    ds = xr.Dataset({"tos": (("member", "lead"), np.ones((1, 2)), {"units": "degC"})}, coords=dict(
        member=[1], lead=[1, 2], init_time=t0, valid_start=("lead", np.array(["2027-01-01", "2027-02-01"], "datetime64[ns]")),
        valid_end=("lead", np.array(["2027-02-01", "2027-03-01"], "datetime64[ns]"))))
    with pytest.raises(ValueError, match="cannot downscale"):
        S.aggregate(ds, L.parse("2w").periods("2027-01-01"))

def test_nan_mask_is_kept_not_zero_filled():
    ds = _daily(n=7)
    ds["t2m"][:, :, 0, 0] = np.nan
    w = S.aggregate(ds, L.parse("1w").periods("2027-01-01"))
    assert np.isnan(w.t2m.values[:, 0, 0, 0]).all() and np.isfinite(w.t2m.values[:, 0, 1, 1]).all()

# ------------------------------------------------------------------------------------------------- adapters end to end
@pytest.mark.skipif(not os.path.exists(os.path.join(os.environ.get("ORCA_EXAMPLE", "/home/user/data/orca_example"), "salt.nc")),
                    reason="official ORCA-DL example_data not downloaded")
def test_orca_dl_adapter_end_to_end_on_godas_example(tmp_path):
    pytest.importorskip("weatherai.inference"); pytest.importorskip("torch")
    from tests._stubs import ORCA_EXAMPLE, orca_persistence_forecaster
    f, inp = orca_persistence_forecaster(ORCA_EXAMPLE)
    from weatherpre import runners
    req = runners.RunRequest("orca-dl", ["tos", "thetao"], L.parse("3m"), api._init("1980-02-01"), members=[1, 2],
                             source=f"dir:{ORCA_EXAMPLE}", out_dir=tmp_path, options={"forecaster": f})
    native = runners.get("orca-dl")(req)
    assert str(native.init_time.values)[:10] == "1980-02-01" and native.attrs["ic_period"].startswith("1980-01")
    out = S.aggregate(native, L.parse("3m").periods("1980-02-01"))
    assert list(out.period.values) == ["m1 1980-02", "m2 1980-03", "m3 1980-04"] and out.sizes["member"] == 2
    tos0 = inp.fields["tos"]
    np.testing.assert_allclose(out.tos.isel(member=0, period=2).values, tos0, equal_nan=True, atol=1e-4)   # persistence stub
    assert np.isnan(out.tos.values[:, :, ~np.isfinite(tos0)]).all()                                     # land stays NaN
    assert out.attrs["member_kind"] == "checkpoint_seed" and out.tos.attrs["proxy_for"] == "sst"
    assert S.validate(native) == [] and os.path.exists(tmp_path / "orca-dl_ic1980-01.weatherpre.json")

@pytest.mark.skipif(not os.path.exists(os.path.join(os.environ.get("FUXI_SAMPLE", "/home/user/data/fuxi_s2s_sample"), "mask.nc")),
                    reason="official FuXi-S2S sample not downloaded")
def test_fuxi_s2s_adapter_end_to_end_on_official_sample(tmp_path):
    pytest.importorskip("onnxruntime"); pytest.importorskip("weatherai.inference")
    from tests._stubs import FUXI_SAMPLE, fuxi_stub_onnx
    from weatherai.inference.fuxi_s2s import FuXiS2SForecaster
    f = FuXiS2SForecaster.from_onnx(fuxi_stub_onnx(str(tmp_path / "w")), cache_dir=str(tmp_path / "g"), threads=2)
    from weatherpre import runners
    req = runners.RunRequest("fuxi-s2s", ["t2m", "tp", "sst"], L.parse("2w"), api._init("2020-06-02"), members=2, seed=3,
                             source=f"official-sample:{FUXI_SAMPLE}", out_dir=tmp_path / "run", options={"forecaster": f})
    native = runners.get("fuxi-s2s")(req)
    assert native.sizes["lead"] == 14 and native.sizes["member"] == 2
    assert str(native.init_time.values)[:10] == "2020-06-03"                     # end of the init day's mean
    assert str(native.valid_start.values[0])[:10] == "2020-06-03"
    w = S.aggregate(native, L.parse("2w").periods(native.init_time.values))
    assert list(w.period.values) == ["w1", "w2"]
    # tp: model channel mm/h -> mm/day (x24); stub adds 0.01*k + noise per step to the input value, finite everywhere
    assert 1.0 < float(w.tp.mean()) < 20.0 and w.tp.attrs["units"] == "mm/day"
    assert np.isnan(native.sst.values[:, :, native.valid_sst.values == 0]).all() and native.sst.attrs["units"] == "degC"
    assert 5 < float(native.sst.mean()) < 25                                      # degC (K would be ~287)
    # resume: a second identical request reuses the member files
    files = sorted((tmp_path / "run").rglob("member_*.nc")); t0 = [os.path.getmtime(p) for p in files]
    runners.get("fuxi-s2s")(req)
    assert [os.path.getmtime(p) for p in files] == t0

@pytest.mark.skipif(not os.path.exists(os.path.join(os.environ.get("ORCA_EXAMPLE", "/home/user/data/orca_example"), "salt.nc")),
                    reason="official ORCA-DL example_data not downloaded")
def test_public_forecast_api_orca_target_window(tmp_path, monkeypatch):
    pytest.importorskip("weatherai.inference"); pytest.importorskip("torch")
    from tests._stubs import ORCA_EXAMPLE, orca_persistence_forecaster
    from weatherai import inference as wi
    f, inp = orca_persistence_forecaster(ORCA_EXAMPLE)
    monkeypatch.setattr(wi, "load", lambda *a, **k: f)
    monkeypatch.setenv("WEATHERPRE_WEIGHTS_ORCA_DL", "/unused")          # weights root is not opened by the stand-in
    ds = api.forecast("orca-dl", "tos", init="1980-02-01", target_start="1980-04-01", target_end="1980-06-30",
                      members=[1, 2], source=f"dir:{ORCA_EXAMPLE}", out_dir=tmp_path, verbose=False, checkpoint="/unused")
    assert list(ds.period.values) == ["1980-04", "1980-05", "1980-06"] and ds.sizes["member"] == 2
    assert ds.attrs["product"] == "seasonal" and "valid_tos" in ds and ds.tos.attrs["units"] == "degC"
    assert (tmp_path / "orca-dl_1980-02-01_seasonal.nc").exists()
    with pytest.raises(api.BackendUnavailable, match="input_missing"):
        api.forecast("orca-dl", "tos", "3m", init="1990-02-01", source="orca-example", checkpoint="/unused", verbose=False)
