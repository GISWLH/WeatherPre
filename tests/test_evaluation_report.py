import numpy as np, pandas as pd, pytest, xarray as xr
from weatherpre import hindcast as H, metrics as M, report as R, schema as S

def test_split_overlap_and_model_training_period():
    with pytest.raises(ValueError, match="overlaps"):
        H.Split(train=("1950-01", "2000-12"), calibration=("1991-01", "2010-12"), test=("2011-01", "2022-12"))
    s = H.Split(train=None, calibration=("1991-01", "2010-12"), test=("2011-01", "2022-12"))
    assert s.role("2015-06-01") == "test" and s.role("1995-01-01") == "calibration" and s.role("1980-01-01") is None
    with pytest.raises(ValueError, match="training period"):
        s.check_model(("1958-01", "2018-12"))
    s.check_model(("1850-01", "2010-12"))

def test_no_future_information():
    H.assert_available(["2026-11-30T18"], "2026-12-01T00")
    with pytest.raises(ValueError, match="future"):
        H.assert_available(["2026-12-15"], "2026-12-01")

def _obs():
    t = pd.date_range("1991-01-01", "2022-12-01", freq="MS")
    rng = np.random.default_rng(0)
    a = np.zeros(len(t))
    for i in range(1, len(t)): a[i] = 0.9 * a[i - 1] + 0.4 * rng.standard_normal()      # AR(1) anomalies: persistence has skill
    return pd.Series(27 + np.cos(2 * np.pi * (t.month - 1) / 12) + a, index=t)

def test_persistence_beats_climatology_on_ar1_and_scores_only_test_inits():
    obs = _obs()
    split = H.Split(train=None, calibration=("1991-01", "2010-12"), test=("2011-01", "2022-12"))
    clim = H.climatology(obs, split)
    inits = pd.date_range("2005-02-01", "2022-12-01", freq="MS")              # includes calibration-period inits
    p, c = H.persistence_forecast(obs, clim, inits, [1, 6]), H.climatology_forecast(clim, inits, [1, 6])
    tab = H.evaluate_index(p, obs, split, refs={"climatology": c}, clim=clim)
    assert tab.set_index("lead").n_inits.loc[1] == 144                       # 2011-01..2022-12 only (no calibration-period inits)
    l1 = tab.set_index("lead").loc[1]
    assert l1.msss_vs_climatology > 0.5 and l1.acc > 0.8
    with pytest.raises(ValueError, match="conditional"):
        H.evaluate_index(p, obs, split, experiment_type="conditional_hindcast")

def test_ensemble_scores():
    rng = np.random.default_rng(1)
    centre = xr.DataArray(rng.standard_normal(500), dims="init")
    obs = centre + 0.5 * xr.DataArray(rng.standard_normal(500), dims="init")                       # truth ~ same law as members
    good = centre + 0.5 * xr.DataArray(rng.standard_normal((500, 20)), dims=("init", "member"))
    bad = xr.DataArray(rng.standard_normal((500, 20)), dims=("init", "member")) * 3
    assert float(M.crps_ensemble(good, obs).mean()) < float(M.crps_ensemble(bad, obs).mean())
    one = xr.DataArray(np.array([[1.0, 3.0]]), dims=("init", "member"))
    np.testing.assert_allclose(M.crps_ensemble(one, xr.DataArray([2.0], dims="init")).values, [1.0 - 0.5 * 4 / 2])   # fair estimator
    ss = M.spread_skill(good, obs)
    assert 0.85 < float(ss["ratio"]) < 1.15                                                         # calibrated ensemble
    rel = M.reliability(np.array([0.1, 0.1, 0.9, 0.9]), np.array([0, 0, 1, 1]))
    assert rel["obs_freq"][1] == 0.0 and rel["obs_freq"][9] == 1.0

def test_box_index_ignores_nan_and_weights_area():
    lat = np.arange(-10, 10.1, 2.5); lon = np.arange(180, 250, 2.5)
    da = xr.DataArray(np.full((lat.size, lon.size), 28.0), coords={"latitude": lat, "longitude": lon}, dims=("latitude", "longitude"))
    da[0, :] = np.nan
    assert abs(float(M.box_index(da, "nino34")) - 28.0) < 1e-9

def _common(name, group, lat=np.arange(-63.5, 64, 1.0)):
    lon = np.arange(0.5, 360, 1.0)
    a = np.full((2, 3, lat.size, lon.size), 25.0); a[:, :, :10, :20] = np.nan
    t = np.array(["2027-01-01", "2027-02-01", "2027-03-01", "2027-04-01"], "datetime64[ns]")
    ds = xr.Dataset({"tos": (("member", "period", "latitude", "longitude"), a, {"units": "degC"}),
                     "valid_tos": (("latitude", "longitude"), np.isfinite(a[0, 0]).astype("int8"))},
                    coords=dict(member=[1, 2], period=["m1", "m2", "m3"], latitude=lat, longitude=lon, init_time=t[0],
                                valid_start=("period", t[:3]), valid_end=("period", t[1:]), completeness=("period", [1.0, 1.0, 1.0])))
    return S.finalize(ds, model=name, backend="test", source="synthetic", product="seasonal", time_semantics="monthly mean",
                      member_kind="checkpoint_seed", independence_group=group)

def test_report_bundle_statuses_duplicates_and_coverage(tmp_path):
    a, b = _common("orca-dl", "orca-dl"), _common("orca-dl-seed2", "orca-dl")
    tab = R.export_bundle({"orca-dl": a, "orca-dl-seed2": b}, str(tmp_path), region=(-90, 90, 0, 360))
    assert set(tab[tab.model == "orca-dl"].status) == {"comparison_only"}
    assert set(tab[tab.model == "orca-dl-seed2"].status) == {"excluded"} and tab[tab.model == "orca-dl-seed2"].duplicate_of.iloc[0] == "orca-dl"
    cov = tab.coverage.iloc[0]
    assert 0.85 < cov < 0.9                       # 63.5S-63.5N is ~89.5 % of the sphere, minus the NaN block
    hc = pd.DataFrame({"lead": [1, 2, 3], "n_inits": [100, 100, 100], "msss_vs_climatology": [0.4, 0.1, -0.2]})
    t2 = R.export_bundle({"orca-dl": a}, str(tmp_path / "b"), region=(-60, 60, 0, 360), hindcasts={"orca-dl": hc}, calibrated={"orca-dl": True})
    assert list(t2.status) == ["eligible", "eligible", "comparison_only"]
    import json
    meta = json.load(open(tmp_path / "models.json"))
    assert meta["independent_models"] == ["orca-dl"]
