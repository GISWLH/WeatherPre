import datetime as dt
import numpy as np, pytest
from weatherpre import api, catalog as C, leads as L
from weatherpre.adapters import canglong_earth as CE

LEVELS = ["200 hPa", "300 hPa", "500 hPa", "700 hPa", "850 hPa"]

def _manifest(weeks=6):
    def var(i, layered=False):
        if layered:
            return {"id": i, "layers": [{"index": k, "label": lab, "files": [{"time": t, "path": f"data/fields/{i}/l{k:02d}_t{t:02d}.bin"}
                                                                           for t in range(weeks)]} for k, lab in enumerate(LEVELS)]}
        return {"id": i, "layers": [{"index": 0, "label": "surface", "files": [{"time": t, "path": f"data/fields/{i}/l00_t{t:02d}.bin"}
                                                                              for t in range(weeks)]}]}
    ids = ["2m_temperature", "mean_sea_level_pressure", "large_scale_rain_rate", "convective_rain_rate"]
    return {"attributes": {"forecast_start_date": "2026-10-08", "model_version": "test"},
            "times": [{"index": t, "week": t + 1} for t in range(weeks)],
            "variables": [var(i) for i in ids] + [var("upper_geopotential", True), var("upper_temperature", True)]}

def _value(path: str) -> float:
    t = int(path[-6:-4])
    if "2m_temperature" in path: return 280.0 + t
    if "mean_sea_level_pressure" in path: return 101300.0
    if "large_scale" in path: return 1e-5
    if "convective" in path: return -3e-5 if t == 0 else 2e-5                  # negative total in week 1 -> clipped to 0
    if "upper_geopotential" in path: return 9.80665 * (5500.0 + 100 * int(path[-10:-8]))
    return 270.0 + int(path[-10:-8])                                          # upper_temperature, by layer index

@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(CE, "release", lambda: {"version": "v1", "source": "x.nc"})
    monkeypatch.setattr(CE, "manifest", lambda version: _manifest())
    def field(version, path, cache):
        a = np.full((CE.NLAT, CE.NLON), _value(path), dtype="<f4")
        a[0] += 1.0                                                           # first row = 90N in the published files
        return a
    monkeypatch.setattr(CE, "_field", field)

def test_catalog_and_resolve():
    m = C.get("cas-canglong")
    assert m.name == "canglong" and m.scales == (C.S2S,) and m.hosted
    assert api.resolve("canglong", "latest", L.parse("6w"))[0] == "canglong"

def test_schema_units_and_orientation(fake):
    u = CE.forecast("latest", [1, 2, 3, 4, 5, 6], ["z500", "t850", "t2m", "msl", "tp"])
    assert u.attrs["scale"] == "s2s" and list(u.week.values) == [1, 2, 3, 4, 5, 6]
    assert str(u.init_time.values)[:10] == "2026-10-08" and str(u.valid_start.values[1])[:10] == "2026-10-15"
    assert float(u.latitude[0]) == -90.0 and float(u.latitude[-1]) == 90.0
    np.testing.assert_allclose(float(u.t2m.sel(week=3).isel(latitude=0, longitude=0)), 282.0)
    np.testing.assert_allclose(float(u.t2m.sel(week=3).isel(latitude=-1, longitude=0)), 283.0)   # +1 row was 90N
    np.testing.assert_allclose(float(u.z500.isel(week=0, latitude=5, longitude=5)), 5700.0, rtol=1e-6)
    np.testing.assert_allclose(float(u.t850.isel(week=0, latitude=5, longitude=5)), 274.0)
    np.testing.assert_allclose(float(u.msl.isel(week=0, latitude=5, longitude=5)), 1013.0, rtol=1e-6)
    assert float(u.tp.sel(week=1).min()) == 0.0
    np.testing.assert_allclose(float(u.tp.sel(week=2).isel(latitude=5, longitude=5)), 3e-5 * 86400, rtol=1e-5)
    assert u.tp.attrs["units"] == "mm/day"

def test_weeks_beyond_run_and_other_inits(fake):
    with pytest.warns(UserWarning, match="beyond"):
        u = CE.forecast("latest", [5, 6, 7, 8], ["t2m"])
    assert list(u.week.values) == [5, 6]
    u = CE.forecast(dt.datetime(2026, 10, 8), [1], ["t2m"])
    assert u.sizes["week"] == 1
    with pytest.raises(api.BackendUnavailable, match="latest run"):
        CE.forecast(dt.datetime(2026, 9, 1), [1], ["t2m"])

@pytest.mark.network
def test_canglong_earth_live():
    u = api.forecast("canglong", "t2m,tp", "6w", verbose=False)
    assert u.sizes["week"] >= 1 and u.sizes["latitude"] == 181
    assert 270 < float(u.t2m.mean()) < 300 and 0 <= float(u.tp.min())
