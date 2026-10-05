import numpy as np, pandas as pd, pytest, xarray as xr
from weatherpre import s2s
from weatherpre.comparison import Comparison
from weatherpre import leads as L

def _weather(grid=1.5, hours=range(0, 337, 6), init="2020-10-01"):
    lat = np.arange(-90, 90.01, grid); lon = np.arange(0, 360, grid)
    lead = np.array(list(hours))
    t2m = 280 + 0.01 * lead[:, None, None] + 0 * lat[None, :, None] + 0 * lon[None, None, :]
    tp = 2.0 * lead[:, None, None] / 24 + 0 * lat[None, :, None] + 0 * lon[None, None, :]     # 2 mm/day, accumulated
    u = xr.Dataset({"t2m": (("lead", "latitude", "longitude"), t2m.astype("float32"), {"units": "K"}),
                    "tp": (("lead", "latitude", "longitude"), tp.astype("float32"), {"units": "mm"})},
                   coords=dict(lead=lead, latitude=lat, longitude=lon))
    it = np.datetime64(init, "ns")
    u = u.assign_coords(init_time=it, valid_time=("lead", it + lead.astype("timedelta64[h]").astype("timedelta64[ns]")))
    u.attrs.update(model="toy", backend="test", source="synthetic", init=init)
    return u

def test_weekly_means_and_precip_rate():
    w = s2s.weekly_from_leads(_weather(), [1, 2])
    assert list(w.week.values) == [1, 2] and w.attrs["scale"] == "s2s"
    np.testing.assert_allclose(float(w.t2m.sel(week=1).mean()), 280 + 0.01 * np.mean(range(0, 168, 6)), rtol=1e-6)
    np.testing.assert_allclose(float(w.tp.sel(week=2).mean()), 2.0, rtol=1e-5)
    assert str(w.valid_start.values[1])[:10] == "2020-10-08" and str(w.valid_end.values[0])[:10] == "2020-10-08"

def test_incomplete_week_is_skipped():
    with pytest.warns(UserWarning, match="week 3"):
        w = s2s.weekly_from_leads(_weather(hours=range(0, 400, 6)), [1, 2, 3])
    assert list(w.week.values) == [1, 2]

def test_regrid_to_wb2_grid():
    u = s2s.to_1p5(_weather(grid=0.5, hours=[0, 6]))
    assert u.sizes["latitude"] == 121 and u.sizes["longitude"] == 240
    assert float(abs(u.t2m.isel(lead=0) - 280).max()) < 1e-3

def test_to_wb2_schema():
    w = s2s.weekly_from_leads(_weather(), [1, 2])
    d = s2s.to_wb2(w)
    assert d.sizes["time"] == 1 and list(d.prediction_timedelta.values.astype("timedelta64[D]").astype(int)) == [0, 7]
    np.testing.assert_allclose(float(d.total_precipitation_24hr.mean()), 0.002, rtol=1e-5)    # m/day

def _cmp():
    rows = []
    for init, bump in (("2020-10-01", 0.0), ("2020-10-05", 1.0)):
        for m, base in (("a", 1.0), ("b", 2.0)):
            for wk in (1, 2):
                rows.append(dict(init=init, model=m, variable="t2m", week=wk, rmse=base + wk + bump, acc=0.9 / wk - 0.1 * base))
    return Comparison("s2s", ["t2m"], L.parse("w1-2"), {}, pd.DataFrame(rows), "test")

def test_summary_pools_mse_over_inits():
    s = _cmp().summary()
    a1 = s[(s.model == "a") & (s.week == 1)]
    np.testing.assert_allclose(a1.rmse.item(), np.sqrt((2.0 ** 2 + 3.0 ** 2) / 2))
    assert a1.n_inits.item() == 2

def test_table_and_ranking():
    c = _cmp()
    t = c.table()
    assert list(t.columns) == ["w1", "w2"] and t.index[0] == "a"
    r = c.ranking()
    assert r.index[0] == "a" and r.loc["a", "mean_rank"] == 1.0
    assert c.ranking("acc").index[0] == "a"
    assert "t2m RMSE" in repr(c)
