"""Live-data smoke tests (WeatherBench 2 on GCS, NOAA S3). Run with WEATHERPRE_NETWORK=1 pytest -m network."""
import pytest
import weatherpre as wp

pytestmark = pytest.mark.network

def test_wb2_weather_forecast():
    ds = wp.forecast("graphcast", "z500", "48h", init="2020-10-03", verbose=False)
    assert ds.sizes["lead"] == 8 and 4500 < float(ds.z500.mean()) < 6000

def test_s2s_ifs_ext_beats_climatology_in_week1():
    c = wp.compare(["ifs-ext", "climatology"], "t2m", "w1-2", init="2020-10-01", verbose=False)
    t = c.table()
    assert t.loc["ifs-ext", "w1"] < t.loc["climatology", "w1"]

def test_gefs_two_leads():
    ds = wp.forecast("gefs", "t2m", [6, 12], init="2020-10-03", verbose=False)
    assert ds.sizes["latitude"] == 361

def test_earth2studio_persistence():
    pytest.importorskip("earth2studio")
    ds = wp.forecast("e2s:Persistence", "z500", "12h", init="2020-10-03", verbose=False)
    assert list(ds.lead.values) == [6, 12]

def test_single_lead_and_single_week_are_scored():
    c = wp.compare(["graphcast"], "z500", "24", init="2020-10-03", verbose=False)
    assert list(c.table().columns) == ["+24h"]
    s = wp.compare(["climatology"], "t2m", "w1", init="2020-10-01", verbose=False)
    assert list(s.table().columns) == ["w1"]
