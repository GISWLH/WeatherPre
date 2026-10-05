import datetime as dt
import pytest
from weatherpre import api, leads as L

W, S = L.parse("7d"), L.parse("6w")

@pytest.mark.parametrize("model,init,lv,backend", [
    ("graphcast", "2020-10-03", W, "wb2"), ("ifs", "2020-10-03", W, "wb2"), ("pangu", "2019-06-01T12", W, "wb2"),
    ("gefs", "2020-10-03", W, "gefs"), ("gefs", "2020-10-01", S, "gefs"), ("cfsv2", "2021-01-01", S, "cfs"),
    ("ifs-ext", "2020-10-01", S, "wb2-ext"), ("climatology", "latest", S, "baseline"), ("persistence", "2020-10-01", S, "baseline"),
    ("aifs", "latest", W, "opendata"), ("gfs", "latest", W, "noaa"), ("aigfs", "latest", W, "noaa"),
])
def test_routes(model, init, lv, backend):
    assert api.resolve(model, init, lv)[0] == backend

def test_wrong_scale_explains_alternatives():
    with pytest.raises(api.BackendUnavailable, match="ifs-ext"):
        api.resolve("graphcast", "2020-10-03", S)
    with pytest.raises(api.BackendUnavailable, match="S2S"):
        api.resolve("ifs-ext", "2020-10-01", W)

@pytest.mark.parametrize("model,init,lv", [("aifs", "2020-10-03", W), ("gefs", "2019-01-01", W), ("ifs-ext", "latest", S),
                                            ("weathernext", "latest", W), ("nope", "latest", W)])
def test_unavailable(model, init, lv):
    with pytest.raises(api.BackendUnavailable):
        api.resolve(model, init, lv)

def test_gpu_only_model_without_earth2studio():
    try:
        import earth2studio  # noqa: F401
        pytest.skip("earth2studio installed")
    except ImportError:
        with pytest.raises(api.BackendUnavailable, match="Earth2Studio"):
            api.resolve("fcn3", "2020-10-03", W)

def test_variables():
    assert api.variables(None) == list(api.VARIABLES)
    assert api.variables("T2M,precip") == ["t2m", "tp"]
    assert api.variables(["z", "mslp"]) == ["z500", "msl"]
    with pytest.raises(ValueError):
        api.variables("humidity")

def test_init_parsing():
    assert api._init("2020-10-03") == dt.datetime(2020, 10, 3)
    assert api._init("2020-10-03T12") == dt.datetime(2020, 10, 3, 12)
    assert api._init("latest") == "latest"
    assert api._looks_like_init("2020-10-03") and not api._looks_like_init("t2m")
