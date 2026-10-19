import numpy as np
import pytest
import xarray as xr
from weatherpre import catalog as C


def test_research_catalog_and_forecast_guard():
    from weatherpre.api import resolve, BackendUnavailable
    for name, role in (("pbc", "calibration"), ("genfocal", "downscaling"), ("translocation", "diagnostic")):
        spec = C.get(name)
        assert spec.role == role and spec.status == "research" and not spec.hosted
        with pytest.raises(BackendUnavailable, match="research method"):
            resolve(name, "2020-10-03")
    assert C.canonical("gray-swan") == "translocation"


def test_pbc_xarray_coordinate_preservation():
    pytest.importorskip("weatherai")
    import torch
    import weatherpre as wp
    torch.manual_seed(3)
    train = [torch.rand(20, 2, 4) for _ in range(4)]
    model = wp.load_research("pbc", lite=True).fit(train[0], torch.rand_like(train[0]), *train[1:])
    coords = {"time": np.arange(20), "latitude": [20., 30.], "threshold": [.2, .4, .6, .8]}
    arrays = [xr.DataArray(a.numpy(), dims=("time", "latitude", "threshold"), coords=coords) for a in train]
    result = wp.calibrate(model, *arrays)
    assert result.dims == ("time", "latitude", "category")
    np.testing.assert_allclose(result.sum("category"), 1, atol=1e-6)
    np.testing.assert_array_equal(result.latitude, arrays[0].latitude)
    with pytest.raises(ValueError):
        wp.calibrate(model, arrays[0], arrays[1].assign_coords(latitude=[10., 40.]), *arrays[2:])


def test_lite_research_factories():
    pytest.importorskip("weatherai")
    import weatherpre as wp
    assert wp.load_research("genfocal", lite=True, channels=2).super_resolution is not None
    assert wp.load_research("gray-swan", lite=True).backbone is not None


def test_default_dispatch_is_official_without_fallback(monkeypatch):
    import weatherai
    import weatherpre as wp
    def unavailable(**kwargs):
        raise RuntimeError("official checkpoint unavailable")
    monkeypatch.setattr(weatherai.GenFocal, "from_pretrained", unavailable)
    with pytest.raises(RuntimeError, match="official checkpoint unavailable"):
        wp.load_research("genfocal")
    with pytest.raises(TypeError, match="work_dir"):
        wp.load_research("pbc")
