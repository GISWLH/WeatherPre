import pytest
from weatherpre import catalog as C

def test_every_model_is_consistent():
    for m in C.MODELS.values():
        assert set(m.scales) <= {C.WEATHER, C.S2S, C.SEASONAL}, m.name
        assert m.kind in ("AI", "NWP", "baseline"), m.name
        if "e2s" in m.routes: assert m.e2s, m.name
        assert m.status in ("hosted", "gpu", "run", "blocked")
        assert m.validated_max_days is None or m.technical_max_days is None or m.validated_max_days <= m.technical_max_days
        assert m.state("report_eligible") in ("yes", "no"), m.name

def test_both_scales_have_hosted_models():
    assert len(C.for_scale(C.WEATHER, hosted_only=True)) >= 8
    assert {"ifs-ext", "gefs", "cfsv2", "climatology", "persistence"} <= {m.name for m in C.for_scale(C.S2S, hosted_only=True)}

def test_aliases():
    assert C.canonical("IFS") == "ifs-hres" and C.canonical("aifs") == "aifs-single" and C.canonical("clim") == "climatology"
    with pytest.raises(KeyError):
        C.get("no-such-model")

E2S_CLASSES = {"ACE2ERA5", "AIFS", "AIFS2", "AIFS2ENS", "AIFSENS", "Atlas", "Aurora", "Aurora1p5_6h", "DLESyMLatLon", "FCN3",
               "FengWu", "FuXi", "FuXiS2S", "GenCastMini", "GraphCastOperational", "Pangu6", "SFNO", "UCast",
               "WeatherNext2Cyclones", "SamudrACE"}   # earth2studio 0.19 `earth2studio.models.px` exports

def test_e2s_class_names():
    used = {m.e2s for m in C.MODELS.values() if m.e2s}
    try:
        import earth2studio.models.px as px
        assert all(hasattr(px, c) for c in used), [c for c in used if not hasattr(px, c)]
    except ImportError:
        assert used <= E2S_CLASSES, used - E2S_CLASSES

def test_table():
    t = C.table("s2s")
    assert {"model", "status", "route", "period"} <= set(t.columns) and len(t) >= 5
