import pytest
from weatherpre import leads as L

@pytest.mark.parametrize("spec,scale,first,last,n", [
    ("48h", "weather", 6, 48, 8), ("7d", "weather", 12, 168, 14), ("15d", "weather", 24, 360, 15),
    ("hours", "weather", 6, 48, 8), ("week", "weather", 12, 168, 14), ("15days", "weather", 24, 360, 15),
    ("24,48,72", "weather", 24, 72, 3), ("6-48/6", "weather", 6, 48, 8), (36, "weather", 36, 36, 1),
    ("6w", "s2s", 1, 6, 6), ("45d", "s2s", 1, 6, 6), ("week3-4", "s2s", 3, 4, 2), ("w3", "s2s", 3, 3, 1),
    ("s2s", "s2s", 1, 6, 6), ("5W", "s2s", 1, 5, 5), ("W5", "s2s", 5, 5, 1),
    ("2m", "seasonal", 1, 2, 2), ("6mo", "seasonal", 1, 6, 6), ("m2-4", "seasonal", 2, 4, 3),   # calendar months (was 2m -> 8 weeks)
])
def test_parse(spec, scale, first, last, n):
    lv = L.parse(spec)
    assert lv.scale == scale
    xs = lv.hours if scale == "weather" else lv.weeks if scale == "s2s" else lv.months
    assert (xs[0], xs[-1], len(xs)) == (first, last, n)

def test_boundary_is_15_days():
    assert L.parse("15d").scale == "weather"
    assert L.parse("16d").scale == "s2s"
    with pytest.raises(ValueError):
        L.parse(lead_hours=[24, 400])

def test_legacy_keywords():
    assert L.parse(lead_days=15).hours[-1] == 360
    assert L.parse(preset="hours").hours == tuple(range(6, 49, 6))
    assert L.parse(lead_hours=[6, 12]).hours == (6, 12)

def test_default_and_bad():
    assert L.parse().hours[-1] == 168
    with pytest.raises(ValueError):
        L.parse("soon")
