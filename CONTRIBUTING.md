# Contributing

WeatherPre stays small by **assembling** existing pieces (hosted forecasts, upstream inference packages, WeatherBench-X
metrics) instead of re-implementing them. Contributions that keep that spirit are very welcome.

## Dev setup

```bash
git clone https://github.com/GISWLH/WeatherPre && cd WeatherPre
python -m venv .venv && . .venv/bin/activate      # Python >= 3.11
pip install -U pip setuptools wheel && pip install -e ".[dev]"
pytest -q                                          # offline unit tests (< 5 s)
WEATHERPRE_NETWORK=1 pytest -q -m network          # live-data smoke tests (WeatherBench 2 GCS, NOAA S3)
ruff check weatherpre tests scripts
```

## Adding a model

1. Add one `Model(...)` line to [`weatherpre/catalog.py`](weatherpre/catalog.py): scale(s), route(s), grid, lead, period,
   licence, reference. That alone makes it appear in `weatherpre models` and in the README table
   (`python scripts/gen_model_table.py`).
2. Route:
   * **Earth2Studio** model: set `e2s="<ClassName>"` and `routes=("e2s",)`. Nothing else is needed.
   * **Hosted data**: add an adapter in `weatherpre/adapters/` that returns the weather schema
     (`lead` [h], `latitude`, `longitude`; z500 [m], t850 [K], t2m [K], msl [hPa], tp [mm since init]) and a branch in
     `api.resolve` / `api._forecast_weather`. S2S weekly means come for free via `s2s.weekly_from_leads`.
3. Add a test in `tests/` (offline if possible; mark live-data tests with `@pytest.mark.network`).
4. Never commit weights or data; check the licence (several model weights are non-commercial).

## Pull requests

Small and focused, with a real output (a table or figure) when behaviour changes. Scores must come from WeatherBench-X.
