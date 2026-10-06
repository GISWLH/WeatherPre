# Changelog

## 0.3.0 (2026-10-05)

**Two scales, one call.** Leads up to 15 days are *weather* (instantaneous fields at lead hours); beyond 15 days are
*S2S* (weekly means, week k = days 7(k-1)..7k after init).

* New simple API: `wp.forecast(model, variable, lead, init=...)` and `wp.compare(models, variable, lead, init=...)`
  returning a `Comparison` (`.table()`, `.ranking()`, `.plot()`, `.plot_maps()`, `.summary()`); `init="A..B"` averages
  scores over several inits. Old keyword calls (`lead_days=`, `lead_hours=`, `preset=`, `init=` as 2nd positional) still work.
* Lead strings: `48h`, `7d`, `15d`, `6w`, `45d`, `week3-4`, `w3`, `24,48,72`.
* Model catalog (`weatherpre/catalog.py`) drives routing, `weatherpre models`, and the README table.
* New hosted models: **NOAA GEFSv12** ensemble mean (0.5°, 35 d, 2020-09 → now), **NOAA CFSv2** (9 months, 2020 → now),
  **ECMWF extended range** ensemble mean (WeatherBench 2, 2016–2022), and S2S baselines **climatology** / **persistence**.
* S2S verification: weekly means vs ERA5 weekly means, RMSE + ACC (1990–2017 climatology), WeatherBench-X.
* Earth2Studio route rewritten and tested end-to-end (CPU, `Persistence`): AIFS v2, AIFS-ENS v2, Aurora 1.5,
  FourCastNet 3, Atlas, U-CAST, WeatherNext 2, GenCast-mini, SFNO, FengWu, FuXi-S2S, DLESyM, ACE2, plus any
  `"e2s:<ClassName>"`.
* New CLI: `weatherpre forecast MODEL [VAR] [LEAD]`, `weatherpre compare MODELS [VAR] [LEAD]`, `weatherpre models --scale s2s`.
  The old `compare --models ...` command is now `compare-wb2`.
* Fixed: interpreter crash at exit when eccodes was imported before PROJ.
* Tests (offline + opt-in live data), GitHub Actions CI, CONTRIBUTING, CITATION.

## 0.2.0
One-call `forecast()`, horizon comparison, publication-style maps, Aurora on HF ZeroGPU.
