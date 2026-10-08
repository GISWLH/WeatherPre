# Changelog

## 0.4.0 (unreleased, 2026-10-08)

**AI S2S / seasonal / ocean model runs, with explicit availability and evidence.**

* Catalog: products (weather / subseasonal / seasonal / scenario), time step, variables, technical vs validated horizon,
  members, requirements, independence groups, evidence (`real_data_verified`, `report_eligible`). New: ORCA-DL, UniCM
  (blocked), SamudrACE (research only); NeuralGCM gains the official-JAX run route. `weatherpre models` shows every model.
* `weatherpre.availability`: per-request reasons (`not_installed`, `weights_missing`, `input_missing`,
  `input_not_published`, `gpu_insufficient`, `out_of_period`, `beyond_technical_horizon`, `beyond_validated_horizon`,
  `proxy_variable`, `forcing_missing`, `blocked`, `research_only`, ...); automatic S2S candidates come from the catalog.
* Leads: `6m` = six calendar months (was 175 days); `m2-4`; `target_start` / `target_end` windows; incomplete periods are
  reported. FuXi-S2S requests beyond 42 days are refused for every backend.
* Variables: `sst`, `sst_anom`, `tos` (ORCA-DL 5 m proxy, never renamed), `thetao`, `so`, `zos`, `uo`, `vo`, `siconc`.
* WeatherAI-backed runners (ORCA-DL, FuXi-S2S official ONNX, ACE2, NeuralGCM) + input sources (WB2/ARCO ERA5 daily means
  in the FuXi-S2S layout, GODAS, OISST); common schema with members, masks, period bounds and completeness; WeatherPre +
  WeatherAI run manifests; resumable per-member files; `weatherpre.planner` with resource limits; `check` / `plan` CLI.
* `forecast()` passes `source`, `checkpoint`, `members`, `seed`, `device`, `model_backend`, `forcing`,
  `forcing_provenance`, `out_dir` through (Earth2Studio `source` was previously rejected).
* Precipitation: per-model semantics instead of a blanket ×1000 + cumulative sum; lead 0 no longer accumulated.
  Earth2Studio FuXiS2S daily means are not pushed through the 6-hourly weekly path; DLESyM ocean leads masked to 48 h.
* Evaluation: `weatherpre.hindcast` (non-overlapping train / calibration / test, no-future-data checks, conditional
  hindcasts refused unless flagged), `weatherpre.metrics` (RMSE, ACC, MSSS, fair CRPS, spread–skill, reliability, Niño
  boxes); `weatherpre.report.export_bundle` for SST_S2S (docs/SST_S2S_HANDOFF.md).

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
