# Validation record — AI S2S / seasonal path (2026-10-08)

Environment: cloud box, 4 CPU cores, 15 GB RAM, **no GPU**; network: PyPI, GitHub, Google Cloud Storage and NOAA S3
reachable; **Hugging Face, Zenodo and NOAA PSL blocked** (so ORCA-DL / FuXi-S2S / ACE2 weights and GODAS files could not
be downloaded); no SSH access to xuanze / yongqiang. Base commits: WeatherPre `2996237`, WeatherAI `4ce651c`.

Levels are kept apart: offline tests ≠ real-data input checks ≠ real-data inference ≠ full-length runs ≠ skill.

## 1. Reported problems: reproduced at `2996237`, fixed now

| # | problem (reproduced) | now | test |
|---|---|---|---|
| 1 | `models --scale s2s` hid GPU models without `--all` | every model listed with status, technical / validated horizon, evidence; `--hosted-only` filters | `test_listing_shows_all_s2s_models_and_states_are_separate` |
| 2 | auto S2S candidates = hand-kept NWP + baselines | derived from the catalog + availability (`availability.candidates`); model runs only via an explicit plan | `test_auto_s2s_candidates_come_from_the_catalog` |
| 3 | AI S2S candidates U-CAST, FuXi-S2S, DLESyM, ACE2 registered without conditions | each has products, horizons, members, requirements, evidence; U-CAST is weather-scale only (Earth2Studio badge medium-range); ACE2 is a *scenario* model | `test_reasons_are_distinct` |
| 4 | public API rejected `sst` | `sst`, `sst_anom`, `tos` (proxy), `thetao`, `so`, `zos`, `uo`, `vo`, `siconc` | `test_sst_is_a_variable_and_never_t2m_or_proxy` |
| 5 | no ORCA-DL / UniCM route | ORCA-DL runner (WeatherAI); UniCM `blocked` with the concrete reason | `test_orca_dl_adapter_end_to_end_on_godas_example`, `test_reasons_are_distinct` |
| 6 | `source=` rejected by `forecast` | passed through to Earth2Studio (and used by the WeatherAI runners) | `test_e2s_source_is_passed_through` |
| 7 | `6m` → 175 days | six calendar months (Dec–May from 2026-12-01 = 182 d); `target_start/target_end` windows | `test_6m_is_calendar_months_not_175_days`, `test_explicit_target_window_and_incomplete_week_note` |
| 8 | FuXi-S2S `6m` accepted by the resolver once dependencies were present | refused in `resolve` and `availability` for every backend beyond 42 days (`8w`, `7w`, `6m`); WeatherAI also raises `HorizonError` | `test_fuxi_42_day_limit_in_resolver_and_availability` |
| 9 | GPU path only checked with Persistence | stated per model as evidence; model runs need weights + a plan | listing / `check` |
| 10 | blanket `×1000` + cumsum for precipitation | per-model semantics; FuXi-S2S ×24 from mm/h; `tp06/12/24` only when the window equals the step; unknown → dropped with warning; lead 0 excluded | `test_e2s_precip_semantics` |
| 11 | members / forcing / model-specific initialisation not expressible | `members`, `seed`, `device`, `checkpoint`, `source`, `model_backend`, `forcing`, `forcing_provenance`, `out_dir` | adapter tests |

Additional problem found and fixed: Earth2Studio FuXiS2S returns daily means labelled at day start; the 6-hourly weekly
path would have mixed the input day into week 1 → refused with a pointer to the WeatherAI backend. DLESyM's ocean output
is only valid every 48 h → other leads are masked.

## 2. Offline tests

`pytest tests` → **76 passed, 6 skipped** (skips = network-marked). Includes end-to-end adapter runs on **real input
data** with stand-in networks (they test plumbing, not forecasts):
* ORCA-DL on the official GODAS January-1980 example: real units / NaN masks through WeatherAI's normalisation, two
  members, monthly periods, persistence stub reproduces the input `tos` exactly, land stays NaN, manifest written.
* FuXi-S2S on the official ERA5 sample (2020-06-01/02) through the ONNX runner (stand-in graph with the official
  interface): init_time = 2020-06-03, 14 daily leads → 2 complete weeks, tp in mm/day, sst in °C with the official land
  mask, resume skips existing members.

## 3. Real-data checks (no model weights needed)

| check | result | reproduce |
|---|---|---|
| FuXi-S2S inputs built from public ERA5 (WB2 hourly 1.5° + ARCO 100 m winds) vs the official sample, 76 channels × 2 days | passes WeatherAI input checks; median RMSE/std 0.044; worst tp 0.35 and upper-level v ≈ 0.2 (regridding-method differences); area-weighted global means within 1.2 % (z500 0.0003 %, tp 1.2 %); sst land mask identical (9848 points) | `python scripts/validate_fuxi_inputs.py <FuXi-S2S/data> results/validation/fuxi_inputs_2020-06-02.json` |
| FuXi-S2S tp / ttr units (WeatherAI) | sample tp = mean hourly accumulation (1.152e-4 vs 1.150e-4 m; daily total 2.76e-3) → mm/h; ttr/3600 = −244.80 vs −244.82 W/m² | WeatherAI `scripts/fuxi_s2s_tp_units_check.py` |
| OISST v2.1 (NOAA S3) Nov 2015 Niño-3.4 | 29.57 °C from 30 daily files | `WEATHERPRE_NETWORK=1 pytest tests/test_network.py -k oisst` |
| Hosted products after the changes | WB2 GraphCast, IFS-ext vs climatology, GEFS, single-lead scoring: 4 passed | `WEATHERPRE_NETWORK=1 pytest -m network tests` |

Whether the remaining input differences change FuXi-S2S forecasts could not be tested (no weights here).

## 4. Real-data inference

* **ORCA-DL, FuXi-S2S, ACE2 with official weights: not run** (weights unreachable). WeatherAI's earlier records remain the
  evidence (ORCA-DL 1 GODAS init; FuXi-S2S 1 step + 3-step chain; ACE2 one 8-day case); SST_S2S's ORCA-DL monthly run and
  FuXi-S2S 11 × 42 ONNX run are external and were not reproduced.
* NeuralGCM: official JAX forecaster on the official demo snapshot (WeatherAI test, CPU, 185 s).

Reproduce on a box with the weights:
```
export WEATHERPRE_WEIGHTS_ORCA_DL=/ckpt/orca_dl WEATHERPRE_WEIGHTS_FUXI_S2S=/ckpt/fuxi_s2s
weatherpre forecast orca-dl tos 11m --init 1980-02-01 --source orca-example --members 1 --out-dir runs/orca_1980
#   expect seed-1 lat-weighted RMSE vs GODAS 5 m of 0.43 degC (+1 mo) ... 0.96 (+11 mo) (WeatherAI docs/results/orca_dl_real_case.json)
weatherpre forecast fuxi-s2s t2m,tp,sst 6w --init 2020-06-02 --source official-sample:/data/FuXi-S2S/data --members 11 \
    --device cuda --out-dir runs/fuxi_2020-06-02
scripts/remote/run_model.sh fuxi-s2s t2m,tp,sst 6w 2022-12-01 --members 11 --seed 0      # on xuanze / yongqiang
```

## 5. Full-length runs

Not run (no weights, no GPU): FuXi-S2S 42 d × 11 members, ORCA-DL 7–24 months × 8 seeds, ACE2 multi-month forced runs.

## 6. Skill

**No AI-model skill was measured.** Baseline only, to show the evaluation path works on real data and to set the bar:
monthly Niño-3.4 from ERA5 SST (WB2 6-hourly 1.5°, 24 samples per month), calibration 1991–2010, test inits 2011–2022.
Anomalies vs 1991–2010: Dec 1997 +2.81, Dec 2015 +2.65, Dec 2010 −1.71, Dec 2020 −1.19 °C.

| lead (months) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| persistence RMSE (°C) | 0.249 | 0.410 | 0.547 | 0.674 | 0.778 | 0.870 |
| persistence anomaly corr. | 0.954 | 0.874 | 0.775 | 0.658 | 0.545 | 0.435 |
| persistence MSSS vs climatology | 0.906 | 0.744 | 0.542 | 0.307 | 0.079 | −0.142 |

`python scripts/nino34_baseline_hindcast.py results/validation/nino34_baseline`
(→ `results/validation/nino34_baseline/nino34_baseline_hindcast.json`). An AI SST model is worth weighting only if it
beats these numbers on the same split, with its training period not overlapping the test years (`Split.check_model`).
