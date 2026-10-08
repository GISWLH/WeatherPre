# AI sub-seasonal, seasonal and ocean models in WeatherPre

This page covers the model-run path (WeatherAI / official ONNX / official JAX / Earth2Studio): what each model can run,
what has been verified, the conventions WeatherPre applies, and how to add a model. Everything here is checked by
`tests/test_ai_s2s.py` and `tests/test_evaluation_report.py`; real-data checks are listed in [VALIDATION.md](VALIDATION.md).

## Responsibilities

| | WeatherAI | WeatherPre |
|---|---|---|
| model code, weights loading, normalisation, model-specific inference, masks | ✔ (`weatherai.inference`) | |
| parity with the official implementation, checkpoint cards (inputs, units, grids, licences, evidence) | ✔ (`weatherai.inference.cards`) | reads them |
| input acquisition (ERA5, GODAS, OISST), backend choice, run plans, chunked / resumable runs | | ✔ (`sources/`, `planner.py`, `runners.py`) |
| common output schema, period aggregation, evaluation, report hand-off | | ✔ (`schema.py`, `hindcast.py`, `metrics.py`, `report.py`) |

WeatherPre never copies research scripts: it calls `weatherai.inference.load(model, ...).run(...)` and converts the
model-native `ForecastResult` (arrays + coordinates + masks + run manifest) to its schema.

## Status (2026-10-08)

`weatherpre models` lists every model (no `--all` needed); `weatherpre check MODEL VAR LEAD --init DATE` gives the reasons.
Four states are kept apart: **catalogued** (listed), **runnable** (route + requirements met *here*), **real-data verified**
(evidence), **report-eligible** (independent hindcast + calibration: no AI model today).

| model | route | product | time step | technical max | validated | real-data evidence | blocking conditions today |
|---|---|---|---|---|---|---|---|
| **ORCA-DL** | weatherai (torch) | seasonal months | 1 month | no code limit (official 24 mo) | — | partial: 1 GODAS init (WeatherAI); SST_S2S run (external) | weights (HF `JayKuo/ORCA-DL-data`, licence unstated); GODAS month must be published (~10 d) |
| **FuXi-S2S** | weatherai (official ONNX; torch secondary); e2s | subseasonal days | 1 day | **42 days** (refused beyond) | — | partial: official sample (WeatherAI); 11×42 ONNX (SST_S2S, external) | weights (Zenodo, CC-BY-NC-ND); inputs from WB2 ERA5 only to 2023-01-10, later via `file:` |
| **ACE2-ERA5** | weatherai (torch); e2s | scenario (forced) | 6 h | none | — | partial: one 8-day case | forcing for every step + provenance; weights |
| **NeuralGCM** | wb2 (hosted 2020); weatherai (official JAX) | weather | any | 15 d hosted | — | partial: official JAX demo run (WeatherAI) | caller-prepared model-grid dataset + forcing policy |
| **DLESyM** | e2s | subseasonal / seasonal | 6 h atm / 48 h ocean | none | — | none | Earth2Studio + one ≥40 GB GPU (vendor badge) |
| **U-CAST** | e2s | weather (medium-range per Earth2Studio) | 12 h | none | — | none | one ≥40 GB GPU (vendor badge) |
| **UniCM** | — | — | 1 month | — | — | none | **blocked**: no published checkpoint (GitHub HEAD 67fe4c1) |
| **SamudrACE** | e2s | — | 6 h / 5 d | — | — | none | **research only**: CM4 piControl model years, not real 2026 initial conditions |

"technical max" = what the code / checkpoint can run; "validated" = independently verified skill (none yet).
The xuanze / yongqiang RTX 4090 cards (24 GB each) fit ORCA-DL (2.4 GB peak), FuXi-S2S (5.4 GB native fp32) and ACE2
(3.1 GB); the Earth2Studio 40 GB badge excludes DLESyM / U-CAST there (memory of two cards is never added).

## Conventions

* **Products.** weather (≤15 d, lead hours) · subseasonal (weeks; week k = [t0+7(k−1), t0+7k)) · seasonal (**calendar
  months**: month 1 = first full month starting at/after t0; `6m` is six calendar months, not 175 days) · scenario
  (prescribed boundary conditions). Explicit windows: `target_start="2027-01-01", target_end="2027-06-30"` (inclusive end
  date), aggregated by `month` (default), `week` or as one `period`.
* **init_time (t0) of mean-state models** is the end of the initial-condition period: FuXi-S2S initialised from the daily
  means of D−1 and D → t0 = D+1 00Z (lead day 1 = D+1); ORCA-DL initialised from the November monthly mean → t0 = 1 Dec,
  lead 1 = December. Periods are computed from t0, so FuXi-S2S `6w` needs exactly 42 daily steps.
* **Completeness.** Native intervals are averaged with their overlap as weight; periods not fully covered are dropped and
  listed in `attrs['incomplete_periods']` (`require_complete=False` keeps them with `completeness < 1`). Monthly means are
  never split into weeks. Hosted products that end early (GEFS 35 d) return the complete weeks and report the rest
  (`partial_coverage`); model runs refuse requests beyond their technical horizon.
* **Variables.** `sst` [degC] = a sea-surface-temperature product (ERA5, OISST, FuXi-S2S's ERA5-like sst). `tos` [degC] =
  ORCA-DL's top layer, initialised from GODAS **5 m** potential temperature: a proxy, returned only under the name `tos`
  (requesting `sst` from ORCA-DL fails with `proxy_variable`). `t2m` is never used for either. Ocean: `thetao`, `so`
  (with `depth`), `zos`, `uo`, `vo`.
* **Precipitation.** FuXi-S2S's channel is the daily mean of hourly ERA5 accumulations in mm/h (verified against WB2
  hourly ERA5; Earth2Studio documents the same) → WeatherPre `tp` = 24 × channel [mm/day]. Earth2Studio: `tp06/12/24`
  are window accumulations in m and must match the model step; `tp` is used only where its meaning is known (FuXiS2S);
  otherwise it is dropped with a warning. Lead 0 never contributes to accumulations.
* **Members** are always stored (`member` dim). ORCA-DL members are training seeds of one model (`member_kind =
  checkpoint_seed`); FuXi-S2S members are stochastic draws with noise from `rng([seed, member, step])`; one
  `independence_group` per model, so seeds or a second implementation are never counted as extra models.
* **Masks.** Values outside a model's domain / land mask are NaN, with `valid_<var>` masks; never zero-filled (ORCA-DL
  covers 63.5S–63.5N only).

## Running

```python
import weatherpre as wp
wp.availability.check("fuxi-s2s", "t2m,tp,sst", "6w", "2022-12-01")            # reasons, no download
ds = wp.forecast("fuxi-s2s", "t2m,tp,sst", "6w", init="2022-12-01", members=11, seed=0,
                 checkpoint="/ckpt/fuxi_s2s", device="cuda", out_dir="runs/fuxi")   # official ONNX, one file per member
ds = wp.forecast("orca-dl", "tos,thetao", init="2026-12-01", target_start="2027-01-01", target_end="2027-06-30",
                 members=[1, 2, 3, 4, 5, 6, 7, 8], checkpoint="/ckpt/orca_dl", source="godas", out_dir="runs/orca")
plan = wp.planner.plan(["fuxi-s2s", "orca-dl", "ace2"], "t2m", "6w", "2022-12-01", members=11)   # no GPU unless allowed
```

CLI: `weatherpre check`, `weatherpre plan`, `weatherpre forecast ... --source --checkpoint --members --seed --device
--model-backend --target-start --target-end --out-dir`. Weights roots: `WEATHERPRE_WEIGHTS_<MODEL>` or
`~/.config/weatherpre/weights.json`. Remote GPU runs: [REMOTE.md](REMOTE.md).

Earth2Studio's `source` argument now reaches the adapter (`wp.forecast("e2s:Persistence", ..., source="WB2ERA5_121x240")`).

## Adding a model

1. A `catalog.Model` entry: routes, products / scales, time step, variables, `technical_max_days` (+ note),
   `validated_max_days` (None until a hindcast exists), members, requirements, `independence_group`, evidence.
2. A runner `@runners.register("name")` returning the common schema at native resolution (`schema.finalize`, members kept,
   `valid_start` / `valid_end`, masks, units). `api.forecast` aggregates it; plots, scores and `report.export_bundle` need
   no change. Add availability rules (inputs / latency) in `availability.input_reason` if the model needs special inputs.
