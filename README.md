# WeatherPre

**Near-real-time & forward forecasts from frontier AI weather models: one call, any model, any date. Assembled from hosted data, upstream packages and WeatherBench-X, nothing re-implemented.**

近实时 / 前向预报：一行代码、指定模型和起报日期。只组装托管数据、官方推理库和 WeatherBench-X，不重复造轮子。

Route order / 路线优先级：**① hosted data 托管数据 → ② upstream package on GPU (HF / Colab) → ③ our port [`GISWLH/WeatherAI`](https://github.com/GISWLH/WeatherAI)** · [`docs/SURVEY.md`](docs/SURVEY.md) · [`docs/RESULTS.md`](docs/RESULTS.md) · [`docs/GPU.md`](docs/GPU.md)

## Quick start / 快速开始

```bash
pip install -e .            # Python >= 3.11
weatherpre forecast graphcast 2020-10-03 --days 15 --plot     # CLI -> data/forecasts/*.nc + png
weatherpre forecast aifs latest --days 15 --plot               # newest ECMWF Open Data cycle
weatherpre horizon 2020-10-03 --preset week --out results/x    # multi-model table + plots
```

```python
import weatherpre as wp

ds = wp.forecast("aifs", init="latest", lead_days=15)           # xarray.Dataset (lead, latitude, longitude)
ds = wp.forecast("graphcast", init="2020-10-03", lead_days=7)   # 2020 -> hosted WeatherBench 2
ds = wp.forecast("ifs", init="2020-10-03", lead_hours=[6, 12, 24, 48])
ds = wp.forecast("aurora", init="2020-10-03", lead_days=15)     # 0.25 deg, runs on HF ZeroGPU (HF_TOKEN)

ds.z500.isel(lead=2).plot()                                     # z500 [m]; also t850, t2m, msl [hPa], tp [mm] if available
wp.plot_maps(ds, ("z500", "t2m"), [24, 120, 240], "maps.png")   # cartopy maps, labelled model / init / lead
fc, table, info = wp.compare("auto", "2020-10-03", preset="week")   # all suitable models + RMSE table
```

`lead_hours=[..]` | `lead_days=N` (24 h steps) | `preset="hours"` (6..48 h, 6 h) / `"week"` (12..168 h, 12 h) / `"15days"` (24..360 h, 24 h). `init` = date, `YYYY-MM-DDTHH` or `"latest"`. Variables: `z500` [m], `t850` [K], `t2m` [K], `msl` [hPa], `tp` [mm, ECMWF models]; coords `lead` [h], `valid_time`, scalar `init_time`.

### Backend auto-selection / 自动选路

| Request | Backend | Notes |
|---|---|---|
| `init="latest"`, `aifs` / `ifs` | ECMWF Open Data | last ~4 days only, 00/12Z to 360 h, 06/18Z to ~144 h |
| `init="latest"`, `aigfs` / `gfs` | NOAA S3 | to 384 h |
| 2018–22 date, `hres graphcast pangu fuxi gencast neuralgcm ifs-ens` | WeatherBench 2 (hosted) | 00/12Z inits, 1.5°, lead limits below |
| any date, `aurora` | HF ZeroGPU Space (upstream `microsoft-aurora`) | ERA5 initial conditions up to 2021, IFS analysis after |
| 2020 date, `aifs` / `aigfs` / `gfs` | **raises `BackendUnavailable`** with the options | no hosted archive; AIFS needs Earth2Studio on a Py >= 3.11 GPU ([Colab](notebooks/colab_earth2studio.ipynb), not run) |

## Now = 2020-10-03: what to expect next / 前向预报示例

"It is 2020-10-03 00Z, give me the next hours / week / 15 days." Best hosted models per horizon are picked by `wp.best_models(init, preset)`. All numbers below are real runs (`scripts/make_examples.py`), scored against ERA5 (WeatherBench 2) with WeatherBench-X (area-weighted RMSE). For a past date this is a **hindcast check**; for a real "now" the valid times are in the future and **cannot be scored**.

### Next 48 h / 未来 6–48 小时 (`preset="hours"`)

Ran: `fuxi` (≤48 h), `gencast` (≤48 h), `graphcast` (≤48 h), `ifs-hres` (≤48 h), `neuralgcm` (≤48 h), `pangu` (≤48 h).

![hours maps](docs/img/2020-10-03_hours_z500_models.png)
![hours scores](docs/img/2020-10-03_hours_scores.png)

| z500 RMSE [m] | +6 h | +12 h | +24 h | +48 h |
|---|---:|---:|---:|---:|
| fuxi | 1.6 | **2.7** | 4.2 | 8.3 |
| gencast | – | 2.9 | 3.9 | 7.7 |
| graphcast | **1.4** | 2.8 | 4.0 | 7.7 |
| ifs-hres | 2.3 | 3.6 | 4.7 | 9.2 |
| neuralgcm | – | 3.0 | **3.8** | **7.1** |
| pangu | 1.5 | 3.0 | 4.5 | 7.3 |

| t2m RMSE [K] | +6 h | +12 h | +24 h | +48 h |
|---|---:|---:|---:|---:|
| fuxi | 0.4 | 0.4 | 0.5 | 0.7 |
| gencast | – | **0.4** | **0.4** | **0.6** |
| graphcast | **0.4** | 0.4 | 0.5 | 0.6 |
| ifs-hres | 0.7 | 0.8 | 0.8 | 0.9 |
| pangu | 0.4 | 0.5 | 0.5 | 0.7 |

Single-model maps, ECMWF IFS HRES z500 / t2m / msl: [`docs/img/2020-10-03_hours_hres_maps.png`](docs/img/2020-10-03_hours_hres_maps.png). CSV: [`results/examples/2020-10-03_hours_scores.csv`](results/examples/2020-10-03_hours_scores.csv).

### Next week / 未来一周 (`preset="week"`)

Ran: `fuxi` (≤168 h), `gencast` (≤168 h), `graphcast` (≤168 h), `ifs-ens` (≤168 h), `ifs-hres` (≤168 h), `neuralgcm` (≤168 h), `pangu` (≤168 h).

![week maps](docs/img/2020-10-03_week_z500_models.png)
![week scores](docs/img/2020-10-03_week_scores.png)

| z500 RMSE [m] | +24 h | +72 h | +120 h | +168 h |
|---|---:|---:|---:|---:|
| fuxi | 4.2 | 14.1 | 27.1 | 40.3 |
| gencast | 3.9 | 13.4 | 26.6 | 37.5 |
| graphcast | 4.0 | 12.9 | **24.9** | 44.0 |
| ifs-ens | 4.8 | 15.0 | 26.5 | **37.0** |
| ifs-hres | 4.7 | 16.2 | 33.2 | 47.4 |
| neuralgcm | **3.8** | 12.8 | 25.3 | 40.7 |
| pangu | 4.5 | **12.7** | 25.5 | 42.7 |

| t2m RMSE [K] | +24 h | +72 h | +120 h | +168 h |
|---|---:|---:|---:|---:|
| fuxi | 0.5 | 0.8 | 1.1 | 1.4 |
| gencast | **0.4** | 0.8 | **1.0** | **1.3** |
| graphcast | 0.5 | **0.8** | 1.0 | 1.5 |
| ifs-ens | 0.8 | 1.0 | 1.2 | 1.5 |
| ifs-hres | 0.8 | 1.1 | 1.3 | 1.8 |
| pangu | 0.5 | 0.9 | 1.2 | 1.6 |

CSV: [`results/examples/2020-10-03_week_scores.csv`](results/examples/2020-10-03_week_scores.csv).

### Next 15 days / 未来 15 天 (`preset="15days"`)

Ran: `fuxi` (≤360 h), `gencast` (≤360 h), `ifs-ens` (≤360 h), `neuralgcm` (≤360 h), `graphcast` (≤240 h), `ifs-hres` (≤240 h), `pangu` (≤240 h).

![15-day maps](docs/img/2020-10-03_15days_z500_models.png)
![15-day scores](docs/img/2020-10-03_15days_scores.png)

| z500 RMSE [m] | +24 h | +120 h | +240 h | +360 h |
|---|---:|---:|---:|---:|
| fuxi | 4.2 | 27.1 | 62.2 | 83.7 |
| gencast | 3.9 | 26.6 | 59.8 | 84.4 |
| graphcast | 4.0 | **24.9** | 71.1 | – |
| ifs-ens | 4.8 | 26.5 | **57.1** | **79.3** |
| ifs-hres | 4.7 | 33.2 | 77.2 | – |
| neuralgcm | **3.8** | 25.3 | 71.2 | 119.2 |
| pangu | 4.5 | 25.5 | 74.4 | – |

| t2m RMSE [K] | +24 h | +120 h | +240 h | +360 h |
|---|---:|---:|---:|---:|
| fuxi | 0.5 | 1.1 | 2.0 | 2.5 |
| gencast | **0.4** | **1.0** | 1.8 | 2.4 |
| graphcast | 0.5 | 1.0 | 2.2 | – |
| ifs-ens | 0.8 | 1.2 | **1.8** | **2.3** |
| ifs-hres | 0.8 | 1.3 | 2.5 | – |
| pangu | 0.5 | 1.2 | 2.3 | – |

CSV: [`results/examples/2020-10-03_15days_scores.csv`](results/examples/2020-10-03_15days_scores.csv).

### Latest cycle (live) / 最新一轮

`wp.forecast("aifs", init="latest", lead_days=15)`: AIFS-single, init 2026-10-03 00Z, ECMWF Open Data, 0.25°, 15 leads (+24 … +360 h).

![AIFS latest](docs/img/aifs_latest_maps.png)

Four-model view (`aifs-single`, `ifs-hres`, `aigfs`, `gfs`, same cycle, +120 h; [t2m](docs/img/latest_week_t2m_models.png)):

![latest z500](docs/img/latest_week_z500_models.png)

No verification exists yet for these valid times, so only the **inter-model spread** (RMSE of each model against the 4-model mean, a consensus not a skill score) is given: [`results/examples/latest_week_consensus.csv`](results/examples/latest_week_consensus.csv). Leads whose valid time already has an IFS analysis are scored against it as a proxy truth.

## Models / 模型

Status: ✅ ran in this repo (outputs in `results/`) · ⚠️ route exists, **not run** · ❌ blocked. Resolution of hosted WeatherBench 2 (WB2) data here is 1.5°.

| Model | Type | Route | Lead · grid | Licence note | Status |
|---|---|---|---|---|---|
| **IFS HRES** | NWP, medium-range | hosted: ECMWF Open Data (latest ~4 d only) · WB2 (2016–22) | ≤360 h (00/12Z) · 0.25° / 1.5° | CC-BY-4.0 | ✅ |
| **AIFS-single** | AI, medium-range | hosted: ECMWF Open Data (latest ~4 d only; no archive) | ≤360 h (00/12Z) · 0.25° | CC-BY-4.0 | ✅ latest · ❌ 2020 |
| AIFS-ENS | AI, ensemble | hosted: Open Data (dir exists) | not checked | CC-BY-4.0 | ⚠️ |
| **AIGFS** (NOAA) | AI, medium-range | hosted: NOAA S3 (daily dirs since 2026-04-16) | ≤384 h · 0.25° | public domain | ✅ |
| **GFS** (NOAA) | NWP baseline | hosted: NOAA S3 (0.25° present 2022-01, absent 2021-03) | ≤384 h · 0.25° | public domain | ✅ |
| **GraphCast** *Science* 2023 | AI, medium-range | hosted: WB2 (2019-11…2021-01) | ≤240 h · 1.5° | upstream weights NC (re-check) | ✅ |
| **Pangu-Weather** *Nature* 2023 | AI, medium-range | hosted: WB2 (2018–22) | ≤240 h · 1.5° | upstream NC (re-check) | ✅ |
| **FuXi** *npj CAS* 2023 | AI, medium-range | hosted: WB2 (2020) | ≤360 h · 1.5° | upstream NC (re-check) | ✅ |
| **GenCast** *Nature* 2025 | AI, ensemble (mean in WB2) | hosted: WB2 (2020) | ≤360 h, 12 h steps · 1.5° | upstream NC (re-check) | ✅ mean |
| **NeuralGCM** *Nature* 2024 | hybrid, det./ens. | hosted: WB2 (2020) | ≤360 h, 12 h steps · 1.5° | re-check | ✅ det. |
| **IFS-ENS** | NWP ensemble (mean) | hosted: WB2 (2018–22) | ≤360 h · 1.5° | WB2 terms | ✅ mean |
| **Aurora** *Nature* 2025 | AI foundation, 0.25° | upstream `microsoft-aurora` on **HF ZeroGPU** (ERA5 or IFS-analysis initial conditions); [Colab](notebooks/colab_aurora.ipynb) | any steps (chunked) · 0.25° | MIT weights | ✅ HF · ⚠️ Colab |
| FourCastNet3 / SFNO, Pangu, FuXi, FengWu, GraphCast-op., AIFS(2) | AI, medium-range | Earth2Studio ([wrapper](weatherpre/adapters/earth2studio_run.py), [Colab](notebooks/colab_earth2studio.ipynb)) | model-specific | per model | ⚠️ not run (needs Py ≥ 3.11 GPU) |
| GenCast-mini, WN-Cyclones-mini, ACE2-ERA5, DLESyM, SamudrACE | AI, ensemble / climate | Earth2Studio, or our [WeatherAI](https://github.com/GISWLH/WeatherAI) ports | 1° | Apache-2.0 / per model | ⚠️ not run |
| StormCast, CorrDiff | AI, km-scale (CONUS / downscaling) | Earth2Studio · WeatherAI port | 3 km | Apache-2.0 | ⚠️ not run |
| FuXi-S2S *Nat. Commun.* 2024 | S2S | Earth2Studio · WeatherAI port | daily, 42 d | **CC-BY-NC-ND** | ⚠️ not run |
| ORCA-DL, UniCM | ocean / SST | WeatherAI port only | monthly | no licence / no public ckpt | ⚠️ not run |
| TropiCycloneNet | tropical cyclone | WeatherAI port only | 24 h track | CC-BY-4.0 ckpt | ⚠️ not run |
| WeatherNext 2 / 3 | AI, ensemble 0.25° | Google GCS/BigQuery/EE, allow-list | ≤15 d | experimental ToS | ❌ needs Google allow-list |

## Limits / 局限

* **15-day coverage differs per model**: GraphCast, Pangu, IFS HRES stop at 240 h in WeatherBench 2; FuXi, IFS-ENS (ensemble mean), GenCast (ensemble mean, 12 h steps) and NeuralGCM (12 h steps) reach 360 h; NeuralGCM has no t2m. Missing leads are skipped with a warning.
* ECMWF Open Data keeps ~4 days: **no archive**. AIFS / IFS for a 2020 date can only come from WB 2 (IFS HRES) or from running the model yourself (Earth2Studio, not run here).
* WB 2 inits are 00/12Z; hosted data is 1.5° (Aurora / Open Data / NOAA: 0.25°). Scores of different grids are compared against the matching ERA5 grid and are not strictly like-for-like.
* Everything is scored against ERA5, including IFS HRES (its own analysis would score it better). `–` = lead not provided (12 h-step models have no +6 h).
* Future valid times cannot be scored; latest-cycle tables show model spread only.
* Aurora 2020 uses ERA5 initial conditions and the ERA5-pretrained checkpoint, inside its training period: a pipeline check, not out-of-sample skill. HF ZeroGPU is quota-limited and may abort a long rollout.
* Colab notebooks and the Earth2Studio wrapper are provided but **not executed**.

## Licences / 许可

Code Apache-2.0. ECMWF Open Data CC-BY-4.0 (credit ECMWF); NOAA public domain; ERA5 per Copernicus terms. NC / ND weights are never committed. Regenerate every image: `python scripts/make_examples.py aifs_latest hours week days15 latest_models`.
