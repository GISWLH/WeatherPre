# WeatherPre

**Near-real-time forecasts from frontier AI weather models, assembled from existing hosted data, upstream inference packages and WeatherBench-X — nothing re-implemented.**

近实时前沿 AI 天气模型预报：只组装现成的托管数据、官方推理库和 WeatherBench-X 评测，不重复造轮子。

Route order / 路线优先级：**① hosted data 托管数据 → ② upstream package on GPU (HF / Colab) 官方包 GPU 推理 → ③ our port [`GISWLH/WeatherAI`](https://github.com/GISWLH/WeatherAI)**。Survey: [`docs/SURVEY.md`](docs/SURVEY.md) · numbers: [`docs/RESULTS.md`](docs/RESULTS.md)

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

## Usage / 用法

```bash
pip install -e .          # Python ≥ 3.11
weatherpre models         # model list and route
weatherpre latest --model aifs-single --lead 0-120/24
weatherpre compare --models hres,graphcast,pangu,fuxi,gencast --init 2020-10-01T00..2020-10-31T12 --lead 24-240/24
```

Output layout = WeatherBench 2 (`time, prediction_timedelta, level, latitude, longitude`); scoring by WeatherBench-X (area-weighted RMSE, ACC). GPU runs: HF Space `LonghaoWang/weatherai-graphcast-smoke` ("WeatherPre" tab, [`docs/GPU.md`](docs/GPU.md)) or Colab.

## Limits / 局限

* ECMWF Open Data keeps ~4 days: **no archive**. Hosted 2020 forecasts exist only for the WB2 models above (00/12Z inits; WB2 stops at 240 h for GraphCast/Pangu/HRES, 360 h for FuXi/GenCast/NeuralGCM/IFS-ENS).
* Latest-date truth = IFS analysis (proxy); future valid times cannot be scored. ERA5 lags ~5 days and is not wired for live.
* Aurora/Oct-2020 uses the ERA5-pretrained checkpoint inside its training period: a pipeline check, not out-of-sample skill.

## Licences / 许可

Code Apache-2.0. ECMWF Open Data CC-BY-4.0 (credit ECMWF); NOAA public domain; ERA5 per Copernicus terms. NC / ND weights are never committed.
