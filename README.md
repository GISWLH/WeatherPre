# WeatherPre

近实时对比前沿气象/气候 AI 模型的预报。**不从零搭建**:只组装现成的托管数据、官方推理库和评测库。
Near-real-time forecasts from frontier weather/climate AI models, **assembled from existing hosted data, upstream inference packages and WeatherBench-X** — no model or metric is re-implemented here.

路线优先级 / Route order: **① 托管预报产品 hosted data → ② 官方/成熟推理框架 + 真实初值 → ③ 自建 [`GISWLH/WeatherAI`](https://github.com/GISWLH/WeatherAI) 移植**(最后手段)。详见 / see [`docs/SURVEY.md`](docs/SURVEY.md).

## 已跑通 / What has actually run

| Route | Model | Init | 数据来源 / Source | 状态 |
|---|---|---|---|---|
| ① hosted, historic | IFS HRES, GraphCast, Pangu, FuXi, GenCast(mean), NeuralGCM, IFS-ENS(mean) | 2020-10-01 … 10-31, 00/12Z (62) | WeatherBench 2 GCS zarr, ERA5 truth, 1.5° | ✅ [`results/oct2020`](results/oct2020) |
| ① hosted, latest | AIFS-single, IFS HRES (ECMWF Open Data); AIGFS, GFS (NOAA S3) | 2026-10-02 06Z (GFS 12Z) | `ecmwf-opendata`, HTTPS byte-range | ✅ forecast.nc + maps [`results/latest`](results/latest) |
| ① hosted, verified | the four above | 2026-09-29 06Z | truth = IFS analysis (proxy, not ERA5) | ✅ [`results/live_2026-09-29T06`](results/live_2026-09-29T06) |
| ② inference on HF GPU | Aurora 0.25° (`microsoft-aurora`, upstream) | 2020-10-01 00Z, ERA5 IC | ZeroGPU Space tab, 48 h in 49 s | ✅ [`results/aurora_2020-10-01`](results/aurora_2020-10-01) |
| ② inference on HF GPU | Aurora 0.25° HRES-T0 fine-tuned | 2026-10-02 06Z, IFS-analysis IC | same | ✅ 48 h forecast produced (not yet verifiable: valid times are in the future) |
| ② Colab | Aurora; Earth2Studio models | – | [`notebooks/`](notebooks) | ⚠️ written, **not executed on Colab** |
| ① WeatherNext 2/3 | – | – | needs Google allow-list | ❌ blocked (user action) |

Numbers are in the result folders (`metrics.csv`, `skill.png`) and summarised in [`docs/RESULTS.md`](docs/RESULTS.md).

## 用法 / Usage

```bash
pip install -e .                       # needs Python >= 3.11; eccodes for GRIB
weatherpre models                      # list models and their route
weatherpre latest --model aifs-single --lead 0-120/24      # newest complete cycle -> data/.../forecast.nc + maps.png
weatherpre run --model aigfs --init 2026-10-02T06 --lead 0-240/24
weatherpre run --model graphcast --init 2020-10-01T00 --lead 24-240/24     # extract from WeatherBench 2
weatherpre compare --models hres,graphcast,pangu,fuxi,gencast --init 2020-10-01T00..2020-10-31T12 --lead 24-240/24
weatherpre compare --models aifs-single,aigfs,ifs-hres,gfs --init 2026-09-29T06 --lead 24-72/24 \
                   --extra aurora=data/hf/aurora_finetuned_2026-09-29T12.nc   # add a forecast produced on HF/Colab
```

All forecasts use the WeatherBench 2 layout (`time, prediction_timedelta, level, latitude, longitude`; `geopotential`, `temperature`, `2m_temperature`, `mean_sea_level_pressure`), scored with **WeatherBench-X** (area-weighted RMSE, ACC).

GPU 推理 / GPU inference: HF Space [`LonghaoWang/weatherai-graphcast-smoke`](https://huggingface.co/spaces/LonghaoWang/weatherai-graphcast-smoke) (ZeroGPU; "WeatherPre" accordion; code in [`hf/`](hf)) or the Colab notebooks. Tokens are never stored in the repo.

## 局限 / Limits (read before quoting numbers)

* AIFS-single/IFS Open Data keep only ~4 days → **no archive**; "latest" means latest cycle, historic scoring uses WeatherBench 2.
* Recent-date truth is the **IFS analysis** (proxy): it favours IFS-derived systems and penalises GFS/AIGFS at short leads. Not an ERA5 score; no ACC (no 0.25° climatology wired for live).
* Oct 2020 Aurora run uses the ERA5-pretrained checkpoint on ERA5 — Oct 2020 lies in its training period, so this is a pipeline check, **not** an out-of-sample skill claim.
* Historic WeatherBench 2 forecasts are different systems with different initial conditions (ERA5- vs IFS-initialised) and resolutions (1.5° here).
* Not run: WeatherNext, FourCastNet3/Pangu/FuXi/GraphCast live inference, ensembles/CRPS, S2S/ocean models (see SURVEY for routes).

## 许可 / Licences

Code: Apache-2.0. Data: ECMWF Open Data CC-BY-4.0 (attribute ECMWF); NOAA public domain; WeatherBench 2 / ERA5 per Copernicus terms. Aurora weights MIT. Several upstream weights are non-commercial (e.g. FuXi-S2S CC-BY-NC-ND) — never committed; check each licence before redistributing outputs.
