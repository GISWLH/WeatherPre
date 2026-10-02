# SURVEY / 调研 (2026-10-03)

Goal / 目标: near-real-time forecasts from frontier AI weather/climate models, **assembled from existing tools, nothing rebuilt**.
Route preference / 路线优先级: **(1) hosted data/API → (2) established framework + real initial conditions → (3) our own `GISWLH/WeatherAI` ports**.

Status legend / 状态: ✅ checked today (URL/listing/PyPI/GitHub/HF fetched) · ⚠️ partly checked · ❌ blocked · ❓ not checked.
Box = CPU, 15 GB RAM, ~20 GB disk, no GPU. HF = ZeroGPU Space `LonghaoWang/weatherai-graphcast-smoke` (reuse; 10-Space limit reached). Colab = free T4 / Pro.

## 1. Hosted forecasts (no inference needed) / 现成预报产品

| Source | What | Access | Licence | Status |
|---|---|---|---|---|
| ECMWF Open Data | IFS HRES/ENS, **AIFS-single**, **AIFS-ENS** 0.25°, 4 runs/day (00/06/12/18z; list showed 00z+06z of 2026-10-02 at probe time) | `ecmwf-opendata` (PyPI 0.3.34, Apache-2.0) or plain HTTPS `data.ecmwf.int/forecasts/YYYYMMDD/HHz/{ifs,aifs-single,aifs-ens}/0p25/` + `.index` byte-range. Only ~4 days retained (listing began 2026-09-29) → **latest only, no archive** | CC-BY-4.0 | ✅ |
| NOAA **AIGFS** (operational AI-GFS) | 0.25° pres/sfc GRIB2 each 6 h (`aigfs.tHHz.pres.f000…`, ~74 MB/file), plus `init/` GDAS zarr | public S3 bucket `noaa-nws-graphcastgfs-pds` prefix `aigfs.YYYYMMDD/HH/model/atmos/{grib2,init}` (listing: daily dirs 2026-04-16 → 2026-10-02) | US Gov public domain | ✅ |
| NOAA GraphCast-GFS (experimental) | same bucket, `graphcastgfs.YYYYMMDD/` 2024-02-05 → 2026-05-05 (last dir; looks superseded by AIGFS – inference) | S3 | public domain | ✅ |
| NOAA GFS / GEFS analysis+forecasts | truth proxy for latest dates | `noaa-gfs-bdp-pds` S3 (dirs `gfs.20261001/`, `gfs.20261002/` seen) | public domain | ✅ |
| **WeatherBench 2 hosted forecasts** (GCS `gs://weatherbench2/datasets/`, anonymous HTTPS ok) | Pre-computed 2020 (and some 2018/2022) forecasts: `graphcast`/`graphcast_v2` (2018/2020/2022), `pangu`, `fuxi` (2020), `keisler`, `sphericalcnn`, `neuralgcm_deterministic`, `neuralgcm_ens`, `gencast` (2020), `hres`, `ifs_ens`, `ens`, `ifs_extended_range`, `aurora` (2022), `fgn` (2022); resolutions 64x32 / 240x121 / 1440x721. Truth: `era5`, `hres_t0`, climatologies | zarr via `gcsfs`/`xarray` | per-dataset (ERA5: Copernicus licence; model outputs: see each dir / WB2 docs) | ✅ listing |
| **Google WeatherNext 2 / 3** | 64-member 0.25° ensemble to 15 d (WN2) on GCS zarr / BigQuery / Earth Engine; WN3 current | **allow-list via "WeatherNext Data Request" form with a Google account** (see developers.google.com/weathernext). Custom inference on Vertex needs billing + GPU quota | experimental, own ToS | ❌ needs user to apply |
| `google-deepmind/weathernext` | code + public WeatherNext-Cyclones **Mini** 1° ckpt (used by Earth2Studio) | GitHub Apache-2.0 | code Apache-2.0, ckpt terms to check | ✅ repo / ⚠️ ckpt terms |
| NVIDIA hosted forecasts | no public hosted real-time forecast feed found | — | — | ❌ none found (Earth2Studio does inference locally) |

## 2. Frameworks (run the model ourselves) / 推理框架

| Framework | Covers | Install | Status |
|---|---|---|---|
| **NVIDIA Earth2Studio** (Apache-2.0, 1.1k★, active 2026-10-02) | Model wrappers in `earth2studio/models/px`: AIFS, AIFS2, AIFS-ENS, AIFS2-ENS, Aurora, Aurora1.5, FourCastNet3 (fcn3), SFNO, FuXi, FengWu, Pangu, GraphCast (operational, small), GenCast-Mini, WeatherNext2-Cyclones-Mini, FuXi-S2S, ACE2-ERA5, SamudrACE, DLESyM, Atlas, U-CAST, StormCast (+CONUS), StormScope, CBottle. Data sources in `earth2studio/data`: **WB2_ERA5, ARCO, CDS, GFS, GDAS, IFS (ECMWF open), GEFS, HRRR, GOES, IBTrACS** … → real initial conditions for any date | `pip install earth2studio` (PyPI 0.19.0, Py ≥3.11) + per-model extras (`earth2studio[aurora]`, `[graphcast]`, `[aifs]`, `[fcn3]`, `[pangu]`, `[fuxi]`, `[gencast]` …) | ✅ cloned + PyPI |
| ECMWF `ai-models` + plugins | `ai-models` 0.7.4, `ai-models-graphcast` 0.1.0, `-panguweather` 0.0.9, `-fourcastnet` 0.0.7, `-fourcastnetv2` 0.0.3, `-gencast` 0.0.6 (last uploads 2024-12 / 2025-02) – **plugin releases are stale vs Earth2Studio/anemoi**; needs CDS/MARS credentials for ICs | pip | ✅ PyPI, ⚠️ stale |
| ECMWF `anemoi-inference` / `anemoi-models` | AIFS (single/ENS) official runner; 0.12.0, Py 3.11–3.13, Apache-2.0; weights HF `ecmwf/aifs-single-1.1`, `aifs-single-2.0`, `aifs-ens-1.0` (CC-BY-4.0, ungated) | pip | ✅ |
| `microsoft-aurora` 2.0.1 | Aurora (HF `microsoft/aurora`, MIT weights; repo licence flagged NOASSERTION by GitHub) | pip | ✅ |
| `geoarches` 0.2.2 (BSD-3) | ArchesWeather / ArchesWeatherGen; HF `gcouairon/ArchesWeather` (ungated, BSD) | pip | ✅ |
| `fme` 2026.5.1 (Apache-2.0, `ai2cm/ace`) | ACE2-ERA5, SamudrACE; HF `allenai/ACE2-ERA5` (Apache-2.0) | pip | ✅ |
| `neuralgcm` 1.2.2 / `dinosaur` 1.5.0 | NeuralGCM; checkpoints on GCS | pip | ✅ PyPI, ⚠️ ckpt terms |
| NVIDIA `physicsnemo` (Apache-2.0) | CorrDiff, StormCast, FCN3 training | pip | ✅ repo |
| **WeatherBench 2** (`google-research/weatherbench2`, Apache-2.0) | metrics (RMSE, ACC, CRPS, SEEPS, spectra) on zarr | `pip install git+https://github.com/google-research/weatherbench2` (**not on PyPI**) | ✅ repo |
| **WeatherBench-X** (`google-research/weatherbenchX`, Apache-2.0, pushed 2026-10-02) | newer metric library, supports gridded + sparse obs; **not on PyPI** | `pip install git+https://github.com/google-research/weatherbenchX` | ✅ repo |

## 3. Per-model decision / 逐模型方案

GPU: "box" = feasible without GPU; "HF" = ZeroGPU (≈120 s/call, 1 GPU); "Colab" = T4/A100. Sizes are my estimates unless marked checked.

| Model (journal) | Best route today | Latest-date IC | Oct-2020 route | Where | Licence note |
|---|---|---|---|---|---|
| ECMWF IFS HRES | route 1: Open Data (latest) / WB2 `hres` (2020) | n/a (it is the NWP) | WB2 `hres` | box | CC-BY-4.0 / WB2 |
| **AIFS-single** (ECMWF) | route 1: Open Data hosted | hosted | WB2 has none → Earth2Studio `aifs` with WB2/ARCO ERA5 IC *or* anemoi; weights ungated | box for download; inference HF/Colab | CC-BY-4.0 |
| AIFS-ENS | route 1: Open Data hosted | hosted | Earth2Studio `aifsens` | box (download) | CC-BY-4.0 |
| **NOAA AIGFS** | route 1: S3 hosted | hosted | not available (starts 2026-04) | box | public domain |
| GraphCast (*Science* 2023) | WB2 hosted (2018/2020/2022) for Oct 2020; Earth2Studio `graphcast_operational` (GFS/IFS IC) for latest; hosted GraphCast-GFS stops 2026-05 | GFS via Earth2Studio | WB2 `graphcast/2020` (240x121 / 64x32 / raw) | box (read) / HF,Colab (run) | weights CC-BY-NC-SA 4.0 (upstream; ❓ re-verify) |
| Pangu-Weather (*Nature* 2023) | WB2 hosted 2018-22, `pangu_hres_init` 2020-22; Earth2Studio `pangu` for latest | GFS via Earth2Studio | WB2 `pangu` | box / HF | upstream NC licence (❓ verify before redistribution) |
| FuXi (*npj CAS* 2023) | WB2 hosted 2020; Earth2Studio `fuxi` | ERA5 / GFS | WB2 `fuxi` | box / HF | upstream NC (❓) |
| FengWu (*Commun. Earth Environ.* 2025) | Earth2Studio `fengwu` (ONNX from HF `NickGeneva/earth_ai`, no licence on card) | GFS | Earth2Studio w/ WB2_ERA5 | HF/Colab | ⚠️ unclear; keep out of repo |
| Aurora (*Nature* 2025) | `microsoft-aurora` or Earth2Studio `aurora`; WB2 `aurora` only 2022 | GFS/IFS via E2S | run on HF/Colab | HF / Colab | MIT weights |
| FourCastNet3 / SFNO | Earth2Studio `fcn3`/`sfno`; HF `nvidia/fourcastnet3` Apache-2.0 | GFS/ARCO | same | HF/Colab | Apache-2.0 |
| GenCast (*Nature* 2025) | WB2 hosted 2020 (ensemble + mean); Earth2Studio `gencast_mini` (1°) to run | ERA5/IFS | WB2 `gencast/2020` | box (read) | upstream CC-BY-NC-SA (❓) |
| NeuralGCM (*Nature* 2024) | WB2 hosted `neuralgcm_deterministic` / `neuralgcm_ens` 2020; `neuralgcm` pip to run | ERA5 only (IC from ERA5) | WB2 | box (read) | ❓ ckpt terms |
| WeatherNext Gen/2 (FGN) | WB2 `fgn` (2022 only); real-time **needs allow-list** (❌) | — | — | — | ToS |
| WeatherNext Cyclones (*Nature* 2026) | Earth2Studio `weathernext2_cyclones` (Mini 1°); HF `kashif/weathernext2` is a mirror (cc-by-4.0 card) | GFS | — | HF/Colab | ⚠️ check |
| ACE2-ERA5 (*npj CAS* 2025) | Earth2Studio `ace2` or `fme`; HF Apache-2.0 | ERA5 (needs forcings) | WB2_ERA5 | HF/box (1°) | Apache-2.0 |
| ArchesWeather(Gen) (*Sci. Adv.* 2026) | `geoarches`; HF BSD | ERA5/IFS | geoarches | HF | BSD |
| StormCast (*Sci. Adv.* 2026) / CorrDiff | Earth2Studio `stormcast` (HRRR IC, CONUS) ; HF Apache-2.0 | HRRR via E2S | HRRR archive | HF | Apache-2.0 |
| FuXi-S2S (*Nat. Commun.* 2024) | Earth2Studio `fuxi_s2s`; data on HF dataset `FudanFuXi/FuXi-S2S` (**CC-BY-NC-ND**; gated "auto") | ERA5 daily | same | box/HF | **NC-ND: no redistribution** |
| ORCA-DL, TropiCycloneNet, UniCM, GenFocal | our `GISWLH/WeatherAI` ports (route 3) | needs ocean/TC data | — | HF | see WeatherAI docs |
| DLESyM, SamudrACE, Atlas, U-CAST | Earth2Studio wrappers (HF Apache-2.0 / "other" for Atlas) | ERA5 | — | HF | check each |

## 4. Truth & evaluation / 真值与评测

* Historic (Oct 2020): ERA5 from **WB2** `era5/*` (no key) — compare against hosted WB2 forecast zarrs with **WeatherBench-X / WB2 metrics**. CDS API / ARCO-ERA5 (`gs://gcp-public-data-arco-era5`) also possible but CDS needs a key (blocker if used). ERA5 in WB2 ends 2022/2023 → not for "latest".
* Latest: no ERA5 yet (5-day lag, key needed). Proxy truth = **IFS HRES analysis (step 0 of later cycles, open data)** or GFS analysis (S3). Mark as *analysis-proxy*, not ERA5.
* Do **not** re-implement metrics: use WeatherBench-X / WB2.

## 5. Constraints → what runs where

| Task | Box | HF ZeroGPU | Colab |
|---|---|---|---|
| Download + plot hosted forecasts (AIFS, AIGFS, IFS) | ✅ | – | ✅ |
| Read WB2 hosted 2020 forecasts + metrics at 240x121 | ✅ (streaming) | – | ✅ |
| Inference of 0.25° models (Aurora, FCN3, AIFS, GraphCast op., Pangu) | ❌ no GPU/RAM | ✅ ≤120 s/call, preload weights | ✅ |
| 1° models (GraphCast small, GenCast mini, ACE2, WN-Cyclones mini) | ⚠️ CPU maybe | ✅ | ✅ |

## 6. Blockers needing the user / 需要你操作

1. **WeatherNext** real-time data: fill the WeatherNext Data Request form with a Google account (allow-list).
2. **CDS API key** (only if ERA5 beyond WB2 range or ai-models/MARS flows are wanted) — not needed for the first demos.
3. Weights/licences flagged ❓ must be re-read before any redistribution; NC/ND weights are never committed.
