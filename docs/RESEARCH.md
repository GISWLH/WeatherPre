# Official research methods in the forecast universe

`load_research` now defaults to **official author implementations**. Lite
experiments require `lite=True`; errors never trigger a lite fallback.

| Name | Role | Default source and weights |
|---|---|---|
| `pbc` | Probabilistic S2S calibration | [Author PBC scripts](https://github.com/mouatadid/pbc); historical fitting, no universal checkpoint |
| `genfocal` | Climate debiasing / downscaling | [Author GenFocal JAX/Flax](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/genfocal); published ReFlow/raw and SR/EMA parameters |
| `translocation` / `gray-swan` | Extreme-event diagnostics | [Official GraphCast](https://github.com/google-deepmind/graphcast) pretrained 0.25°/13-level model and author-referenced ERF |

```python
import weatherpre as wp
pbc = wp.load_research('pbc', work_dir='pbc-work')
pbc.run('climatology', 'era5-f1_mslp', 19, target_dates='20200106')

# Linux/Python 3.12 plus WeatherAI/hf_official/requirements.txt:
genfocal = wp.load_research('genfocal', stage='super_resolution')
# Evaluate the official EMA network with author-normalized arrays:
# residual = genfocal(x, sigma, cond)
# genfocal.trajectory_sampler(...) invokes the original full trajectory sampler.

diagnostic = wp.load_research('translocation')
# inputs, targets, forcings = diagnostic.prepare(era5_dataset)
# forecast = diagnostic(inputs, targets * float('nan'), forcings)
# erf = wp.diagnose(diagnostic, inputs, targets_template=targets * float('nan'),
#                  forcings=forcings, location={'batch':0,'time':-1,'lat':260,'lon':220})
```

PBC executes the author's unchanged scripts and original data layout. Prepare
its forecast/reforecast/observation histories before correction or fitting.
The paper's windows are days 19–25 and 26–32, distinct from generic weekly
aggregations. Its published archive contains code, not universal fitted weights.

GenFocal requires stage-specific input conditioning, statistics and sampling.
The default checkpoint is CONUS super-resolution; use `stage='debiasing'` for
the released ReFlow network. Full physical downscaling requires the original
trajectory pipeline; a denoiser call is not a climate forecast. `wp.downscale`
is currently the explicit lite tensor helper, so call official sampler methods
on the official model. `wp.calibrate` likewise remains a lite tensor helper;
official PBC is run with `pbc.run` and produces author-format NetCDF files.

All three remain in `wp.models()` / `weatherpre models --all` with distinct
roles and `research` status. They are excluded from automatic deterministic
forecast rankings. The gray-swan study is not a new forecasting architecture.

For untrained experimental models only:

```python
pbc_lite = wp.load_research('pbc', lite=True)
genfocal_lite = wp.load_research('genfocal', lite=True, channels=4)
diagnostic_lite = wp.load_research('translocation', lite=True)
```

`examples/research_methods.py` is explicitly synthetic/lite. Source revisions,
checkpoint URLs, licenses and actual test scope are recorded in
[WeatherAI's integration report](https://github.com/GISWLH/WeatherAI/blob/main/docs/research_20261019.md).
The [official ZeroGPU Space](https://huggingface.co/spaces/LonghaoWang/weatherai-official)
uses a dedicated Linux/Python 3.12 runtime. Existing Aurora endpoints and old
lite smoke tests remain separate. Use WeatherAI's `scripts/run_official_hf.py`
to prepare author assets and record an actual pretrained GPU result.
