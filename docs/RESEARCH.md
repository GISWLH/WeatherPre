# Research methods in the forecast universe

The catalog includes three papers with explicit roles and `research` status:

| Name | Role | Paper | Available path |
|---|---|---|---|
| `pbc` | Probabilistic S2S calibration | [Nature 2026](https://doi.org/10.1038/s41586-026-11126-x) | Historical fit, CDF projection and five-category probabilities |
| `genfocal` | Climate regional downscaling | [Nature Machine Intelligence 2026](https://doi.org/10.1038/s42256-026-01308-7) | Two-stage ReFlow + EDM native lite; explicit normalized tensor inputs |
| `translocation` (`gray-swan`) | Extreme-event diagnostics | [Science Advances 2026](https://doi.org/10.1126/sciadv.ady9406) | GraphCast lite / supplied differentiable backbone and signed ERF |

These are not interchangeable deterministic weather models. `forecast(...)`
rejects these names with a pointer to the research API. They appear in
`wp.models()` / `weatherpre models --all` but are excluded from auto comparison
and hosted-model counts. GenFocal is a climate trajectory method with no
weather/S2S forecast scale. The gray-swan paper introduces a mechanism study,
not an independent forecast architecture.

```bash
pip install "weatherpre[research] @ git+https://github.com/GISWLH/WeatherPre"
```

```python
import weatherpre as wp
print(wp.models()[['model', 'role', 'status']])
pbc = wp.load_research('pbc')
genfocal = wp.load_research('genfocal', channels=4)
diagnostic = wp.load_research('translocation')
```

## PBC: fit, then calibrate

Prepared PyTorch training tensors are `(sample, *grid, threshold)`; four CDF
thresholds yield five category probabilities. Supply historical raw CDFs,
observed indicators, climatology and two lagged observations. Every observation
must have been available at forecast issuance. Seasonal windows, 20-year
climatologies and the paper's rolling hyperparameter selection remain the
data provider's responsibility. The Nature lead windows are days 19-25 and
26-32, which differ from WeatherPre's standard week3/week4 aggregates.

```python
pbc.fit(raw_train, observed_train, climatology_train, lag1_train, lag2_train)
# xarray inputs have matching dates/grid and a 'threshold' dimension.
probability = wp.calibrate(pbc, raw_cdf, climatology_cdf, lag1_cdf, lag2_cdf)
# -> DataArray(time, latitude, longitude, category), units=1
```

For separate Debias++ and Persistence++ thresholds, use the WeatherAI tensor
API's `debias_raw`, `debias_observed` and `debias_target` arguments. Use
`pbc.state_dict()` to save fitted parameters. There is no universal pretrained
PBC checkpoint and ensemble means alone are insufficient input.

## GenFocal and gray-swan diagnostics

```python
genfocal.eval()
result = wp.downscale(genfocal, normalized_climate, condition,
                      output_size=(8, 32, 16), flow_steps=8, diffusion_steps=16)
# condition keys: 'channel:mean', 'channel:std'; tensors B,T,lon,lat,C.
# Supply baseline, residual_mean and residual_std together for physical units.

diagnostic.eval()
erf = wp.diagnose(diagnostic, x, channel=0, location=(3, 4))
# Lite x: B,4,8,16. 'gradient' retains the sign; channel 0 is synthetic.
# For a real model: wp.load_research('translocation', lite=False, backbone=trained_model)
```

Lite networks have random weights and do not produce scientifically validated
forecasts. GenFocal's exact JAX architecture/checkpoint factory is available
separately as `weatherai.models.genfocal.GenFocalOfficial`; it needs the official
Linux/JAX runtime. The native lite U-ViT is not checkpoint compatible.
See [WeatherAI model details and evidence](https://github.com/GISWLH/WeatherAI/blob/main/docs/research_20261019.md).

## HF ZeroGPU

The existing [shared Space](https://huggingface.co/spaces/LonghaoWang/weatherai-graphcast-smoke)
adds `/research_gpu`: PBC CPU/CUDA agreement and valid probabilities, GenFocal
two-stage backward/sampling, GraphCast ERF. These are synthetic smoke checks,
not trained-model forecast-skill tests. Existing `/weatherpre_aurora` remains.
Run WeatherAI's `scripts/run_research_hf.py` to save timestamped GPU evidence.
