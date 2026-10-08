# Research runs on the shared GPU boxes (xuanze, yongqiang)

Known configuration (re-check with `nvidia-smi` before every run): **xuanze** 2 × RTX 4090 24 GB, **yongqiang** 1 × RTX 4090
24 GB. A job must fit on **one** card; memory of two cards is never added.

Rules
* Research runs are separate from operational services: own user directory (`WP_ROOT`, default `~/weatherpre-research`),
  own virtual environments, own caches and outputs. Do not touch running services, published forecasts or shared envs.
* One environment per model family (dependencies conflict): `venv-torch` (ORCA-DL, ACE2, FuXi-S2S native), `venv-onnx`
  (FuXi-S2S official ONNX; `onnxruntime-gpu` matching the CUDA driver), `venv-jax` (NeuralGCM, `jax[cuda12]`). Each needs
  `pip install -e WeatherAI -e WeatherPre` plus `weatherpre[ai-torch|ai-onnx|ai-jax]`.
* Weights are downloaded by the user into a local cache (`WEATHERPRE_WEIGHTS_<MODEL>`), never committed or redistributed
  (FuXi-S2S: CC-BY-NC-ND; ORCA-DL: licence unstated).
* `weatherpre plan ... --allow-gpu --max-gpu-jobs 1` before anything heavy; automatic selection never starts GPU jobs.

`scripts/remote/run_model.sh MODEL VARS LEAD INIT [forecast options]` does: `nvidia-smi` snapshot → picks the card with the
most free memory and compares it with the model's measured peak (+20 %) → falls back to CPU if it does not fit →
`weatherpre check` → `weatherpre forecast --out-dir` (members one at a time, one file + manifest per member, resumable:
re-running skips members already written with the same configuration hash) → `/usr/bin/time -v` and a second `nvidia-smi`
snapshot next to the outputs. Peak GPU memory of WeatherAI runs is also in each member's manifest (`memory.peak_gpu_mb`).

Not available from the cloud session that wrote this (no SSH access): nothing here has been executed on xuanze or
yongqiang. Reproduction commands are in [VALIDATION.md](VALIDATION.md).
