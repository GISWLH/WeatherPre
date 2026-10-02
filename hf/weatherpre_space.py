"""WeatherPre tab for the existing ZeroGPU Space (LonghaoWang/weatherai-graphcast-smoke).
Clones GISWLH/WeatherPre at run time, builds real initial conditions on CPU (ERA5 from WeatherBench 2, or IFS open-data
analysis), then runs the *upstream* microsoft-aurora package on the GPU. No token is needed or written."""
from __future__ import annotations
import datetime as dt, os, subprocess, sys, time, traceback
import gradio as gr
import spaces

REPO = "/tmp/WeatherPre"

def _ensure_repo():
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/GISWLH/WeatherPre", REPO], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "pull", "-q"], check=False)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

@spaces.GPU(duration=120)
def _aurora_gpu(batch, steps, variant, out):
    import torch
    from weatherpre.adapters import aurora_run
    t = time.time()
    ds = aurora_run.forecast(batch, int(steps), variant, "cuda")
    ds.to_netcdf(out)
    return f"gpu={torch.cuda.get_device_name(0)} peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB forecast_s={time.time()-t:.1f}"

def run_aurora(init: str, source: str, steps: int, variant: str):
    """init like 2020-10-01T00 ; source era5|ifs ; variant small|pretrained|finetuned"""
    try:
        _ensure_repo()
        from weatherpre.adapters import aurora_run
        from weatherpre.common import parse_time
        t0 = time.time(); t = parse_time(init)
        batch = aurora_run.batch_from_era5(t) if source == "era5" else aurora_run.batch_from_ifs(t, __import__("pathlib").Path("/tmp/ifs"))
        # fetch the checkpoint on CPU, outside the GPU window
        from huggingface_hub import hf_hub_download
        hf_hub_download("microsoft/aurora", {"small": "aurora-0.25-small-pretrained.ckpt", "pretrained": "aurora-0.25-pretrained.ckpt", "finetuned": "aurora-0.25-finetuned.ckpt"}[variant])
        out = f"/tmp/aurora_{variant}_{t:%Y%m%dT%H}_{source}.nc"
        prep = time.time() - t0
        msg = _aurora_gpu(batch, steps, variant, out)
        return out, f"OK init={init} source={source} variant={variant} steps={steps} prep_s={prep:.1f} {msg}"
    except Exception:
        return None, traceback.format_exc()[-3000:]

def tab():
    with gr.Accordion("WeatherPre: Aurora forecast from real initial conditions (ZeroGPU)", open=False):
        i = gr.Textbox("2020-10-01T00", label="init (UTC, YYYY-MM-DDTHH)")
        s = gr.Dropdown(["era5", "ifs"], value="era5", label="initial conditions: era5 (WB2, historic) / ifs (open-data analysis, last ~4 days)")
        n = gr.Slider(1, 20, 4, step=1, label="6-hour steps")
        v = gr.Dropdown(["small", "pretrained", "finetuned"], value="pretrained", label="Aurora checkpoint")
        b = gr.Button("Run Aurora")
        f = gr.File(label="forecast.nc"); t = gr.Textbox(label="status", lines=6)
        b.click(run_aurora, [i, s, n, v], [f, t], api_name="weatherpre_aurora")
