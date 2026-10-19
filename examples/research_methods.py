"""Runnable synthetic research example; no trained-model forecast skill claim.

Install weatherpre[research], then run python examples/research_methods.py.
"""
import numpy as np
import torch
import xarray as xr
import weatherpre as wp

torch.manual_seed(19)
torch.set_num_threads(2)

# Historical CDFs already prepared on matching thresholds and grids.
history = [torch.rand(32, 2, 4) for _ in range(4)]
observed = (torch.rand(32, 2, 1) < torch.tensor([.2, .4, .6, .8])).float()
model = wp.load_research("pbc", lite=True).fit(history[0], observed, *history[1:])
coords = {"time": np.arange(32), "latitude": [25., 30.], "threshold": [.2, .4, .6, .8]}
arrays = [xr.DataArray(v.numpy(), dims=("time", "latitude", "threshold"), coords=coords) for v in history]
print(wp.calibrate(model, *arrays).isel(time=0))

gen = wp.load_research("genfocal", lite=True, channels=2).eval()
x = torch.randn(1, 2, 8, 8, 2)
cond = {k: torch.zeros_like(x) for k in gen.debiasing.cfg.cond_keys}
result = wp.downscale(gen, x, cond, output_size=(4, 16, 8), flow_steps=2, diffusion_steps=3)
print("GenFocal normalized synthetic output:", tuple(result["output"].shape))

diagnostic = wp.load_research("translocation", lite=True)
erf = wp.diagnose(diagnostic, torch.randn(1, 4, 8, 16), location=(3, 4))
print("GraphCast lite signed ERF:", tuple(erf["gradient"].shape))
