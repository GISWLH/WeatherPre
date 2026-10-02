# GPU routes / GPU 路线

* **HF Space (used, tested)** `LonghaoWang/weatherai-graphcast-smoke` → "WeatherPre" accordion (`hf/weatherpre_space.py`). API: `gradio_client.Client(space, token=$HF_TOKEN).predict(init, "era5"|"ifs", steps, "small"|"pretrained"|"finetuned", api_name="/weatherpre_aurora")`. ZeroGPU limit 120 s per GPU call; the checkpoint is fetched on CPU outside the GPU window. Space runs Python 3.10 → Earth2Studio (needs ≥3.11) is not installable there; only upstream packages that support 3.10 (microsoft-aurora, …).
* **Colab (written, not executed)** `notebooks/colab_aurora.ipynb` (same module as the HF run), `notebooks/colab_earth2studio.ipynb` (Earth2Studio models; API as documented upstream, wrapper `weatherpre/adapters/earth2studio_run.py` untested).
* **Box** CPU only (15 GB RAM): Aurora-small at 0.25° was OOM-killed on CPU, so inference stays on HF/Colab.
