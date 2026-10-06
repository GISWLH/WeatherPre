import os, pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs internet (WeatherBench 2 GCS / NOAA S3); run with WEATHERPRE_NETWORK=1")

def pytest_collection_modifyitems(config, items):
    if os.environ.get("WEATHERPRE_NETWORK"): return
    skip = pytest.mark.skip(reason="network test: set WEATHERPRE_NETWORK=1")
    for it in items:
        if "network" in it.keywords: it.add_marker(skip)
