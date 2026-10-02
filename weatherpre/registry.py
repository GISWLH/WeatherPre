"""Model registry: name -> access route. Nothing here re-implements a model; every entry points to an
existing hosted dataset, an upstream client library or an upstream inference framework."""
B = "gs://weatherbench2/datasets/"
G = "240x121_equiangular_with_poles_conservative"

WB2_TRUTH = B + f"era5/1959-2022-6h-{G}.zarr"
WB2_CLIM = B + f"era5-hourly-climatology/1990-2019_6h_{G}.zarr"

# route 1a: hosted forecasts in WeatherBench 2 (historic; anonymous GCS read)
WB2_HOSTED = {
    "hres":           dict(path=B + f"hres/2016-2022-0012-{G}.zarr", years="2016-2022", note="IFS HRES"),
    "graphcast":      dict(path=B + f"graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-{G}.zarr", years="2019-11..2021-01", note="GraphCast (Science 2023)"),
    "pangu":          dict(path=B + f"pangu/2018-2022_0012_{G}.zarr", years="2018-2022", note="Pangu-Weather (Nature 2023)"),
    "fuxi":           dict(path=B + f"fuxi/2020-{G}.zarr", years="2020", note="FuXi (npj CAS 2023)"),
    "gencast":        dict(path=B + f"gencast/2020-{G}_mean.zarr", years="2020", note="GenCast ens. mean (Nature 2025)"),
    "neuralgcm":      dict(path=B + f"neuralgcm_deterministic/2020-{G}.zarr", years="2020", note="NeuralGCM det. (Nature 2024)"),
    "neuralgcm-ens":  dict(path=B + f"neuralgcm_ens/2020-{G}_mean.zarr", years="2020", note="NeuralGCM ens. mean"),
    "ifs-ens":        dict(path=B + f"ifs_ens/2018-2022-{G}_mean.zarr", years="2018-2022", note="IFS ENS mean"),
}

# route 1b: live hosted products (latest cycles only)
LIVE = {
    "aifs-single": dict(adapter="ecmwf", cycles=(0, 6, 12, 18), note="ECMWF AIFS-single, CC-BY-4.0"),
    "ifs-hres":    dict(adapter="ecmwf", cycles=(0, 6, 12, 18), note="ECMWF IFS HRES, CC-BY-4.0"),
    "aigfs":       dict(adapter="noaa",  cycles=(0, 6, 12, 18), note="NOAA operational AI-GFS"),
    "gfs":         dict(adapter="noaa",  cycles=(0, 6, 12, 18), note="NOAA GFS (NWP baseline)"),
}

# route 2: upstream frameworks (needs GPU: HF Space / Colab). Earth2Studio class names verified in its repo.
EARTH2STUDIO = {
    "e2s-aifs": "AIFS", "e2s-aifs2": "AIFS2", "e2s-aifs-ens": "AIFSENS", "e2s-aurora": "Aurora", "e2s-fcn3": "FCN3",
    "e2s-sfno": "SFNO", "e2s-pangu": "Pangu24", "e2s-fuxi": "FuXi", "e2s-graphcast-operational": "GraphCastOperational",
    "e2s-graphcast-small": "GraphCastSmall", "e2s-gencast-mini": "GenCastMini", "e2s-ace2": "ACE2ERA5",
    "e2s-weathernext2-cyclones-mini": "WeatherNext2CyclonesMini",
}
