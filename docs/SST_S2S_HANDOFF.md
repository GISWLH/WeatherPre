# Hand-off to SST_S2S (2027-01 – 2027-06 climate report)

SST_S2S is not part of this change and was not modified. This page is the interface WeatherPre provides and the
downstream changes SST_S2S needs before any AI model can influence the report.

## What WeatherPre changes do — and do not — do

* They make ORCA-DL and FuXi-S2S selectable and runnable without editing research scripts, with members, masks, units,
  manifests and an explicit eligibility decision per model / variable / period.
* They **do not** add models to the report. SST_S2S currently hard-codes the NMME model list and a 90 % NMME + 10 % CAS
  SST weighting; until that is configuration-driven, a WeatherPre export changes nothing in the report.
* No AI model is report-eligible today: none has an independent hindcast + calibration for the report variables and
  leads. Exports mark them `comparison_only`.

## Files (`weatherpre.report.export_bundle`)

```
out/report_2027H1/
  models.json        conventions, region, independent_models, one row per model x variable x period (see below)
  models.csv         same rows as a table
  orca-dl.nc         weatherpre-common-1: tos (member, period, latitude, longitude) degC, valid_tos mask, valid_start/end
  fuxi-s2s.nc        t2m / tp / sst (member, period, latitude, longitude), valid_sst
  *.weatherpre.json  run manifests (code revisions of WeatherPre and WeatherAI, weights / input / output sha256, timings, memory)
```

Row fields: `model, independence_group, variable, period, valid_start, valid_end, status (executed | comparison_only |
eligible | excluded), reasons[] (blocking), notes[] (caveats, e.g. "tos is a 5 m proxy"), members, coverage (area fraction
of the report region with data), completeness, duplicate_of, experiment_type, file`.

## Required SST_S2S changes

1. **Configuration-driven model selection** instead of the fixed NMME list, e.g.

   ```yaml
   models:
     nmme:      {source: nmme, role: weighted}
     cas_sst:   {source: cas, role: weighted}
     orca-dl:   {source: weatherpre, bundle: out/report_2027H1, role: comparison}     # becomes weighted only if eligible
     fuxi-s2s:  {source: weatherpre, bundle: out/report_2027H1, role: comparison}
   variables: [sst, t2m, tp]
   periods: [2027-01, 2027-02, 2027-03, 2027-04, 2027-05, 2027-06]
   ```
2. **Eligibility per variable and period**: read `models.json`; only rows with `status == "eligible"` may enter weighted
   products; `comparison_only` rows can be shown beside them; `excluded` rows are listed with their reasons. FuXi-S2S
   reaches at most 42 days after its init, so for January–June it can cover at most the first weeks; ORCA-DL gives `tos`
   only (a 5 m proxy, never relabelled `sst`).
3. **Calibration and weights**: replace the fixed 90/10 split by weights estimated on a calibration period that does not
   overlap the evaluation period (`weatherpre.hindcast.Split` enforces this), per variable / lead; record them.
4. **Spatial gaps**: when a model has no data at a grid point (ORCA-DL poleward of 63.5°, land), renormalise the weights
   of the remaining models there and store the effective weight map; never treat NaN as zero.
5. **No double counting**: use `independence_group` — the 8 ORCA-DL seeds or a second FuXi-S2S implementation are one
   model.

## Migration example

```python
import json, xarray as xr
meta = json.load(open("out/report_2027H1/models.json"))
use = {(r["model"], r["variable"], r["period"]) for r in meta["rows"] if r["status"] == "eligible"}
show = [r for r in meta["rows"] if r["status"] == "comparison_only"]
for r in show:
    ds = xr.open_dataset(f"out/report_2027H1/{r['file']}")
    field = ds[r["variable"]].sel(period=r["period"])                 # all members; ds[r["variable"]].mean("member") for a map
    mask = ds.get(f"valid_{r['variable']}")                            # 1 = data, 0 = outside domain / land
```
