import argparse, sys, datetime as dt
from pathlib import Path
from . import registry as R
from .common import parse_time, parse_leads

def main(argv=None):
    try:
        return _main(argv)
    except Exception as e:
        if type(e).__name__ == "BackendUnavailable":
            print(f"weatherpre: {e}", file=sys.stderr); raise SystemExit(2)
        raise

def _main(argv=None):
    ap = argparse.ArgumentParser("weatherpre", description="Near-real-time AI weather forecasts from existing hosted data / frameworks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="list models and their access route")
    p = sub.add_parser("run", help="produce forecast.nc for one model/init"); p.add_argument("--model", required=True)
    p.add_argument("--init", required=True); p.add_argument("--lead", default="0-120/24"); p.add_argument("--out", default="data")
    p = sub.add_parser("latest", help="newest complete cycle of a live hosted model"); p.add_argument("--model", default="aifs-single")
    p.add_argument("--lead", default="0-120/24"); p.add_argument("--out", default="data")
    p = sub.add_parser("compare", help="WeatherBench-X RMSE/ACC across models")
    p.add_argument("--models", required=True); p.add_argument("--init", required=True,
        help="single init, or START..END (historic, 12h steps)"); p.add_argument("--lead", default="24-240/24"); p.add_argument("--out", default="results/compare")
    p.add_argument("--extra", action="append", default=[], help="NAME=forecast.nc produced elsewhere (e.g. HF/Colab run)")
    p = sub.add_parser("forecast", help="one call: weatherpre forecast aifs 2020-10-03 --days 15")
    p.add_argument("model"); p.add_argument("init", nargs="?", default="latest")
    g = p.add_mutually_exclusive_group(); g.add_argument("--days", type=int); g.add_argument("--hours", help="e.g. 6,12,24 or 6-48/6"); g.add_argument("--preset", choices=["hours", "week", "15days"])
    p.add_argument("--out", default="data"); p.add_argument("--plot", action="store_true", help="also write a maps PNG next to the netCDF")
    p = sub.add_parser("horizon", help="multi-model table + maps for hours/week/15days")
    p.add_argument("init", nargs="?", default="latest"); p.add_argument("--preset", default="week", choices=["hours", "week", "15days"])
    p.add_argument("--models", default="auto"); p.add_argument("--out", default="results/horizon")
    a = ap.parse_args(argv)
    if a.cmd == "models":
        print("# hosted WeatherBench2 (historic, key-free):");  [print(f"  {k:14s} {v['years']:18s} {v['note']}") for k, v in R.WB2_HOSTED.items()]
        print("# live hosted products:");  [print(f"  {k:14s} {v['note']}") for k, v in R.LIVE.items()]
        print("# Earth2Studio (GPU, Colab/HF):");  [print(f"  {k:26s} -> earth2studio.models.px.{v}") for k, v in R.EARTH2STUDIO.items()]
        return
    if a.cmd == "forecast":
        from . import api, maps
        ds = api.forecast(a.model, a.init, lead_hours=parse_leads(a.hours) if a.hours else None, lead_days=a.days, preset=a.preset, cache=a.out)
        out = Path(a.out) / "forecasts"; out.mkdir(parents=True, exist_ok=True)
        stem = f"{ds.attrs['model']}_{ds.attrs['init']}_{int(ds.lead.max())}h"
        ds.to_netcdf(out / f"{stem}.nc"); print(ds); print("wrote", out / f"{stem}.nc")
        if a.plot:
            ls = [int(h) for h in ds.lead.values[:: max(1, len(ds.lead) // 4)]][:4]
            print("wrote", maps.plot_maps(ds, variables=("z500", "t2m", "tp"), leads=ls, out=str(out / f"{stem}.png")))
        return
    if a.cmd == "horizon":
        from . import api, horizon, maps
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        fc, table, info = horizon.compare(a.models if a.models == "auto" else a.models.split(","), a.init, a.preset)
        tag = f"{a.preset}_{next(iter(fc.values())).attrs['init']}"
        table.round(3).to_csv(out / f"{tag}_score.csv", index=False)
        print("truth:", info["truth"], "| skipped:", info["skipped"]); print(table.round(2).to_string(index=False)[:3000])
        return
    from . import pipeline as P
    if a.cmd == "run":
        print(P.run(a.model, parse_time(a.init), parse_leads(a.lead), a.out))
    elif a.cmd == "latest":
        print(P.latest(a.model, parse_leads(a.lead), a.out))
    elif a.cmd == "compare":
        models = [m for m in a.models.split(",") if m]; leads = parse_leads(a.lead)
        if ".." in a.init:
            s, e = a.init.split(".."); s, e = parse_time(s), parse_time(e)
            inits = []; t = s
            while t <= e: inits.append(t); t += dt.timedelta(hours=12)
            df = P.compare_historic(models, inits, leads, Path(a.out))
        else:
            init = parse_time(a.init)
            df = P.compare_live(models, init, leads, Path(a.out), extra=dict(x.split("=", 1) for x in a.extra))
        print(df.groupby(["model", "lead_h"]).mean(numeric_only=True).round(3).to_string())

if __name__ == "__main__":
    main()
