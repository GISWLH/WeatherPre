import argparse, datetime as dt
from pathlib import Path
from . import registry as R
from .common import parse_time, parse_leads

def main(argv=None):
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
    a = ap.parse_args(argv)
    if a.cmd == "models":
        print("# hosted WeatherBench2 (historic, key-free):");  [print(f"  {k:14s} {v['years']:18s} {v['note']}") for k, v in R.WB2_HOSTED.items()]
        print("# live hosted products:");  [print(f"  {k:14s} {v['note']}") for k, v in R.LIVE.items()]
        print("# Earth2Studio (GPU, Colab/HF):");  [print(f"  {k:26s} -> earth2studio.models.px.{v}") for k, v in R.EARTH2STUDIO.items()]
        return
    from . import pipeline as P
    if a.cmd == "run":
        print(P.run(a.model, parse_time(a.init), parse_leads(a.lead), a.out))
    elif a.cmd == "latest":
        print(P.latest(a.model, parse_leads(a.lead), a.out))
    elif a.cmd == "compare":
        models = a.models.split(","); leads = parse_leads(a.lead)
        if ".." in a.init:
            s, e = a.init.split(".."); s, e = parse_time(s), parse_time(e)
            inits = []; t = s
            while t <= e: inits.append(t); t += dt.timedelta(hours=12)
            df = P.compare_historic(models, inits, leads, Path(a.out))
        else:
            init = parse_time(a.init)
            df = P.compare_live(models, init, leads, Path(a.out))
        print(df.groupby(["model", "lead_h"]).mean(numeric_only=True).round(3).to_string())

if __name__ == "__main__":
    main()
