"""weatherpre command line.

    weatherpre models [--scale weather|s2s]
    weatherpre forecast MODEL [VARIABLE] [LEAD] [--init DATE] [--plot]
    weatherpre compare  MODELS [VARIABLE] [LEAD] [--init DATE|START..END] [--out DIR]

LEAD: 48h | 7d | 15d (weather, <= 15 days)  ·  6w | 45d | week3-4 (S2S, weekly means)."""
import argparse, sys, datetime as dt
from pathlib import Path
from .common import parse_time, parse_leads

def main(argv=None):
    try:
        rc = _main(argv)
    except Exception as e:
        if type(e).__name__ in ("BackendUnavailable", "ValueError", "KeyError"):
            print(f"weatherpre: {e}", file=sys.stderr); rc = 2
        else:
            raise
    return rc or 0

def _main(argv=None):
    ap = argparse.ArgumentParser("weatherpre", description="Weather (<= 15 d) and S2S (> 15 d) forecasts from AI + NWP models, "
                                 "with accuracy comparison. Examples: `weatherpre forecast graphcast z500 7d --init 2020-10-03`, "
                                 "`weatherpre compare ifs-ext,gefs,climatology t2m 6w --init 2020-10-01`.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("models", help="list models, scale, route and status"); p.add_argument("--scale", choices=["weather", "s2s"])
    p.add_argument("--all", action="store_true", help="include GPU-only and blocked models")

    p = sub.add_parser("forecast", help="one model: weatherpre forecast aifs t2m 15d")
    p.add_argument("model"); p.add_argument("variable", nargs="?", default="all", help="z500,t850,t2m,msl,tp or 'all'")
    p.add_argument("lead", nargs="?", default="7d", help="48h | 7d | 15d | 6w | week3-4 | 24,48,72")
    p.add_argument("--init", default="latest"); p.add_argument("--out", default="data")
    p.add_argument("--plot", action="store_true", help="also write a maps PNG next to the netCDF")
    g = p.add_mutually_exclusive_group(); g.add_argument("--days", type=int, help=argparse.SUPPRESS)
    g.add_argument("--hours", help=argparse.SUPPRESS); g.add_argument("--preset", choices=["hours", "week", "15days"], help=argparse.SUPPRESS)

    p = sub.add_parser("compare", help="several models + scores: weatherpre compare graphcast,pangu,ifs z500 7d --init 2020-10-03")
    p.add_argument("models", help="comma list or 'auto'"); p.add_argument("variable", nargs="?", default="z500,t2m")
    p.add_argument("lead", nargs="?", default="7d"); p.add_argument("--init", default="latest", help="date, 'latest' or START..END")
    p.add_argument("--out", default="results/compare"); p.add_argument("--no-plot", action="store_true")

    # legacy commands (kept working, not advertised)
    p = sub.add_parser("horizon"); p.add_argument("init", nargs="?", default="latest")
    p.add_argument("--preset", default="week", choices=["hours", "week", "15days"]); p.add_argument("--models", default="auto"); p.add_argument("--out", default="results/horizon")
    p = sub.add_parser("run"); p.add_argument("--model", required=True)
    p.add_argument("--init", required=True); p.add_argument("--lead", default="0-120/24"); p.add_argument("--out", default="data")
    p = sub.add_parser("latest"); p.add_argument("--model", default="aifs-single")
    p.add_argument("--lead", default="0-120/24"); p.add_argument("--out", default="data")
    p = sub.add_parser("compare-wb2", help=argparse.SUPPRESS)
    p.add_argument("--models", required=True); p.add_argument("--init", required=True); p.add_argument("--lead", default="24-240/24")
    p.add_argument("--out", default="results/compare"); p.add_argument("--extra", action="append", default=[])
    a = ap.parse_args(argv)

    if a.cmd == "models":
        from . import catalog as C
        t = C.table(a.scale)
        if not a.all: t = t[t.status == "hosted"]
        import pandas as pd
        with pd.option_context("display.width", 200, "display.max_colwidth", 40):
            print(t.drop(columns=["label"]).to_string(index=False))
        if not a.all: print("\n(+ GPU models via Earth2Studio / HF: `weatherpre models --all`)")
        return 0
    if a.cmd == "forecast":
        from . import api, maps
        ds = api.forecast(a.model, None if a.variable == "all" else a.variable, a.lead, init=a.init, cache=a.out,
                          lead_hours=parse_leads(a.hours) if a.hours else None, lead_days=a.days, preset=a.preset)
        out = Path(a.out) / "forecasts"; out.mkdir(parents=True, exist_ok=True)
        span = f"w{int(ds.week.max())}" if "week" in ds.dims else f"{int(ds.lead.max())}h"
        stem = f"{ds.attrs['model']}_{ds.attrs['init']}_{span}".replace(":", "-")
        ds.to_netcdf(out / f"{stem}.nc"); print(ds); print("wrote", out / f"{stem}.nc")
        if a.plot:
            if "week" in ds.dims:
                for v in ds.data_vars:
                    print("wrote", maps.plot_s2s({ds.attrs["model"]: ds}, v, int(ds.week.max()), str(out / f"{stem}_{v}.png")))
            else:
                ls = [int(h) for h in ds.lead.values[:: max(1, len(ds.lead) // 4)]][:4]
                print("wrote", maps.plot_maps(ds, variables=tuple(ds.data_vars), leads=ls, out=str(out / f"{stem}.png")))
        return 0
    if a.cmd == "compare":
        from .comparison import compare
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        c = compare(a.models, a.variable, a.lead, init=a.init)
        print(c); print("\nranking:\n", c.ranking().round(3).to_string())
        tag = f"{c.scale}_{c.inits[0] if c.inits else a.init}".replace(":", "-")
        if len(c.scores):
            c.to_csv(out / f"{tag}_scores.csv"); print("wrote", out / f"{tag}_scores.csv")
            if not a.no_plot:
                print("wrote", c.plot(str(out / f"{tag}_scores.png")))
                print("wrote", c.plot_maps(str(out / f"{tag}_maps.png")))
        print("truth:", c.truth, "| skipped:", c.skipped or "-")
        return 0
    if a.cmd == "horizon":
        from . import horizon
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        fc, table, info = horizon.compare(a.models if a.models == "auto" else a.models.split(","), a.init, a.preset)
        tag = f"{a.preset}_{next(iter(fc.values())).attrs['init']}"
        table.round(3).to_csv(out / f"{tag}_score.csv", index=False)
        print("truth:", info["truth"], "| skipped:", info["skipped"]); print(table.round(2).to_string(index=False)[:3000])
        return 0
    from . import pipeline as P
    if a.cmd == "run":
        print(P.run(a.model, parse_time(a.init), parse_leads(a.lead), a.out))
    elif a.cmd == "latest":
        print(P.latest(a.model, parse_leads(a.lead), a.out))
    elif a.cmd == "compare-wb2":
        models = [m for m in a.models.split(",") if m]; leads = parse_leads(a.lead)
        if ".." in a.init:
            s, e = a.init.split(".."); s, e = parse_time(s), parse_time(e)
            inits = []; t = s
            while t <= e: inits.append(t); t += dt.timedelta(hours=12)
            df = P.compare_historic(models, inits, leads, Path(a.out))
        else:
            df = P.compare_live(models, parse_time(a.init), leads, Path(a.out), extra=dict(x.split("=", 1) for x in a.extra))
        print(df.groupby(["model", "lead_h"]).mean(numeric_only=True).round(3).to_string())
    return 0

if __name__ == "__main__":
    sys.exit(main())
