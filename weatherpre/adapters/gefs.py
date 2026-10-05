"""NOAA GEFSv12 on public S3 (bucket noaa-gefs-pds, anonymous HTTPS, byte-range via the .idx sidecar).

0.5° ensemble mean (``geavg``), 00Z cycles reach 840 h (35 days), 06/12/18Z cycles 384 h. Archive from 2020-09-23
(GEFSv12). If the ensemble-mean file is missing for a lead the control member (``gec00``) is used for the whole
request, and the Dataset says so in its attrs."""
from __future__ import annotations
import datetime as dt, re, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
FIRST = dt.datetime(2020, 9, 23)
MAX_LEAD = 840                                        # 00Z runs; 06/12/18Z stop at 384 h
WANT = [r":HGT:500 mb:", r":TMP:850 mb:", r":TMP:2 m above ground:", r":PRMSL:mean sea level:"]
APCP = r":APCP:surface:"

def url(init: dt.datetime, h: int, member="avg") -> str:
    d, c = f"{init:%Y%m%d}", f"{init:%H}"
    return f"{BUCKET}/gefs.{d}/{c}/atmos/pgrb2ap5/ge{member}.t{c}z.pgrb2a.0p50.f{h:03d}"

def _get(u, rng=None) -> bytes:
    req = urllib.request.Request(u, headers={"Range": f"bytes={rng}"} if rng else {})
    for attempt in range(6):                     # S3 answers 503 "Slow Down" under load: back off and retry
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (500, 502, 503, 504) or attempt == 5: raise
        except Exception:
            if attempt == 5: raise
        time.sleep(2 ** attempt)

def exists(init, h, member="avg") -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url(init, h, member) + ".idx", method="HEAD"), timeout=30); return True
    except urllib.error.HTTPError:
        return False

def latest_init(max_lead=0, now: dt.datetime | None = None) -> dt.datetime:
    now = (now or dt.datetime.utcnow()).replace(minute=0, second=0, microsecond=0)
    t = now - dt.timedelta(hours=now.hour % 6)
    if max_lead > 384: t = t.replace(hour=0)                     # only 00Z runs go beyond 16 days
    for _ in range(16):
        if (max_lead <= 384 or t.hour == 0) and exists(t, int(max_lead)):
            return t
        t -= dt.timedelta(hours=6 if max_lead <= 384 else 24)
    raise RuntimeError("no complete GEFS cycle found in the last 4 days")

def pick_member(init, max_lead) -> str:
    return "avg" if exists(init, max_lead, "avg") else "c00"

def _one(init, h, member, out: Path, precip: bool) -> Path:
    f = out / f"gefs-{member}_{init:%Y%m%dT%H}_f{h:03d}{'p' if precip else ''}.grib2"
    if f.exists() and f.stat().st_size: return f
    u = url(init, h, member)
    idx = _get(u + ".idx").decode().strip().splitlines()
    offs = [int(l.split(":")[1]) for l in idx]
    pats = WANT + ([APCP] if precip and h > 0 else [])
    buf = b""
    for pat in pats:
        for i, l in enumerate(idx):
            if re.search(pat, l):
                buf += _get(u, f"{offs[i]}-{offs[i + 1] - 1 if i + 1 < len(offs) else ''}"); break
    f.write_bytes(buf)
    return f

def fetch(init: dt.datetime, leads, out: Path, member="avg", precip=False, workers=8) -> list[Path]:
    """One small GRIB per lead with z500, t850, t2m, msl (+ 6 h precipitation bucket if `precip`)."""
    if init < FIRST:
        raise ValueError(f"GEFSv12 0.5° archive starts {FIRST:%Y-%m-%d}")
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda h: _one(init, int(h), member, out, precip), leads))

def to_accumulated(ds):
    """GEFS APCP comes in 6 h buckets; turn it into 'accumulated since init' like the other models."""
    if "total_precipitation" not in ds: return ds
    tp = ds.total_precipitation.fillna(0).cumsum("prediction_timedelta")
    return ds.assign(total_precipitation=tp)
