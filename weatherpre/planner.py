"""Run plans with resource limits: automatic selection never starts unbounded GPU work.

    plan = weatherpre.planner.plan(["fuxi-s2s", "orca-dl", "ace2"], "sst,t2m", "6w", "2022-12-01", limits=Limits(max_gpu_jobs=1))
    print(plan.table())        # what would run, where, how long, and why the rest will not
    plan.execute(dry_run=False, out_dir="runs/")     # only jobs marked "run"

A job fits a GPU only if its peak memory fits ONE device (several GPUs' memory is never added). CPU-capable WeatherAI models
fall back to CPU when allowed. Estimates come from measured peaks / per-step times (catalog, WeatherAI cards); unknown
estimates are marked so.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from . import availability as A, catalog as C, leads as L

# seconds per model step (measured where noted; None = unknown)
STEP_SECONDS = {"fuxi-s2s": {"gpu": 0.67, "cpu": 25.0},         # native fp32 HF ZeroGPU / 8-core CPU (WeatherAI docs)
                "orca-dl": {"gpu": 0.12, "cpu": 6.0},           # 7 steps 0.82 s GPU; CPU ~ 11 steps / ~60 s (estimate)
                "ace2": {"gpu": 0.11, "cpu": 7.4}}              # 32 steps 3.5 s GPU / 236 s CPU (WeatherAI docs)
STEPS_PER_DAY = {"fuxi-s2s": 1, "orca-dl": 1 / 30.4, "ace2": 4}

@dataclass
class Limits:
    allow_gpu: bool = False            # automatic plans do not use GPUs unless this is set
    max_gpu_jobs: int = 1
    max_gpu_mem_gb: float | None = None  # per job; defaults to the largest detected device
    allow_cpu: bool = True
    max_cpu_hours: float = 6.0
    max_members: int = 11

@dataclass
class Job:
    model: str
    decision: str                      # run | skip
    device: str | None
    members: int
    est_hours: float | None
    reasons: list[str] = field(default_factory=list)

@dataclass
class Plan:
    request: dict
    jobs: list[Job]

    def table(self):
        import pandas as pd
        return pd.DataFrame([dict(model=j.model, decision=j.decision, device=j.device, members=j.members,
                                  est_hours=None if j.est_hours is None else round(j.est_hours, 2), reasons="; ".join(j.reasons))
                             for j in self.jobs])

    def to_run(self) -> list[Job]:
        return [j for j in self.jobs if j.decision == "run"]

    def execute(self, dry_run: bool = True, **forecast_kw):
        """Run the 'run' jobs one after another (members sequentially inside each job). Returns {model: Dataset | error}."""
        if dry_run: return {j.model: "dry-run" for j in self.to_run()}
        from .api import forecast
        out = {}
        r = self.request
        for j in self.to_run():
            try:
                out[j.model] = forecast(j.model, r["variables"], r["lead"], r["init"], members=j.members,
                                        device="cuda" if j.device == "gpu" else "cpu", **forecast_kw)
            except Exception as e:                                  # keep going, report
                out[j.model] = f"{type(e).__name__}: {e}"
        return out

def plan(models="auto", variables=None, lead="6w", init=None, *, members: int = 1, limits: Limits | None = None,
         env: A.Environment | None = None, **check_kw) -> Plan:
    limits = limits or Limits()
    env = env or A.Environment.detect()
    lv = lead if isinstance(lead, L.Leads) else L.parse(lead)
    names = [m.name for m in C.MODELS.values()] if models == "auto" else [C.canonical(m) for m in models]
    gpu_budget = limits.max_gpu_jobs
    jobs = []
    for name in names:
        m = C.get(name)
        a = A.check(name, variables, lv, init, env=env, members=members, **check_kw)
        reasons = [f"{r.code}: {r.message}" for r in a.reasons]
        if not a.ok:
            jobs.append(Job(name, "skip", None, members, None, reasons)); continue
        if m.hosted and a.backend not in ("weatherai", "e2s"):
            jobs.append(Job(name, "run", "none (hosted)", members, 0.0, reasons)); continue
        if models == "auto" and m.research_only:
            jobs.append(Job(name, "skip", None, members, None, reasons + ["research_only"])); continue
        n_mem = min(members, limits.max_members)
        if n_mem < members: reasons.append(f"members capped at {limits.max_members}")
        days = a.required_days or 0.0
        st = STEP_SECONDS.get(name, {})
        steps = days * STEPS_PER_DAY.get(name, 0) or None
        mem_cap = limits.max_gpu_mem_gb or env.max_gpu_gb
        fits_gpu = bool(env.gpus) and (m.gpu_mem_gb or 1e9) <= mem_cap
        if limits.allow_gpu and fits_gpu and gpu_budget > 0 and a.backend in ("weatherai", "e2s"):
            gpu_budget -= 1
            est = steps * st["gpu"] * n_mem / 3600 if steps and "gpu" in st else None
            jobs.append(Job(name, "run", "gpu", n_mem, est, reasons)); continue
        if a.backend == "weatherai" and m.cpu_ok and limits.allow_cpu:
            est = steps * st["cpu"] * n_mem / 3600 if steps and "cpu" in st else None
            if est is not None and est > limits.max_cpu_hours:
                jobs.append(Job(name, "skip", "cpu", n_mem, est, reasons + [f"CPU estimate {est:.1f} h > limit {limits.max_cpu_hours} h"]))
            else:
                jobs.append(Job(name, "run", "cpu", n_mem, est, reasons + (["estimate unknown"] if est is None else [])))
            continue
        why = "GPU not allowed in this plan (Limits.allow_gpu)" if not limits.allow_gpu else \
              "no GPU slot left" if gpu_budget <= 0 else f"needs one GPU with >= {m.gpu_mem_gb} GB"
        jobs.append(Job(name, "skip", None, n_mem, None, reasons + [why]))
    return Plan(dict(variables=variables, lead=lead, init=init, members=members), jobs)
