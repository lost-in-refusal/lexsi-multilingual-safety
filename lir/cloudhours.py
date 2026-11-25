"""The cloud GPU time of the project, from the job records of the organisation's Hugging Face account.

  python -m lir.cloudhours     # results/lir/stage2/cloud_jobs.{json,md}

Reads the list of jobs of the namespace the credits belong to (a login with access is needed) and sums the running
time by GPU type. The prices are the list prices `hf jobs hardware` printed on Nov 25, 2026; the account's invoice
is the authority on cost.
"""
from __future__ import annotations

import time

from lir.common import ROOT, write_json
from lir.hfjob import NAMESPACE

OUT = ROOT / "results" / "lir" / "stage2"
PRICE = {"l4x1": 0.80, "l40sx1": 1.80, "a100-large": 2.50}   # dollars an hour


def summary(jobs: list[dict]) -> dict:
    """jobs: one dict a job with flavor, stage and running_secs. Hours and list-price cost by GPU type, every job counted
    whatever its end, because a failed job is paid for too."""
    by = {}
    for j in jobs:
        d = by.setdefault(j["flavor"], {"jobs": 0, "completed": 0, "running_hours": 0.0})
        d["jobs"] += 1
        d["completed"] += int(j["stage"] == "COMPLETED")
        d["running_hours"] += (j["running_secs"] or 0) / 3600
    for flavor, d in by.items():
        d["running_hours"] = round(d["running_hours"], 2)
        d["list_price_dollars"] = round(d["running_hours"] * PRICE[flavor], 2) if flavor in PRICE else None
    known = [d["list_price_dollars"] for d in by.values() if d["list_price_dollars"] is not None]
    return {"by_gpu": by, "jobs": len(jobs), "running_hours": round(sum(d["running_hours"] for d in by.values()), 2),
            "list_price_dollars": round(sum(known), 2)}


def md(res: dict) -> str:
    lines = ["### Cloud GPU time (Hugging Face Jobs)", "",
             f"From the job records of the `{NAMESPACE}` account, read on {res['read_utc']} UTC. Running time is what the record "
             "gives for each job; every job is counted, also the ones that failed or were stopped. Cost is running time at the list "
             "price of the GPU type, not the invoice.", "",
             "| GPU type | Jobs | Completed | Running hours | List price an hour | Cost at list price |", "|---|---:|---:|---:|---:|---:|"]
    for flavor, d in sorted(res["by_gpu"].items()):
        price = f"${PRICE[flavor]:.2f}" if flavor in PRICE else "?"
        cost = f"${d['list_price_dollars']:.2f}" if d["list_price_dollars"] is not None else "?"
        lines.append(f"| `{flavor}` | {d['jobs']} | {d['completed']} | {d['running_hours']:.2f} | {price} | {cost} |")
    lines += [f"| all | {res['jobs']} | | {res['running_hours']:.2f} | | ${res['list_price_dollars']:.2f} |", "",
              "| Job | GPU type | Started (UTC) | End | Running minutes |", "|---|---|---|---|---:|"]
    lines += [f"| `{j['name']}` | `{j['flavor']}` | {j['started_utc'] or ''} | {j['stage'].lower()} | {(j['running_secs'] or 0) / 60:.1f} |"
              for j in res["job_list"]]
    return "\n".join(lines)


def main() -> None:
    from huggingface_hub import HfApi

    api, jobs = HfApi(), []
    for j in api.list_jobs(namespace=NAMESPACE):
        d = api.inspect_job(job_id=j.id, namespace=NAMESPACE)
        jobs.append({"id": d.id, "name": (d.labels or {}).get("name") or "", "flavor": str(d.flavor), "stage": d.status.stage,
                     "started_utc": d.started_at.strftime("%Y-%m-%d %H:%M") if d.started_at else None,
                     "running_secs": d.durations.running_secs if d.durations else None})
    jobs.sort(key=lambda j: j["started_utc"] or "")
    res = dict(summary(jobs), read_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()), namespace=NAMESPACE, prices=PRICE, job_list=jobs)
    write_json(OUT / "cloud_jobs.json", res)
    (OUT / "cloud_jobs.md").write_text(md(res) + "\n")
    print(md(res))


if __name__ == "__main__":
    main()
