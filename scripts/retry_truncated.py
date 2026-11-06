"""Drop truncated rows from an arm's task outputs so lir.evaluate regenerates them at the current cap."""
import json
import shutil
import sys
from pathlib import Path

d = Path("runs/eval") / sys.argv[1]
gen, grd = d / "task.gen.jsonl", d / "task.guard.jsonl"
rows = [json.loads(l) for l in open(gen)]
bad = {r["uid"] for r in rows if r["truncated"]}
shutil.copy(gen, d / "task.gen.cap512.jsonl")
with open(gen, "w") as f:
    for r in rows:
        if r["uid"] not in bad:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
if grd.exists():
    shutil.copy(grd, d / "task.guard.cap512.jsonl")
    keep = [l for l in open(grd) if json.loads(l)["uid"] not in bad]
    open(grd, "w").writelines(keep)
print(f"{sys.argv[1]}: {len(bad)} truncated task rows removed for regeneration")
