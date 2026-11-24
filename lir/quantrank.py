"""Preregistration section 15a: the circuit plan against random plans of the same shape, on perplexity.

  python -m lir.quantrank measure --size 3-27-6   # perplexity of the circuit plan, 43 random plans and the reversed plan -> runs/quant/ranks.json
  python -m lir.quantrank report                  # R1 -> results/lir/stage2/quantization_ranks.{json,md}

Section 15 evaluates three random plans for safety, and its gate measured the perplexity of those three: at every tier
size it tried, the circuit plan's perplexity was the lowest of the four. This block asks whether that holds against
many random plans. It draws 40 more of the same tier size with seeds the gate did not use and measures perplexity
only, with the gate's own code and text: no attack prompt is answered and no guard reads anything. The circuit plan
and the gate's three random plans are measured again in the same run, so every value that is compared comes from one
session. R1 counts the random plans, the gate's three included, at or below the circuit plan.

`measure` needs the environment of lir.quantize (see there). `report` reads files only.
"""
from __future__ import annotations

import argparse
import json
import time

from lir import quantize as q
from lir.common import device, empty_cache, git_commit, write_json

NEW_SEEDS = tuple(range(6, 86, 2))     # 40 plans. The gate used 0, 2 and 4; seeds two apart share no draw
ALL_SEEDS = tuple(q.RANDOM_SEEDS) + NEW_SEEDS   # the 43 random plans R1 counts
ALPHA = 0.05
OUT_RUNS = q.OUT_RUNS
OUT = q.OUT
R1_ROWS = {True: "At the same 4.0 bits, the model kept a lower perplexity under the circuit plan than under random plans of "
                 "the same shape",
           False: "The circuit plan's perplexity could not be told apart from that of random plans of the same shape"}


def parse_size(text: str) -> tuple[int, int, int]:
    rung = tuple(int(x) for x in text.split("-"))
    if rung not in q.RUNGS:
        raise SystemExit(f"not a tier size of section 15: {text}")
    return rung


def reversed_plan(scores: dict[str, float], tiers: dict[str, int], n_layers: int = q.N_LAYERS) -> dict:
    """The circuit plan upside down: the layers the scores rank lowest at 8 bits and those they rank highest at 2."""
    from circuitkit.applications.quantization.quant_utils import build_quantization_plan, compute_layer_scores

    attn, mlp = compute_layer_scores(*q.split_scores(scores), n_layers, aggregation="mean")
    flip = lambda s: ({k: -v for k, v in s.items()} if isinstance(s, dict) else [-v for v in s])
    plan = build_quantization_plan(flip(attn), flip(mlp), high_fraction=tiers["high"] / n_layers, mid_fraction=tiers["mid"] / n_layers)
    for comp in ("attn", "mlp"):
        got = {t: sum(x == t for x in plan[comp].values()) for t in tiers}
        if got != tiers:
            raise SystemExit(f"reversed/{comp}: tier sizes {got}, wanted {tiers}")
    return plan


def plans_to_measure(scores: dict[str, float], rung, seeds: tuple[int, ...] = ALL_SEEDS) -> dict[str, dict]:
    """name -> plan, in the order they are measured: the circuit plan, the random plans (the gate's three first) and
    the reversed plan. A random plan that repeats the circuit plan or another random plan is refused."""
    tiers = dict(zip(q.BITS, rung))
    plans = q.layer_plans(scores, tiers=tiers, seeds=tuple(seeds))
    out = {"circuit": plans["circuit"]}
    key = lambda p: json.dumps(p, sort_keys=True)
    seen = {key(plans["circuit"])}
    for s in seeds:
        p = plans[f"random-{s}"]
        if key(p) in seen:
            raise SystemExit(f"random-{s} repeats a plan that is already there")
        seen.add(key(p))
        out[f"random-{s}"] = p
    out["reversed"] = reversed_plan(scores, tiers)
    return out


def measure(a) -> None:
    """Perplexity, as the gate measures it, of every plan of one tier size. Each value is saved as it ends, so a second
    call goes on where the first stopped."""
    from lir import generate

    rung = parse_size(a.size)
    scores, rows = q.node_scores(), q.task_rows()
    plans = plans_to_measure(scores, rung)
    out = OUT_RUNS / "ranks.json"
    OUT_RUNS.mkdir(parents=True, exist_ok=True)
    got = json.loads(out.read_text()) if out.exists() else {}
    for name, plan in plans.items():
        key = f"{name}@{q.rung_name(rung)}"
        if key in got:
            continue
        if a.before and time.time() >= a.before:
            print(f"not started: {key}; it is {time.strftime('%F %T')} and the limit was {time.strftime('%F %T', time.localtime(a.before))}")
            break
        tok, model = generate.load_model(None)
        info = {"device": str(device()), "commit": git_commit(), "layers": q.tier_lists(plan)}
        info.update(q.quantized(model, plan))
        t0 = time.time()
        info["perplexity"] = q.perplexity(tok, model, rows)
        info["seconds"] = round(time.time() - t0, 1)
        got[key] = info
        write_json(out, got)
        del model
        empty_cache()
        print(f"{key}: perplexity {info['perplexity']['all']:.2f}", flush=True)


def rank_result(circuit: float, random: dict[str, float], alpha: float = ALPHA) -> dict:
    """R1. circuit: the circuit plan's perplexity; random: the perplexity of each random plan counted. If the circuit
    plan were one more random plan, every rank among them would be as likely as any other, so the chance that at most
    `at_or_below` random plans reach its value is (at_or_below + 1) / (n + 1). A value that is not a number counts as
    above the circuit plan's."""
    vals = sorted(random.values(), key=lambda v: (v != v, v))
    c = sum(v <= circuit for v in vals)
    n = len(vals)
    p = (c + 1) / (n + 1)
    finite = [v for v in vals if v == v and v != float("inf")]
    med = finite[len(finite) // 2] if len(finite) % 2 else (finite[len(finite) // 2 - 1] + finite[len(finite) // 2]) / 2
    return {"circuit": circuit, "plans": n, "at_or_below": c, "p": p, "passes": p < alpha, "report": R1_ROWS[p < alpha],
            "lowest": finite[0], "median": med, "highest": finite[-1]}


def result(gate: dict, ranks: dict, decides: str | None = None, seeds: tuple[int, ...] = ALL_SEEDS) -> dict:
    """R1 for every tier size at which the circuit plan and all the random plans are measured in `ranks`, the raw file
    of `measure`. gate: the raw file of the gate, whose values are shown beside and decide nothing here. decides: the
    tier size whose result carries the sentence."""
    ppl = lambda d, key: d[key]["perplexity"]["all"]
    res = {"seeds": list(seeds), "alpha": ALPHA, "decides": decides, "bf16": ppl(gate, "bf16"), "int4": ppl(gate, "int4"), "sizes": {}}
    for rung in q.RUNGS:
        size = q.rung_name(rung)
        names = [f"random-{s}@{size}" for s in seeds]
        if f"circuit@{size}" not in ranks or not all(n in ranks for n in names):
            continue
        values = {n.split("@")[0]: ppl(ranks, n) for n in names}
        r = rank_result(ppl(ranks, f"circuit@{size}"), values)
        r["random_plans"] = values
        r["in_the_gate"] = {a: ppl(gate, f"{a}@{size}") for a in q.MIXED if f"{a}@{size}" in gate}
        if f"reversed@{size}" in ranks:
            r["reversed"] = ppl(ranks, f"reversed@{size}")
        r["commits"] = sorted({ranks[k].get("commit") for k in [f"circuit@{size}", *names]}, key=str)
        res["sizes"][size] = r
    return res


def report_md(res: dict) -> str:
    n = len(res["seeds"])
    lines = ["### The circuit plan against random plans of the same shape, on perplexity (section 15a: R1)", "",
             f"{n} random plans of a tier size, the three of section 15's gate and {n - 3} more drawn with seeds the gate did not "
             "use, each quantized to 4.0 bits and written out as the arms of section 15 are. Perplexity is the gate's: the "
             "reference answers of the 300 held-out task prompts, given the prompt. **No attack prompt is answered and no "
             "guard reads anything, so this says nothing about safety: it does not stand in for Q1 or Q2 of section 15.** "
             "The circuit plan and the gate's three random plans were measured again in the same run as the new plans. "
             "Registered after the gate had been read, where the circuit plan had the lowest perplexity of four plans at "
             "every tier size tried; that is why the gate's three plans are counted too. The unquantized model's perplexity "
             f"is {res['bf16']:.2f} and uniform 4-bit's {res['int4']:.2f}.", ""]
    for size, r in res["sizes"].items():
        h, m, l = size.split("-")
        head = f"With {h}, {m} and {l} layers at 8, 4 and 2 bits"
        count = (f"the circuit plan's perplexity is {r['circuit']:.2f}; {r['at_or_below']} of the {r['plans']} random plans are at or "
                 f"below it (p = {r['p']:.3f})")
        tail = (f" The random plans run from {r['lowest']:.2f} to {r['highest']:.2f}, with a median of {r['median']:.2f}."
                + (f" The reversed plan, with the circuit's ranking upside down: {r['reversed']:.2f}." if "reversed" in r else "")
                + (" In the gate, in another session: the circuit plan " + f"{r['in_the_gate']['circuit']:.2f}" + ", its three random plans "
                   + ", ".join(f"{v:.2f}" for a, v in r["in_the_gate"].items() if a != "circuit") + "." if "circuit" in r["in_the_gate"] else ""))
        if size == res["decides"]:
            lines.append(f"- **R1. {head}, the size of the arms of section 15: {count}. {r['report']}.**{tail}")
        else:
            lines.append(f"- Beside, with no sentence. {head}: {count}.{tail}")
    if res["decides"] not in res["sizes"]:
        lines.append("- R1 is not read: the tier size of the arms does not have all its plans measured.")
    lines += ["", "p is the chance, if the circuit plan were one more random plan, that so few random plans would reach its value: the "
              f"number at or below it plus one, over the number of plans plus one. With {n} plans it is below 0.05 only if at most "
              "one random plan is at or below the circuit plan.", "",
              "The plans differ in where the 8-bit layers go as well as the 2-bit ones, so this does not say which of the two "
              "matters. Not done: a comparison with a ranking that knows nothing of refusal, such as one by the size of the "
              "weights or by the position of a layer. So this does not show that the refusal circuit in particular is what "
              "matters, only that its ranking carries information about which layers bear fewer bits."]
    return "\n".join(lines)


def report(a) -> None:
    gate_p, ranks_p = OUT_RUNS / "gate.json", OUT_RUNS / "ranks.json"
    if not gate_p.exists() or not ranks_p.exists():
        print("pending, nothing written: the gate or the plans of this block are not measured")
        return
    used = q.chosen().get("rung")
    res = result(json.loads(gate_p.read_text()), json.loads(ranks_p.read_text()), q.rung_name(used) if used else None)
    if not res["sizes"]:
        print("pending, nothing written: no tier size has all its plans measured")
        return
    md = report_md(res)
    write_json(OUT / "quantization_ranks.json", res)
    (OUT / "quantization_ranks.md").write_text(md + "\n")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["measure", "report"])
    ap.add_argument("--size", help="measure: the tier size, as 3-27-6")
    ap.add_argument("--before", type=float, default=0, help="measure: start no plan at or after this time (epoch seconds)")
    a = ap.parse_args()
    if a.what == "measure":
        if not a.size:
            raise SystemExit("measure needs --size")
        measure(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
