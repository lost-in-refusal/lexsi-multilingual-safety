"""Lost in Refusal demo: compare Tiny Aya Global before and after fine-tuning.

Reads the published evaluation outputs (no model is loaded), so it runs on a free CPU Space.
"""
import json
import os
import random
from collections import defaultdict

import gradio as gr
from huggingface_hub import snapshot_download

REPO = "lost-in-refusal/lost-in-refusal-eval-outputs"
ARMS = {"base": "Tiny Aya Global", "plain": "+ plain LoRA", "safegrad": "+ SafeGrad LoRA"}
LANGS = {"en": "English", "ko": "Korean", "zh": "Mandarin", "sw": "Swahili (never fine-tuned)"}
SETS = {"attack": "MultiJail attack prompts", "xstest": "XSTest safe prompts (over-refusal)"}

root = os.environ.get("LIR_EVAL_DIR") or snapshot_download(
    REPO, repo_type="dataset", allow_patterns=["*/attack.guard.jsonl", "*/xstest.guard.jsonl", "results.md"])
rows = defaultdict(dict)  # (set, family, lang) -> arm -> record
for arm in ARMS:
    for s in SETS:
        try:
            with open(f"{root}/{arm}/{s}.guard.jsonl", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    rows[(s, r["family"], r["lang"])][arm] = r
        except FileNotFoundError:
            pass
keys = sorted(k for k, v in rows.items() if len(v) == len(ARMS))
try:
    results_md = open(f"{root}/results.md", encoding="utf-8").read()
except FileNotFoundError:
    results_md = ""


def badge(r, s):
    if s == "attack":
        bad = r["safety"] != "Safe"
        return ("🔴 judged unsafe" if bad else "🟢 safe") + f" ({r['safety']})"
    return "🟠 refused a safe prompt" if r["refusal"] == "Yes" else "🟢 answered"


def show(set_name, lang, only_changed, seed):
    pool = [k for k in keys if k[0] == set_name and k[2] == lang]
    if only_changed:
        pool = [k for k in pool if len({(rows[k][a]["safety"], rows[k][a]["refusal"]) for a in ARMS}) > 1]
    if not pool:
        return "No examples for this filter.", "", "", ""
    k = random.Random(seed).choice(pool)
    v = rows[k]
    prompt = f"**Prompt** ({LANGS[lang]}, `{k[1]}`):\n\n> {v['base']['prompt']}"
    outs = [f"**{badge(v[a], set_name)}**\n\n{v[a]['response']}" for a in ARMS]
    return prompt, *outs


with gr.Blocks(title="Lost in Refusal") as demo:
    gr.Markdown("# Lost in Refusal\nBenign fine-tuning of Tiny Aya Global in English, Korean and Mandarin, and what "
                "it does to refusals in every language. All responses are greedy generations from our runs; verdicts "
                "are Qwen3Guard-Gen-4B. Some prompts and responses are harmful by design.")
    with gr.Row():
        set_name = gr.Dropdown(list(SETS), value="attack", label="Prompt set",
                               info="; ".join(f"{k}: {v}" for k, v in SETS.items()))
        lang = gr.Dropdown(list(LANGS), value="ko", label="Language")
        only = gr.Checkbox(value=True, label="Only prompts where the verdict changes between models")
        seed = gr.Slider(0, 999, value=0, step=1, label="Example")
    prompt = gr.Markdown()
    cols = []
    with gr.Row():
        for v in ARMS.values():
            with gr.Column():
                gr.Markdown(f"### {v}")
                cols.append(gr.Markdown())
    for c in (set_name, lang, only, seed):
        c.change(show, [set_name, lang, only, seed], [prompt, *cols])
    demo.load(show, [set_name, lang, only, seed], [prompt, *cols])
    with gr.Accordion("Results table", open=False):
        gr.Markdown(results_md)

demo.launch()
