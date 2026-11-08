from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen2.5-0.5B"
ITEMS = ROOT / "cache" / "s1_items.json"
OUT = ROOT / "results"
N_HS, N_MMLU, N_GSM = 200, 200, 50


def hs_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def prepare() -> None:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)
    rng = np.random.default_rng(0)
    hs = load_dataset("Rowan/hellaswag", split="validation")
    hs_items = []
    for i in sorted(rng.choice(len(hs), N_HS, replace=False).tolist()):
        r = hs[i]
        ctx = r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()
        hs_items.append({"ctx": hs_clean(ctx), "endings": [hs_clean(e) for e in r["endings"]], "label": int(r["label"])})
    rng = np.random.default_rng(0)
    mm = load_dataset("cais/mmlu", "all", split="test")
    mm_items = []
    for i in sorted(rng.choice(len(mm), N_MMLU, replace=False).tolist()):
        r = mm[i]
        subject = r["subject"].replace("_", " ")
        body = "\n".join(f"{l}. {c.strip()}" for l, c in zip("ABCD", r["choices"]))
        prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n{r['question'].strip()}\n{body}\nAnswer:"
        mm_items.append({"prompt": prompt, "label": int(r["answer"])})
    gtrain = load_dataset("openai/gsm8k", "main", split="train")
    shots = "".join(f"Question: {gtrain[i]['question']}\nAnswer: {gtrain[i]['answer']}\n\n" for i in range(5))
    gtest = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(0)
    g_items = []
    for i in sorted(rng.choice(len(gtest), N_GSM, replace=False).tolist()):
        r = gtest[i]
        g_items.append({"prompt": shots + f"Question: {r['question']}\nAnswer:", "target": r["answer"].split("####")[-1].strip()})
    ITEMS.parent.mkdir(parents=True, exist_ok=True)
    ITEMS.write_text(json.dumps({"hellaswag": hs_items, "mmlu": mm_items, "gsm8k": g_items}))


class Truncated:
    def __init__(self, model, tok, keep):
        import torch

        self.torch = torch
        self.body = model.model
        self.tok = tok
        self.keep = torch.as_tensor(keep, device=model.device)
        self.w = model.lm_head.weight.detach()[:, self.keep].clone()

    def logits(self, ids, past=None):
        out = self.body(input_ids=ids, past_key_values=past, use_cache=past is not None)
        h = out.last_hidden_state.index_select(-1, self.keep)
        return (h @ self.w.T).float(), out.past_key_values

    def loglik(self, context: str, cont: str) -> float:
        torch = self.torch
        c = self.tok(context, add_special_tokens=False).input_ids
        x = self.tok(context + cont, add_special_tokens=False).input_ids
        n = len(x) - len(c)
        ids = torch.tensor([x], device=self.keep.device)
        with torch.no_grad():
            lg, _ = self.logits(ids)
        lp = torch.log_softmax(lg[0, -n - 1 : -1], -1)
        return float(lp.gather(-1, ids[0, -n:, None]).sum())

    def next_token_logits(self, prompt: str):
        torch = self.torch
        ids = torch.tensor([self.tok(prompt, add_special_tokens=False).input_ids], device=self.keep.device)
        with torch.no_grad():
            lg, _ = self.logits(ids)
        return lg[0, -1]

    def greedy(self, prompt: str, max_new: int = 256) -> str:
        from transformers import DynamicCache

        torch = self.torch
        ids = torch.tensor([self.tok(prompt, add_special_tokens=False).input_ids], device=self.keep.device)
        past = DynamicCache()
        out = []
        with torch.no_grad():
            lg, past = self.logits(ids, past)
            for _ in range(max_new):
                nxt = int(lg[0, -1].argmax())
                if nxt == self.tok.eos_token_id:
                    break
                out.append(nxt)
                text = self.tok.decode(out)
                if "Question:" in text or "\n\n" in text:
                    break
                lg, past = self.logits(torch.tensor([[nxt]], device=ids.device), past)
        return self.tok.decode(out).split("Question:")[0].split("\n\n")[0]


def strict(text: str) -> str | None:
    m = re.search(r"#### (\-?[0-9\.\,]+)", text)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def run() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = json.loads(ITEMS.read_text())
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    h = model.config.hidden_size
    variants = {"full": np.arange(h), "first_removed": np.arange(h // 2, h), "last_removed": np.arange(h // 2)}
    letters = [tok(" " + l, add_special_tokens=False).input_ids[0] for l in "ABCD"]
    rows = []
    for name, keep in variants.items():
        m = Truncated(model, tok, keep)
        start = time.perf_counter()
        acc, accn = [], []
        for it in items["hellaswag"]:
            conts = [" " + e for e in it["endings"]]
            ll = np.array([m.loglik(it["ctx"], c) for c in conts])
            lens = np.array([len(c.encode()) for c in conts])
            acc.append(int(ll.argmax() == it["label"]))
            accn.append(int((ll / lens).argmax() == it["label"]))
        rows += [
            {"task": "HellaSwag", "variant": name, "metric": "acc", "value": float(np.mean(acc)), "n": len(acc)},
            {"task": "HellaSwag", "variant": name, "metric": "acc_norm", "value": float(np.mean(accn)), "n": len(accn)},
        ]
        mm = [int(int(m.next_token_logits(it["prompt"])[letters].argmax()) == it["label"]) for it in items["mmlu"]]
        rows.append({"task": "MMLU", "variant": name, "metric": "acc", "value": float(np.mean(mm)), "n": len(mm)})
        gens = []
        for it in items["gsm8k"]:
            g = m.greedy(it["prompt"])
            gens.append({"target": it["target"], "pred": strict(g), "text": g})
        em = [int(x["pred"] == x["target"]) for x in gens]
        rows.append({"task": "GSM8K", "variant": name, "metric": "strict_em", "value": float(np.mean(em)), "n": len(em)})
        (OUT / f"s1_gsm8k_{name}.json").write_text(json.dumps(gens, indent=1))
        print(name, round(time.perf_counter() - start, 1), [r for r in rows if r["variant"] == name], flush=True)
    OUT.mkdir(exist_ok=True)
    with (OUT / "s1_llm.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
