import argparse
import time
import os
import sys
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm import tqdm
from huggingface_hub import HfApi

PROMPT_TMPL = (
    "You are a sentiment classifier.\n"
    "Task: Given the sentence, output only one word: positive or negative.\n"
    "Sentence: {text}\n"
    "Answer:"
)

def normalize_pred(s: str) -> str:
    s = s.strip().lower()
    tok = s.split()[0] if s else ""
    tok = tok.strip(" .,!?:;\"'")
    if "pos" in tok:
        return "positive"
    if "neg" in tok:
        return "negative"
    if tok in ("positive", "negative"):
        return tok
    return "unknown"

def log(msg: str):
    print(msg, flush=True)

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/oft_sst2.yaml")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--disable_tqdm", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    model_name = args.ckpt or cfg["model_name"]
    device = args.device

    log(f"[INFO] model={model_name}, device={device}")
    log(f"[INFO] HF_ENDPOINT env={os.environ.get('HF_ENDPOINT')}")
    try:
        log(f"[INFO] huggingface_hub endpoint={HfApi().endpoint}")
    except Exception as e:
        log(f"[WARN] cannot query HfApi endpoint: {e}")

    # Print cache locations (helps debug locks / wrong cache)
    log(f"[INFO] HF_HOME={os.environ.get('HF_HOME')}")
    log(f"[INFO] TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}")
    log(f"[INFO] default cache (~/.cache/huggingface) exists={os.path.exists(os.path.expanduser('~/.cache/huggingface'))}")

    t0 = time.time()

    log("[INFO] loading tokenizer... (start)")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"[INFO] loading tokenizer... (done) elapsed={time.time()-t0:.1f}s")

    log("[INFO] loading model weights... (start)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if cfg.get("fp16", True) else None,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    log(f"[INFO] loading model weights... (done) elapsed={time.time()-t0:.1f}s")

    ds = load_dataset("glue", "sst2")
    val_ds = ds["validation"].shuffle(seed=cfg.get("seed", 42))
    eval_n_cfg = int(cfg.get("eval_samples", 800))
    eval_n = min(eval_n_cfg, len(val_ds))
    eval_ds = val_ds.select(range(eval_n))
    log(f"[INFO] dataset ready: eval_n={eval_n} (val_size={len(val_ds)})")

    acc = evaluate.load("accuracy")
    preds, refs = [], []

    iterator = eval_ds
    if not args.disable_tqdm:
        iterator = tqdm(
            eval_ds,
            total=eval_n,
            desc="Evaluating",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.0,
            miniters=1,
            leave=True,
        )

    t_eval = time.time()
    for i, ex in enumerate(iterator, start=1):
        prompt = PROMPT_TMPL.format(text=ex["sentence"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = normalize_pred(gen)

        ref = int(ex["label"])
        pred_int = 1 if pred == "positive" else 0
        preds.append(pred_int)
        refs.append(ref)

        if args.log_every > 0 and (i % args.log_every == 0 or i == eval_n):
            correct = sum(p == r for p, r in zip(preds, refs))
            running_acc = correct / len(preds)
            elapsed = time.time() - t_eval
            speed = len(preds) / max(elapsed, 1e-6)
            log(f"[PROGRESS] {i}/{eval_n} run_acc={running_acc:.4f} speed={speed:.2f} ex/s")

            if not args.disable_tqdm and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"run_acc": f"{running_acc:.4f}", "ex/s": f"{speed:.2f}"})

    result = acc.compute(predictions=preds, references=refs)
    log(f"Model: {model_name}")
    log(f"Eval samples: {eval_n}")
    log(f"Accuracy: {result['accuracy']:.4f}")

if __name__ == "__main__":
    main()