import os
import argparse
import yaml
import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import OFTConfig, get_peft_model

PROMPT_TMPL = (
    "You are a sentiment classifier.\n"
    "Task: Given the sentence, output only one word: positive or negative.\n"
    "Sentence: {text}\n"
    "Answer:"
)

def build_example(sentence: str, label: int):
    ans = "positive" if int(label) == 1 else "negative"
    prompt = PROMPT_TMPL.format(text=sentence)
    full = prompt + " " + ans
    return prompt, full

def preprocess(tokenizer, max_length: int):
    def _fn(batch):
        prompts = []
        full_texts = []
        for s, y in zip(batch["sentence"], batch["label"]):
            p, f = build_example(s, y)
            prompts.append(p)
            full_texts.append(f)

        tok_full = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tok_prompt = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        labels = []
        for ids_full, ids_prompt in zip(tok_full["input_ids"], tok_prompt["input_ids"]):
            # labels 长度必须 == input_ids 长度
            lab = [-100] * len(ids_full)
            # prompt 部分 mask 掉，只让 answer 部分产生 loss
            start = min(len(ids_prompt), len(ids_full))
            for j in range(start, len(ids_full)):
                lab[j] = ids_full[j]
            labels.append(lab)

        tok_full["labels"] = labels

        # attention_mask 必须存在（tokenizer 默认会给）
        # 这里确保字段齐全
        if "attention_mask" not in tok_full:
            tok_full["attention_mask"] = [[1] * len(x) for x in tok_full["input_ids"]]

        return tok_full
    return _fn

@dataclass
class CausalLMDataCollator:
    """Pad input_ids/attention_mask/labels manually to avoid tokenizer.pad incompatibilities."""
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def print_trainable_params(model):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    ratio = trainable / total if total > 0 else 0.0
    print(f"[PARAM] trainable={trainable:,} total={total:,} ratio={ratio:.6f}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/oft_sst2.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))

    model_name = cfg["model_name"]
    max_length = int(cfg.get("max_length", 256))

    # 国内镜像（你也可以在 shell 里 export HF_ENDPOINT）
    os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", ""))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if cfg.get("fp16", True) else None,
        device_map="cuda",
        trust_remote_code=True,
    )

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # 训练时建议关 cache

    # OFT 配置：只用 oft_block_size，避免与 r 冲突
    oft_cfg = OFTConfig(
        oft_block_size=int(cfg.get("oft_block_size", 32)),
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    model = get_peft_model(model, oft_cfg)
    print_trainable_params(model)

    ds = load_dataset("glue", "sst2")
    train_n = int(cfg.get("train_samples", 20000))
    eval_n = int(cfg.get("eval_samples", 800))

    train_ds = ds["train"].shuffle(seed=int(cfg.get("seed", 42))).select(range(min(train_n, len(ds["train"]))))
    eval_ds = ds["validation"].shuffle(seed=int(cfg.get("seed", 42))).select(range(min(eval_n, len(ds["validation"]))))

    train_ds = train_ds.map(preprocess(tokenizer, max_length), batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess(tokenizer, max_length), batched=True, remove_columns=eval_ds.column_names)

    # 自定义 collator：彻底解决 labels padding 兼容问题
    data_collator = CausalLMDataCollator(pad_token_id=tokenizer.pad_token_id)

    # ---- TrainingArguments compatibility (transformers versions differ) ----
    common_kwargs = dict(
        output_dir=cfg.get("output_dir", "outputs/checkpoints/oft_sst2"),
        logging_dir=cfg.get("log_dir", "outputs/logs/oft_sst2"),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("lr", 2e-4)),
        num_train_epochs=float(cfg.get("epochs", 1)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        fp16=bool(cfg.get("fp16", True)),
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=0,  # 先稳定跑通；之后可改回 2
    )

    try:
        args_tr = TrainingArguments(
            **common_kwargs,
            eval_strategy="steps",
            eval_steps=200,
        )
    except TypeError:
        args_tr = TrainingArguments(
            **common_kwargs,
            evaluation_strategy="steps",
            eval_steps=200,
        )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args_tr.output_dir)
    tokenizer.save_pretrained(args_tr.output_dir)

if __name__ == "__main__":
    main()