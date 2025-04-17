#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pass_at_k_two_pass.py
=====================
Evaluate a 24‑Point‑Game model with multi‑k Pass@k using a fixed two‑stage
generation strategy:

Stage‑1: generate at most `max_think_tokens` tokens;
Stage‑2: for any sample whose <think> block is still open, force‑close
         it with "</think>\n<answer>" and generate up to 64 extra tokens.

Only two forward passes are required per prompt, regardless of
`num_samples`, and no custom StoppingCriteria are needed.
"""

import os
import re
import json
import argparse
import torch
import torch.distributed as dist
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_from_disk
from build_prompt import is_expr_equal_to_24

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #
PASS_K_VALUES = [1, 2, 4, 8, 16]          # Pass@k values to report
ANSWER_MAX_TOKENS = 64                   # cap for Stage‑2 continuation


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True, type=str)
    p.add_argument("--temperature", required=True, type=float)
    p.add_argument("--top_k", default=1, type=int)
    p.add_argument("--max_tokens", required=True, type=int,
                   help="maximum tokens for Stage‑1 (<think> part)")
    p.add_argument("--input_path", required=True, type=str)
    p.add_argument("--output_path", required=True, type=str)
    p.add_argument("--num_samples", default=max(PASS_K_VALUES), type=int,
                   help="completions per prompt; must be ≥ max(PASS_K_VALUES)")
    p.add_argument("--max_think_tokens", required=True, type=int,
                   help="token budget for generated <think> block")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data loader
# --------------------------------------------------------------------------- #
def load_data(path: str):
    """Load first 100 test samples as list of dicts."""
    ds = load_from_disk(path)["test"].select(range(100))
    return ds.to_list()


# --------------------------------------------------------------------------- #
# Two‑stage generator
# --------------------------------------------------------------------------- #
def generate_two_pass(model,
                      tokenizer,
                      prompt_txt: str,
                      args,
                      device):
    """
    Produce `args.num_samples` completions for one prompt using two passes.
    """
    # ---- Stage‑1 : batch generation for <think> ----
    prompt_ids = tokenizer(prompt_txt,
                           return_tensors="pt").to(device).input_ids
    stage1 = model.generate(
        prompt_ids,
        num_return_sequences=args.num_samples,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_think_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    outputs = [""] * args.num_samples
    cont_prompts, cont_indices = [], []

    for i, seq in enumerate(stage1):
        gen_part = tokenizer.decode(seq[prompt_ids.shape[1]:],
                                    skip_special_tokens=True)
        if "</think>" in gen_part:
            outputs[i] = gen_part                      # finished in Stage‑1
        else:
            forced_tail = "</think>\n<answer>"
            cont_prompts.append(prompt_txt + gen_part + forced_tail)
            cont_indices.append(i)
            outputs[i] = gen_part + forced_tail        # placeholder

    # ---- Stage‑2 : continue only where needed ----
    if cont_prompts:
        cont_inputs = tokenizer(cont_prompts,
                                return_tensors="pt",
                                padding=True).to(device)
        cont_batch = model.generate(
            **cont_inputs,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=ANSWER_MAX_TOKENS,
            pad_token_id=tokenizer.eos_token_id
        )
        base_len = cont_inputs["input_ids"].shape[1]
        for j, seq in enumerate(cont_batch):
            gen_rest = tokenizer.decode(seq[base_len:],
                                        skip_special_tokens=True)
            outputs[cont_indices[j]] += gen_rest

    return outputs


# --------------------------------------------------------------------------- #
# Distributed inference
# --------------------------------------------------------------------------- #
def inference(dataset, args, accelerator: Accelerator):
    """
    Run inference on a shard (rank::world) and gather results.
    """
    device = accelerator.device
    rank = accelerator.process_index
    world = accelerator.num_processes
    data_shard = dataset[rank::world]

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16"
        else torch.bfloat16
    ).to(device)
    mdl, tok = accelerator.prepare(mdl, tok)
    base = mdl.module if hasattr(mdl, "module") else mdl

    shard_out = []
    for sample in tqdm(data_shard, desc=f"Inference‑rank{rank}", unit="item"):
        sys_prompt = tok.apply_chat_template(sample["prompt"],
                                             add_generation_prompt=True,
                                             tokenize=False)
        shard_out.append({
            **sample,
            "outputs": generate_two_pass(base, tok, sys_prompt, args, device)
        })

    # gather across ranks
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * world
        dist.all_gather_object(gathered, shard_out)
        return ([x for sh in gathered for x in sh]
                if accelerator.is_main_process else None)
    return shard_out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(preds, num_samples: int):
    total = len(preds)
    pass_cnt = {k: 0 for k in PASS_K_VALUES if k <= num_samples}

    for item in tqdm(preds, desc="Scoring", unit="item"):
        flags = []
        for txt in item["outputs"]:
            m = re.search(r"<answer>(.*?)</answer>", txt, re.S)
            if not m:
                flags.append(0.0)
                continue
            expr = m.group(1).strip()
            ref_digits = sorted(map(int, re.findall(r"\d+", item["expression"])))
            gen_digits = sorted(map(int, re.findall(r"\d+", expr)))
            if ref_digits != gen_digits:
                flags.append(0.0)
                continue
            try:
                flags.append(1.0 if is_expr_equal_to_24(expr) else 0.0)
            except Exception:
                flags.append(0.0)

        item["correctness_per_sample"] = flags
        for k in pass_cnt:
            if any(flags[:k]):
                pass_cnt[k] += 1
            item[f"pass@{k}"] = int(any(flags[:k]))

    metrics = {"number": total}
    metrics.update({f"pass@{k}": pass_cnt[k] / total for k in pass_cnt})
    return metrics, preds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    os.makedirs(args.output_path, exist_ok=True)

    data = load_data(args.input_path)
    preds = inference(data, args, accelerator)

    if accelerator.is_main_process:
        metrics, annotated = compute_metrics(preds, args.num_samples)

        with open(os.path.join(args.output_path, "inference.jsonl"),
                  "w", encoding="utf-8") as fout:
            for row in annotated:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(os.path.join(args.output_path, "metrics.json"),
                  "w", encoding="utf-8") as fout:
            json.dump(metrics, fout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()