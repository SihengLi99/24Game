import os
import re
import json
import argparse

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from datasets import load_from_disk
from build_prompt import is_expr_equal_to_24

# Compute pass@k for multiple k values in one run
PASS_K_VALUES = [1, 2, 4, 8, 16]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--num_samples", type=int, default=max(PASS_K_VALUES),
        help=f"Number of completions per prompt; should be >= max({PASS_K_VALUES})"
    )
    return parser.parse_args()


def load_data(path):
    dataset = load_from_disk(path)["test"]
    
    dataset = dataset.select(range(100))
    
    print(dataset)
    return dataset.to_list()

def inference_transformers(
    dataset,
    model_name_or_path,
    temperature,
    top_k,
    max_tokens,
    num_samples,
    accelerator
):
    device = accelerator.device
    world_size = accelerator.num_processes
    process_index = accelerator.process_index

    # Shard dataset across GPUs/processes
    dataset_shard = dataset[process_index::world_size]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.float16 if accelerator.mixed_precision == 'fp16' else torch.bfloat16
    )
    model.to(device)

    model, tokenizer = accelerator.prepare(model, tokenizer)
    # unwrap DDP wrapper for generation
    gen_model = model.module if hasattr(model, "module") else model

    results_shard = []
    for item in tqdm(dataset_shard, desc="Inference", unit="item"):
        item["prompt_processed"] = tokenizer.apply_chat_template(
            item["prompt"],
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = tokenizer(
            item["prompt_processed"], return_tensors="pt", padding=True
        ).to(device)

        generate_kwargs = {
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "max_new_tokens": max_tokens,
            "num_return_sequences": num_samples,
            "pad_token_id": tokenizer.eos_token_id
        }
        outputs = gen_model.generate(**inputs, **generate_kwargs)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]
        gen_tokens = outputs[:, input_len:]
        texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        results_shard.append({**item, "outputs": texts})

        # Gather Python objects across processes using torch.distributed.all_gather_object
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        # gather lists of dicts from each process
        dist.all_gather_object(gathered, results_shard)
        if accelerator.is_main_process:
            # flatten list of lists
            results = [item for shard in gathered for item in shard]
        else:
            results = None
    else:
        # single process or non-distributed
        results = results_shard
    return results


def compute_metrics(dataset, num_samples):
    total = len(dataset)
    pass_counts = {k: 0 for k in PASS_K_VALUES if k <= num_samples}

    for item in tqdm(dataset, desc="Computing Metrics", unit="item"):
        correctness_list = []
        for out in item["outputs"]:
            match = re.search(r"<answer>(.*?)</answer>", out, re.DOTALL)
            if not match:
                correctness_list.append(0.0)
            else:
                model_expr = match.group(1).strip()
                
                # Compare digits from both the model's expression and the reference expression.
                ref_expr = item["expression"]
                ref_digits_str = re.findall(r"\d+", ref_expr)
                model_digits_str = re.findall(r"\d+", model_expr)

                ref_digits = sorted(int(x) for x in ref_digits_str)
                model_digits = sorted(int(x) for x in model_digits_str)

                if ref_digits != model_digits:
                    correctness_list.append(0.0)
                else:
                    try:
                        if is_expr_equal_to_24(model_expr):
                            correctness_list.append(1.0)
                        else:
                            correctness_list.append(0.0)
                    except Exception:
                        # Handle any exceptions that occur during expression evaluation
                        correctness_list.append(0.0)
                    
        item["correctness_per_sample"] = correctness_list
        for k in list(pass_counts):
            if any(correctness_list[:k]):
                pass_counts[k] += 1
            item[f"pass@{k}"] = int(any(correctness_list[:k]))

    metrics = {"number": total}
    for k, count in pass_counts.items():
        metrics[f"pass@{k}"] = float(count / total)

    return metrics, dataset


def main():
    args = parse_args()
    accelerator = Accelerator()

    os.makedirs(args.output_path, exist_ok=True)

    data = load_data(args.input_path)
    results = inference_transformers(
        data,
        args.model_name_or_path,
        args.temperature,
        args.top_k,
        args.max_tokens,
        args.num_samples,
        accelerator
    )

    # Only main process writes files and computes metrics
    if accelerator.is_main_process:
        metrics, annotated = compute_metrics(results, args.num_samples)

        # Write detailed results
        with open(os.path.join(args.output_path, "inference.jsonl"), "w", encoding="utf-8") as f:
            for item in annotated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Write aggregated metrics
        with open(os.path.join(args.output_path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
