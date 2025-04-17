import os
import json
import argparse
import numpy as np

import datasets
import transformers
from vllm import LLM, SamplingParams

from build_sft_dataset import evaluate_final_answer

# Compute pass@k for multiple k values in one run (e.g., 1,2,4,8,16)
PASS_K_VALUES = [1, 2, 4, 8, 16]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--num_samples", type=int, default=max(PASS_K_VALUES),
        help=f"Number of completions per prompt; should be >= max({PASS_K_VALUES})"
    )
    return parser.parse_args()


def load_data(path):
    dataset = datasets.load_from_disk(path)["test"]
    def process(item):
        item["prompt"] = item["prompt"][0]["content"]
        return item
    dataset = dataset.map(process, num_proc=8)
    print(dataset)
    return dataset.to_list()


def inference_vllm(
    dataset,
    model_name_or_path,
    temperature,
    top_k,
    max_tokens,
    tensor_parallel_size,
    num_samples=1
):
    # Request num_samples completions per prompt for Pass@k
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
        n=num_samples
    )
    model = LLM(
        model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    for item in dataset:
        messages = [{"role": "user", "content": item["prompt"]}]
        item["prompt_processed"] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

    outputs = model.generate(
        [item["prompt_processed"] for item in dataset],
        sampling_params
    )

    results = []
    for item, output in zip(dataset, outputs):
        assert item["prompt_processed"] == output.prompt
        # Collect all completions
        texts = [out.text for out in output.outputs]
        results.append({**item, "outputs": texts})

    return results


def compute_metrics(dataset, num_samples):
    total = len(dataset)
    # Initialize counters for each k
    pass_counts = {k: 0 for k in PASS_K_VALUES if k <= num_samples}

    for item in dataset:
        # Evaluate correctness for each sample
        correctness_list = [evaluate_final_answer(out)[0] for out in item["outputs"]]
        item["correctness_per_sample"] = correctness_list
        # For each requested k, mark pass if any of first k are correct
        for k in list(pass_counts):
            if any(correctness_list[:k]):
                pass_counts[k] += 1
            # record per-item pass@k
            item[f"pass@{k}"] = int(any(correctness_list[:k]))

    # Build metrics summary
    metrics = {"number": total}
    for k, count in pass_counts.items():
        metrics[f"pass@{k}"] = float(count / total)

    return metrics, dataset


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    dataset = load_data(args.input_path)
    results = inference_vllm(
        dataset,
        args.model_name_or_path,
        args.temperature,
        args.top_k,
        args.max_tokens,
        args.tensor_parallel_size,
        num_samples=args.num_samples
    )
    metrics, dataset = compute_metrics(results, args.num_samples)

    # Write detailed results
    with open(os.path.join(args.output_path, "inference.jsonl"), "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write aggregated metrics
    with open(os.path.join(args.output_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
