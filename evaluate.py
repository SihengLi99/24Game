import os
import re
import json
import argparse
import numpy as np
from enum import Enum

import datasets
import transformers
from vllm import LLM, SamplingParams

from build_response import evaluate_final_answer

def parse_args():
    
    args = argparse.ArgumentParser()
    args.add_argument("--model_name_or_path", type=str)
    args.add_argument("--temperature", type=float)
    args.add_argument("--top_k", type=int, default=1)
    args.add_argument("--max_tokens", type=int)
    args.add_argument("--tensor_parallel_size", type=int, default=1)
    
    args.add_argument("--prompt", type=str, default=None)
    
    args.add_argument("--stage", type=str)
    
    args.add_argument("--input_path", type=str)
    args.add_argument("--output_path", type=str)
    
    return args.parse_args()

def load_data(path):
    
    dataset = datasets.load_from_disk(path)["test"]
    print(dataset)
    return dataset.to_list()

def inference_vllm(dataset, model_name_or_path, temperature, top_k, max_tokens, tensor_parallel_size, system_prompt=None, add_response=False):
    
    sampling_params = SamplingParams(temperature=temperature, top_k=top_k, max_tokens=max_tokens)
    model = LLM(model_name_or_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    
    for item in dataset:        
        if system_prompt is None:
            messages = [
                {"role": "user", "content": item["prompt"]}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]}
            ]
        if add_response is True:
            messages.append({"role": "assistant", "content": item["response"]})
                    
        item["prompt_processed"] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    
    outputs = model.generate([item["prompt_processed"] for item in dataset], sampling_params)
    results = []
    for item, output in zip(dataset, outputs):
        assert item["prompt_processed"] == output.prompt
        output = output.outputs[0].text
        
        results.append({**item, "output": output})
            
    return results
def compute_metrics(dataset):
    
    for item in dataset:
        item["correctness"], _ = evaluate_final_answer(item["output"])
    
    metrics = {
        "number": len(dataset),
        "correctness": np.mean([item["correctness"] for item in dataset]),
    }
    return metrics, dataset
    
def main(args):
    
    os.makedirs(args.output_path, exist_ok=True)
    
    dataset = load_data(args.input_path)  
    results = inference_vllm(dataset, args.model_name_or_path, args.temperature, args.top_k, args.max_tokens, args.tensor_parallel_size)
    metrics, dataset = compute_metrics(results)
    
    with open(os.path.join(args.output_path, "inference.jsonl"), "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")  
             
    json.dump(metrics, open(os.path.join(args.output_path, "metrics.json"), "w", encoding="utf-8"), indent=4, ensure_ascii=False) 
            
if __name__ == "__main__":
    
    args = parse_args()