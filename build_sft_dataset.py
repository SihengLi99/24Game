import random
import math
import re
from fractions import Fraction
from datasets import load_from_disk, Dataset, DatasetDict
import os
from tqdm import tqdm  # progress bar

from build_prompt import is_expr_equal_to_24
from build_response import solve_24_response

def extract_numbers_from_expression(expression):
    """
    Extracts numeric values from a given arithmetic expression using regex.
    Expects exactly four numbers; otherwise, raises an assertion error.
    """
    numbers = re.findall(r'\d+(?:\.\d+)?', expression)
    numbers = [int(num) for num in numbers]
    assert len(numbers) == 4, f"Expected exactly 4 numbers in expression, got {len(numbers)} in: {expression}"
    return numbers

def evaluate_final_answer(response_text, tolerance=1e-6):
    """
    Extracts the final answer from the response (within <answer>...</answer> tags) and
    evaluates it using is_expr_equal_to_24.
    Raises an assertion error if the result does not evaluate to 24.
    """
    match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    assert match is not None, "Response is missing <answer> tags."
    answer_expr = match.group(1).strip()
    assert is_expr_equal_to_24(answer_expr, tolerance=tolerance), \
           f"Expression does not evaluate to 24: {answer_expr}"
    return True, 24

def create_sft_examples(hf_dataset, exploration_three_prob=0.2, exploration_two_prob=0.2, error_prob=0.01):
    """
    Processes each record in the HuggingFace dataset to generate an SFT training example.
    Each record must contain a non-empty 'prompt' list and an 'expression' field.
    The function extracts four numbers from the expression, generates a response using the 24 Game solver,
    asserts that the computed final answer equals 24 (using is_expr_equal_to_24 for evaluation),
    and returns a list of examples.
    A progress bar is displayed during processing.
    """
    sft_examples = []
    for record in tqdm(hf_dataset, desc="Constructing SFT examples"):
        # Assert that required keys exist.
        assert "prompt" in record, "Record missing 'prompt' field."
        assert isinstance(record["prompt"], list) and len(record["prompt"]) > 0, "Prompt must be a non-empty list."
        prompt_record = record["prompt"][0]
        assert "content" in prompt_record, "Prompt record missing 'content'."
        prompt_text = prompt_record["content"]

        assert "expression" in record, "Record missing 'expression' field."
        expression = record["expression"]

        # Extract exactly four numbers from the expression.
        numbers = extract_numbers_from_expression(expression)
        
        # Generate response using the 24 Game solver.
        response_text = solve_24_response(numbers, 
                                          exploration_three_prob=exploration_three_prob,
                                          exploration_two_prob=exploration_two_prob,
                                          error_prob=error_prob)
        
        # Assert that the final computed answer is indeed 24 using is_expr_equal_to_24.
        evaluate_final_answer(response_text)
        
        # Compose the SFT example with messages.
        sft_example = {
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text}
            ],
            "expression": expression,
        }
        sft_examples.append(sft_example)
    return sft_examples

def analyze_statistics(dataset_path, model_name):
    """
    Analyzes token statistics for the SFT dataset by loading it from disk, applying the chat template using
    the tokenizer's method, and then tokenizing the result.
    A progress bar is shown while processing the examples.
    
    Returns a dictionary containing average token count, maximum tokens, and median token count.
    """
    from transformers import AutoTokenizer
    import statistics
    from datasets import load_from_disk
    from tqdm import tqdm

    # Load the dataset from disk.
    sft_dataset = load_from_disk(dataset_path)["train"]
    print(sft_dataset[0])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_counts = []
    for example in tqdm(sft_dataset, desc="Analyzing token statistics"):
        text = tokenizer.apply_chat_template(example["messages"],
                                             add_generation_prompt=False,
                                             tokenize=False)
        tokens = tokenizer.tokenize(text)
        token_counts.append(len(tokens))
    avg = sum(token_counts) / len(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    median = statistics.median(token_counts) if token_counts else 0
    stats = {
        "average_tokens": avg,
        "max_tokens": max_tokens,
        "median_tokens": median
    }
    print(f"Token Statistics: Average tokens: {avg:.2f}, Max tokens: {max_tokens}, Median tokens: {median:.2f}")
    return stats

def main(dataset_name, exploration_three_prob=0.2, exploration_two_prob=0.2, error_prob=0.01):
    # Set the directory for the HuggingFace dataset (saved using datasets.save_to_disk).
    input_dataset_path = f"./data/{dataset_name}"  # Adjust this directory as needed.
    assert os.path.exists(input_dataset_path), f"Input dataset directory '{input_dataset_path}' does not exist."
    
    # Load the HuggingFace dataset.
    hf_dataset = load_from_disk(input_dataset_path)["train"]
    
    # Create SFT examples from the loaded dataset.
    sft_examples = create_sft_examples(hf_dataset, 
                                       exploration_three_prob=exploration_three_prob,
                                       exploration_two_prob=exploration_two_prob,
                                       error_prob=error_prob)
    print(f"Generated {len(sft_examples)} SFT examples.")
    
    # Convert examples to a HuggingFace Dataset.
    sft_dataset = DatasetDict({"train": Dataset.from_list(sft_examples)})
    
    # Save the SFT dataset in HuggingFace format.
    output_dir = f"./data/{dataset_name}_sft_{exploration_three_prob}_{exploration_two_prob}_{error_prob}"
    sft_dataset.save_to_disk(output_dir)
    print(f"SFT dataset saved in HuggingFace format to '{output_dir}'")

if __name__ == "__main__":
    
    # Uncomment the following line to run the main function.
    main(dataset_name="24_game_100000_direct", exploration_three_prob=0.2, exploration_two_prob=0.2, error_prob=0.01)
    
    # Uncomment and adjust the following line to analyze token statistics.
    analyze_statistics("./data/24_game_100000_direct_sft_0.2_0.2_0.01", "/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-3B")