import random
import math
import re
from fractions import Fraction
from datasets import load_from_disk, Dataset, DatasetDict
import os
from tqdm import tqdm  # progress bar

from build_prompt import is_expr_equal_to_24

# --- Formatting Functions ---
def format_num(num):
    """
    Returns a string representation for a Fraction num.
    - If num is an integer (denom==1), returns the integer as a string.
    - Otherwise, returns a string in the form "numerator/denominator".
    """
    if isinstance(num, Fraction):
        if num.denominator == 1:
            return str(num.numerator)
        else:
            return f"{num.numerator}/{num.denominator}"
    else:
        # Fallback for non-Fraction values.
        s = f"{num:.4f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

def format_state(items):
    """
    Constructs a state string for the current list of (value, expression) pairs.
    If the expression is a bare number it is left as is; otherwise it shows the inner expression and its fraction value.
    """
    state_str = ""
    for v, e in items:
        if state_str:
            state_str += ","
        if "(" not in e and ")" not in e:
            state_str += f"{e}"
        else:
            # Remove the outermost parentheses and display its computed Fraction.
            state_str += f"{e[1:-1]}={format_num(v)}"
    return state_str

def format_value(items):
    """
    Constructs a string for the current list of (value, expression) pairs,
    using only the value part. The resulting string is a comma-separated list
    of the formatted values.
    """
    values_str = ""
    for v, _ in items:
        if values_str:
            values_str += ","
        values_str += format_num(v)
    return values_str

# --- 24 Game Solver Code with Fraction Arithmetic ---
def dfs_24_items(items, logs, used_error=False, error_prob=0.25, memo=None):
    """
    Uses depth-first search (DFS) with memoization on a list of items.
    Each item is a tuple (Fraction value, expression string).
    The solver combines items using +, -, *, / to obtain 24 exactly.
    
    A logging error may be injected with probability error_prob and immediately corrected in the logs.
    """
    if memo is None:
        memo = {}
        
    # Build a unique key for the state using sorted string representations of the Fraction values.
    key = tuple(sorted(format_num(val) for val, _ in items))
    if key in memo and len(key) > 2:
        key_str = "(" + ",".join(format_num(val) for val, _ in items) + ")"
        logs.append(f"Memo: {key_str} has been computed before.")
        return False, None

    if len(items) == 1:
        num, expr = items[0]
        if num == Fraction(24, 1):
            logs.append(f"\nFinal: {expr} = {format_num(num)} (24)")
            return True, expr
        else:
            logs.append(f"\n{expr} = {format_num(num)} ≠ 24. Backtracking.\n")
            memo[key] = False
            return False, None

    # Try combining every two items.
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, expr_a = items[i]
            b, expr_b = items[j]
            # Generate candidate operations, using one ordering for commutative operations.
            candidates = []
            candidates.append((a + b, f"({expr_a}+{expr_b})", "+", a, b))
            candidates.append((a - b, f"({expr_a}-{expr_b})", "-", a, b))
            candidates.append((b - a, f"({expr_b}-{expr_a})", "-", b, a))
            candidates.append((a * b, f"({expr_a}*{expr_b})", "*", a, b))
            if b != Fraction(0, 1):
                candidates.append((a / b, f"({expr_a}/{expr_b})", "/", a, b))
            if a != Fraction(0, 1):
                candidates.append((b / a, f"({expr_b}/{expr_a})", "/", b, a))
    
            if len(items) == 2:
                # For the last two items, check whether any candidate equals 24.
                for result, expr, op, val1, val2 in candidates:
                    if result == Fraction(24, 1):
                        logs.append(f"Find: {format_num(val1)} {op} {format_num(val2)} = {format_num(result)}")
                        logs.append(f"Final: {expr}={format_num(result)}")
                        return True, expr
                memo[key] = False
                return False, None
            if len(items) == 3:
                c, expr_c = [items[k] for k in range(len(items)) if k not in (i, j)][0]
                logs.append(f"Trying {a},{b}, keeping {c}")
                
            for correct_val, op_str, op_symbol, val1, val2 in candidates:
                made_error = False
                displayed_val = correct_val
                # Optionally inject an error into the log.
                if (not used_error) and (random.random() < error_prob):
                    made_error = True
                    if random.choice([True, False]):
                        # Multiply by a random factor (converted to Fraction) for display only.
                        factor = Fraction(int(random.uniform(50, 150)), 100)  # factor between 0.5 and 1.5
                        displayed_val = correct_val * factor
                    else:
                        offset = Fraction(int(random.uniform(-15, 15)), 1)
                        displayed_val = correct_val + offset

                logs.append(f"{format_num(val1)} {op_symbol} {format_num(val2)} = {format_num(displayed_val)}")
                if made_error:
                    logs.append(f"Wait! {format_num(val1)} {op_symbol} {format_num(val2)} = {format_num(correct_val)}")
                
                # Build the new state by removing items i and j and adding the new computed item.
                new_item = (correct_val, op_str)
                new_items = [items[k] for k in range(len(items)) if k not in (i, j)] + [new_item]
                # # Normalize the state by sorting on the Fraction value.
                # new_items.sort(key=lambda x: x[0])
                
                if len(new_items) == 1:
                    num, expr = new_items[0]
                    if num == Fraction(24, 1):
                        logs.append(f"Final: {expr} = {format_num(num)} (24)")
                        return True, expr
                    else:
                        continue
                else:
                    if len(new_items) > 2:
                        state_str = format_state(new_items)
                        logs.append(f"State: {state_str}")
                    found, final_expr = dfs_24_items(new_items, logs,
                                                     used_error=(used_error or made_error),
                                                     error_prob=error_prob,
                                                     memo=memo)
                    if found:
                        return True, final_expr
                    
                    revert_str = format_state(items)
                    if len(new_items) > 2:
                        logs.append(f"Reverting: {revert_str}")
                    
    memo[key] = False
    return False, None

def solve_24_response(numbers, error_prob=0.25):
    """
    Solves the 24 Game for a given set of four integers using DFS with Fraction arithmetic.
    Returns a chain-of-thought (wrapped in <think>...</think>) and the final answer (wrapped in <answer>...</answer>).
    """
    logs = []
    # Create the initial state: represent each number as a Fraction.
    items = [(Fraction(n), format_num(Fraction(n))) for n in numbers]
    
    logs.append("State: " + format_state(items))
    
    found, final_expr = dfs_24_items(items, logs, used_error=False, error_prob=error_prob, memo=None)
    
    if not found:
        logs.append("Tried all combinations; no solution found.")
        final_thought = "\n".join(logs)
        return f"<think>\n{final_thought}\n</think>\n<answer>No valid expression found.</answer>"
    
    final_thought = "\n".join(logs)
    response = f"<think>\n{final_thought}\n</think>\n<answer>{final_expr}</answer>"
    return response

# --- End of 24 Game Solver Code ---

def extract_numbers_from_expression(expression):
    """
    Extracts numeric values from a given arithmetic expression using regex.
    Expects exactly four numbers; otherwise, raises an assertion error.
    """
    numbers = re.findall(r'\d+(?:\.\d+)?', expression)
    numbers = [float(num) for num in numbers]
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

def create_sft_examples(hf_dataset, error_prob=0.25):
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
        response_text = solve_24_response(numbers, error_prob=error_prob)
        
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

def main(dataset_name, error_prob):
    # Set the directory for the HuggingFace dataset (saved using datasets.save_to_disk).
    input_dataset_path = f"./data/{dataset_name}"  # Adjust this directory as needed.
    assert os.path.exists(input_dataset_path), f"Input dataset directory '{input_dataset_path}' does not exist."
    
    # Load the HuggingFace dataset.
    hf_dataset = load_from_disk(input_dataset_path)["train"]
    
    # Create SFT examples from the loaded dataset.
    sft_examples = create_sft_examples(hf_dataset, error_prob=error_prob)
    print(f"Generated {len(sft_examples)} SFT examples.")
    
    # Convert examples to a HuggingFace Dataset.
    sft_dataset = DatasetDict({"train": Dataset.from_list(sft_examples)})
    
    # Save the SFT dataset in HuggingFace format.
    output_dir = f"./data/{dataset_name}_sft_{error_prob}"
    sft_dataset.save_to_disk(output_dir)
    print(f"SFT dataset saved in HuggingFace format to '{output_dir}'")

if __name__ == "__main__":
    # Example input: four numbers between 0 and 99.
    example_numbers = [35, 12, 53, 16]
    response_text = solve_24_response(example_numbers, error_prob=0.01)
    print(response_text)
    
    # Uncomment the following line to run the main function.
    # main(dataset_name="24_game_200000_direct", error_prob=0.01)
    
    # Uncomment and adjust the following line to analyze token statistics.
    # analyze_statistics("./data/24_game_100000_direct_sft_0.01", "/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-3B")