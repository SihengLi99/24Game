import random
from itertools import permutations
from fractions import Fraction
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import concurrent.futures
import os

def ops_expr(a_val, a_expr, b_val, b_expr):
    """
    Use Fraction arithmetic to combine a and b in different ways.
    Returns a list of (value, expression) pairs.
    Skips division if the denominator is 0.
    """
    results = []
    results.append((a_val + b_val, f"({a_expr}+{b_expr})"))
    results.append((a_val - b_val, f"({a_expr}-{b_expr})"))
    results.append((b_val - a_val, f"({b_expr}-{a_expr})"))
    results.append((a_val * b_val, f"({a_expr}*{b_expr})"))
    if b_val != 0:
        results.append((a_val / b_val, f"({a_expr}/{b_expr})"))
    if a_val != 0:
        results.append((b_val / a_val, f"({b_expr}/{a_expr})"))
    return results

def find_expression_for_24(nums):
    """
    Use Fraction arithmetic to search for a valid expression that equals 24 exactly.
    Returns the expression as a string if found; otherwise, returns None.
    """
    target = Fraction(24, 1)
    for perm in permutations(nums):
        A = (Fraction(perm[0]), str(perm[0]))
        B = (Fraction(perm[1]), str(perm[1]))
        C = (Fraction(perm[2]), str(perm[2]))
        D = (Fraction(perm[3]), str(perm[3]))

        # 1. ((a op b) op c) op d
        for r1_val, r1_expr in ops_expr(A[0], A[1], B[0], B[1]):
            for r2_val, r2_expr in ops_expr(r1_val, r1_expr, C[0], C[1]):
                for r3_val, r3_expr in ops_expr(r2_val, r2_expr, D[0], D[1]):
                    if r3_val == target:
                        return r3_expr

        # 2. (a op (b op c)) op d
        for r1_val, r1_expr in ops_expr(B[0], B[1], C[0], C[1]):
            for r2_val, r2_expr in ops_expr(A[0], A[1], r1_val, r1_expr):
                for r3_val, r3_expr in ops_expr(r2_val, r2_expr, D[0], D[1]):
                    if r3_val == target:
                        return r3_expr

        # 3. (a op b) op (c op d)
        for r1_val, r1_expr in ops_expr(A[0], A[1], B[0], B[1]):
            for r2_val, r2_expr in ops_expr(C[0], C[1], D[0], D[1]):
                for r3_val, r3_expr in ops_expr(r1_val, r1_expr, r2_val, r2_expr):
                    if r3_val == target:
                        return r3_expr

        # 4. a op ((b op c) op d)
        for r1_val, r1_expr in ops_expr(B[0], B[1], C[0], C[1]):
            for r2_val, r2_expr in ops_expr(r1_val, r1_expr, D[0], D[1]):
                for r3_val, r3_expr in ops_expr(A[0], A[1], r2_val, r2_expr):
                    if r3_val == target:
                        return r3_expr

        # 5. a op (b op (c op d))
        for r1_val, r1_expr in ops_expr(C[0], C[1], D[0], D[1]):
            for r2_val, r2_expr in ops_expr(B[0], B[1], r1_val, r1_expr):
                for r3_val, r3_expr in ops_expr(A[0], A[1], r2_val, r2_expr):
                    if r3_val == target:
                        return r3_expr
    return None

def is_expr_equal_to_24(expression, tolerance=1e-6):
    """
    Evaluate the given expression using eval with Fraction arithmetic,
    and determine whether the result is equal to 24 within the specified tolerance.
    Returns True if the absolute difference is below tolerance, otherwise False.
    """
    try:
        # Evaluate the expression with Fraction in the namespace for precise arithmetic.
        result_val = eval(expression, {"__builtins__": None}, {"Fraction": Fraction})
    except ZeroDivisionError:
        return False
    except Exception:
        return False
    return abs(float(result_val) - 24) < tolerance

def worker_generate_candidate(value_min, value_max):
    """
    Continuously generate 4 random integers until finding one with a valid 24 game solution.
    Returns (nums, sorted_nums, expression) when a valid candidate is found.
    """
    while True:
        nums = [random.randint(value_min, value_max) for _ in range(4)]
        expr = find_expression_for_24(nums)
        if expr is not None:
            return (nums, tuple(sorted(nums)), expr)

def parallel_generate_candidates(n, value_min, value_max, existing_set, n_processes=None, description="samples"):
    """
    Generate n candidate samples in parallel.
    The candidate combination (sorted) is ensured to be unique (not in existing_set).
    Parameter n_processes specifies the number of processes to use; if None, it will auto-detect.
    Returns a list of (nums, expression) pairs and adds the candidates to existing_set.
    """
    results = []
    num_workers = n_processes if n_processes is not None else (os.cpu_count() or 4)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Initially submit some tasks.
        futures = [executor.submit(worker_generate_candidate, value_min, value_max) for _ in range(num_workers * 4)]
        with tqdm(total=n, desc=f"Generating {description} samples") as pbar:
            while len(results) < n:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    candidate = fut.result()  # (nums, sorted_nums, expr)
                    if candidate[1] not in existing_set:
                        results.append(candidate)
                        existing_set.add(candidate[1])
                        pbar.update(1)
                    futures.remove(fut)
                    futures.append(executor.submit(worker_generate_candidate, value_min, value_max))
    return [(nums, expr) for nums, sorted_nums, expr in results]

def create_24_data_without_overlap(n_train, n_test, value_min=1, value_max=99, seed=None, n_processes=None):
    """
    Generate training and test data in parallel ensuring no overlapping number combinations.
    Each candidate is a list of 4 integers.
    Parameter n_processes can specify the number of parallel processes.
    """
    if seed is not None:
        random.seed(seed)

    train_data = []
    test_data = []
    candidate_set = set()

    train_candidates = parallel_generate_candidates(n_train, value_min, value_max, candidate_set, n_processes, description="train")
    for nums, expr in train_candidates:
        train_data.append(nums)

    test_candidates = parallel_generate_candidates(n_test, value_min, value_max, candidate_set, n_processes, description="test")
    for nums, expr in test_candidates:
        test_data.append(nums)

    return train_data, test_data

def build_prompt(numbers, prompt_type='direct'):
    """
    Build a prompt text for the given 4 numbers.
    The prompt instructs how to combine them to compute 24.
    """
    nums_str = ", ".join(str(n) for n in numbers)
    base = (
        f"Below are four integers: {nums_str}. "
        "You must use each integer exactly once, combining them with +, -, *, /, and parentheses to alter operation order, "
        "so that the final result equals 24. "
        "Important: begin your response with <think>...</think> to present your reasoning, then enclose your final expression in <answer>...</answer>. "
        "For example:\n"
        "<think>...</think>\n"
        "<answer>(4*(6+2)-8)</answer>\n"
        "End your response with <|endoftext|>.\n"
    )
    if prompt_type == 'deliberation':
        extra = (
            "You may think carefully and proceed with deliberation. In your <think>...</think> section, "
            "you can use different cognitive behaviors:\n"
            "- Reflection: revisit your reasoning steps.\n"
            "- Verification: check intermediate results.\n"
            "- Revision: correct any mistakes.\n"
            "- Backtracking: revert to a previous step if needed.\n"
            "Make sure to present your chain of thought in detail."
        )
        prompt = base + extra
    else:
        prompt = base
    return prompt

def create_prompted_data(nums_list, prompt_type='direct'):
    """
    For a list of valid 24 game quadruples (nums_list),
    generate a prompt and the corresponding 24 game expression.
    Uses is_expr_equal_to_24 to verify the expression evaluates to 24 within tolerance.
    """
    records = []
    for nums in nums_list:
        prompt_text = build_prompt(nums, prompt_type)
        expression = find_expression_for_24(nums)
        if not is_expr_equal_to_24(expression, tolerance=1e-6):
            raise AssertionError(f"Expression does not evaluate to 24 within tolerance: {expression}")

        records.append({
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            "expression": expression
        })
    return records

def build_hf_dataset(n_train, n_test, prompt_type='direct', dataset_name='24_point_game', n_processes=None):
    """
    Generate non-overlapping training and test data,
    build a Hugging Face dataset, and save it to disk.
    Parameter n_processes specifies the number of processes for data generation.
    After building the dataset, an extra verification step ensures each expression evaluates to 24.
    """
    train_nums, test_nums = create_24_data_without_overlap(n_train, n_test, seed=42, n_processes=n_processes)
    train_records = create_prompted_data(train_nums, prompt_type=prompt_type)
    test_records = create_prompted_data(test_nums, prompt_type=prompt_type)

    # Extra verification: Check that every expression evaluates to 24
    for record in train_records + test_records:
        expr = record["expression"]
        if not is_expr_equal_to_24(expr, tolerance=1e-6):
            raise AssertionError(f"Verification failed, expression not equal to 24: {expr}")

    train_dataset = Dataset.from_list(train_records)
    test_dataset = Dataset.from_list(test_records)
    ds = DatasetDict({"train": train_dataset, "test": test_dataset})
    ds.save_to_disk(f"./data/{dataset_name}")
    print(f"Hugging Face Dataset saved to: {dataset_name}")
    return ds

if __name__ == "__main__":
    # Specify the desired number of processes, for example, 8
    desired_processes = 64

    n_train = 1000000
    prompt_type = "direct"
    ds = build_hf_dataset(
        n_train=n_train,  
        n_test=1000,     
        prompt_type=prompt_type,
        dataset_name=f"24_game_{n_train}_{prompt_type}",
        n_processes=desired_processes
    )

    print(ds)
    print("\nSample training record:\n", ds["train"][0])
    print("\nSample test record:\n", ds["test"][0])