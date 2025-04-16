#!/usr/bin/env python
# coding: utf-8

"""
Two reward functions for a 24-style game:
1) format_reward_func: Checks that the completion includes a <think>...</think> block and an <answer>...</answer> block.
   Additionally, penalizes responses where the <think> block is lazy (e.g. exactly '...' or empty).
2) correctness_reward_func: Checks if the <answer> expression uses the same digits as the reference expression,
   and if it evaluates to 24.
"""

import re

def format_reward_func(completions, **kwargs):
    """
    Checks each completion for:
      1) A <think>...</think> block.
      2) An <answer>...</answer> block.
      3) A non-lazy <think> block (i.e. not simply "..." or empty).
      
    Returns a list of float rewards:
      1.0 if all conditions are met,
      0.0 if either block is missing or if the <think> block is lazy.
    """
    # Removed the requirement for an end tag like <|endoftext|>.
    pattern = re.compile(r'(?s)^<think>.*?</think>.*?<answer>.*?</answer>\s*$')
    rewards = []
    for completion in completions:
        # Ensure each completion is a list with one dictionary.
        assert len(completion) == 1, f"Unexpected completion structure: {completion}"
        output = completion[0]["content"]
        
        # First, check if the overall format is correct.
        if not pattern.match(output):
            rewards.append(0.0)
            continue
        
        # Extract the content inside the <think> block.
        think_match = re.search(r'(?s)<think>(.*?)</think>', output)
        if think_match:
            think_content = think_match.group(1).strip()
            # If the think block is exactly "..." or empty, penalize it.
            if think_content == "..." or think_content == "":
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def correctness_reward_func(completions, expression=None, **kwargs):
    """
    Checks if the model's <answer> expression is correct by:
      1) ensuring it uses the same digits as the reference expression (from expression[i]),
      2) and evaluating to 24.
    
    Args:
      completions: A list of completions, where each completion is a list containing one dictionary with a "content" key.
      expression: A list of reference expressions (strings) that yield 24 and use a specific set of digits.
                  expression[i] corresponds to completions[i].
    
    Returns:
      A list of float rewards (1.0 if both digit match and evaluation to 24 is successful, otherwise 0.0).
    """
    if expression is None:
        return [0.0] * len(completions)

    answer_pattern = re.compile(r'(?s)<answer>(.*?)</answer>')
    epsilon = 1e-6

    rewards = []
    for i, completion in enumerate(completions):
        assert len(completion) == 1, f"Unexpected completion structure: {completion}"
        output = completion[0]["content"]
        
        match = answer_pattern.search(output)
        if not match:
            rewards.append(0.0)
            continue
        
        model_expr = match.group(1).strip()
        # Compare digits from both the model's expression and the reference expression.
        ref_expr = expression[i]
        ref_digits_str = re.findall(r"\d+", ref_expr)
        model_digits_str = re.findall(r"\d+", model_expr)

        ref_digits = sorted(int(x) for x in ref_digits_str)
        model_digits = sorted(int(x) for x in model_digits_str)

        if ref_digits != model_digits:
            rewards.append(0.0)
            continue

        try:
            val = eval(model_expr)
            if abs(val - 24) < epsilon:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    return rewards


REWARD_FUNCS_REGISTRY = {
    "format": format_reward_func,
    "correctness": correctness_reward_func,
}

if __name__ == "__main__":
    """
    A few sample completions to verify the reward functions.
    We supply a matching 'expression' list with reference expressions.
    """

    # 1) Correct format & correct expression => format=1.0, correctness=1.0
    completion_1 = [[{
        "content": "<think>Some detailed reasoning about how I combine these numbers step by step.</think><answer>(4*(6+2)-8)</answer>"
    }]]
    reference_expr_1 = ["(4*(6+2)-8)"]  # yields 24, digits: 4,6,2,8

    # 2) Format ok, but expression evaluates to 25 => format=1.0, correctness=0.0
    completion_2 = [[{
        "content": "<think>Some detailed reasoning here.</think><answer>((4*6)+1)</answer>"
    }]]
    reference_expr_2 = ["(4*(6+2)-8)"]  # reference expects 24 but model expression yields 25

    # 3) Lazy response in <think> block => format=0.0, correctness (if digits match) can still be 1.0
    completion_3 = [[{
        "content": "<think>...</think><answer>(8/(3-(8/3)))</answer>"
    }]]
    reference_expr_3 = ["(8/(3-(8/3)))"]  # digits: 8,3,8,3 => 24

    # 4) Correct format, expression yields 24 but has mismatched digits => correctness=0.0
    completion_4 = [[{
        "content": "<think>Some detailed reasoning here.</think><answer>(8/(3-(8/3)))</answer>"
    }]]
    reference_expr_4 = ["(4*(6+2)-8)"]  # reference digits: 4,6,2,8 vs model digits: 8,3,8,3

    completions = completion_1 + completion_2 + completion_3 + completion_4
    expressions = reference_expr_1 + reference_expr_2 + reference_expr_3 + reference_expr_4

    fmt_scores = format_reward_func(completions)
    corr_scores = correctness_reward_func(completions, expression=expressions)

    print("Format Rewards:", fmt_scores)
    print("Correctness Rewards:", corr_scores)

    """
    Explanation:
    1) => format=1.0, correctness=1.0
    2) => format=1.0, correctness=0.0 (because model expression evaluates to 25)
    3) => format=0.0, correctness=1.0 (lazy <think> block)
    4) => format=1.0, correctness=0.0 (digit mismatch vs. reference)
    """