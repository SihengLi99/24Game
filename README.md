# 24 Point Game 🎯

> **Author:** Siheng Li  
> **Date:** 2025‑04‑17  
> A lightweight 24-point solver with O(1)-style inference, trained using SFT, SFT + RL, and pure RL, achieving up to 60 % accuracy.

---

## Table of Contents

1. [Data Preparation](#1-data-preparation-🔍)  
2. [Training](#2-training-🏋️)  
3. [Evaluation](#3-evaluation-📏)  
4. [Results & Analysis](#4-results--analysis-📊)  
5. [Contact & Acknowledgements](#5-contact--acknowledgements)

---

## 1 · Data Preparation 🔍 <a id="1-data-preparation-🔍"></a>

We generate **100 000** valid four‑integer puzzles (values ∈ [1 … 99]) that can yield exactly **24**.

| Step | Description |
|------|-------------|
| **Enumeration** | List every 4‑integer combination in `[1, 99]`. |
| **Filtering** | Keep only tuples with **≥ 1** valid 24‑point expression. |
| **Scripts** | `build_prompt.py`, `build_response.py`, `build_sft_dataset.py` |

### 1.1 Finding Valid Combinations

```python
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
```


```python

    train_data = []
    test_data = []

    # Define a candidate set to ensure that there is no overlap between the training and test sets.
    candidate_set = set()

    train_candidates = parallel_generate_candidates(n_train, value_min, value_max, candidate_set, n_processes, description="train")
    for nums, expr in train_candidates:
        train_data.append(nums)

    test_candidates = parallel_generate_candidates(n_test, value_min, value_max, candidate_set, n_processes, description="test")
    for nums, expr in test_candidates:
        test_data.append(nums)
```


### 1.2 Human‑Like Thought Processes 🧠

```python
def solve_24_response(numbers,
             exploration_three_prob=0.3,
             exploration_two_prob=0.3,
             error_prob=0.1):
    """
    Solve the 24 Game for four numbers with:
      - separate probabilistic exploration at 3‑ and 2‑node stages
      - injected calculation errors
      - clear human-like logs
    """
    items = [(Fraction(n), str(n)) for n in numbers]
    thought = [f"Starting with values: {numbers}"]
    answer_expr = None

    for i in range(4):
        if answer_expr:
            break
        for j in range(i + 1, 4):
            if answer_expr:
                break
            a, expr_a = items[i]
            b, expr_b = items[j]
            rest = [items[k] for k in range(4) if k not in (i, j)]

            ops = [
                (a + b, f"({expr_a}+{expr_b})"),
                (a - b, f"({expr_a}-{expr_b})"),
                (b - a, f"({expr_b}-{expr_a})"),
                (a * b, f"({expr_a}*{expr_b})")
            ]
            if b != 0:
                ops.append((a / b, f"({expr_a}/{expr_b})"))
            if a != 0:
                ops.append((b / a, f"({expr_b}/{expr_a})"))

            for res1, subexpr1 in ops:
                if answer_expr:
                    break
                three = rest + [(res1, subexpr1)]
                new_values = ", ".join([fmt(v) for v, _ in three])
                thought.append(f"Try {subexpr1[1:-1]} = {fmt(res1)}, new values: [{new_values}]")

                final_found, final_expr, final_op, final_val_l, final_expr_l, final_val_r, final_expr_r = check_whether_can_make_24(three)
                
                if final_found is False:
                    if random.random() < exploration_three_prob:
                        for r2, expr2, rem in generate_pairwise_ops(three):
                            if random.random() < exploration_two_prob:   
                                ok2, _thought = exploration_two([(r2, expr2), rem], error_prob)
                                assert ok2 is False
                                thought.extend(_thought)
                                thought.append("-> Cannot reach 24.")
                    else:
                        thought.append("It seems cannot obtain 24.")
                else:
                    for r2, expr2, rem in generate_pairwise_ops(three):
                        ok2, _thought = exploration_two([(r2, expr2), rem], error_prob)
                        if ok2 is True:
                            thought.extend(_thought)
                            final_val = {
                                '+': final_val_l + final_val_r,
                                '-': final_val_l - final_val_r,
                                '*': final_val_l * final_val_r,
                                '/': final_val_l / final_val_r if final_val_r != 0 else None
                            }[final_op]
                            thought.append(
                                f"Compute {final_expr_l}{final_op}{final_expr_r} = "
                                f"{fmt(final_val_l)} {final_op} {fmt(final_val_r)} = {fmt(final_val)}"
                            )
                            answer_expr = final_expr[1:-1]
                            thought.append(f"Found a solution: {answer_expr}")
                            break
                        else:
                            if random.random() < exploration_two_prob:   
                                thought.extend(_thought)
                                thought.append("-> Cannot reach 24.")

    thought = ["<think>"] + thought + ["</think>"] + [f"<answer>{answer_expr}</answer>"]
    
    return "\n".join(thought)
```

<details>
<summary>Example</summary>

```text
<think>
Starting with values: [35, 12, 53, 16]
Try 35+12 = 47, new values: [53, 16, 47]
It seems cannot obtain 24.
Try 35-12 = 23, new values: [53, 16, 23]
It seems cannot obtain 24.
Try 12-35 = -23, new values: [53, 16, -23]
It seems cannot obtain 24.
Try 35*12 = 420, new values: [53, 16, 420]
Consider 53+16 = 69
Remaining 35*12 = 420
-> Cannot reach 24.
Consider 53-16 = 33
Wait! 53-16 = 37
Remaining 35*12 = 420
-> Cannot reach 24.
Consider 16/53 = 16/53
Remaining 35*12 = 420
-> Cannot reach 24.
Consider 53+(35*12) = 473
Remaining 16
-> Cannot reach 24.
Consider 53-(35*12) = -367
Remaining 16
-> Cannot reach 24.
Consider (35*12)-53 = 367
Remaining 16
-> Cannot reach 24.
Consider 53/(35*12) = 53/420
Remaining 16
-> Cannot reach 24.
Consider 16+(35*12) = 436
Remaining 53
-> Cannot reach 24.
Consider 16-(35*12) = -404
Remaining 53
-> Cannot reach 24.
Consider 16/(35*12) = 4/105
Remaining 53
-> Cannot reach 24.
Consider (35*12)/16 = 105/4
Remaining 53
-> Cannot reach 24.
Try 35/12 = 35/12, new values: [53, 16, 35/12]
Consider 53-16 = 37
Remaining 35/12 = 35/12
-> Cannot reach 24.
Consider 53*16 = 848
Remaining 35/12 = 35/12
-> Cannot reach 24.
Consider 16-(35/12) = 157/12
Remaining 53
-> Cannot reach 24.
Consider (35/12)-16 = -157/12
Remaining 53
-> Cannot reach 24.
Consider (35/12)/16 = 35/192
Remaining 53
-> Cannot reach 24.
Try 12/35 = 12/35, new values: [53, 16, 12/35]
It seems cannot obtain 24.
Try 35+53 = 88, new values: [12, 16, 88]
It seems cannot obtain 24.
Try 35-53 = -18, new values: [12, 16, -18]
It seems cannot obtain 24.
Try 53-35 = 18, new values: [12, 16, 18]
Consider 12-16 = -1
Wait! 12-16 = -4
Remaining 53-35 = 18
-> Cannot reach 24.
Consider 12/16 = 3/4
Remaining 53-35 = 18
Compute (53-35)/(12/16) = 18 / 3/4 = 24
Found a solution: (53-35)/(12/16)
</think>
<answer>(53-35)/(12/16)</answer>
```

</details>

---

## 2 · Training 🏋️ <a id="2-training-🏋️"></a>

### 2.1 Supervised Fine‑Tuning (SFT) 📚

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model_args.model_name_or_path,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=get_peft_config(model_args),
    processing_class=tokenizer,
)
```

```bash
# Supports multi-node, multi-GPU training.
sbatch train_sft.slurm
```

### 2.2 Reinforcement Learning (GRPO) 🤖

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model_args.model_name_or_path,
    reward_funcs=reward_funcs,  # discourages “hacks”
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=get_peft_config(model_args),
    processing_class=tokenizer,
)
```

```python
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
```

```bash
# Supports multi-node, multi-GPU training and enables inference using VLLM.
sbatch train_grpo.slurm
```

---

## 3 · Evaluation 📏 <a id="3-evaluation-📏"></a>

We sample **16** completions per prompt and compute **Pass@ K** for  
*K ∈ {1, 2, 4, 8, 16}*.

```python
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
```

```bash
bash evaluate.sh
```

---

## 4 · Results & Analysis 📊 <a id="4-results--analysis-📊"></a>

### 4.1 Main Results

| Model | Pass@1 | Pass@2 | Pass@4 | Pass@8 | Pass@16 |
|-------|-------:|-------:|-------:|-------:|--------:|
| Qwen2.5‑1.5B‑Instruct | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 |
| DeepSeek‑R1‑Distill‑Qwen‑1.5B | 0.04 | 0.08 | 0.12 | 0.21 | 0.33 |
| Qwen2.5‑1.5B + SFT  | 0.40 | 0.66 | 0.80 | 0.88 | 0.91 |
| Qwen2.5‑1.5B + SFT + RL  | **0.60** | **0.84** | **0.91** | **0.92** | **0.94** |
| Qwen2.5‑1.5B + RL only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

> **Notes:**
> 
> - *All inference results are available for local inspection.*
> - *For **Qwen2.5‑1.5B + SFT + RL**, we observed a clear upward trend in the reward curve during RL training. However, due to limited computational resources, a more extensive exploration is left to future work.*
> - *For the **RL-only** approach, we experimented with models up to **Qwen2.5‑14B**. Nonetheless, the reward signal showed no significant improvement over time, indicating limited learning progress.*

### 4.2 Impact of Test‑Time Compute ⏱️

```python
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
```

```bash
bash analyze.sh
```


The following figure illustrates how increasing `max_think_tokens` (i.e., test-time compute) improves the model’s reasoning accuracy across various Pass@k metrics:

![Test-time Compute vs Accuracy](assets/test_time_compute_vs_accuracy.png)

---

## 5 · Contact & Acknowledgements <a id="5-contact--acknowledgements"></a>

| | |
|---|---|
| **Name** | Siheng Li |
| **Email** | sihengli24@gmail.com |
| **GitHub** | <https://github.com/SihengLi99/24Game> |

Feel free to open an issue or reach out—happy solving ! 🎉
