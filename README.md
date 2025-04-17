# 24 Point Game 🎯

> **Author:** Siheng Li  
> **Date:** 2025‑04‑17  
> A simple *O(1)* 24‑point solver, trained with **SFT**, **SFT + RL**, and **pure RL**, surpassing **70 %** accuracy.

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
| **Scripts** | `build_prompt.py`, `build_response.py` |

### 1.1 Finding Valid Combinations

```python
from fractions import Fraction
from itertools import permutations
from your_module import ops_expr

def find_expression_for_24(nums):
    """Return a string expression that equals 24, or None."""
    target = Fraction(24, 1)

    for perm in permutations(nums):
        A = (Fraction(perm[0]), str(perm[0]))
        B = (Fraction(perm[1]), str(perm[1]))
        C = (Fraction(perm[2]), str(perm[2]))
        D = (Fraction(perm[3]), str(perm[3]))

        # Pattern 1: ((a op b) op c) op d
        for v1, e1 in ops_expr(A[0], A[1], B[0], B[1]):
            for v2, e2 in ops_expr(v1, e1, C[0], C[1]):
                for v3, e3 in ops_expr(v2, e2, D[0], D[1]):
                    if v3 == target:
                        return e3
        # … Patterns 2–5 omitted for brevity
    return None
```

### 1.2 Human‑Like Thought Processes 🧠

```python
from fractions import Fraction
import random
from your_module import (
    fmt,
    generate_pairwise_ops,
    check_whether_can_make_24,
    exploration_two,
)

def solve_24_response(
    numbers,
    exploration_three_prob=0.30,
    exploration_two_prob=0.30,
    error_prob=0.10,
):
    """Return <think>…</think><answer>…</answer> with logs & optional errors."""
    items = [(Fraction(n), str(n)) for n in numbers]
    thought = [f"Starting with values: {numbers}"]
    answer_expr = None

    # … recursive solver logic …

    thought = ["<think>", *thought, "</think>", f"<answer>{answer_expr}</answer>"]
    return "\n".join(thought)
```

<details>
<summary>Example (with “Wait” logs)</summary>

```text
<think>
Starting with values: [35, 12, 53, 16]
Try 35+12 = 47 → [53, 16, 47] – cannot obtain 24
Try 35-12 = 23 → [53, 16, 23] – cannot obtain 24
Try 35*12 = 420 → [53, 16, 420]
Wait.
Consider 53-16 = 37, remaining 420 → cannot reach 24
…
Compute (53-35)/(12/16) = 18 ÷ 3/4 = 24 ✓
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

```bash
sbatch train_grpo.slurm
```

---

## 3 · Evaluation 📏 <a id="3-evaluation-📏"></a>

We sample **16** completions per prompt and compute **Pass@ K** for  
*K ∈ {1, 2, 4, 8, 16}*.

```python
def compute_metrics(dataset, num_samples=16):
    # calculate Pass@K and attach correctness flags
    return metrics, dataset_with_flags
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
| Qwen2.5‑1.5B + SFT + RL  | | | | | |
| Qwen2.5‑1.5B + RL only | | | | | |

### 4.2 Impact of Test‑Time Compute ⏱️

Discuss how increasing *K* improves accuracy at the cost of latency and compute. Include any Pareto observations or scaling laws you observed.

---

## 5 · Contact & Acknowledgements <a id="5-contact--acknowledgements"></a>

| | |
|---|---|
| **Name** | Siheng Li |
| **Email** | sihengli24@gmail.com |
| **GitHub** | <https://github.com/your-username/24-point-game> |

Feel free to open an issue or reach out—happy solving ! 🎉
