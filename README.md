# Option 1: 24 Point Game 🎯

---

## 📘 Project Overview
**Title:** 24 Point Game  
**Author:** Siheng Li  
**Date:** 2025‑04‑17  

> A simple O1 model for the 24 Point Game, trained via supervised fine‑tuning (SFT), SFT + reinforcement learning (SFT+RL), and pure RL, achieving over **70%** accuracy.

---

## 📝 Experiment Report

### 1. Data Preparation 🔍
In this section, we describe how we generated **100 000** valid 4‑integer puzzles (from 1 to 99) that evaluate exactly to 24.

- **Source of puzzles:**  
  Enumerate all 4‑integer combinations in `[1, 99]` and filter those with at least one valid 24‑point expression.  
- **Generation scripts:**  
  - `build_prompt.py`  
  - `build_response.py`  

#### 1.1 Finding Valid Number Combinations
```python
from fractions import Fraction
from itertools import permutations
from your_module import ops_expr

def find_expression_for_24(nums):
    """
    Search for any expression on `nums` that evaluates to exactly 24.
    Returns the expression string if found; otherwise, returns None.
    """
    target = Fraction(24, 1)
    for perm in permutations(nums):
        A = (Fraction(perm[0]), str(perm[0]))
        B = (Fraction(perm[1]), str(perm[1]))
        C = (Fraction(perm[2]), str(perm[2]))
        D = (Fraction(perm[3]), str(perm[3]))

        # Try all 5 parenthesization patterns...
        # Pattern 1: ((a op b) op c) op d
        for r1_val, r1_expr in ops_expr(A[0], A[1], B[0], B[1]):
            for r2_val, r2_expr in ops_expr(r1_val, r1_expr, C[0], C[1]):
                for r3_val, r3_expr in ops_expr(r2_val, r2_expr, D[0], D[1]):
                    if r3_val == target:
                        return r3_expr
        # ... (patterns 2–5 omitted for brevity)
    return None



⸻

1.2 Generating Human‑Like Thought Processes 🧠

We inject probabilistic exploration, occasional errors, and clear “thinking” logs to emulate a human solver.

from fractions import Fraction
import random
from your_module import fmt, generate_pairwise_ops, check_whether_can_make_24, exploration_two

def solve_24_response(numbers,
                      exploration_three_prob=0.3,
                      exploration_two_prob=0.3,
                      error_prob=0.1):
    """
    Solve the 24 Game for four numbers with:
      - probabilistic exploration at 3-node and 2-node stages
      - injected calculation errors
      - clear human-like thought logs
    """
    items = [(Fraction(n), str(n)) for n in numbers]
    thought = [f"Starting with values: {numbers}"]
    answer_expr = None

    # Combine pairs and explore recursively...
    # (core logic omitted for brevity)

    thought = ["<think>"] + thought + ["</think>", f"<answer>{answer_expr}</answer>"]
    return "\n".join(thought)



⸻

Example Thought Process (with “Wait”) ⏳

<think>
Starting with values: [35, 12, 53, 16]
Try 35+12 = 47, new values: [53, 16, 47]
It seems cannot obtain 24.
Try 35-12 = 23, new values: [53, 16, 23]
It seems cannot obtain 24.
Try 35*12 = 420, new values: [53, 16, 420]
Wait.
Consider 53-16 = 37
Remaining 35*12 = 420
-> Cannot reach 24.
...
Compute (53-35)/(12/16) = 18 ÷ 3/4 = 24
Found a solution: (53-35)/(12/16)
</think>
<answer>(53-35)/(12/16)</answer>



⸻

🏋️ 2. Training

2.1 Supervised Fine‑Tuning (SFT) 📚

trainer = SFTTrainer(
    model=model_args.model_name_or_path,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=get_peft_config(model_args),
)

sbatch train_sft.slurm



⸻

2.2 Reinforcement Learning (RL) 🤖

Using GRPO with carefully designed reward functions to discourage hacking:

trainer = GRPOTrainer(
    model=model_args.model_name_or_path,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=get_peft_config(model_args),
    processing_class=tokenizer,
)

sbatch train_grpo.slurm



⸻

📏 3. Evaluation

We measure Pass@K by sampling 16 outputs per prompt and checking correctness at K = 1,2,4,8,16.

def compute_metrics(dataset, num_samples=16):
    # Compute pass@k for each sample...
    return metrics, dataset_with_flags

bash evaluate.sh



⸻

📊 4. Results & Analysis

Main Results 🎯

Model	Pass@1	Pass@2	Pass@4	Pass@8	Pass@16
Qwen2.5-1.5B-Instruct
DeepSeek-R1-Distill-Qwen-1.5B	
Qwen2.5-1.5B + SFT					
Qwen2.5-1.5B + SFT + RL					
Qwen2.5-1.5B + RL only					

Fill in the table above with your experimental numbers.

⸻

Impact of Test‑Time Compute ⏱️

Describe how increasing the number of samples at inference (Pass@K) affects final accuracy and latency.

⸻

📬 Contact & Acknowledgements

Feel free to open an issue or reach out:
	•	Name: Siheng Li
	•	Email: sihengli24@gmail.com
	•	GitHub: github.com/your-username/24-point-game

Happy solving! 🎉

