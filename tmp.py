import random
from fractions import Fraction

# --- 24 Game Solver with Clear Logs and Separate Exploration Rates ---

def fmt(frac: Fraction) -> str:
    """Format a Fraction as integer or numerator/denominator."""
    return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"


def generate_pairwise_ops(items):
    """
    Generate all pairwise operations (+, -, *, /) for three items.
    Returns list of (result: Fraction, expr: str, rem: (Fraction, str)).
    """
    candidates = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, expr_a = items[i]
            b, expr_b = items[j]
            rem = items[3 - i - j]
            candidates.extend([
                (a + b, f"({expr_a}+{expr_b})", rem),
                (a - b, f"({expr_a}-{expr_b})", rem),
                (b - a, f"({expr_b}-{expr_a})", rem),
                (a * b, f"({expr_a}*{expr_b})", rem)
            ])
            if b != 0:
                candidates.append((a / b, f"({expr_a}/{expr_b})", rem))
            if a != 0:
                candidates.append((b / a, f"({expr_b}/{expr_a})", rem))
    return candidates


def can_make_24_two(items):
    """
    Check if two items combine to 24 via one operation.
    Returns (True, expr, op_char, val_left, expr_left, val_right, expr_right) if possible.
    """
    (a, expr_a), (b, expr_b) = items
    ops = [
        (a + b, f"({expr_a}+{expr_b})", '+', a, expr_a, b, expr_b),
        (a - b, f"({expr_a}-{expr_b})", '-', a, expr_a, b, expr_b),
        (b - a, f"({expr_b}-{expr_a})", '-', b, expr_b, a, expr_a),
        (a * b, f"({expr_a}*{expr_b})", '*', a, expr_a, b, expr_b)
    ]
    if b != 0:
        ops.append((a / b, f"({expr_a}/{expr_b})", '/', a, expr_a, b, expr_b))
    if a != 0:
        ops.append((b / a, f"({expr_b}/{expr_a})", '/', b, expr_b, a, expr_a))

    for res, expr, op_char, val_l, expr_l, val_r, expr_r in ops:
        if res == Fraction(24, 1):
            return True, expr, op_char, val_l, expr_l, val_r, expr_r
    return False, None, None, None, None, None, None


def check_whether_can_make_24(three):

    for r2, expr2, rem in generate_pairwise_ops(three):
        ok2, final_expr, final_op, val_l, expr_l, val_r, expr_r = \
            can_make_24_two([(r2, expr2), rem])
        if ok2:
            return True, final_expr, final_op, val_l, expr_l, val_r, expr_r
    return False, None, None, None, None, None, None

def exploration_two(two, error_prob):
    
    ok2, expr2, op_char, val_l, expr_l, val_r, expr_r = \
        can_make_24_two(two)

    (r2, expr2), rem = two
    # simulate human error/delay
    made_error = False
    disp_val = r2
    if random.random() < error_prob:
        made_error = True
        disp_val = r2 + Fraction(random.randint(-5, 5), 1)

    thought = []
    rem_val, rem_expr = rem
    thought.append(f"Consider {expr2[1:-1]} = {fmt(disp_val)}")
    if made_error:
        thought.append(f"Wait! {expr2[1:-1]} = {fmt(r2)}")
    if rem_expr == fmt(rem_val):
        thought.append(f"Remaining {rem_expr}")
    else:
        thought.append(f"Remaining {rem_expr[1:-1]} = {fmt(rem_val)}")
        
    return ok2, thought

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
                                '/': final_val_l / final_val_r
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


if __name__ == '__main__':
    thought = solve_24_response([35, 12, 53, 16],
                exploration_three_prob=0.3,
                exploration_two_prob=0.3,
                error_prob=0.01)
    
    print(thought)