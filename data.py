"""
data.py — dataset loading + prompt formatting.

Deliberately torch-free and tokenizer-free so you can build/debug it on your
local box with no GPU stack installed. Tokenization (which needs the
tokenizer) lives in train.py instead, keeping the CPU/GPU split clean.

Each example in the jsonl is:
    {"db_id": "store", "question": "...", "gold_sql": "SELECT ..."}
"""
import json
import random
from typing import List, Dict, Tuple


def load_examples(path: str) -> List[Dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def split(examples: List[Dict], eval_fraction: float, seed: int
          ) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]  # train, eval


def load_schema(db_id: str, schema_dir: str = "sample_data/schemas") -> str:
    """Return a textual schema for the prompt. Falls back to a stub if the
    schema file is absent so the harness still runs end-to-end."""
    import os
    path = os.path.join(schema_dir, f"{db_id}.sql")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return f"-- schema for {db_id} not found"


# ---- prompt construction (pure string logic — the part you tune for quality)
PROMPT_TEMPLATE = """You are an expert at translating questions into SQLite SQL.

Database schema:
{schema}

Question: {question}

Write a single SQL query that answers the question. Return only the SQL.
SQL:"""


def format_prompt(question: str, schema: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema, question=question.strip())


def build_supervised_text(example: Dict, schema: str) -> Tuple[str, str]:
    """Returns (prompt, completion). train.py masks the prompt in the loss so
    the model is only trained on producing the SQL completion."""
    prompt = format_prompt(example["question"], schema)
    completion = " " + example["gold_sql"].strip()
    return prompt, completion


# ---- preference data (objective = dpo) ------------------------------------
def load_preference_examples(path: str) -> List[Dict]:
    """Each line: {db_id, question, chosen, rejected}. Same loader shape as
    SFT; only the fields differ."""
    return load_examples(path)


def build_preference_text(example: Dict, schema: str):
    """Returns (prompt, chosen, rejected) — the triple DPO consumes."""
    prompt = format_prompt(example["question"], schema)
    chosen = " " + example["chosen"].strip()
    rejected = " " + example["rejected"].strip()
    return prompt, chosen, rejected


def make_preference_pairs(examples: List[Dict],
                          candidates_by_index: Dict[int, List[str]],
                          db_dir: str) -> List[Dict]:
    """Turn SFT examples + candidate SQLs into DPO pairs using EXECUTION truth
    — no human labels. Two kinds of preference signal, ranked LEXICOGRAPHICALLY:

      1. correctness first  (execution_reward: 1.0 match / 0.3 runs-wrong / 0.0 error)
      2. efficiency as a TIEBREAK among equally-correct candidates
         (sql_cost: cheaper plan preferred)

    So `chosen` is the most-correct candidate and, among the correct ones, the
    cheapest; `rejected` is the worst. We keep a pair when there's a real signal:
    a correctness gap, OR both candidates correct but different cost (the
    "optimized vs resource-heavy SQL" preference). Correctness ALWAYS dominates —
    we never prefer a cheap wrong query over an expensive right one, and we don't
    manufacture cost preferences among non-correct candidates.

    CPU-only (just runs/compiles SQL), so you build and verify the preference
    dataset locally before any GPU time. `candidates_by_index` maps example index
    -> list of candidate SQL strings (e.g. sampled from the SFT model).
    """
    import os
    from eval_harness import execution_reward, sql_cost

    pairs = []
    for i, ex in enumerate(examples):
        cands = candidates_by_index.get(i, [])
        if len(cands) < 2:
            continue
        db_path = os.path.join(db_dir, f"{ex['db_id']}.db")
        scored = [{"sql": c,
                   "reward": execution_reward(db_path, c, ex["gold_sql"]),
                   "cost": sql_cost(db_path, c)}
                  for c in cands]
        # higher reward first; among equal reward, LOWER cost (cheaper) first.
        scored.sort(key=lambda s: (s["reward"], -s["cost"]), reverse=True)
        best, worst = scored[0], scored[-1]

        correctness_gap = best["reward"] > worst["reward"]
        efficiency_gap = (best["reward"] == worst["reward"] == 1.0
                          and best["cost"] < worst["cost"])
        if correctness_gap or efficiency_gap:
            pairs.append({
                "db_id": ex["db_id"], "question": ex["question"],
                "chosen": best["sql"], "rejected": worst["sql"],
            })
    return pairs


if __name__ == "__main__":
    # Local smoke test — no torch needed.
    ex = load_examples("sample_data/examples.jsonl")
    print(f"Loaded {len(ex)} examples")
    tr, ev = split(ex, 0.2, 42)
    print(f"train={len(tr)} eval={len(ev)}")
    schema = load_schema(ex[0]["db_id"])
    p, c = build_supervised_text(ex[0], schema)
    print("\n--- sample prompt ---\n" + p)
    print("\n--- completion ---\n" + c)
