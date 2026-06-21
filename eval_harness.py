"""
eval_harness.py — execution accuracy for text-to-SQL.

This is the honest metric: run the predicted SQL and the gold SQL against the
actual database and compare RESULT SETS, not strings. Exact-match string
comparison lies (different-but-equivalent SQL scores as wrong; wrong SQL that
happens to match a token scores as right).

Pure-CPU, stdlib-only (sqlite3). Build and debug this locally before any GPU
work — every method in the plan is scored through this one function, so it
must be trustworthy first.
"""
import os
import sqlite3
from typing import List, Dict, Optional


def execute_sql(db_path: str, sql: str, timeout: float = 5.0):
    """Returns (result_set_or_None, error_or_None). Result set is an
    order-insensitive frozenset of row tuples."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return frozenset(rows), None
    except Exception as e:  # noqa: BLE001 — any SQL/DB error means "failed"
        return None, str(e)


def execution_match(db_path: str, pred_sql: str, gold_sql: str) -> Optional[bool]:
    """True/False if comparable; None if the GOLD itself fails (bad example —
    excluded from accuracy so a broken gold doesn't penalize the model)."""
    gold_rows, gold_err = execute_sql(db_path, gold_sql)
    if gold_err is not None:
        return None
    pred_rows, pred_err = execute_sql(db_path, pred_sql)
    if pred_err is not None:
        return False
    return pred_rows == gold_rows


def evaluate(predictions: List[str],
             examples: List[Dict],
             db_dir: str) -> Dict:
    """Score a batch. predictions[i] corresponds to examples[i]."""
    assert len(predictions) == len(examples)
    correct = 0
    counted = 0
    pred_errors = 0
    skipped_bad_gold = 0
    failures = []

    for pred, ex in zip(predictions, examples):
        db_path = os.path.join(db_dir, f"{ex['db_id']}.db")
        result = execution_match(db_path, pred, ex["gold_sql"])
        if result is None:
            skipped_bad_gold += 1
            continue
        counted += 1
        if result:
            correct += 1
        else:
            _, err = execute_sql(db_path, pred)
            if err is not None:
                pred_errors += 1
            failures.append({"question": ex.get("question"),
                             "pred": pred, "gold": ex["gold_sql"],
                             "error": err})

    acc = correct / counted if counted else 0.0
    return {
        "execution_accuracy": round(acc, 4),
        "correct": correct,
        "counted": counted,
        "pred_errors": pred_errors,          # predictions that didn't even run
        "skipped_bad_gold": skipped_bad_gold,
        "failures": failures,
    }


def execution_reward(db_path: str, pred_sql: str, gold_sql: str) -> float:
    """Programmatic, execution-grounded reward for preference tuning.

    The elegant bit for text-to-SQL: the SAME engine that evaluates a model
    also supplies the training signal for DPO/PPO, so you need no human
    preference labels and no separate learned reward model.

        1.0  -> runs AND result set matches gold
        0.3  -> runs but wrong result (valid SQL, semantically off)
        0.0  -> fails to execute (or gold itself is broken)

    DPO: label candidates by this reward, pair a high one (chosen) with a low
    one (rejected). PPO: use it directly as the scalar reward.
    """
    gold_rows, gold_err = execute_sql(db_path, gold_sql)
    if gold_err is not None:
        return 0.0
    pred_rows, pred_err = execute_sql(db_path, pred_sql)
    if pred_err is not None:
        return 0.0
    return 1.0 if pred_rows == gold_rows else 0.3


def sql_cost(db_path: str, sql: str, timeout: float = 5.0) -> int:
    """A cheap, deterministic proxy for how much WORK a query does: the number of
    SQLite VDBE bytecode operations in its compiled plan (via EXPLAIN).

    Why bytecode-op count and not wall-clock time: on the tiny sample DB, timing
    is pure noise, but the *plan* still reflects complexity (a needless cross-join
    or subquery compiles to more ops). Independent of data size, fully
    reproducible. Returns a large sentinel for SQL that won't even compile, so an
    uncompilable query is treated as infinitely expensive.

    This does NOT touch execution accuracy (the honest eval stays result-set
    correctness only). It exists solely to enrich preference-pair construction:
    among EQUALLY-CORRECT candidates, prefer the cheaper one. See
    data.make_preference_pairs.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        n_ops = len(conn.execute("EXPLAIN " + sql).fetchall())
        conn.close()
        return n_ops
    except Exception:  # noqa: BLE001 — uncompilable SQL -> infinitely expensive
        return 10 ** 9


def clean_sql(raw: str) -> str:
    """Strip the model's wrapping (code fences, trailing prose) down to the
    SQL. Generation is messy; normalize before executing."""
    text = raw.strip()
    if "```" in text:
        # take the content of the first fenced block
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("sql"):
                p = p[3:].strip()
            if p:
                text = p
                break
    # cut at first semicolon if the model rambled afterward
    if ";" in text:
        text = text.split(";")[0] + ";"
    return text.strip()


if __name__ == "__main__":
    # Local smoke test against the sample DB — no model, no GPU.
    # Verifies the metric logic itself: a correct query, an equivalent-but-
    # reordered query (should still match), and a wrong query.
    db_dir = "sample_data/dbs"
    examples = [
        {"db_id": "store", "question": "How many customers?",
         "gold_sql": "SELECT COUNT(*) FROM customers"},
        {"db_id": "store", "question": "All customer names",
         "gold_sql": "SELECT name FROM customers"},
        {"db_id": "store", "question": "Total revenue",
         "gold_sql": "SELECT SUM(amount) FROM orders"},
    ]
    predictions = [
        "SELECT COUNT(*) FROM customers",                 # exact correct
        "SELECT name FROM customers ORDER BY id DESC",    # reordered -> set match
        "SELECT COUNT(*) FROM orders",                    # wrong
    ]
    report = evaluate(predictions, examples, db_dir)
    print("Execution accuracy:", report["execution_accuracy"])
    print("Correct:", report["correct"], "/", report["counted"])
    print("Prediction errors:", report["pred_errors"])
    for f in report["failures"]:
        print("  FAIL:", f["pred"], "| err:", f["error"])
