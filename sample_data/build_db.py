"""
sample_data/build_db.py — build the local sample data so the CPU spine runs.

This is the no-GPU foundation: it materializes everything the eval harness and
data loader read, from a SINGLE source of truth (the ROWS + EXAMPLES below):

    sample_data/dbs/store.db       the SQLite DB execution accuracy runs against
    sample_data/schemas/store.sql  textual schema injected into the prompt
    sample_data/examples.jsonl     SFT data  {db_id, question, gold_sql}
    sample_data/preferences.jsonl  DPO data  {db_id, question, chosen, rejected}

Idempotent: rerun any time to rebuild from scratch. Pure stdlib (sqlite3 + json),
so it runs on the local 16GB no-GPU box. Paths are resolved relative to THIS
file, so it works regardless of the directory you invoke it from.

    python sample_data/build_db.py
"""
import json
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(HERE, "dbs")
SCHEMA_DIR = os.path.join(HERE, "schemas")
EXAMPLES_PATH = os.path.join(HERE, "examples.jsonl")
PREFERENCES_PATH = os.path.join(HERE, "preferences.jsonl")

DB_ID = "store"

# ---- the schema (also written verbatim into the prompt) --------------------
SCHEMA_SQL = """\
CREATE TABLE customers (
    id   INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount      REAL,
    status      TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);"""

# ---- the data (one source of truth for the DB AND the answers) -------------
CUSTOMERS = [
    (1, "Asha",   "Pune"),
    (2, "Ravi",   "Mumbai"),
    (3, "Priya",  "Pune"),
    (4, "Karan",  "Delhi"),
    (5, "Meera",  "Pune"),
    (6, "Vikram", "Mumbai"),
]

ORDERS = [
    # id, customer_id, amount, status
    (1, 1, 250.0, "shipped"),
    (2, 2, 100.0, "pending"),
    (3, 3, 400.0, "shipped"),
    (4, 1, 150.0, "cancelled"),
    (5, 4, 300.0, "shipped"),
    (6, 5,  50.0, "pending"),
]

# ---- SFT examples: {db_id, question, gold_sql} -----------------------------
# Spread across COUNT / SUM / AVG / MIN / MAX / WHERE / DISTINCT / ORDER BY so
# the tiny eval set actually exercises a range of SQL, not one template.
EXAMPLES = [
    ("How many customers are there?",
     "SELECT COUNT(*) FROM customers"),
    ("List the names of all customers.",
     "SELECT name FROM customers"),
    ("What is the total amount of all orders?",
     "SELECT SUM(amount) FROM orders"),
    ("How many customers are from Pune?",
     "SELECT COUNT(*) FROM customers WHERE city = 'Pune'"),
    ("What is the average order amount?",
     "SELECT AVG(amount) FROM orders"),
    ("How many orders have status 'shipped'?",
     "SELECT COUNT(*) FROM orders WHERE status = 'shipped'"),
    ("List all distinct cities.",
     "SELECT DISTINCT city FROM customers"),
    ("What are the names of customers from Mumbai?",
     "SELECT name FROM customers WHERE city = 'Mumbai'"),
    ("What is the largest order amount?",
     "SELECT MAX(amount) FROM orders"),
    ("What is the smallest order amount?",
     "SELECT MIN(amount) FROM orders"),
    ("How many orders are there?",
     "SELECT COUNT(*) FROM orders"),
    ("List order amounts greater than 200.",
     "SELECT amount FROM orders WHERE amount > 200"),
    ("List customer names in alphabetical order.",
     "SELECT name FROM customers ORDER BY name"),
    ("What is the total order amount for shipped orders?",
     "SELECT SUM(amount) FROM orders WHERE status = 'shipped'"),
]

# ---- DPO preference pairs: {db_id, question, chosen, rejected} -------------
# chosen = executes-and-matches; rejected = errors or wrong result. In the real
# pipeline these come from data.make_preference_pairs scoring sampled candidates
# with execution_reward; hand-written here so the DPO path has data on day one.
PREFERENCES = [
    ("How many customers are there?",
     "SELECT COUNT(*) FROM customers",
     "SELECT customers FROM COUNT"),
    ("List the names of all customers.",
     "SELECT name FROM customers",
     "SELECT name FROM custmers"),
    ("What is the total amount of all orders?",
     "SELECT SUM(amount) FROM orders",
     "SELECT amount FROM orders"),
    ("How many customers are from Pune?",
     "SELECT COUNT(*) FROM customers WHERE city = 'Pune'",
     "SELECT COUNT(*) FROM customers"),
    ("What is the average order amount?",
     "SELECT AVG(amount) FROM orders",
     "SELECT SUM(amount) FROM orders"),
    ("How many orders have status 'shipped'?",
     "SELECT COUNT(*) FROM orders WHERE status = 'shipped'",
     "SELECT COUNT(*) FROM orders WHERE status = shipped"),
]


def build_database():
    os.makedirs(DB_DIR, exist_ok=True)
    db_path = os.path.join(DB_DIR, f"{DB_ID}.db")
    if os.path.exists(db_path):
        os.remove(db_path)  # rebuild from scratch — keep it idempotent

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    cur.executemany("INSERT INTO customers (id, name, city) VALUES (?, ?, ?)",
                    CUSTOMERS)
    cur.executemany(
        "INSERT INTO orders (id, customer_id, amount, status) VALUES (?, ?, ?, ?)",
        ORDERS)
    conn.commit()
    conn.close()
    return db_path


def write_schema():
    os.makedirs(SCHEMA_DIR, exist_ok=True)
    path = os.path.join(SCHEMA_DIR, f"{DB_ID}.sql")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SCHEMA_SQL + "\n")
    return path


def write_examples():
    with open(EXAMPLES_PATH, "w", encoding="utf-8") as f:
        for question, gold_sql in EXAMPLES:
            f.write(json.dumps(
                {"db_id": DB_ID, "question": question, "gold_sql": gold_sql}) + "\n")
    return EXAMPLES_PATH


def write_preferences():
    with open(PREFERENCES_PATH, "w", encoding="utf-8") as f:
        for question, chosen, rejected in PREFERENCES:
            f.write(json.dumps(
                {"db_id": DB_ID, "question": question,
                 "chosen": chosen, "rejected": rejected}) + "\n")
    return PREFERENCES_PATH


def _sanity_check(db_path):
    """Run each gold query against the freshly built DB so a broken example is
    caught HERE, locally, before it ever reaches the model or the eval harness."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    bad = []
    for question, gold_sql in EXAMPLES:
        try:
            cur.execute(gold_sql).fetchall()
        except Exception as e:  # noqa: BLE001
            bad.append((question, gold_sql, str(e)))
    conn.close()
    return bad


def main():
    db_path = build_database()
    schema_path = write_schema()
    ex_path = write_examples()
    pref_path = write_preferences()

    bad = _sanity_check(db_path)
    if bad:
        print("WARNING: some gold queries do not execute against the DB:")
        for q, sql, err in bad:
            print(f"  [{q}] {sql}  ->  {err}")
    else:
        print(f"All {len(EXAMPLES)} gold queries execute cleanly.")

    print("Built:")
    print(f"  {db_path}")
    print(f"  {schema_path}")
    print(f"  {ex_path}   ({len(EXAMPLES)} SFT examples)")
    print(f"  {pref_path}   ({len(PREFERENCES)} preference pairs)")


if __name__ == "__main__":
    main()
