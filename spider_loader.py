"""
spider_loader.py — wire the Spider text-to-SQL benchmark into the lab's format.

Spider (Yu et al. 2018) ships ~10k questions over ~200 multi-table SQLite DBs, with
a database-disjoint train/dev split. This converts an unzipped Spider directory into
exactly what the harness already consumes — nothing else changes:

    <out>/train.jsonl          {db_id, question, gold_sql}   (from train_spider.json)
    <out>/dev.jsonl            {db_id, question, gold_sql}   (from dev.json — the eval set)
    <out>/dbs/<db_id>.db       one SQLite DB per db_id used
    <out>/schemas/<db_id>.sql  CREATE TABLE text for the prompt

CPU-only (json + sqlite3 + shutil) — build/verify the dataset locally, no GPU. Each
gold query is EXECUTED while loading; any that error are dropped, so the eval set holds
only runnable gold (no silent skips later).

Getting Spider:
  * Kaggle: add a public "spider" dataset as Input; it mounts read-only under
    /kaggle/input/<slug> — pass that path as --spider_dir, and --out_dir spider_data
    (which lands in the writable working dir).
  * Local: download + unzip the Spider release; pass the folder as --spider_dir. It must
    contain train_spider.json, dev.json, and database/<db_id>/<db_id>.sqlite.

Usage:
    python spider_loader.py --spider_dir /path/to/spider --out_dir spider_data \
        --max_train 500 --max_eval 200
"""
import argparse
import json
import os
import shutil
import sqlite3


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _schema_text(db_path):
    """CREATE TABLE statements, matching the sample_data/schemas/*.sql style."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    conn.close()
    return "\n\n".join(r[0].strip() for r in rows) + "\n"


def _gold_runs(db_path, sql):
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute(sql).fetchall()
        conn.close()
        return True
    except Exception:  # noqa: BLE001 — any SQL/DB error means "drop this example"
        return False


def convert_split(records, spider_dir, out_dir, max_n, max_schema_chars, db_status):
    """Convert one Spider split into lab examples, materializing each db_id's DB +
    schema once. `db_status` is shared across splits so DBs aren't rebuilt."""
    dbs_dir = os.path.join(out_dir, "dbs")
    sch_dir = os.path.join(out_dir, "schemas")
    os.makedirs(dbs_dir, exist_ok=True)
    os.makedirs(sch_dir, exist_ok=True)

    kept, dropped_gold = [], 0
    for ex in records:
        if max_n and len(kept) >= max_n:
            break
        db_id = ex["db_id"]
        gold = (ex.get("query") or "").strip()
        question = ex["question"]
        out_db = os.path.join(dbs_dir, f"{db_id}.db")

        # Materialize the DB + schema once per db_id, recording whether we accept it.
        if db_id not in db_status:
            src = os.path.join(spider_dir, "database", db_id, f"{db_id}.sqlite")
            if not os.path.exists(src):
                db_status[db_id] = "missing"
            else:
                shutil.copyfile(src, out_db)
                schema = _schema_text(out_db)
                if len(schema) > max_schema_chars:
                    os.remove(out_db)            # too big for the prompt budget
                    db_status[db_id] = "toobig"
                else:
                    with open(os.path.join(sch_dir, f"{db_id}.sql"),
                              "w", encoding="utf-8") as f:
                        f.write(schema)
                    db_status[db_id] = "ok"

        if db_status[db_id] != "ok":
            continue
        if not gold or not _gold_runs(out_db, gold):
            dropped_gold += 1
            continue
        kept.append({"db_id": db_id, "question": question, "gold_sql": gold})

    return kept, dropped_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider_dir", required=True,
                    help="unzipped Spider folder (train_spider.json, dev.json, database/)")
    ap.add_argument("--out_dir", default="spider_data")
    ap.add_argument("--max_train", type=int, default=500,
                    help="cap train examples (0 = all). Keep small for a free T4.")
    ap.add_argument("--max_eval", type=int, default=200,
                    help="cap eval (dev) examples (0 = all)")
    ap.add_argument("--max_schema_chars", type=int, default=3000,
                    help="skip DBs whose schema text exceeds this (prompt budget)")
    args = ap.parse_args()

    train_src = os.path.join(args.spider_dir, "train_spider.json")
    dev_src = os.path.join(args.spider_dir, "dev.json")
    for p in (train_src, dev_src):
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p} — is --spider_dir the unzipped Spider folder?")

    db_status = {}
    train_kept, tg = convert_split(_load_json(train_src), args.spider_dir, args.out_dir,
                                   args.max_train, args.max_schema_chars, db_status)
    dev_kept, dg = convert_split(_load_json(dev_src), args.spider_dir, args.out_dir,
                                 args.max_eval, args.max_schema_chars, db_status)

    _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_kept)
    _write_jsonl(os.path.join(args.out_dir, "dev.jsonl"), dev_kept)

    n_ok = sum(v == "ok" for v in db_status.values())
    n_big = sum(v == "toobig" for v in db_status.values())
    n_missing = sum(v == "missing" for v in db_status.values())
    print(f"train: kept {len(train_kept)} ({tg} dropped: gold didn't run)")
    print(f"dev  : kept {len(dev_kept)} ({dg} dropped: gold didn't run)")
    print(f"DBs  : {n_ok} used, {n_big} skipped (schema too big), {n_missing} missing")
    print(f"wrote {args.out_dir}/train.jsonl, dev.jsonl, dbs/, schemas/")
    print("\nRun it:  python run.py --config lora --data spider")


if __name__ == "__main__":
    main()
