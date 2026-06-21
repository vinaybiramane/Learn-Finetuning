"""
results_logger.py — append one row per run, print the cross-method comparison.

This is the shared ledger that makes the whole lab a COMPARISON rather than a
pile of runs. Both axes (method = mechanism, objective) plus the cost/quality
columns land in one CSV so any two techniques line up side by side.

Deliberately torch-free (stdlib csv only) so it runs on the local no-GPU box —
`python run.py --show` works without the training stack installed.
"""
import csv
import os
from typing import Dict, List

# Column order in the CSV. Identity + both axes first, then the knobs that vary
# in ablations, then the measured cost/quality outcomes.
FIELDNAMES = [
    "name", "method", "objective",
    "lora_r", "lora_alpha", "epochs",
    "trainable_params", "total_params", "trainable_pct",
    "train_time_s", "peak_vram_mb",
    "execution_accuracy", "inference_latency_ms",
]

# Subset shown in the printed table (the CSV keeps everything).
DISPLAY_COLUMNS = [
    ("name", "name"),
    ("method", "mech"),
    ("objective", "obj"),
    ("trainable_pct", "train%"),
    ("peak_vram_mb", "vram_mb"),
    ("train_time_s", "time_s"),
    ("execution_accuracy", "exec_acc"),
    ("inference_latency_ms", "lat_ms"),
]


def log_result(cfg, metrics: Dict, csv_path: str) -> None:
    """Append a single run's row. Creates the file (with header) on first use."""
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    row = {
        "name": cfg.name,
        "method": cfg.method,
        "objective": cfg.objective,
        "lora_r": getattr(cfg, "lora_r", None),
        "lora_alpha": getattr(cfg, "lora_alpha", None),
        "epochs": getattr(cfg, "epochs", None),
    }
    for key in ("trainable_params", "total_params", "trainable_pct",
                "train_time_s", "peak_vram_mb",
                "execution_accuracy", "inference_latency_ms"):
        row[key] = metrics.get(key)

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _read_rows(csv_path: str) -> List[Dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(value: str) -> str:
    """Trim float noise for the table; pass everything else through."""
    if value is None or value == "":
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "-"
    return f"{f:.4g}"


def print_comparison(csv_path: str) -> None:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"No results yet at {csv_path}. Run a method first, e.g. "
              f"`python run.py --config lora`.")
        return

    rows = _read_rows(csv_path)
    if not rows:
        print(f"No results yet at {csv_path}.")
        return

    headers = [label for _, label in DISPLAY_COLUMNS]
    table = [[_fmt(r.get(key)) for key, _ in DISPLAY_COLUMNS] for r in rows]

    widths = [len(h) for h in headers]
    for line in table:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], len(cell))

    def render(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(render(headers))
    print("  ".join("-" * w for w in widths))
    for line in table:
        print(render(line))


if __name__ == "__main__":
    # Self-contained smoke test — no torch, no GPU, no real run needed.
    # Logs two synthetic rows to a throwaway CSV and prints the table, then
    # cleans up. Verifies log_result + print_comparison agree on the schema.
    import tempfile
    from types import SimpleNamespace

    tmp = os.path.join(tempfile.gettempdir(), "_peft_lab_results_smoke.csv")
    if os.path.exists(tmp):
        os.remove(tmp)

    fake_lora = SimpleNamespace(name="lora_r16", method="lora", objective="sft",
                                lora_r=16, lora_alpha=32, epochs=3)
    fake_dpo = SimpleNamespace(name="dpo_lora", method="lora", objective="dpo",
                               lora_r=16, lora_alpha=32, epochs=3)
    log_result(fake_lora, {
        "trainable_params": 1_376_256, "total_params": 1_543_714_304,
        "trainable_pct": 0.0892, "train_time_s": 412.3, "peak_vram_mb": 5821.0,
        "execution_accuracy": 0.83, "inference_latency_ms": 142.5,
    }, tmp)
    log_result(fake_dpo, {
        "trainable_params": 1_376_256, "total_params": 1_543_714_304,
        "trainable_pct": 0.0892, "train_time_s": 638.1, "peak_vram_mb": 6402.0,
        "execution_accuracy": 0.88, "inference_latency_ms": 141.9,
    }, tmp)

    print_comparison(tmp)
    os.remove(tmp)
    print("\n(results_logger smoke test ok)")
