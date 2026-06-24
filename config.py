"""
config.py — the single place every experiment is defined.

Design principle: everything that must stay FIXED across methods for an
apples-to-apples comparison lives here as a default and is NOT touched per
method (base_model, dataset, seq_len, seed, data slice, eval metric).
Only the method-specific knobs change between runs.

This module is intentionally torch-free so it imports instantly on your
local 16GB box for inspecting / building configs without the GPU stack.
"""
import os
from dataclasses import dataclass, field, replace
from typing import Tuple


def _writable_base() -> str:
    """Where run outputs (runs/, results/) should go.

    Kaggle and Colab mount the *code* read-only (under /kaggle/input, /content),
    so writing 'runs/' next to the code fails with a read-only-filesystem error.
    Send outputs to the platform's writable dir instead; fall back to the current
    directory locally (so local behaviour is unchanged)."""
    for base in ("/kaggle/working", "/content"):
        if os.path.isdir(base) and os.access(base, os.W_OK):
            return base
    return "."


_OUTPUT_BASE = _writable_base()
DEFAULT_RUNS_DIR = os.path.join(_OUTPUT_BASE, "runs")
DEFAULT_RESULTS_CSV = os.path.join(_OUTPUT_BASE, "results", "comparison.csv")


@dataclass
class ExperimentConfig:
    # ---- identity -----------------------------------------------------------
    name: str

    # TWO ORTHOGONAL AXES (they combine freely):
    #   mechanism = HOW MANY params change (PEFT family)
    #   objective = WHAT you train on (the loss / data shape)
    # e.g. mechanism=qlora x objective=dpo  ==  "QLoRA-based DPO".
    method: str                  # mechanism: lora|qlora|bitfit|ia3|prompt_tuning|prefix_tuning|full
    objective: str = "sft"       # objective: sft | dpo | ppo

    # ---- FIXED across all experiments (do not vary these per method) ---------
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    dataset_path: str = "sample_data/examples.jsonl"          # SFT: {db_id,question,gold_sql}
    pref_dataset_path: str = "sample_data/preferences.jsonl"  # DPO: {db_id,question,chosen,rejected}
    db_dir: str = "sample_data/dbs"
    schema_dir: str = "sample_data/schemas"     # CREATE TABLE text injected into the prompt
    eval_dataset_path: str = ""        # if set (e.g. Spider dev), used as eval instead of a random split
    eval_fraction: float = 0.2          # held-out slice for execution accuracy (when no eval_dataset_path)
    max_seq_len: int = 1024
    seed: int = 42

    # ---- training (shared defaults; keep identical for fair comparison) ------
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    bf16: bool = True

    # ---- LoRA / QLoRA --------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Keep target modules IDENTICAL between lora and qlora so the ONLY
    # difference measured is 4-bit quantization, nothing else.
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    load_in_4bit: bool = False          # flipped True by the qlora preset

    # ---- prompt / prefix tuning ---------------------------------------------
    num_virtual_tokens: int = 20

    # ---- preference tuning (objective = dpo | ppo) --------------------------
    dpo_beta: float = 0.1               # KL strength in the DPO loss
    ppo_steps: int = 100                # PPO rollout/update steps
    ppo_reward: str = "execution"       # programmatic reward from the eval harness
    # Reward for PPO and pair-labelling for DPO both come from EXECUTION truth
    # (does the SQL run and match gold?) — no human labels, no learned reward
    # model needed. See data.py / eval_harness.execution_reward.

    # ---- output --------------------------------------------------------------
    # Resolved to a WRITABLE location (e.g. /kaggle/working on Kaggle) so runs
    # don't try to write next to read-only mounted code. Local: ./runs, ./results.
    output_dir: str = DEFAULT_RUNS_DIR
    results_csv: str = DEFAULT_RESULTS_CSV


# ---------------------------------------------------------------------------
# PRESETS — `python run.py --config lora` loads one of these.
# Phase 3a: lora, qlora (revision-as-foundation).
# Phase 3b: bitfit, ia3, prompt_tuning, prefix_tuning (plug into same rig).
# Phase 2 baseline: full.
# ---------------------------------------------------------------------------
def _presets():
    base = dict()  # all defaults come from the dataclass

    lora = ExperimentConfig(name="lora_r16", method="lora", **base)

    qlora = ExperimentConfig(
        name="qlora_r16", method="qlora", load_in_4bit=True, **base
    )

    # Selective: train only bias terms (~0.1% params). No new params.
    bitfit = ExperimentConfig(name="bitfit", method="bitfit", **base)

    # Additive (multiplicative scaling vectors on K, V, FFN).
    ia3 = ExperimentConfig(name="ia3", method="ia3", **base)

    # Additive (soft tokens at input only — expect this to struggle at 1.5B).
    prompt_tuning = ExperimentConfig(
        name="prompt_tuning", method="prompt_tuning", **base
    )

    # Additive (virtual K/V at every layer).
    prefix_tuning = ExperimentConfig(
        name="prefix_tuning", method="prefix_tuning", **base
    )

    # Phase 2 baseline — update all weights (expensive; small model only).
    full = ExperimentConfig(name="full_ft", method="full", lr=1e-5, **base)

    # ---- Phase 4: PREFERENCE TUNING (objective axis) --------------------
    # Same mechanisms, different objective — the axes compose with no
    # special-casing. Standard recipe: SFT-LoRA first, then DPO-LoRA.
    dpo = ExperimentConfig(
        name="dpo_lora", method="lora", objective="dpo", lr=5e-6, **base
    )
    # QLoRA-based DPO — mechanism=qlora x objective=dpo.
    qdpo = ExperimentConfig(
        name="dpo_qlora", method="qlora", objective="dpo",
        load_in_4bit=True, lr=5e-6, **base
    )
    ppo = ExperimentConfig(
        name="ppo_lora", method="lora", objective="ppo", lr=1e-5, **base
    )

    return {
        "lora": lora,
        "qlora": qlora,
        "bitfit": bitfit,
        "ia3": ia3,
        "prompt": prompt_tuning,
        "prefix": prefix_tuning,
        "full": full,
        "dpo": dpo,
        "qdpo": qdpo,
        "ppo": ppo,
    }


PRESETS = _presets()


def get_config(name: str) -> ExperimentConfig:
    if name not in PRESETS:
        raise SystemExit(
            f"Unknown config '{name}'. Available: {', '.join(PRESETS)}"
        )
    return PRESETS[name]


# Convenience for rank ablations: python run.py --config lora --lora_r 8
def with_overrides(cfg: ExperimentConfig, **overrides) -> ExperimentConfig:
    return replace(cfg, **overrides)


# ---------------------------------------------------------------------------
# DATASETS — swap the data the harness runs on without touching any method.
# `python run.py --config lora --data spider`. Build Spider first with
# spider_loader.py. The mechanism/objective axes are unaffected.
# ---------------------------------------------------------------------------
DATASETS = {
    "sample": {
        "dataset_path": "sample_data/examples.jsonl",
        "pref_dataset_path": "sample_data/preferences.jsonl",
        "db_dir": "sample_data/dbs",
        "schema_dir": "sample_data/schemas",
        "eval_dataset_path": "",                       # random held-out slice
    },
    "spider": {
        "dataset_path": "spider_data/train.jsonl",
        # DPO pairs aren't generated for Spider yet (needs candidate sampling),
        # so DPO on Spider would still read the sample pairs; SFT/PPO work fully.
        "pref_dataset_path": "sample_data/preferences.jsonl",
        "db_dir": "spider_data/dbs",
        "schema_dir": "spider_data/schemas",
        "eval_dataset_path": "spider_data/dev.jsonl",  # Spider's real dev split
    },
}


def with_dataset(cfg: ExperimentConfig, name: str) -> ExperimentConfig:
    if name not in DATASETS:
        raise SystemExit(
            f"Unknown dataset '{name}'. Available: {', '.join(DATASETS)}"
        )
    return replace(cfg, **DATASETS[name])
