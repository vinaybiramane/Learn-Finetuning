"""
run.py — orchestrator. One command runs one method end-to-end and appends a
row to the comparison table.

    # Phase 3a (revision-as-foundation):
    python run.py --config lora
    python run.py --config qlora

    # LoRA rank ablation (your saturation curve):
    python run.py --config lora --lora_r 4   --name lora_r4
    python run.py --config lora --lora_r 8   --name lora_r8
    python run.py --config lora --lora_r 32  --name lora_r32
    python run.py --config lora --lora_r 64  --name lora_r64

    # Phase 3b (same rig, other families):
    python run.py --config bitfit
    python run.py --config ia3
    python run.py --config prompt
    python run.py --config prefix

    # See the accumulated table any time:
    python run.py --show
"""
import argparse

import data as datamod
import eval_harness
import results_logger
from config import get_config, with_overrides, with_dataset, DEFAULT_RESULTS_CSV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="preset name (lora, qlora, bitfit, ...)")
    ap.add_argument("--lora_r", type=int)
    ap.add_argument("--lora_alpha", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--name", help="override run name (for ablations)")
    ap.add_argument("--data", default="sample", help="dataset: sample | spider")
    ap.add_argument("--show", action="store_true",
                    help="print comparison table and exit")
    args = ap.parse_args()

    if args.show or not args.config:
        results_logger.print_comparison(DEFAULT_RESULTS_CSV)
        return

    cfg = get_config(args.config)
    cfg = with_dataset(cfg, args.data)          # point at sample or spider data
    overrides = {k: v for k, v in vars(args).items()
                 if k in {"lora_r", "lora_alpha", "epochs", "name"}
                 and v is not None}
    if overrides:
        cfg = with_overrides(cfg, **overrides)

    # Import GPU modules lazily so --show works without torch installed.
    from model_factory import build_model_and_tokenizer, count_trainable
    from train import train, generate_predictions

    print(f"=== {cfg.name} (mechanism={cfg.method}, objective={cfg.objective}) "
          f"on {cfg.base_model} ===")

    model, tokenizer = build_model_and_tokenizer(cfg)
    param_stats = count_trainable(model)
    print(f"trainable: {param_stats['trainable_params']:,} "
          f"({param_stats['trainable_pct']:.3f}%)")

    # --- OBJECTIVE DISPATCH: same model, same eval, different loss/data ----
    if cfg.objective == "sft":
        train_ex, eval_ex = datamod.load_split(cfg)
        print(f"train={len(train_ex)}  eval={len(eval_ex)}")
        train_stats = train(model, tokenizer, train_ex, cfg)

    elif cfg.objective == "dpo":
        from preference import train_dpo
        pref = datamod.load_preference_examples(cfg.pref_dataset_path)
        # eval still uses the SFT (gold) set for execution accuracy
        _, eval_ex = datamod.load_split(cfg)
        print(f"preference pairs={len(pref)}  eval={len(eval_ex)}")
        train_stats = train_dpo(model, tokenizer, pref, cfg)

    elif cfg.objective == "ppo":
        from preference import train_ppo
        train_ex, eval_ex = datamod.load_split(cfg)
        print(f"train={len(train_ex)}  eval={len(eval_ex)}  (execution reward)")
        train_stats = train_ppo(model, tokenizer, train_ex, cfg)

    else:
        raise SystemExit(f"Unknown objective '{cfg.objective}'")

    preds, latency = generate_predictions(model, tokenizer, eval_ex, cfg)
    preds = [eval_harness.clean_sql(p) for p in preds]
    report = eval_harness.evaluate(preds, eval_ex, cfg.db_dir)
    print(f"execution accuracy: {report['execution_accuracy']} "
          f"({report['correct']}/{report['counted']})")

    metrics = {
        **param_stats,
        **train_stats,
        "execution_accuracy": report["execution_accuracy"],
        "inference_latency_ms": round(latency, 1) if latency else None,
    }
    results_logger.log_result(cfg, metrics, cfg.results_csv)
    print()
    results_logger.print_comparison(cfg.results_csv)


if __name__ == "__main__":
    main()
