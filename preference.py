"""
preference.py — preference-tuning objectives (DPO, PPO).

These reuse the SAME model factory (any PEFT mechanism), the SAME execution
reward, and feed the SAME execution-accuracy eval as SFT. Only the trainer and
data shape differ — which is the whole point of making `objective` its own
axis. Needs the GPU stack (TRL).

Recipe note: preference tuning runs AFTER an SFT pass. In practice you'd load
the SFT-LoRA adapter as the starting policy; here each run trains from the base
for simplicity, but the structure is the place to wire that in.
"""
import time
import torch

import data as datamod
from eval_harness import execution_reward, clean_sql


# ---------------------------------------------------------------------------
# DPO — preference pairs, no reward model, no sampling loop.
# ---------------------------------------------------------------------------
def train_dpo(model, tokenizer, pref_examples, cfg):
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset

    rows = []
    for ex in pref_examples:
        schema = datamod.load_schema(ex["db_id"], cfg.schema_dir)
        prompt, chosen, rejected = datamod.build_preference_text(ex, schema)
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    ds = Dataset.from_list(rows)

    args = DPOConfig(
        output_dir=f"{cfg.output_dir}/{cfg.name}",
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        beta=cfg.dpo_beta,                 # KL strength — the DPO temperature
        bf16=cfg.bf16 and torch.cuda.is_available(),
        max_length=cfg.max_seq_len,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=cfg.seed,
    )
    # ref_model=None -> DPOTrainer uses the frozen base as the implicit
    # reference (PEFT adapters disabled), which is exactly what we want.
    trainer = DPOTrainer(model=model, ref_model=None, args=args,
                         train_dataset=ds, processing_class=tokenizer)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available() else None)
    return {"train_time_s": dt, "peak_vram_mb": peak}


# ---------------------------------------------------------------------------
# PPO — online: sample SQL, score with execution reward, update policy.
# Heavier and finickier than DPO. The reward is programmatic (execution),
# so NO learned reward model is needed — that's the design choice this lab
# takes for text-to-SQL. (Swap in a learned RM here if you ever want the
# classic RLHF setup; that's the one real fork.)
# ---------------------------------------------------------------------------
def train_ppo(model, tokenizer, train_examples, cfg):
    from trl import PPOConfig, PPOTrainer
    import os

    ppo_config = PPOConfig(
        learning_rate=cfg.lr,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.batch_size,
    )
    trainer = PPOTrainer(ppo_config, model, ref_model=None, tokenizer=tokenizer)
    device = next(model.parameters()).device

    gen_kwargs = dict(max_new_tokens=128, do_sample=True, top_p=0.95,
                      temperature=0.7, pad_token_id=tokenizer.pad_token_id)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    step = 0
    while step < cfg.ppo_steps:
        for ex in train_examples:
            if step >= cfg.ppo_steps:
                break
            schema = datamod.load_schema(ex["db_id"], cfg.schema_dir)
            prompt = datamod.format_prompt(ex["question"], schema)
            query_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            response_ids = trainer.generate(query_ids[0], **gen_kwargs)
            response_txt = tokenizer.decode(response_ids.squeeze()[query_ids.shape[1]:],
                                            skip_special_tokens=True)

            # EXECUTION REWARD — the harness supplies the training signal.
            db_path = os.path.join(cfg.db_dir, f"{ex['db_id']}.db")
            reward = execution_reward(db_path, clean_sql(response_txt), ex["gold_sql"])
            reward_t = torch.tensor(reward, device=device)

            trainer.step([query_ids[0]], [response_ids.squeeze()], [reward_t])
            step += 1

    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available() else None)
    return {"train_time_s": dt, "peak_vram_mb": peak}
