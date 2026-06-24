"""
train.py — the SFT objective (the default objective axis).

Two jobs, both shared by EVERY mechanism (lora, qlora, bitfit, ...) unchanged:
  1. train(...):                prompt-masked cross-entropy SFT.
  2. generate_predictions(...): greedy decode for execution-accuracy eval.

Prompt-masked loss is the key detail: labels are -100 over the prompt tokens so
the loss is computed ONLY on the SQL completion. The model is graded on
producing SQL, never on reciting the schema/question back.

GPU side (torch / transformers). DPO/PPO live in preference.py; this file is
SFT only. Tokenization lives here (not data.py) to keep the CPU spine
tokenizer-free.
"""
import time

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

import data as datamod


class _SFTDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _tokenize(examples, tokenizer, cfg):
    """Build {input_ids, attention_mask, labels} with the PROMPT MASKED.

    input_ids = prompt + completion + EOS
    labels    = [-100]*prompt + completion + EOS   (loss on completion only)
    """
    eos = tokenizer.eos_token_id
    rows = []
    for ex in examples:
        schema = datamod.load_schema(ex["db_id"], cfg.schema_dir)
        prompt, completion = datamod.build_supervised_text(ex, schema)

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
        completion_ids = completion_ids + [eos]

        input_ids = (prompt_ids + completion_ids)[: cfg.max_seq_len]
        labels = ([-100] * len(prompt_ids) + completion_ids)[: cfg.max_seq_len]

        rows.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        })
    return rows


def train(model, tokenizer, train_examples, cfg):
    rows = _tokenize(train_examples, tokenizer, cfg)
    dataset = _SFTDataset(rows)

    # Pads input_ids with pad_token and labels with -100 to the batch max.
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, padding="longest"
    )

    cuda = torch.cuda.is_available()
    use_bf16 = bool(cfg.bf16 and cuda and torch.cuda.is_bf16_supported())

    args = TrainingArguments(
        output_dir=f"{cfg.output_dir}/{cfg.name}",
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        bf16=use_bf16,
        fp16=(cuda and not use_bf16),
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=cfg.seed,
        # Paged 8-bit optimizer is the QLoRA pairing; plain AdamW otherwise.
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=dataset, data_collator=collator
    )

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6 if cuda else None

    return {
        "train_time_s": round(dt, 1),
        "peak_vram_mb": round(peak, 1) if peak is not None else None,
    }


def generate_predictions(model, tokenizer, eval_examples, cfg):
    """Greedy decode one SQL per eval example. Returns (predictions, avg latency
    in ms/example). run.py runs clean_sql + execution accuracy on the result.

    Greedy (do_sample=False) so the eval is deterministic and the comparison
    across methods is apples-to-apples.
    """
    model.eval()
    prev_use_cache = model.config.use_cache
    model.config.use_cache = True              # cache OK at inference (off in train)
    device = next(model.parameters()).device

    preds = []
    t0 = time.time()
    for ex in eval_examples:
        schema = datamod.load_schema(ex["db_id"], cfg.schema_dir)
        prompt = datamod.format_prompt(ex["question"], schema)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]   # strip the prompt
        preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    elapsed_ms = (time.time() - t0) * 1000.0
    avg_latency_ms = elapsed_ms / max(1, len(eval_examples))

    model.config.use_cache = prev_use_cache
    return preds, avg_latency_ms
