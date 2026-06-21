"""
model_factory.py — the MECHANISM axis lives here.

Two jobs:
  1. build_model_and_tokenizer(cfg): load the FIXED base model + tokenizer.
  2. apply_peft(model, cfg):          turn it into ONE mechanism.

The whole two-axis design rests on this file: "adding a mechanism = one branch
in apply_peft." Nothing else (data, loss, eval, logging) changes when you add a
family. lora vs qlora differ by EXACTLY one input — cfg.load_in_4bit — with
identical target_modules, so the only thing the comparison measures is 4-bit
quantization.

GPU side (torch / transformers / peft / bitsandbytes). Imported lazily by
run.py so the torch-free CPU spine still works without this stack installed.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    # Causal LMs often ship without a pad token; reuse EOS so batching/padding
    # works. Right-padding is correct for training with a causal mask.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _bnb_config(cfg):
    """QLoRA's 4-bit recipe: NF4 weights + double quantization, with compute
    (the LoRA matmuls) kept in bf16/fp16. This is the ONLY thing that makes
    qlora differ from lora."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # information-optimal 4-bit
        bnb_4bit_use_double_quant=True,         # quantize the quant constants too
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
    )


def build_model_and_tokenizer(cfg):
    tokenizer = build_tokenizer(cfg)
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    model_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
    if cfg.load_in_4bit:
        model_kwargs["quantization_config"] = _bnb_config(cfg)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    model.config.use_cache = False              # required while training
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = apply_peft(model, cfg)
    return model, tokenizer


# ---------------------------------------------------------------------------
# THE MECHANISM SWITCH. One branch per PEFT family. Add a family = add a branch.
# ---------------------------------------------------------------------------
def apply_peft(model, cfg):
    method = cfg.method

    # --- Full fine-tune: no PEFT, every weight trains (Phase 2 baseline). ----
    if method == "full":
        for p in model.parameters():
            p.requires_grad = True
        return model

    # --- Selective / BitFit: train ONLY bias terms (~0.1% of params). No new
    #     parameters, no peft library needed — just a requires_grad mask. ------
    if method == "bitfit":
        for name, p in model.named_parameters():
            p.requires_grad = name.endswith(".bias") or ".bias" in name
        return model

    # --- Everything below uses the peft library. For 4-bit bases, prepare the
    #     model first (casts norms to fp32, enables input grads, etc.). --------
    from peft import (
        get_peft_model, prepare_model_for_kbit_training,
        LoraConfig, IA3Config, PromptTuningConfig, PrefixTuningConfig, TaskType,
    )
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if method in ("lora", "qlora"):
        # Reparameterization: W + (alpha/r)*BA on the target projections.
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    elif method == "ia3":
        # Additive/multiplicative: learned rescaling vectors on K, V, and FFN.
        peft_cfg = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )
    elif method == "prompt_tuning":
        # Additive, input only: train embeddings of N virtual tokens. Expect
        # this to struggle at 1.5B — that's the lesson, not a bug.
        peft_cfg = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=cfg.num_virtual_tokens,
        )
    elif method == "prefix_tuning":
        # Additive, every layer: virtual K/V injected at each attention block.
        peft_cfg = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=cfg.num_virtual_tokens,
        )
    else:
        raise ValueError(
            f"Unknown mechanism '{method}'. "
            f"Expected one of: lora, qlora, bitfit, ia3, prompt_tuning, "
            f"prefix_tuning, full."
        )

    return get_peft_model(model, peft_cfg)


def count_trainable(model):
    """Trainable vs total parameter counts — the headline cost number for the
    comparison table. (Under 4-bit the 'total' counts quantized params, so read
    trainable_pct as relative across mechanisms, not as exact memory.)"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100.0 * trainable / total, 4) if total else 0.0,
    }
