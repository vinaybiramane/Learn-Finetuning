# peft-lab

A hands-on lab for **understanding fine-tuning methods deeply** by running each one
through an *identical* harness on a text-to-SQL task and comparing them apples-to-apples.
It covers the **PEFT family** (LoRA, QLoRA, BitFit, IA3, prompt/prefix tuning) and the
**preference-tuning objectives** (DPO, PPO) — with the math behind each built up from
scratch in a series of runnable notebooks.

The goal is *learning* and an honest comparison, not chasing SOTA.

## Two orthogonal axes

Everything composes along two axes that vary independently:

| axis | question | options |
|---|---|---|
| **mechanism** | *how many* parameters change | `lora` · `qlora` · `bitfit` · `ia3` · `prompt_tuning` · `prefix_tuning` · `full` |
| **objective** | *what* you train on (loss + data) | `sft` · `dpo` · `ppo` |

They combine freely (e.g. `qlora × dpo`). Adding a mechanism is one branch in
`model_factory.apply_peft`; adding an objective is one trainer + one dispatch branch.
Neither touches the other, and the **evaluation is shared by all combinations** — which
is what keeps the comparison fair.

## The shared spine

- **Execution accuracy** (`eval_harness.py`) scores *any* model the same way — run the
  predicted SQL and the gold SQL against a real SQLite database and compare **result
  sets**, not strings. This is the honest metric every method is judged by.
- **Execution reward** (`eval_harness.execution_reward`) supplies the training signal
  for preference tuning: `1.0` exact match / `0.3` runs-but-wrong / `0.0` fails — plus a
  cost tiebreak (`sql_cost`) that prefers the cheaper of two equally-correct queries.
  So DPO pairs and PPO rewards come from **execution truth — no human labels and no
  learned reward model.** That's the elegant fit text-to-SQL offers.

## The teaching notebooks (`demos/`)

A from-scratch curriculum — each notebook is one idea, with tiny worked numbers verified
in runnable cells. **numpy only, no GPU required.** Read them in order.

| # | notebook | covers |
|---|---|---|
| 01 | what is a weight matrix | neuron → matrix·vector; fine-tuning = changing weights |
| 02 | what is rank | independent directions → a low-rank matrix as `B·A` |
| 03 | training & the low-rank change | gradient descent → `ΔW` → why it's low rank |
| 04 | LoRA assembled | `W + (α/r)·B·A`, zero-init, parameter savings |
| 05 | what is quantization | bits → allowed values → rounding error |
| 06 | 4-bit and NF4 | NF4 vs FP4 grids, scales, blocks (outliers), double quantization |
| 07 | QLoRA assembled | LoRA with a 4-bit NF4 frozen base |
| 08 | PEFT families & BitFit | the additive/selective/reparameterization map; bias-only training |
| 09 | how a transformer reads text | tokens → embeddings → attention Q/K/V |
| 10 | IA3 | learned multipliers on K/V/FFN |
| 11 | prompt tuning | trainable soft tokens at the input |
| 12 | prefix tuning | virtual K/V at every layer + a real cross-method parameter comparison |
| 13 | from imitation to preference | sequence log-probability; why SFT alone is brittle |
| 14 | preference pairs from execution | building chosen/rejected from the execution reward |
| 15 | sigmoid & Bradley-Terry | modeling "A beats B" as a probability |
| 16 | DPO assembled | implicit reward `β·log(π/π_ref)`, the reference leash, the loss |
| 17 | DPO vs RLHF | the reward model, the 3-stage recipe, and where PPO fits |
| 18 | PPO | online policy-gradient, the KL leash, clipping |

## Repo layout

```
config.py          experiment presets; the single place every run is defined
data.py            dataset loading, prompt formatting, preference-pair builder   (CPU)
eval_harness.py    execution accuracy + execution reward + sql_cost              (CPU)
results_logger.py  appends one row per run; prints the comparison table          (CPU)
sample_data/       build_db.py builds the sample SQLite DB + examples            (CPU)
model_factory.py   builds the base model and applies one PEFT mechanism          (GPU)
train.py           SFT trainer (prompt-masked loss) + generation                 (GPU)
preference.py      DPO and PPO trainers                                          (GPU)
run.py             orchestrator: one command runs one method end-to-end
demos/             the teaching notebooks (numpy only)
```

The CPU-side modules are deliberately torch-free so the data, metric, and notebooks all
run on a laptop with no GPU.

## Quickstart

**Notebooks (local, no GPU):**
```bash
pip install numpy jupyter
jupyter lab demos/        # open 01 and read in order
```

**The CPU spine (local, no GPU):**
```bash
python sample_data/build_db.py     # build the sample DB + examples
python eval_harness.py             # verify the execution-accuracy metric
python data.py                     # inspect prompt formatting
```

**Training (needs a GPU — e.g. Kaggle/Colab; see `KAGGLE.md`):**
```bash
pip install -r requirements.txt
python run.py --config lora                        # SFT with LoRA
python run.py --config qlora                       # same, 4-bit base
python run.py --config lora --lora_r 8 --name lora_r8   # rank ablation
python run.py --config dpo                         # mechanism=lora × objective=dpo
python run.py --show                               # comparison table
```

Base model (fixed for fair comparison): `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

## Status

- ✅ **Teaching notebooks (01–18)** — complete; every code cell runs (numpy only).
- ✅ **CPU spine** — data, execution-accuracy metric, reward, and preference-pair builder
  run locally and are smoke-tested.
- 🚧 **GPU training runs** — the harness (`model_factory.py`, `train.py`, `preference.py`,
  `run.py`) is implemented but has **not yet been benchmarked end-to-end on a GPU**, so
  `results/` is empty and no execution-accuracy numbers are published yet. Validating the
  rig and producing the comparison table is the next step. (TRL/transformers APIs drift,
  so expect minor signature fixups on a first run.)

In other words: the *concepts and the local spine* are solid and verified; the *measured
outcomes* are still to come.

## Notes

- `lora` and `qlora` keep identical `target_modules` so the only measured difference is
  4-bit quantization.
- The sample data is synthetic.
- No license file is included yet — add one before relying on this for anything beyond
  learning.
