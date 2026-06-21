# Running on Kaggle (free 2×T4)

The CPU spine runs locally; **training needs a GPU**. Quickest path: a Kaggle
notebook with a GPU accelerator. ~6 GB for 1.5B QLoRA fits a single free T4.

## 1. Get the code into the notebook
Either upload this folder as a Kaggle Dataset, or clone from GitHub if you push
it there. Then in the first cell:

```python
%cd /kaggle/working
!git clone <your-repo-url> peft-lab   # or: copy from an attached dataset
%cd peft-lab
```

## 2. Enable GPU + install
- Notebook → Settings → **Accelerator: GPU T4 x2** (or x1).
- Internet must be **On** (Settings) to download the base model + pip wheels.

```python
!pip install -q -r requirements.txt
```

## 3. Build the sample data (CPU, runs anywhere)
```python
!python sample_data/build_db.py
```

## 4. Train
```python
!python run.py --config lora      # SFT with LoRA
!python run.py --config qlora     # same, 4-bit base (QLoRA)
```

The only difference between the two is `load_in_4bit` — identical
`target_modules` — so the delta you read off the table is the cost of 4-bit
quantization, nothing else.

## 5. Compare
```python
!python run.py --show
```

## Notes
- First run downloads `Qwen/Qwen2.5-Coder-1.5B-Instruct` (~3 GB) — a minute or two.
- `bitsandbytes` (QLoRA) needs CUDA; the plain `lora` run does not strictly need
  it but it's in requirements for convenience.
- `results/comparison.csv` is written under the repo dir — download it from the
  notebook's output, or persist by saving a Kaggle Dataset version.
- LoRA rank ablation, same rig: `!python run.py --config lora --lora_r 8 --name lora_r8`
