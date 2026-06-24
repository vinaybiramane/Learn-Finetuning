# Running on Kaggle (free 2×T4)

The CPU spine runs locally; **training needs a GPU**. Quickest path: a Kaggle
notebook with a GPU accelerator. ~6 GB for 1.5B QLoRA fits a single free T4.

## 1. Get the code into a WRITABLE place

Kaggle mounts attached **Datasets read-only** under `/kaggle/input`, and training
writes a `runs/` dir — so the simplest path is to put the code in the writable
`/kaggle/working`:

```python
%cd /kaggle/working
!git clone <your-repo-url> peft-lab
%cd peft-lab
```

> Attached this repo as a **Dataset** instead? It's read-only. The code now routes
> all outputs to `/kaggle/working` automatically (see `config._writable_base`), so
> you can run it straight from the dataset dir:
> ```python
> %cd /kaggle/input/<your-dataset>/        # cwd must hold sample_data/ for data loading
> !python run.py --config lora             # outputs land in /kaggle/working
> ```
> If you're on an older copy without that fix, copy it to working first:
> `shutil.copytree('/kaggle/input/<your-dataset>', '/kaggle/working/lab', dirs_exist_ok=True)`.

## 2. Enable GPU + install
- Notebook → Settings → **Accelerator: GPU T4 x2** (or x1).
- Internet must be **On** (Settings) to download the base model + pip wheels.

```python
!pip install -q -r requirements.txt
```

## 3. Sample data — already included
The sample `store.db` + examples ship in the repo, so there's nothing to build.
(Only regenerate with `python sample_data/build_db.py` if you're in a writable copy
and changed the data — it can't write into a read-only `/kaggle/input` mount.)

## 4. Train
```python
!python run.py --config lora      # SFT with LoRA
!python run.py --config qlora     # same, 4-bit base (QLoRA)
```

The only difference between the two is `load_in_4bit` — identical `target_modules` —
so the delta you read off the table is the cost of 4-bit quantization, nothing else.

## 5. Compare
```python
!python run.py --show
```

## Real data: Spider (optional, makes `exec_acc` meaningful)
The bundled sample is a 14-row smoke test — too small for a real accuracy signal.
For a meaningful comparison, wire in **Spider** (real multi-table databases):

1. Add a public **Spider** dataset to your notebook (Input → Add Data → search
   "spider"). It mounts read-only; find the folder that contains `train_spider.json`,
   `dev.json`, and `database/` — usually `/kaggle/input/<slug>/spider/`.
2. Convert it into the lab's format (writes to the writable working dir):
   ```python
   !python spider_loader.py --spider_dir /kaggle/input/<slug>/spider \
       --out_dir spider_data --max_train 500 --max_eval 200
   ```
3. Run any method on it by adding `--data spider`:
   ```python
   !python run.py --config lora  --data spider
   !python run.py --config qlora --data spider
   !python run.py --show
   ```

`--data spider` uses Spider's real **dev split** as the eval set (not a random
slice), so `exec_acc` finally discriminates. Start with a few hundred examples on a
free T4; raise `--max_train/--max_eval` once it's working. (SFT and PPO work on
Spider; DPO needs generated preference pairs, which is a separate step.)

## Notes
- First run downloads `Qwen/Qwen2.5-Coder-1.5B-Instruct` (~3 GB) — a minute or two.
- `bitsandbytes` (QLoRA) needs CUDA; the plain `lora` run does not strictly need it
  but it's in requirements for convenience.
- Outputs are written under **`/kaggle/working`** (`runs/` and `results/comparison.csv`).
  Download `results/comparison.csv` from the notebook output, or save a Dataset version.
- LoRA rank ablation, same rig: `!python run.py --config lora --lora_r 8 --name lora_r8`
- TRL/transformers APIs drift — if `DPOTrainer`/`PPOTrainer` raise a signature error on
  a first run, it's usually a one-line kwarg fixup.
