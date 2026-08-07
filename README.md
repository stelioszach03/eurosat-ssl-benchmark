# EuroSAT — 7 learning paradigms on identical splits, and what each one costs

**NKUA deep-learning coursework.** One Colab notebook trains a CNN from scratch, ImageNet
transfer, few-shot prototypes, CLIP zero-shot, SimCLR self-supervision and LoRA on the same
stratified EuroSAT splits with the same seed, and reports accuracy *next to wall-clock cost*.

The headline is the cost column: **SimCLR spent 493 GPU-minutes to reach 89.0 %; LoRA reached
98.2 % in 3.7 minutes.**

[![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square)](https://pytorch.org/)
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B%2F32-412991?style=flat-square)](https://github.com/mlfoundations/open_clip)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-FFD21E?style=flat-square)](https://github.com/huggingface/peft)

**[Open the notebook](eurosat_benchmark.ipynb)** · **[Extracted results CSVs](results/)**

---

## Results

Every number below is transcribed from a **saved output cell** in the committed notebook — no
run required to check it. Cell indices are 0-based.

| Method | Full data | 10-shot | 5-shot | Zero-shot | Train cost | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Baseline CNN (3 conv blocks, 64×64) | 83.70 % | — | — | — | 1.16 min | cell 100 |
| **EfficientNet-B0, full fine-tune** | **98.37 %** | **81.83 %** | **78.17 %** | — | 2.17 min | cells 28, 100 |
| Prototype classifier (cosine) | — | 80.10 % | 77.36 % | — | ~0 min | cell 100 |
| CLIP ViT-B/32 zero-shot | — | — | — | 39.68 % | 0 min | cell 100 |
| SimCLR → linear eval | 89.01 % | — | — | — | 493.23 min | cell 100 |
| SimCLR → fine-tune | 95.19 % | 78.02 % | 72.94 % | — | 493.26 min | cell 100 |
| **LoRA ViT-B/16 (r=8, q/v)** | **98.22 %** | — | — | — | **3.70 min** | cells 91, 93, 94 |

Machine-readable: [`results/benchmark_results.csv`](results/benchmark_results.csv) ·
[`results/transfer_learning_results.csv`](results/transfer_learning_results.csv) ·
[`results/lora_vs_full_finetune.csv`](results/lora_vs_full_finetune.csv)

### LoRA vs full fine-tuning of the same backbone (cell 94)

| | Trainable params | % of model | Train time | Test accuracy |
|---|---:|---:|---:|---:|
| ViT-B/16 LoRA (r=8, q/v) | 310,292 | 0.360 % | 3.70 min | 98.22 % |
| ViT-B/16 full fine-tune | 85,806,346 | 99.65 % | 13.69 min | 97.93 % |

276× fewer trainable parameters, 3.70× faster, 0.30 points *more* accurate.

**The 0.30-point margin is not real.** The transfer-learning sweep (cell 28) reports the same
`vit_b_16 / finetune` configuration at **98.17 %** in 13.44 min, from a different run. The
~0.25-point spread between two runs of an identical recipe is the same size as the margin
LoRA "wins" by. The defensible claim is *LoRA matches full fine-tuning at 0.36 % of the
trainable parameters and a quarter of the time* — not that it beats it.

### What is actually worth knowing

1. **Self-supervision did not pay off at this scale.** SimCLR (NT-Xent, τ=0.5, 12,000
   unlabelled images, 30 epochs, ≈490 min elapsed) reached 89.0 % linear / 95.2 % fine-tuned.
   LoRA on ImageNet features reached 98.2 % in 3.7 min. That is a ~133× compute ratio for a
   worse result: on 27,000 images, ImageNet features already transfer.
2. **CLIP zero-shot is weak on satellite imagery** — 39.68 % with the best of four prompt
   strategies (`"a high resolution satellite photo of {}"`), and **0.000 accuracy on
   HerbaceousVegetation** — it never predicts that class at all.
3. **Frozen backbones are not competitive here.** Best frozen is ViT-B/16 at 96.99 % vs
   98.37 % for EfficientNet-B0 fully fine-tuned (cell 28).
4. **Prototype classification is nearly free and nearly as good few-shot** — 80.10 % at
   10-shot with zero training, ~1.7 points behind fine-tuned transfer.

## Setup

- **Data:** [EuroSAT](https://github.com/phelber/EuroSAT) RGB, 27,000 Sentinel-2 images,
  10 land-use classes.
- **Splits:** 18,900 / 4,050 / 4,050 (70/15/15), stratified, seed 42. Disjointness asserted
  in-notebook (cell 20): `train∩val: 0`, `train∩test: 0`, `val∩test: 0`.
- **Single seed.** Every number is one run.
- Colab GPU. SimCLR needs the strongest runtime available; everything else fits a T4.

## How to run

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm \
            open_clip_torch transformers peft accelerate
jupyter notebook eurosat_benchmark.ipynb
```

Or upload `eurosat_benchmark.ipynb` to Colab, pick a GPU runtime, and run all cells — it
downloads the dataset, builds the splits, trains and evaluates.

Runtime: **~9 h** end to end, or **~30 min** if the SimCLR section is skipped (it is 493 of
those minutes).

## What this is not

- **This is coursework, not independent research.** The notebook writes to
  `MyDrive/EuroSAT_DL_Assignment/…`; it was submitted for an NKUA deep-learning course.
  It is presented here because the methodology is sound, not as original work.
- **One seed, no error bars, no significance test.** As shown above, two runs of the same
  ViT fine-tune differ by 0.25 points, so any gap below roughly half a point in this README
  is noise.
- **Not tuned.** Fixed 6–8 epochs, fixed learning rates, no hyperparameter search for any
  paradigm. SimCLR in particular is generally under-trained at 30 epochs, so "SimCLR loses"
  means *at this budget*, not in general.
- **RGB only.** EuroSAT also ships 13 multispectral bands; none are used, which is exactly
  where domain-specific pretraining would be expected to help most.
- **It is a notebook.** No package, no tests, no CI. The committed CSVs under
  [`results/`](results/) are transcriptions of notebook outputs, not the product of a
  reproducible pipeline.
- **The dataset is not redistributed here**; the notebook downloads it.

## Layout

```text
eurosat_benchmark.ipynb   110 cells, 52 with saved outputs — the whole project
results/                  CSVs transcribed from those outputs, with cell references
LICENSE                   MIT
```

## License

MIT — see [`LICENSE`](LICENSE).
