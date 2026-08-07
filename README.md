# EuroSAT — 7 learning paradigms on identical splits

**NKUA deep-learning coursework.** One Colab notebook trains a CNN from scratch, ImageNet transfer, few-shot prototypes, CLIP zero-shot, SimCLR and LoRA on the same stratified EuroSAT splits with the same seed, and reports accuracy next to wall-clock cost.

[![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)

## Results

Every number is transcribed from a saved output cell in the committed notebook — no run required to check it. Cell indices are 0-based.

| Method | Full data | 10-shot | 5-shot | Zero-shot | Train cost | Cell |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN (3 conv blocks, 64×64) | 83.70% | — | — | — | 1.16 min | 100 |
| EfficientNet-B0, full fine-tune | **98.37%** | 81.83% | 78.17% | — | 2.17 min | 28, 100 |
| Prototype classifier (cosine) | — | 80.10% | 77.36% | — | ~0 min | 100 |
| CLIP ViT-B/32 zero-shot | — | — | — | 39.68% | 0 min | 100 |
| SimCLR → linear eval | 89.01% | — | — | — | 493.23 min | 100 |
| SimCLR → fine-tune | 95.19% | 78.02% | 72.94% | — | 493.26 min | 100 |
| LoRA ViT-B/16 (r=8, q/v) | 98.22% | — | — | — | **3.70 min** | 91, 93, 94 |

Machine-readable: [`results/`](results/). LoRA vs full fine-tuning of the same backbone (cell 94): **310,292 trainable parameters, 0.360% of 86,108,948**, 3.70 min, 98.22% — against 85,806,346 params, 13.69 min, 97.93%.

**The 0.30-point LoRA margin is not real.** Cell 28 reports the same `vit_b_16 / finetune` configuration at 98.17% from a different run — a ~0.25-point spread between two runs of an identical recipe, the same size as the margin. The defensible claim is that *LoRA matches full fine-tuning at 0.36% of the trainable parameters and a quarter of the time*, not that it beats it.

Two other things worth knowing: **self-supervision did not pay off at this scale** — SimCLR spent 493 GPU-minutes to reach 89.0% linear / 95.2% fine-tuned, while LoRA on ImageNet features reached 98.2% in 3.7 min, a ~133× compute ratio for a worse result. And **CLIP zero-shot is weak on satellite imagery** — 39.68% with the best of four prompt strategies, including 0.000 accuracy on HerbaceousVegetation, a class it never predicts.

## Run

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm \
            open_clip_torch transformers peft accelerate
jupyter notebook eurosat_benchmark.ipynb
```

Or upload the notebook to Colab, pick a GPU runtime and run all cells — it downloads EuroSAT (27,000 Sentinel-2 RGB images, 10 classes), builds stratified 18,900 / 4,050 / 4,050 splits at seed 42 with disjointness asserted in cell 20, trains and evaluates. Runtime ~9 h end to end, or ~30 min if the SimCLR section is skipped.

## Limitations

- **This is coursework, not independent research.** It was submitted for an NKUA deep-learning course and is presented here because the methodology is sound, not as original work.
- **One seed, no error bars, no significance test.** As shown above, two runs of the same ViT fine-tune differ by 0.25 points, so any gap below roughly half a point here is noise.
- **Not tuned.** Fixed 6–8 epochs, fixed learning rates, no hyperparameter search for any paradigm. SimCLR is under-trained at 30 epochs, so "SimCLR loses" means *at this budget*.
- **RGB only.** EuroSAT also ships 13 multispectral bands; none are used — exactly where domain-specific pretraining would be expected to help most.
- **It is a notebook.** No package, no tests, no CI. The CSVs in `results/` are transcriptions of notebook outputs, not the product of a reproducible pipeline.
- The dataset is not redistributed here; the notebook downloads it.

## License

MIT — see [LICENSE](LICENSE).
