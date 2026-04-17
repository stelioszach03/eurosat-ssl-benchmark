<div align="center">

# EuroSAT Deep Learning Benchmark

**Comprehensive comparison of 7 learning paradigms on satellite imagery: Supervised CNN, Transfer Learning, Few-shot Prototypes, CLIP Zero-shot, SimCLR Self-supervised, and LoRA-efficient Fine-tuning.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/OpenCLIP-ViT--B%2F32-412991?style=flat-square)](https://github.com/mlfoundations/open_clip)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://github.com/huggingface/peft)
[![A100](https://img.shields.io/badge/NVIDIA-A100%2040GB-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)

**[Open Notebook](eurosat_benchmark.ipynb)**

</div>

---

## What this is

A single, self-contained Colab notebook that trains and evaluates **7 different learning approaches** on the [EuroSAT](https://github.com/phelber/EuroSAT) RGB satellite image dataset (10 land-use classes, 27 000 images). Every method is evaluated on the same stratified splits with the same seed, making the comparison fair and reproducible.

---

## Benchmark Results

| Method | Full Data | 10-shot | 5-shot | Zero-shot | Training Cost |
|:-------|----------:|--------:|-------:|----------:|--------------:|
| Baseline CNN (3-block, 64×64) | 83.01% | — | — | — | 1.1 min |
| **EfficientNet-B0 (full FT)** | **98.37%** | 81.83% | 78.17% | — | 2.2 min |
| Prototype Classifier (cosine) | — | 80.10% | 77.36% | — | ~0 min |
| CLIP ViT-B/32 (zero-shot) | — | — | — | 39.68% | 0 min |
| SimCLR → linear eval | 89.01% | — | — | — | 493 min |
| SimCLR → fine-tune | 95.19% | 78.02% | 72.94% | — | 493 min |
| **LoRA ViT-B/16 (r=8, q/v)** | **98.22%** | — | — | — | **3.7 min** |

> **Key finding:** LoRA achieves 98.22% accuracy with only **0.36% trainable parameters** (310K vs 85.8M) and **3.7× faster** training than full ViT fine-tuning — while matching its accuracy.

---

## Methods Covered

### Part 1 — Supervised Learning & Transfer Learning
- **Baseline CNN** — 3 convolutional blocks (Conv → BN → ReLU → MaxPool) + FC classifier, trained from scratch on 64×64 crops.
- **Transfer Learning** — ResNet18, EfficientNet-B0, ViT-B/16 pretrained on ImageNet. Full fine-tuning vs frozen-backbone comparison across all three architectures.
- **Few-shot Transfer** — Top-2 configurations evaluated on 10-shot and 5-shot subsets with stratified sampling.

### Part 2 — Few-shot & Zero-shot
- **Prototype Networks** — Frozen ResNet18 encoder → class prototypes via mean embeddings → nearest-centroid classification with Euclidean and cosine distances.
- **CLIP Zero-shot** — OpenCLIP ViT-B/32 with 4 prompt engineering strategies: basic, domain-specific, ensemble, and multi-template.

### Part 3 — Self-supervised Learning
- **SimCLR** — Contrastive pretraining (NT-Xent loss, τ=0.5) on 12 000 unlabelled images for 30 epochs. Linear evaluation and full fine-tuning on downstream task. t-SNE embedding analysis.

### Part 4 — Parameter-efficient Fine-tuning
- **LoRA** — Low-Rank Adaptation (r=8) on ViT-B/16 query/value projections via HuggingFace PEFT. 0.36% trainable parameters, 3.7 min training, 98.22% test accuracy.

---

## Key Findings

1. **LoRA is the efficiency champion** — 98.22% accuracy with 276× fewer trainable params than full ViT-B/16 fine-tuning, and 3.7× faster.
2. **EfficientNet-B0 full FT is the accuracy champion** — 98.37% on full data, strongest few-shot performance (81.83% on 10-shot).
3. **SimCLR underperforms ImageNet transfer** on EuroSAT — ImageNet features transfer well to satellite imagery; domain-specific pretraining doesn't add enough value at this data scale.
4. **CLIP zero-shot is weak on satellite imagery** (39.68%) — the domain gap between web-crawled training data and remote sensing is substantial.
5. **Prototype classifiers are competitive in few-shot** — 80.10% on 10-shot with zero training, only ~1.7% behind transfer learning.

---

## How to Run

### Google Colab (recommended)

1. Upload `eurosat_benchmark.ipynb` to Colab
2. Select **GPU runtime** (A100 recommended for SimCLR, T4 sufficient for everything else)
3. Run all cells — the notebook handles dataset download, splits, training, and evaluation

### Local

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm open_clip_torch transformers peft accelerate
jupyter notebook eurosat_benchmark.ipynb
```

**Estimated runtime:** ~9 hours (full, including SimCLR pretraining) or ~30 minutes (skip SimCLR).

---

## Repository Structure

```
├── eurosat_benchmark.ipynb    # Complete benchmark notebook (110 cells)
├── assignment.pdf             # Original assignment specification
├── README.md
└── LICENSE
```

---

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) — 27 000 geo-referenced Sentinel-2 satellite images across 10 land-use classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

Splits: **70% train / 15% val / 15% test**, stratified, seed 42, verified leak-free.

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
Built by <strong><a href="https://stelioszach.com">Stelios Zacharioudakis</a></strong> · ML Engineer & Researcher · <a href="mailto:sdi2200243@di.uoa.gr">sdi2200243@di.uoa.gr</a>
</p>
