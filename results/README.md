# Extracted results

These CSVs are transcribed verbatim from the saved outputs of
`../eurosat_benchmark.ipynb`. The `notebook_cell` column gives the 0-indexed cell whose
output each row came from, so every value can be checked without running anything.

- `benchmark_results.csv` — the final 7-paradigm comparison (cell 100).
- `transfer_learning_results.csv` — the frozen-vs-finetune sweep over ResNet18,
  EfficientNet-B0 and ViT-B/16 (cell 28).
- `lora_vs_full_finetune.csv` — the direct LoRA vs full fine-tune comparison (cell 94).

Known inconsistency: `transfer_learning_results.csv` reports `vit_b_16 / finetune` at
`test_acc = 0.981728` in 13.443 min, while `lora_vs_full_finetune.csv` reports the same
configuration at `0.979259` in 13.694 min. The two tables were produced by different runs of
the same recipe. The ~0.25-point spread is the honest scale of single-seed run-to-run noise
here, and it is larger than the 0.30-point margin by which LoRA "beats" full fine-tuning.
