# Reference materials: LCD (arXiv:2608.11786)

Downloaded 2026-09-25 for the plan review (`reports/plan_review.md`).

- **Paper:** N. Thomas, *Language-Conditional Dequantization: Recovering What Quantization Steals from Non-English
  Languages*, arXiv:2608.11786 v1 (CC BY 4.0): https://arxiv.org/abs/2608.11786 (PDF:
  https://arxiv.org/pdf/2608.11786). The PDF is not redistributed in this repository; `paper.pdf` / `paper.txt` are
  git-ignored local copies.
- `code/` — https://anonymous.4open.science/r/LCD-2466 (Apache-2.0, see `code/LICENSE`), incl. `lcd_results/` JSONs.
  Unmodified third-party code; `tests/test_data.py` executes `code/evaluate.py` to check that our data loader selects
  exactly the same documents as LCD's.
- `hf_int3_configs/` — config.json / recipe.yaml / safetensors headers of the author's public INT3 checkpoints
  `nrmlthms/qwen2.5-3b-int3-gptq`, `nrmlthms/llama3.2-3b-int3-gptq` (metadata only; weights not downloaded).

Reference only — not imported by project code.
