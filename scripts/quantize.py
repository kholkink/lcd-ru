#!/usr/bin/env python
"""INT3-ours: GPTQ fake-quant of cfg.model with cfg.base (kind gptq-ours); stores codes in its run dir."""

import _common

from src.pipeline import ensure_quantized

if __name__ == "__main__":
    cfg = _common.config_from_cli(__doc__)
    print(ensure_quantized(cfg["model"], cfg["base"]))
