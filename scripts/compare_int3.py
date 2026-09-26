#!/usr/bin/env python
"""INT3-ours vs INT3-LCD (author's checkpoint): per-module agreement of the integer codes, equality of the group
scales, relative weight difference. Both use sym W3 g128 static minmax scales, so scales should match exactly and
codes differ only through GPTQ's error feedback (calibration batches, numerics).

    python scripts/compare_int3.py --config configs/p1/qwen3b.yaml [compare_path=$LCDRU_RUNS/<quant dir>]
"""

import json
from collections import defaultdict
from pathlib import Path

import _common
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import load_file

from src import pipeline
from src.models import LCD_INT3
from src.quant.gptq import _unpack
from src.utils import Timer, expand_env

if __name__ == "__main__":
    cfg = _common.config_from_cli(__doc__)
    ours = (Path(expand_env(cfg["compare_path"])) if cfg.get("compare_path")
            else pipeline.ensure_quantized(cfg["model"], {**cfg["base_ours"], "kind": "gptq-ours"}))  # fmt: skip
    info = json.loads((Path(ours) / "quant_config.json").read_text())
    t = load_file(str(Path(ours) / "quant.safetensors"))
    repo = LCD_INT3[cfg["model"]]
    agg = defaultdict(lambda: {"codes_equal": 0.0, "n": 0, "scale_equal": 0.0, "ns": 0, "werr_num": 0.0, "werr_den": 0.0})
    with Timer() as tm:
        for fname in [f for f in list_repo_files(repo) if f.endswith(".safetensors")]:
            with safe_open(hf_hub_download(repo, fname), framework="pt") as f:
                for k in f.keys():
                    if not k.endswith(".weight_scale"):
                        continue
                    name = k[: -len(".weight_scale")]
                    lcd_codes = f.get_tensor(name + ".weight").float()
                    lcd_scale = f.get_tensor(k).float()
                    shape = tuple(info["shapes"][name])
                    our_codes = _unpack(t[f"{name}.codes"], shape, 3).float() - t[f"{name}.zero"].repeat_interleave(128, 1)[:, : shape[1]]
                    our_scale = t[f"{name}.scale"]
                    kind = name.rsplit(".", 1)[1]
                    a = agg[kind]
                    a["codes_equal"] += (our_codes == lcd_codes).sum().item()
                    a["n"] += lcd_codes.numel()
                    a["scale_equal"] += (our_scale.to(torch.bfloat16).float() == lcd_scale).sum().item()
                    a["ns"] += lcd_scale.numel()
                    w_ours = our_codes * our_scale.repeat_interleave(128, 1)[:, : shape[1]]
                    w_lcd = lcd_codes * lcd_scale.repeat_interleave(128, 1)[:, : shape[1]]
                    a["werr_num"] += (w_ours - w_lcd).pow(2).sum().item()
                    a["werr_den"] += w_lcd.pow(2).sum().item()
    res = {kind: {"codes_equal": a["codes_equal"] / a["n"], "scales_equal": a["scale_equal"] / a["ns"],
                  "rel_weight_diff": (a["werr_num"] / a["werr_den"]) ** 0.5} for kind, a in sorted(agg.items())}  # fmt: skip
    tot = {k: sum(agg[x][k] for x in agg) for k in ("codes_equal", "n", "scale_equal", "ns")}
    res["all"] = {"codes_equal": tot["codes_equal"] / tot["n"], "scales_equal": tot["scale_equal"] / tot["ns"]}
    for kind, r in res.items():
        print(f"  {kind:10s} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))
    pipeline.record(cfg, "compare_int3", res, tm.seconds, ours=str(ours), lcd_repo=repo)
