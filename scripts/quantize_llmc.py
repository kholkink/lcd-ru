#!/usr/bin/env python
"""INT3 (or W4) with llm-compressor's GPTQ, recipe as in LCD (ref/lcd/code/pipeline.py `run_gptq`): all Linear layers
except lm_head, sym int, group 128, minmax observer, no act-order (the author's checkpoints have `actorder: null`;
llm-compressor 0.14 defaults to "static", so it is switched off explicitly), dampening 0.01, sequential over decoder
layers, max_seq_length 512, calibration texts from a fixed split.

After `oneshot` the integer codes are read back from the model (weight / weight_scale) and stored in our compact
format (src/quant/gptq.py `save_quant`), so loading (`base.kind: gptq-ours` + path) and comparison scripts are shared.

    python scripts/quantize_llmc.py --config configs/p2/qwen3b_llmc.yaml [llmc.qac=false]
"""

import json
import time

import _common
import torch

from src.data import load_texts
from src.models import load_hf, load_tokenizer
from src.pipeline import RUNS, record
from src.quant.gptq import QuantConfig, save_quant
from src.utils import device


def decoder_layer_class(model) -> str:
    return type(model.model.layers[0]).__name__


def extract_state(model, bits: int, group_size: int) -> dict:
    """codes (0 .. 2^b-1, zero = 2^(b-1) for sym) and scales of every quantized Linear."""
    state = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or not hasattr(mod, "weight_scale"):
            continue
        W = mod.weight.detach().float().cpu()
        scale = mod.weight_scale.detach().float().cpu()
        zp = getattr(mod, "weight_zero_point", None)
        zp = zp.detach().float().cpu() if zp is not None else torch.zeros_like(scale)
        s_full = scale.repeat_interleave(group_size, dim=1)[:, : W.shape[1]]
        z_full = zp.repeat_interleave(group_size, dim=1)[:, : W.shape[1]]
        q = torch.round(W / s_full) + z_full  # signed codes (sym: -2^(b-1) .. 2^(b-1)-1)
        off = 2 ** (bits - 1)
        codes = q + off
        if codes.min() < 0 or codes.max() > 2**bits - 1:
            raise ValueError(f"{name}: codes out of range [{codes.min()}, {codes.max()}]")
        err = ((codes - off - z_full) * s_full - W).abs().max().item()
        if err > 1e-2 * s_full.max().item():
            raise ValueError(f"{name}: weights are not on the grid (max err {err})")
        state[name] = {"codes": codes.to(torch.uint8), "scale": scale, "zero": torch.full_like(scale, off) + zp}
    return state


if __name__ == "__main__":
    cfg = _common.config_from_cli(__doc__)
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    lc = cfg["llmc"]
    bits, gs = int(lc.get("bits", 3)), int(lc.get("group_size", 128))
    out = RUNS / lc["name"]
    tok = load_tokenizer(cfg["model"])
    model = load_hf(cfg["model"], device())
    texts = load_texts(lc.get("calib", "calib_en_128"), lc.get("calib_n"))
    scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
        num_bits=bits, type="int", symmetric=bool(lc.get("sym", True)), group_size=gs, strategy="group",
        observer="minmax", actorder=None))  # fmt: skip
    recipe = GPTQModifier(targets="Linear", ignore=["lm_head"], config_groups={"group_0": scheme},
                          dampening_frac=float(lc.get("damp", 0.01)), actorder=None)  # fmt: skip
    t0 = time.time()
    oneshot(model=model, tokenizer=tok, dataset=Dataset.from_dict({"text": texts}), recipe=recipe,
            max_seq_length=int(lc.get("max_seq_length", 512)), num_calibration_samples=len(texts),
            sequential_targets=[decoder_layer_class(model)],
            quantization_aware_calibration=bool(lc.get("qac", True)))  # fmt: skip
    seconds = time.time() - t0
    state = extract_state(model, bits, gs)
    qcfg = QuantConfig(bits=bits, group_size=gs, sym=bool(lc.get("sym", True)), percdamp=float(lc.get("damp", 0.01)),
                       group_params="static", scale_bf16=True)  # fmt: skip
    import llmcompressor

    save_quant(state, qcfg, out, meta={"model": cfg["model"], "tool": f"llmcompressor {llmcompressor.__version__}",
                                       "calib": lc.get("calib"), "qac": lc.get("qac", True), "seconds": seconds})  # fmt: skip
    print(f"llm-compressor GPTQ: {len(state)} modules in {seconds:.0f}s -> {out}")
    record(cfg, "quantize_llmc", {"n_modules": len(state), "out": str(out)}, seconds)
    (out / "llmc_recipe.json").write_text(json.dumps({"recipe": str(recipe), "llmc": lc}, indent=1, default=str))
