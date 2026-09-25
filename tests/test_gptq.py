import pytest
import torch

from src.quant.gptq import GPTQ, QuantConfig, apply_quant, grid_params, quantize_model, save_quant


def calib(tok, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, len(tok), (1, int(torch.randint(20, 60, (1,), generator=g))), generator=g)
            for _ in range(n)]  # fmt: skip


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("bits", [3, 4])
def test_at_most_2_pow_bits_values_per_group(tiny, tok, sym, bits):
    m = tiny(dtype=torch.float32)
    cfg = QuantConfig(bits=bits, group_size=32, sym=sym, scale_bf16=False)
    state = quantize_model(m, calib(tok), cfg, log=None)
    assert len(state) == 3 * 7
    for name, s in state.items():
        w = m.get_submodule(name).weight.data
        for g in range(w.shape[1] // 32):
            blk = w[:, g * 32 : (g + 1) * 32]
            for row in blk[:8]:
                assert row.unique().numel() <= 2**bits
        assert s["codes"].max() <= 2**bits - 1
        if sym:  # symmetric grid: values scale * {-2^(b-1) .. 2^(b-1)-1}
            assert torch.all(s["zero"] == 2 ** (bits - 1))


def test_static_scales_match_lcd_rule():
    torch.manual_seed(0)
    w = torch.randn(16, 128).to(torch.bfloat16).float()
    scale, zero = grid_params(w, 3, True, scale_bf16=True)
    assert torch.equal(scale, (w.abs().amax(1) / 3.5).to(torch.bfloat16).float())
    assert torch.all(zero == 4)


def test_gptq_beats_rtn_on_layer_output():
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 64, bias=False)
    x = torch.randn(512, 256) @ torch.randn(256, 256) * 0.3  # correlated inputs, where GPTQ helps
    y = lin(x)
    g = GPTQ(lin)
    g.add_batch(x)
    w0 = lin.weight.data.clone()
    g.quantize(QuantConfig(bits=3, group_size=128, sym=True, scale_bf16=False))
    err_gptq = (lin(x) - y).norm() / y.norm()
    rtn = torch.zeros_like(w0)
    for c in range(0, 256, 128):
        s, z = grid_params(w0[:, c : c + 128], 3, True)
        rtn[:, c : c + 128] = s[:, None] * (torch.clamp(torch.round(w0[:, c : c + 128] / s[:, None]) + z[:, None], 0, 7) - z[:, None])
    err_rtn = (x @ rtn.t() - y).norm() / y.norm()
    assert err_gptq < 0.8 * err_rtn


@pytest.mark.parametrize("act_order", [False, True])
def test_save_apply_roundtrip(tiny, tok, tmp_path, act_order):
    m = tiny(dtype=torch.bfloat16)
    fresh = tiny(dtype=torch.bfloat16)
    cfg = QuantConfig(bits=3, group_size=32, act_order=act_order)
    state = quantize_model(m, calib(tok), cfg, log=None)
    save_quant(state, cfg, tmp_path / "q")
    apply_quant(fresh, tmp_path / "q")
    for name in state:
        assert torch.equal(fresh.get_submodule(name).weight, m.get_submodule(name).weight), name
    assert torch.equal(fresh.lm_head.weight, m.lm_head.weight)  # untouched


def test_true_sequential_runs(tiny, tok):
    m = tiny()
    state = quantize_model(m, calib(tok, n=3), QuantConfig(group_size=32, true_sequential=True), log=None)
    assert len(state) == 3 * 7
