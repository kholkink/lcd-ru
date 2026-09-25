import pytest
import torch

from src import corrections as C

TEXT = "Съешь же ещё этих мягких французских булок, да выпей чаю."


def logits(model, tok, text=TEXT):
    with torch.no_grad():
        return model(input_ids=tok(text, return_tensors="pt").input_ids).logits


@pytest.mark.parametrize("arch", ["qwen", "llama"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_init_is_bitwise_identical(tiny, tok, clone, arch, dtype):
    base = tiny(arch, dtype)
    ref = logits(base, tok)
    m = clone(base)
    names = C.install(m, rank=2, langs=["ru", "en"])
    assert len(names) == 3 * 7 and not any("lm_head" in n or "embed" in n for n in names)
    C.set_lang(m, "ru")
    assert torch.equal(logits(m, tok), ref)  # A = 0 -> delta is exactly 0
    C.set_lang(m, None)
    assert torch.equal(logits(m, tok), ref)


def test_active_none_ignores_trained_params_and_switching_is_per_language(tiny, tok, clone):
    base = tiny()
    ref = logits(base, tok)
    m = clone(base)
    C.install(m, rank=2, langs=["ru", "en"])
    en_before = {k: v.clone() for k, v in C.state(m, ["en"])["tensors"].items()}
    params = C.trainable(m, "ru")
    C.set_lang(m, "ru")
    opt = torch.optim.SGD(params, lr=0.1)
    ids = tok(TEXT, return_tensors="pt").input_ids
    m(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    assert not torch.equal(logits(m, tok), ref)  # ru correction is now non-zero
    C.set_lang(m, None)
    assert torch.equal(logits(m, tok), ref)  # off -> base model again
    C.set_lang(m, "en")
    assert torch.equal(logits(m, tok), ref)  # en slot untouched (A_en = 0)
    for k, v in C.state(m, ["en"])["tensors"].items():
        assert torch.equal(v, en_before[k])
    assert all(p.grad is None for n, p in m.named_parameters() if ".en" in n or "base" in n or "embed" in n)


def test_save_load_remove_roundtrip(tiny, tok, clone, tmp_path):
    base = tiny()
    ref = logits(base, tok)
    m = clone(base)
    C.install(m, rank=2, langs=["ru"])
    for mod in C.modules(m).values():
        mod.A["ru"].data.normal_(0, 0.1)
    C.set_lang(m, "ru")
    out = logits(m, tok)
    C.save(m, tmp_path / "c.pt", meta={"x": 1})
    m2 = clone(base)
    C.install(m2, rank=2, langs=["ru"])
    assert C.load(m2, tmp_path / "c.pt") == {"x": 1}
    C.set_lang(m2, "ru")
    assert torch.equal(logits(m2, tok), out)
    C.remove(m2)
    assert not C.modules(m2) and torch.equal(logits(m2, tok), ref)


def test_param_count_and_placements(tiny):
    m = tiny(layers=4)
    C.install(m, rank=2, langs=["ru"])
    # q,o: 64+64; k,v: 64+32; gate,up,down: 64+128  -> 1024 per rank per layer
    assert C.n_params(m, "ru") == 2 * 1024 * 4
    for placement, n in [("o_down", 8), ("bottom_half", 14), ("top_half", 14), ("blocks:0,3", 14)]:
        mm = tiny(layers=4)
        assert len(C.install(mm, 2, ["ru"], placement=placement)) == n


def test_init_std_is_std(tiny):
    m = tiny(layers=4)
    torch.manual_seed(0)
    C.install(m, rank=16, langs=["ru"])
    b = torch.cat([mod.B["ru"].flatten() for mod in C.modules(m).values()])
    assert abs(b.std().item() - 0.01) < 5e-4
    assert all(mod.A["ru"].abs().sum() == 0 for mod in C.modules(m).values())
