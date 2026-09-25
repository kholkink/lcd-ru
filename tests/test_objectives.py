import torch

from src import corrections as C
from src.objectives import O3, capture_blocks, kd_loss, lm_loss, normalized_err, position_mask

TEXTS = ["Короткий текст.", "Это заметно более длинный текст, чтобы в батче появился паддинг справа и маска работала."]


def test_hidden_states_aligned_under_padding(tiny, tok):
    m = tiny()
    enc = tok(TEXTS, return_tensors="pt", padding=True, padding_side="right")
    hs_b, hs_1 = [], []
    with torch.no_grad(), capture_blocks(m, hs_b):
        m(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    n = int(enc.attention_mask[0].sum())
    with torch.no_grad(), capture_blocks(m, hs_1):
        m(input_ids=enc.input_ids[:1, :n])
    assert len(hs_b) == 3
    for hb, h1 in zip(hs_b, hs_1):
        assert torch.allclose(hb[0, :n], h1[0], atol=1e-5)
    masks = position_mask(hs_b, enc.attention_mask, skip_first=1, outlier_factor=0)
    assert not masks[0][0, 0] and not masks[0][0, n:].any() and masks[0][0, 1:n].all()


def test_outlier_positions_masked():
    h = torch.ones(1, 10, 8)
    h[0, 3] *= 100.0
    m = position_mask([h], None, skip_first=1, outlier_factor=10.0)[0]
    assert m.tolist() == [[False, True, True, False, True, True, True, True, True, True]]


def test_identical_models_have_zero_losses(tiny, tok, clone):
    t = tiny()
    s = clone(t)
    C.install(s, 2, ["ru"])
    C.set_lang(s, "ru")
    ids = tok(TEXTS[1], return_tensors="pt").input_ids
    hs_t, hs_s = [], []
    with torch.no_grad(), capture_blocks(t, hs_t):
        lt = t(input_ids=ids).logits
    with capture_blocks(s, hs_s):
        ls = s(input_ids=ids).logits
    loss, logs = O3(lam=1.0)(ids, None, ls, list(hs_s), lt, list(hs_t))
    assert logs["kl"] < 1e-6 and logs["hid"] < 1e-10
    assert abs(lm_loss(ls, ids).item() - s(input_ids=ids, labels=ids).loss.item()) < 1e-5
    loss.backward()  # graph reaches the corrections
    assert all(mod.A["ru"].grad is not None for mod in C.modules(s).values())


def test_kd_and_err_positive_for_different_models(tiny, tok):
    a, b = tiny(seed=0), tiny(seed=1)
    ids = tok(TEXTS[1], return_tensors="pt").input_ids
    ha, hb = [], []
    with torch.no_grad(), capture_blocks(a, ha):
        la = a(input_ids=ids).logits
    with torch.no_grad(), capture_blocks(b, hb):
        lb = b(input_ids=ids).logits
    assert kd_loss(lb, la).item() > 0
    m = torch.ones(1, ids.shape[1], dtype=torch.bool)
    assert normalized_err(hb[0], ha[0], m).item() > 0
