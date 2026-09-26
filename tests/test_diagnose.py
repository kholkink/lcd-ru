import torch

from src import corrections as C
from src.eval.diagnose import block_diagnostics

TEXTS = ["Первый текст для диагностики блоков.", "Second text, a bit longer, for the block diagnostics test here."]


def test_identical_models_have_zero_errors(tiny, tok, clone):
    t = tiny()
    s = clone(t)
    C.install(s, 2, ["ru"])
    C.set_lang(s, "ru")
    r = block_diagnostics(s, t, tok, TEXTS, outlier_factor=0)
    assert max(r["e_acc"] + r["e_locT"] + r["e_locS"]) < 1e-10
    assert len(r["lin_e2e"]) == 3 * 7 and max(r["lin_e2e"].values()) < 1e-6 and max(r["lin_local"].values()) < 1e-6


def test_local_vs_accumulated(tiny, tok, clone):
    t = tiny()
    s = clone(t)
    with torch.no_grad():  # perturb only block 0: later blocks have no local error, only propagated error
        for p in s.model.layers[0].parameters():
            if p.dim() == 2:
                p.add_(0.1 * torch.randn_like(p))
    r = block_diagnostics(s, t, tok, TEXTS, outlier_factor=0)
    assert r["e_locT"][0] > 0 and r["e_acc"][0] > 0
    assert abs(r["e_acc"][0] - r["e_locT"][0]) < 1e-9  # block 0: same input for both models
    assert max(r["e_locT"][1:]) < 1e-10 and min(r["e_acc"][1:]) > 0  # clean blocks: error only arrives from below
    assert max(r["e_locS"][1:]) < 1e-10
    assert max(v for k, v in r["lin_local"].items() if ".layers.0." not in k) < 1e-6
    assert min(v for k, v in r["lin_e2e"].items() if ".layers.1." in k) > 0
