import torch

from src import corrections as C
from src.objectives import make
from src.train import doc_schedule, lcd_lr_lambda, train_e2e
from src.utils import set_seed

DOCS = [f"Документ номер {i}: " + "слово " * (10 + 3 * i) for i in range(5)]
CFG = {"steps": 8, "lr": 5e-3, "warmup_ratio": 0.25, "max_len": 64, "log_every": 4, "ckpt_every": 0, "seed": 0}


def student(tiny, seed=0):
    set_seed(seed)
    m = tiny()
    set_seed(seed + 100)
    C.install(m, 2, ["ru"])
    return m


def corr(m):
    return {k: v.clone() for k, v in C.state(m, ["ru"])["tensors"].items()}


def test_lr_schedule_is_lcds():
    f = lcd_lr_lambda(500, 50)
    assert f(0) == 0 and f(25) == 0.5 and f(50) == 1.0 and abs(f(500)) < 1e-12
    assert doc_schedule(3, 7, "fixed", 0) == [0, 1, 2, 0, 1, 2, 0]
    s = doc_schedule(3, 6, "seeded", 1)
    assert sorted(s[:3]) == [0, 1, 2] and s == doc_schedule(3, 6, "seeded", 1)


def test_deterministic_with_seed(tiny, tok, tmp_path):
    runs = []
    for k in range(2):
        m = student(tiny)
        train_e2e(m, tok, DOCS, make("O1"), CFG, "ru", tmp_path / f"r{k}", log=lambda *_: None)
        runs.append(corr(m))
    assert all(torch.equal(runs[0][k], runs[1][k]) for k in runs[0])


def test_resume_equals_uninterrupted(tiny, tok, tmp_path):
    full = student(tiny)
    train_e2e(full, tok, DOCS, make("O1"), CFG, "ru", tmp_path / "full", log=lambda *_: None)
    part = student(tiny)
    train_e2e(part, tok, DOCS, make("O1"), {**CFG, "steps": 8, "ckpt_every": 3}, "ru", tmp_path / "part",
              log=lambda *_: None)  # fmt: skip
    # simulate a crash after the step-6 checkpoint: fresh model, same out dir -> resumes at 6
    ck = torch.load(tmp_path / "part" / "ckpt.pt", weights_only=False)
    assert ck["step"] == 6
    again = student(tiny)
    r = train_e2e(again, tok, DOCS, make("O1"), {**CFG, "ckpt_every": 3}, "ru", tmp_path / "part", log=lambda *_: None)
    assert r["resumed_from"] == 6
    a, b = corr(full), corr(again)
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def test_o1_and_o3_reduce_their_loss(tiny, tok, clone, tmp_path):
    teacher = tiny(seed=0)
    s1 = clone(teacher)
    with torch.no_grad():  # a "quantized" student: perturbed weights
        for n, p in s1.named_parameters():
            if ".layers." in n and p.dim() == 2:
                p.add_(0.05 * torch.randn_like(p))
    s3 = clone(s1)
    C.install(s1, 4, ["ru"])
    C.install(s3, 4, ["ru"])
    cfg = {**CFG, "steps": 40, "lr": 2e-2, "log_every": 10, "doc_order": "fixed"}
    r1 = train_e2e(s1, tok, DOCS[:1], make("O1"), cfg, "ru", tmp_path / "o1", log=lambda *_: None)
    r3 = train_e2e(s3, tok, DOCS[:1], make("O3", lam=1.0), cfg, "ru", tmp_path / "o3", teacher=teacher,
                   log=lambda *_: None)  # fmt: skip
    h1, h3 = r1["history"]["loss"], r3["history"]["loss"]
    assert h1[-1]["loss"] < h1[0]["loss"]
    assert h3[-1]["kl"] < h3[0]["kl"] and h3[-1]["hid"] < h3[0]["hid"]
