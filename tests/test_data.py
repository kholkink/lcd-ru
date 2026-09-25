import ast

import pytest

from src import data


def test_clean_matches_lcd_filter():
    assert data.clean("слишком коротко") is None
    t = "  a\n\nb  " + " w" * 3000
    out = data.clean(t)
    assert out.split()[:2] == ["a", "b"] and len(out.split()) == 2000 and "\n" not in out


def test_hash_selection_is_deterministic_and_disjoint(monkeypatch):
    recs = [{"url": f"http://x/{i}", "text": "слово " * 40} for i in range(400)]
    monkeypatch.setattr(data, "_stream", lambda spec: iter(recs))
    a = data.select_hash({"bucket": 0, "n": 10})
    b = data.select_hash({"bucket": 0, "n": 10})
    c = data.select_hash({"bucket": 1, "n": 10})
    assert [d["id"] for d in a] == [d["id"] for d in b]
    assert not {d["id"] for d in a} & {d["id"] for d in c}


def _lcd_load_texts():
    """LCD's own `load_texts` (ref/lcd/code/evaluate.py), executed from the reference source."""
    src = (data.REPO / "ref/lcd/code/evaluate.py").read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_texts"
            or isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "LANGUAGE_SOURCES" for t in n.targets)]  # fmt: skip
    ns = {}
    exec(compile(ast.Module(keep, []), "lcd_evaluate", "exec"), ns)
    return ns["load_texts"]


@pytest.mark.network
def test_lcd_loader_matches_reference_code():
    ref = _lcd_load_texts()("english", 5, seed=43)
    ours = data.select_lcd({**data.SPECS["calib_en_128"], "n": 5})
    assert [d["text"] for d in ours] == ref
