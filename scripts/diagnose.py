#!/usr/bin/env python
"""Phase 2 diagnostics (src/eval/diagnose.py) for several conditions of one base model against the FP16 teacher.

cfg.diag.conditions: [{cond, corrections: <path or null>, slot: <lang or null>}], cfg.diag.texts: [{split, n, max_len}].
One record per (condition, text set); also written to <run>/diag_<cond>_<split>.json.
"""

import json

import _common

from src import corrections as C
from src import pipeline
from src.data import load_texts
from src.eval.diagnose import block_diagnostics
from src.models import load_hf, load_tokenizer
from src.utils import Timer, device, expand_env

if __name__ == "__main__":
    cfg = _common.config_from_cli(__doc__)
    d = pipeline.run_dir(cfg)
    dg = cfg["diag"]
    tok = load_tokenizer(cfg["model"])
    teacher = load_hf(cfg["model"], device())
    student = pipeline.get_base(cfg["model"], pipeline.base_spec(cfg))
    cc = cfg.get("corrections", {})
    for c in dg["conditions"]:
        if C.modules(student):
            C.remove(student)
        if c.get("corrections"):
            C.install(student, int(cc.get("rank", 2)), list(cc.get("langs", ["ru", "en"])), cc.get("placement", "all"))
            C.load(student, expand_env(c["corrections"]))
        for t in dg["texts"]:
            slot = c.get("slot") if c.get("corrections") else None
            if C.modules(student):
                C.set_lang(student, slot)
            with Timer() as tm:
                r = block_diagnostics(student, teacher, tok, load_texts(t["split"], t.get("n")), int(t.get("max_len", 512)),
                                      linears=bool(dg.get("linears", True)))  # fmt: skip
            (d / f"diag_{c['cond']}_{t['split']}.json").write_text(json.dumps(r))
            e = r["e_acc"]
            print(f"  [{c['cond']}|{slot}] {t['split']}: e_acc first/mid/last {e[0]:.3f}/{e[len(e) // 2]:.3f}/{e[-1]:.3f}; "
                  f"e_locT mid {r['e_locT'][len(e) // 2]:.3f} ({tm.seconds:.0f}s)", flush=True)  # fmt: skip
            pipeline.record(cfg, "diag", r, tm.seconds, cond=c["cond"], split=t["split"], slot=slot)
