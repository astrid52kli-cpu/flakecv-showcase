"""``flakecv demo`` - the acceptance test.

Runs the pipeline on the committed sample data (data/frames, data/labels,
data/optical) and checks every number against :mod:`flakecv.baseline`
(tolerance +-0.005 on means).  Nothing is read from anywhere else, nothing
is trained in advance.

    quick (default, ~10 min)  rules baseline; leave-one-flake-out at n=5
                              (all post-processing variants); registration
                              with the rules mask and with the model mask
    --full (~45 min)          the above plus the whole learning curve, n=1..5,
                              both feature sets

Exit code 0 = everything within tolerance, 1 = something moved.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import baseline as B
from . import curve as C
from . import io, metrics as M, postproc, register as R, rules


class Report:
    def __init__(self, log=print):
        self.rows, self.log = [], log

    def check(self, what, got, want, tol=B.TOL):
        ok = abs(got - want) <= tol + 1e-9
        self.rows.append(dict(what=what, got=round(float(got), 4), want=want, ok=bool(ok)))
        self.log(f"  {'ok  ' if ok else 'FAIL'}  {what:<46} {got:6.3f}   (expected {want:.3f})")
        return ok

    def info(self, what, got):
        self.log(f"        {what:<46} {got}")

    @property
    def passed(self):
        return all(r["ok"] for r in self.rows)


def run(data_root, full=False, log=print, out=None) -> int:
    t0 = time.time()
    rep = Report(log)
    cur = B.CURRENT
    log(f"flakecv demo  ({'full' if full else 'quick'})   data: {data_root}")
    missing = [f for f in B.FRAMES if not io.frame_paths(data_root, f)["label"].is_file()]
    if missing:
        log("\nThe regression check needs all six labelled frames; this copy of the data "
            f"does not have {', '.join(missing)}.\n(The public repository ships three sample "
            "frames only. The expected numbers are listed in flakecv/baseline.py.)")
        return 2

    # ---- load: the six labelled frames, features for both sets ----------
    log("\n[1/4] loading frames and computing features")
    items = C.load_items(data_root, B.FRAMES,
                         feature_sets=("photometric", "framing") if full else ("framing",),
                         log=lambda *a: None)
    for it in items:
        log(f"        {it['name']:<16} {it['px'] * 1000:5.1f} nm/px  {it['hv_kV']:4.0f} kV")

    # ---- rules baseline ---------------------------------------------------
    log("\n[2/4] hand-tuned rules baseline (no training)")
    rs = [M.score(rules.segment(it["img"], it["px"])[0], it["lab"], it["px"]) for it in items]
    for it, s, w in zip(items, rs, cur["rules_iou"]):
        rep.check(f"rules IoU {it['name']}", s["iou"], w)
    rep.check("rules IoU mean", np.mean([s["iou"] for s in rs]), cur["rules_iou_mean"])
    rep.check("rules boundary F mean", np.mean([s["boundary_f"] for s in rs]), cur["rules_boundary_f_mean"])

    # ---- learning curve ---------------------------------------------------
    sizes = (1, 2, 3, 4, 5) if full else (5,)
    log(f"\n[3/4] leave-one-flake-out, n = {', '.join(map(str, sizes))}"
        + ("" if full else "   (--full for the whole curve)"))
    rows = C.run(items, "framing", sizes=sizes, log=lambda *a: None)
    if full:
        rows += C.run(items, "photometric", pipelines={"base": C.PIPELINES["base"]},
                      sizes=sizes, log=lambda *a: None)
    S, SB = C.summarise(rows), C.summarise(rows, "boundary_f")
    name = {"photometric": ("photometric", "base"), "framing": ("framing", "base"),
            "framing+snap": ("framing", "snap")}
    for lab, key in name.items():
        if key not in S:
            continue
        for n in sizes:
            rep.check(f"curve {lab:<13} n={n}  ({S[key][n][1]} runs)", S[key][n][0], cur["curve"][lab][n - 1])
    rep.check("full post-processing n=5 IoU", S[("framing", "full")][5][0], cur["full_n5_iou"])
    rep.check("full post-processing n=5 boundary F", SB[("framing", "full")][5][0], cur["full_n5_boundary_f"])

    # ---- registration -----------------------------------------------------
    log("\n[4/4] optical -> SEM registration, three 20 kV frames (slow: stage-1 search)")
    om = {p.stem: io.imread(p, 1) for p in sorted((Path(data_root) / "optical").glob("*.bmp"))}
    if not om:
        log("        no optical images in data/optical - skipped")
    else:
        # the model mask: the same n=5 leave-one-out fits as above, full post-processing
        from . import model
        got = {"rules": [], "model": []}
        for te in range(3):
            it = items[te]
            # registration was measured on the full-depth TIFF (1-99 % stretch,
            # no 8-bit step); the rules mask is sensitive to that difference
            raw = sorted((Path(data_root) / "raw").glob(it["name"] + ".tif*"))
            if raw:
                g = io.normalize(io.read_sem_tiff(raw[0]).raw)
            else:
                g = it["img"]
                rep.info(f"{it['name']}", "no raw TIFF in data/raw: using the PNG (rules-target number will differ)")
            clf = model.fit([(items[j]["X"]["framing"], items[j]["lab"]) for j in range(6) if j != te], seed=0)
            p = model.predict_proba(clf, it["X"]["framing"])
            targets = dict(rules=rules.segment(g, it["px"])[0],
                           model=postproc.finish(p, it["img"], it["px"], central=True, fill=True))
            for k, tgt in targets.items():
                r = R.om_to_sem(om, g, it["px"], tgt)
                v = M.score(r["mask"], it["lab"], it["px"])["iou"] if r else 0.0
                got[k].append(v)
                rep.info(f"OM->SEM {it['name']} ({k} target)", f"{v:.3f}  {r['om'] if r else '-'}  {r['stage'] if r else ''}")
        rep.check("OM->SEM IoU, rules-mask target, mean", np.mean(got["rules"]), cur["register_rules_target_mean"])
        rep.check("OM->SEM IoU, model-mask target, mean", np.mean(got["model"]), cur["register_model_target_mean"])

    n_fail = sum(not r["ok"] for r in rep.rows)
    log(f"\n{len(rep.rows) - n_fail}/{len(rep.rows)} within +-{B.TOL}   ({time.time() - t0:.0f}s)")
    log("PASS" if rep.passed else "FAIL - a number moved by more than the tolerance; something changed")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(dict(full=full, passed=rep.passed, checks=rep.rows), indent=1))
    return 0 if rep.passed else 1
