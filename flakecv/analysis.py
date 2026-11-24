"""One frame, end to end - shared by the CLI and the app so both compute
exactly the same thing."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import io, metrics as M, model as Mdl, rules


def rules_mask(img01, px_um):
    m, _, _ = rules.segment(img01, px_um)
    return m


def scores(mask, label, px_um):
    return None if label is None else M.score(mask, label, px_um)


def analyse(clf, meta, img01, px_um, label=None, prob=None, **overrides):
    """-> dict(prob, model, rules, scores={'model':..,'rules':..} or {})."""
    if prob is None:
        prob = Mdl.probability_image(clf, meta, img01, px_um)
    mm = Mdl.finish_image(meta, prob, img01, px_um, **overrides)
    rm = rules_mask(img01, px_um)
    sc = {} if label is None else dict(model=M.score(mm, label, px_um), rules=M.score(rm, label, px_um))
    return dict(prob=prob, model=mm, rules=rm, scores=sc)


def load_optical(paths) -> dict:
    """{name: BGR uint8} from image files."""
    import cv2
    return {Path(p).stem: io.imread(p, cv2.IMREAD_COLOR) for p in paths}


def default_optical(data_root) -> list[Path]:
    d = Path(data_root) / "optical"
    return sorted(p for p in d.glob("*") if p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))


def register(img01, px_um, model_mask, optical: dict, om_px=io.OPTICAL_PX_UM["20x"], label=None):
    """OM -> SEM with the model mask as referee.  Adds 'iou_vs_label' if a
    label is given."""
    from . import register as R
    r = R.om_to_sem(optical, img01, px_um, np.asarray(model_mask, np.uint8), om_px=om_px)
    if r is not None and label is not None:
        r["iou_vs_label"] = M.score(r["mask"], label, px_um)["iou"]
    return r


def metrics_rows(name, px_um, res) -> list[dict]:
    """Per-frame metrics as CSV rows (one per method)."""
    out = []
    for method, mask in (("model", res["model"]), ("rules", res["rules"])):
        row = dict(frame=name, method=method, px_um=px_um,
                   area_um2=float(np.asarray(mask > 0).sum()) * px_um * px_um)
        row.update(res["scores"].get(method, {}))
        out.append(row)
    return out
