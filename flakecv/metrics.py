"""Scoring, with UNCERTAIN excluded everywhere.

IoU alone is a poor single number here: losing a one-pixel rim off a thin veil
costs a lot of IoU while the boundary is in fact drawn correctly.  So three
numbers are reported together:

  iou          overall agreement
  boundary_f   F-score of boundary points matched within a PHYSICAL tolerance
               (default 0.25 um) - whether the outline is in the right place
  area_error   relative area error, (pred - ref) / ref - the physical one

Conventions
  ``pred``  any array; > 0 is flake.
  ``ref``   an in-memory three-class label (0 / 1 / 255).  A raw disk PNG
            (0 / 128 / 255) is refused rather than silently mis-scored.
  Pixels where ``ref == UNCERTAIN`` are removed from both pred and ref before
  anything is counted.  An empty union gives NaN, not 0 or 1.

Ported from ``eval/metrics.py``; arithmetic unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import BACKGROUND, FLAKE, UNCERTAIN
from .labels import validate


def _check(pred, ref):
    ref = validate(ref, "ref")
    pred = np.asarray(pred)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    return pred, ref


def _valid(ref):
    return ref != UNCERTAIN


def iou(pred, ref) -> float:
    pred, ref = _check(pred, ref)
    v = _valid(ref)
    p = (pred > 0) & v
    r = (ref == FLAKE) & v
    u = (p | r).sum()
    return float((p & r).sum()) / u if u else float("nan")


def dice(pred, ref) -> float:
    pred, ref = _check(pred, ref)
    v = _valid(ref)
    p = (pred > 0) & v
    r = (ref == FLAKE) & v
    s = p.sum() + r.sum()
    return 2.0 * float((p & r).sum()) / s if s else float("nan")


def area_error(pred, ref, px_um):
    """Returns (relative_error, area_pred_um2, area_ref_um2)."""
    pred, ref = _check(pred, ref)
    v = _valid(ref)
    p = float(((pred > 0) & v).sum()) * px_um ** 2
    r = float(((ref == FLAKE) & v).sum()) * px_um ** 2
    return (p - r) / r if r else float("nan"), p, r


def _edge(m):
    return cv2.morphologyEx(np.asarray(m, np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8))


def boundary_f(pred, ref, px_um, tol_um=0.25) -> float:
    """F-score of boundary points matched within ``tol_um`` micrometres.

    Boundary = 3x3 morphological gradient of each mask; boundary points that
    fall on UNCERTAIN pixels are dropped from both sides.  The tolerance is
    converted to pixels as max(1, round(tol_um / px_um)).
    NaN if either side has no boundary left.
    """
    pred, ref = _check(pred, ref)
    v = _valid(ref)
    pe = _edge((pred > 0).astype(np.uint8)) & v
    re = _edge((ref == FLAKE).astype(np.uint8)) & v
    if pe.sum() == 0 or re.sum() == 0:
        return float("nan")
    tol = max(1, int(round(tol_um / px_um)))
    dt_r = cv2.distanceTransform((re == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dt_p = cv2.distanceTransform((pe == 0).astype(np.uint8), cv2.DIST_L2, 5)
    prec = float((dt_r[pe > 0] <= tol).mean())
    rec = float((dt_p[re > 0] <= tol).mean())
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def score(pred, ref, px_um, tol_um=0.25) -> dict:
    ae, pa, ra = area_error(pred, ref, px_um)
    ref = np.asarray(ref)
    return dict(iou=iou(pred, ref), dice=dice(pred, ref),
                boundary_f=boundary_f(pred, ref, px_um, tol_um),
                area_error=ae, area_pred_um2=pa, area_ref_um2=ra,
                uncertain_frac=float((ref == UNCERTAIN).mean()))


def agreement(ref_a, ref_b, px_um, tol_um=0.25) -> dict:
    """Inter-annotator agreement: the ceiling of any accuracy curve.

    Pixels uncertain in EITHER labelling are excluded, so the two annotators
    are compared only where both committed.
    """
    ref_a = validate(ref_a, "ref_a")
    ref_b = validate(ref_b, "ref_b")
    v = (ref_a != UNCERTAIN) & (ref_b != UNCERTAIN)
    a = (ref_a == FLAKE) & v
    b = (ref_b == FLAKE) & v
    u = (a | b).sum()
    out = dict(iou=float((a & b).sum()) / u if u else float("nan"),
               both_committed=float(v.mean()))
    fake = np.where(v, np.where(b, FLAKE, BACKGROUND), UNCERTAIN).astype(np.uint8)
    out["boundary_f"] = boundary_f(a.astype(np.uint8), fake, px_um, tol_um)
    return out
