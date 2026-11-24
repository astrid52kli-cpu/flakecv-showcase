"""Step-3 candidate detectors on synthetic flakes with known defects, and the
annotation scorer.  (No real crack/bubble labels exist yet.)"""
import math

import cv2
import numpy as np
import pytest

from flakecv import defects as D


def _flake(px_um=0.05, H=400, W=480, seed=0):
    """Flake with a thickness STEP across it (crack/fold) and three dark spots."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 0.20, np.float32)
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (60, 60), (420, 340), 1, -1)
    img[mask > 0] = 0.45
    yy, xx = np.mgrid[0:H, 0:W]
    thick = (mask > 0) & (yy > 0.25 * (xx - 60) + 140)     # step along a slanted line
    img[thick] = 0.55
    spots = [(150, 110), (330, 120), (250, 300)]
    for x, y in spots:
        cv2.circle(img, (x, y), 5, 0.30, -1)
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return np.clip(img, 0, 1), mask, px_um, spots


def test_crack_chord_follows_the_step():
    img, mask, px, _ = _flake()
    res = D.detect(img, mask, px, exclude_metal=False)
    assert res["cracks"], "no crack candidate found"
    c = res["cracks"][0]
    (x0, y0), (x1, y1) = c["p0"], c["p1"]
    # both ends near the true step line y = 0.25 (x - 60) + 140
    for x, y in ((x0, y0), (x1, y1)):
        assert abs(y - (0.25 * (x - 60) + 140)) < 0.6 / px
    assert c["len_um"] > 10


def test_bubbles_find_the_spots():
    img, mask, px, spots = _flake()
    res = D.detect(img, mask, px, exclude_metal=False)
    found = [(b["x"], b["y"]) for b in res["bubbles"][:6]]
    for sx, sy in spots:
        assert min(math.hypot(sx - x, sy - y) for x, y in found) < 0.4 / px


def test_nothing_outside_the_mask():
    img, mask, px, _ = _flake()
    res = D.detect(img, mask, px, exclude_metal=False)
    for b in res["bubbles"]:
        assert mask[int(b["y"]), int(b["x"])] == 1
    assert not (res["line_map"] & (1 - mask)).any()


def test_empty_mask():
    img, mask, px, _ = _flake()
    res = D.detect(img, np.zeros_like(mask), px)
    assert res["cracks"] == [] and res["bubbles"] == [] and res["lines"] == []


def test_parameters_are_physical():
    """Same scene at half the pixel size: the main crack is still found at the
    same physical place."""
    img, mask, px, _ = _flake()
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    bm = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    r1 = D.detect(img, mask, px, exclude_metal=False)["cracks"][0]
    r2 = D.detect(big, bm, px / 2, exclude_metal=False)["cracks"][0]
    assert abs(r1["len_um"] - r2["len_um"]) < 2.0


def test_annotation_roundtrip_and_scoring(tmp_path):
    img, mask, px, spots = _flake()
    res = D.detect(img, mask, px, exclude_metal=False)
    p = tmp_path / "x.defects.json"
    D.save_annotations(p, "x", px, res["cracks"][:1], res["bubbles"][:3])
    doc = D.load_annotations(p)
    assert doc["frame"] == "x" and len(doc["cracks"]) == 1 and len(doc["bubbles"]) == 3
    s = D.score(doc, doc, px)
    assert s["cracks"]["f"] == 1.0 and s["bubbles"]["f"] == 1.0


def test_score_hand_cases():
    px = 0.05
    ref = dict(cracks=[dict(p0=[0, 0], p1=[100, 0])],
               bubbles=[dict(x=10, y=10), dict(x=50, y=50)])
    pred = dict(cracks=[dict(p0=[0, 4], p1=[100, 4]),          # 0.2 um off: match
                        dict(p0=[0, 60], p1=[100, 60])],       # 3 um off: false positive
                bubbles=[dict(x=12, y=10), dict(x=200, y=200)])
    s = D.score(pred, ref, px, crack_tol_um=0.5, bubble_tol_um=0.5)
    assert s["cracks"]["matched"] == 1 and s["cracks"]["precision"] == 0.5 and s["cracks"]["recall"] == 1.0
    assert s["bubbles"]["matched"] == 1 and s["bubbles"]["recall"] == 0.5
    empty = D.score(dict(cracks=[], bubbles=[]), dict(cracks=[], bubbles=[]), px)
    assert math.isnan(empty["cracks"]["f"])
