"""Per-pixel features.  Every length is in micrometres.

Two feature sets:

``"photometric"`` (10) - from ``eval/features.py``.  Kept because the learning
    curve's "10 photometric only" row is measured against it.
``"framing"`` (12, the default) - the 10 plus the framing prior from
    ``eval/features3.py``:

    dist_border  Euclidean distance to the frame edge                    [um]
    geo_border   geodesic distance from the frame edge travelling through
                 the foreground only (MCP_Geometric; cost 1 in foreground,
                 1e4 outside; foreground = above the first of three Otsu
                 thresholds)                                             [um]

The operator always centres the flake, so it never runs off the frame.  A
contact bar is a short geodesic hop from the edge; a centred flake is a long
one, even when the two touch.  This is worth about +0.17 IoU.  It has to be a
feature, not a rule: the flake touches the contacts, so the foreground is one
component that does reach the edge - dropping border-touching components gives
IoU 0.000 on all six frames, and inward flood fills / watersheds leak through
the flake edge (flake vs gold is only 0-3.6 sigma).

Arithmetic is unchanged from the source; the parameters are collected in
:data:`PARAMS` so a saved model can record and check them.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.graph import MCP_Geometric

from .substrate import substrate_stats

PHOTOMETRIC = ["z_substrate", "blur_fine", "blur_coarse", "grad_fine", "grad_coarse",
               "local_std", "rank", "dist_to_bright", "dist_to_line", "line_density"]
FRAMING = PHOTOMETRIC + ["dist_border", "geo_border"]
SETS = {"photometric": PHOTOMETRIC, "framing": FRAMING}
DEFAULT_SET = "framing"

PARAMS = dict(
    gauss_um=(0.15, 0.6),          # blur_* and grad_* scales
    local_um=0.5,                  # local_std window, bright opening
    bright_percentile=97,          # dist_to_bright: the metal
    line_blur_um=0.12, line_min_um=4.0, line_thickness_px=2,
    z_smooth_um=0.06, z_window_k=3.0,
    geo_otsu_classes=3, geo_cost_fg=1.0, geo_cost_bg=1e4,
    geo_start_step_px=4, geo_clip_percentile=99.9,
)


def names(feature_set: str = DEFAULT_SET) -> list[str]:
    try:
        return list(SETS[feature_set])
    except KeyError:
        raise ValueError(f"unknown feature set {feature_set!r}; use one of {list(SETS)}")


def rank01(img) -> np.ndarray:
    """Rank of each pixel in the frame's own intensity distribution, in [0, 1].

    Tied grey levels get the MEAN of the ranks they span, so the result depends
    only on the image.  (The source used an unstable ``np.argsort`` and gave
    tied pixels arbitrary distinct ranks; on 8-bit frames, where one grey level
    holds ~10^5 pixels, that order changed with the numpy version and the CPU,
    moving IoU by up to 0.04 per frame.  Changed 2026-11-21, numbers
    re-baselined.)  Without ties this equals the source's i / (N - 1).
    """
    flat = np.asarray(img).ravel()
    n = flat.size
    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    end = np.cumsum(cnt)                       # one past the last position
    mid = (end - cnt + end - 1) / 2.0 / max(n - 1, 1)
    return mid[inv].astype(np.float32).reshape(np.shape(img))


def _photometric(img01, px_um):
    img = np.asarray(img01, np.float32)
    sm, lvl, sig = substrate_stats(img, px_um)
    f = [(sm - lvl) / max(sig, 1e-6)]
    for s_um in (0.15, 0.6):
        s = max(1.0, s_um / px_um)
        f.append(cv2.GaussianBlur(img, (0, 0), s))
    for s_um in (0.15, 0.6):
        s = max(1.0, s_um / px_um)
        b = cv2.GaussianBlur(img, (0, 0), s)
        gx = cv2.Scharr(b, cv2.CV_32F, 1, 0); gy = cv2.Scharr(b, cv2.CV_32F, 0, 1)
        f.append(np.hypot(gx, gy))
    k = max(3, int(0.5 / px_um) | 1)
    mu = cv2.blur(img, (k, k)); mu2 = cv2.blur(img * img, (k, k))
    f.append(np.sqrt(np.maximum(mu2 - mu * mu, 0)))
    f.append(rank01(img))
    # distance to the brightest class (the metal) and to long straight edges
    hi = (img > np.percentile(img, 97)).astype(np.uint8)
    hi = cv2.morphologyEx(hi, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    f.append(cv2.distanceTransform((hi == 0).astype(np.uint8), cv2.DIST_L2, 5) * px_um)
    lines = np.zeros(img.shape, np.uint8)
    try:
        g8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        g8 = cv2.GaussianBlur(g8, (0, 0), max(1.0, 0.12 / px_um))
        det = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV).detect(g8)[0]
        if det is not None:
            for l in det.reshape(-1, 4):
                if np.hypot(l[2] - l[0], l[3] - l[1]) * px_um >= 4.0:
                    cv2.line(lines, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), 1, 2)
    except cv2.error:
        pass
    f.append(cv2.distanceTransform((lines == 0).astype(np.uint8), cv2.DIST_L2, 5) * px_um)
    f.append(cv2.blur(lines.astype(np.float32), (k * 3 | 1, k * 3 | 1)))
    return np.stack(f, -1).astype(np.float32), dict(level=lvl, sigma=sig)


def framing(img01, px_um):
    """(dist_border, geo_border), each HxW float32 in um."""
    g = np.asarray(img01, np.float32); H, W = g.shape
    b = np.ones((H, W), np.uint8); b[1:-1, 1:-1] = 0
    db = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5) * px_um

    th = threshold_multiotsu(g, 3)
    fg = cv2.morphologyEx((g > th[0]).astype(np.uint8), cv2.MORPH_CLOSE,
                          np.ones((3, 3), np.uint8))
    cost = np.where(fg > 0, 1.0, 1e4).astype(np.float64)
    starts = [(0, x) for x in range(0, W, 4)] + [(H - 1, x) for x in range(0, W, 4)] \
        + [(y, 0) for y in range(0, H, 4)] + [(y, W - 1) for y in range(0, H, 4)]
    geo = MCP_Geometric(cost).find_costs(starts)[0]
    geo = np.nan_to_num(geo, posinf=0.0, nan=0.0).astype(np.float32)
    geo = np.minimum(geo, np.percentile(geo[np.isfinite(geo)], 99.9)) * px_um
    return db.astype(np.float32), geo.astype(np.float32)


def compute(img01, px_um, feature_set: str = DEFAULT_SET):
    """Returns (X [H, W, F] float32, info).  ``info['names']`` lists the channels."""
    nm = names(feature_set)
    X, info = _photometric(img01, px_um)
    if feature_set == "framing":
        db, geo = framing(img01, px_um)
        X = np.concatenate([X, db[..., None], geo[..., None]], axis=-1)
    info["names"] = nm
    info["feature_set"] = feature_set
    return X, info
