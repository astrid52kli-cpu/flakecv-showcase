"""The SiO2 substrate level and spread in a SEM frame.

Ported unchanged from ``sem/subtract.py``.  The iterative median/MAD refinement
was duplicated verbatim in ``eval/features.py::_z`` and ``eval/label.py::prefill``;
it now lives here once (:func:`substrate_stats`).
"""
from __future__ import annotations

import cv2
import numpy as np


def substrate_level(img01, px_um, bins=256, min_weight=0.10, prom=0.25):
    """The substrate is the DARKEST large uniform population in a SEM frame
    (lowest Z, lowest secondary-electron yield), so take the darkest mode of
    the smoothed image that still holds a useful fraction of the frame.

    Returns (level, sigma) in image units.
    """
    sm = cv2.GaussianBlur(img01, (0, 0), max(1.5, 0.10 / px_um))
    h, e = np.histogram(sm, bins=bins, range=(0, 1))
    c = 0.5 * (e[1:] + e[:-1])
    hs = cv2.GaussianBlur(h.astype(np.float32).reshape(-1, 1), (0, 0), 2.0).ravel()
    tot = hs.sum()
    best = None
    for i in range(1, bins - 1):
        if not (hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] > 0):
            continue
        lo, hi = i, i
        while lo > 0 and hs[lo - 1] <= hs[lo]:
            lo -= 1
        while hi < bins - 1 and hs[hi + 1] <= hs[hi]:
            hi += 1
        if hs[i] < prom * hs.max():
            continue
        if hs[lo:hi + 1].sum() / tot < min_weight:
            continue
        if best is None or c[i] < best[0]:
            best = (float(c[i]), max(float(c[hi] - c[lo]) / 4.0, 0.003))
    if best is None:
        return float(np.percentile(sm, 10)), 0.02
    return best


def pixel_noise(img01):
    """std of a single pixel, from nearest-neighbour differences."""
    return float(np.median(np.abs(np.diff(img01, axis=1)))) * 1.4826 / np.sqrt(2)


def substrate_stats(img01, px_um, k=3.0, max_iter=12):
    """Smoothed image plus the substrate (level, sigma), refined on the
    substrate population itself until the spread stops moving.

    The histogram peak width over-states the spread (it contains the flake's
    faint tail); single-pixel noise under-states it (shading, contamination).
    Shrink a +-k sigma window until it converges, floored at the pixel noise.

    Returns (smoothed, level, sigma).
    """
    sm = cv2.GaussianBlur(img01, (0, 0), max(1.0, 0.06 / px_um))
    lvl, sig = substrate_level(img01, px_um)
    lo = pixel_noise(sm)
    for _ in range(max_iter):
        inside = np.abs(sm - lvl) <= k * sig
        if inside.sum() < 2000:
            break
        v = sm[inside]
        nl = float(np.median(v))
        ns = max(float(1.4826 * np.median(np.abs(v - nl))), lo)
        if abs(ns - sig) < 1e-4 * max(sig, 1e-6):
            lvl, sig = nl, ns
            break
        lvl, sig = nl, ns
    return sm, lvl, sig


def substrate_mask(img01, px_um, k=3.0):
    """Pixels within +-k sigma of the substrate level, morphologically cleaned
    at 0.25 um.  Returns (mask uint8, level, sigma)."""
    sm, lvl, sig = substrate_stats(img01, px_um, k=k)
    m = (np.abs(sm - lvl) <= k * sig).astype(np.uint8)
    kk = max(3, int(0.25 / px_um) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((kk, kk), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8))
    return m, lvl, sig
