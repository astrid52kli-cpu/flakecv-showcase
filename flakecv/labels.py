"""Three-class labels.

    0   background   certain: bare substrate, or a bare contact
    1   flake        certain - INCLUDING the part lying on top of a contact
    255 uncertain    a human cannot tell -> EXCLUDED from scoring

On disk the classes are 0 / 128 / 255 in a single-channel PNG, so the file is
readable by eye and by a browser canvas.

The uncertain class is what makes an accuracy curve possible without a perfect
segmentation: the annotator marks where certainty ends, and those pixels count
neither as hits nor as errors (Cityscapes-style ignore label).

Ported from ``eval/label.py``; behaviour unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import BACKGROUND, FLAKE, UNCERTAIN
from .io import imread, imwrite
from .substrate import substrate_stats

DISK = {BACKGROUND: 0, FLAKE: 128, UNCERTAIN: 255}


def save(path, lab) -> None:
    a = np.zeros(np.shape(lab), np.uint8)
    lab = np.asarray(lab)
    a[lab == FLAKE] = 128
    a[lab == UNCERTAIN] = 255
    imwrite(path, a)


def from_disk(a) -> np.ndarray:
    """Disk encoding -> in-memory classes.  Accepts 0/128/255 and 0/1/255
    (the prefill format), and tolerates anti-aliased brush strokes."""
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    out = np.full(a.shape, BACKGROUND, np.uint8)
    out[(a > 60) & (a < 200)] = FLAKE
    out[a >= 200] = UNCERTAIN
    out[a == 1] = FLAKE
    return out


def load(path) -> np.ndarray:
    return from_disk(imread(path, cv2.IMREAD_UNCHANGED))


def stats(lab) -> dict:
    lab = np.asarray(lab)
    return dict(flake=float((lab == FLAKE).mean()),
                background=float((lab == BACKGROUND).mean()),
                uncertain=float((lab == UNCERTAIN).mean()))


def validate(lab, name="label") -> np.ndarray:
    """Refuse anything that is not an in-memory three-class label.

    The common mistake is passing the raw disk PNG (0/128/255): 128 is then
    neither FLAKE nor UNCERTAIN and would silently score as background.
    """
    lab = np.asarray(lab)
    bad = ~np.isin(lab, (BACKGROUND, FLAKE, UNCERTAIN))
    if bad.any():
        vals = np.unique(lab[bad])[:5].tolist()
        raise ValueError(
            f"{name} contains values {vals} outside {{0, 1, 255}}. "
            "If this came straight from a .label.png, load it with "
            "flakecv.labels.load() / from_disk() first.")
    return lab


def prefill(img01, px_um, flake_mask=None, pads=None, k_sure=6.0, k_maybe=2.0,
            pad_rim_um=1.2, near_um=2.5):
    """Three-class starting point for the annotator.

    Certainty follows the distance from the substrate level in units of the
    substrate's own spread: > k_sure sigma is FLAKE, k_maybe..k_sure is
    UNCERTAIN.  ``flake_mask`` / ``pads`` are an algorithm's output to start
    from (pads become background with an uncertain halo).
    Returns (label, info).
    """
    sm, lvl, sig = substrate_stats(img01, px_um)
    z = np.abs(sm - lvl) / max(sig, 1e-6)

    out = np.full(img01.shape, BACKGROUND, np.uint8)
    out[z > k_maybe] = UNCERTAIN
    out[z > k_sure] = FLAKE
    if flake_mask is not None:
        out[(flake_mask > 0) & (z > k_sure)] = FLAKE
    if pads is not None and np.any(pads):
        rim = max(3, int(2 * pad_rim_um / px_um) | 1)
        grown = cv2.dilate(np.asarray(pads, np.uint8), np.ones((rim, rim), np.uint8))
        # a contact itself is background; its halo is nobody's land
        out[grown > 0] = UNCERTAIN
        out[np.asarray(pads, np.uint8) > 0] = BACKGROUND
    k = max(3, int(0.3 / px_um) | 1)
    m = (out == FLAKE).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    out[(out == FLAKE) & (m == 0)] = UNCERTAIN

    # Uncertainty is only meaningful next to something real: keep the band
    # where it touches flake or a contact, drop isolated substrate speckle.
    unc = (out == UNCERTAIN).astype(np.uint8)
    unc = cv2.morphologyEx(unc, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    near = cv2.dilate((out == FLAKE).astype(np.uint8),
                      np.ones((max(3, int(2 * near_um / px_um) | 1),) * 2, np.uint8))
    if pads is not None and np.any(pads):
        near |= cv2.dilate(np.asarray(pads, np.uint8),
                           np.ones((max(3, int(2 * pad_rim_um / px_um) | 1),) * 2, np.uint8))
    out[(out == UNCERTAIN) & ((unc == 0) | (near == 0))] = BACKGROUND
    return out, dict(level=float(lvl), sigma=float(sig),
                     frac_sure=float((out == FLAKE).mean()),
                     frac_uncertain=float((out == UNCERTAIN).mean()))
