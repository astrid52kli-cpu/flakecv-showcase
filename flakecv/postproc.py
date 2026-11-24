"""From probability map to flake mask.

Output path, in order:
  1. threshold the probability (0.5)
  2. morphological open (3x3), then close (0.3 um kernel)
  3. drop components smaller than 6 um^2
  4. ``central`` - keep only the largest component that does not touch the frame
  5. ``fill``    - fill interior holes
  6. ``snap``    - move each boundary point onto the strongest image gradient
                   within +-1.5 um along the outward normal, smooth (sigma 2 px)

!!! CAVEAT - ``central`` and ``fill`` HURT WITH LITTLE TRAINING DATA !!!
With 1-2 training frames the model's largest blob is often an entire gold
contact, and ``central`` is a hard constraint with no way back.  So steps 4-5
are OFF by default below 3 training frames.  Measured, leave-one-flake-out:

                         n=1      n=5
    steps 1-3 only      0.396    0.524
    + fill              0.381    0.542
    + central           0.332    0.534
    + snap only         0.416    0.552
    all of it           0.346    0.565

A guard ("collapse only if the largest blob already holds >=80% of the area")
recovers n=1 only to 0.363.  ``snap`` is safe at every n: it moves the
boundary without changing which region was selected.  Its radius has a real
optimum at 1.5-2.0 um and degrades by 3.0 um.  It does not fix a wrong region:
if the model took a whole contact for flake, snap draws that contact precisely.

Ported from ``eval/postproc.py`` / ``eval/pixelmodel.py``; arithmetic unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_TRAIN_FOR_SHAPE_PRIORS = 3

DEFAULTS = dict(thr=0.5, min_area_um2=6.0, close_um=0.3, search_um=1.5,
                smooth_um=0.08, curve_smooth=2.0)


def shape_priors_default(n_train: int | None) -> bool:
    """central/fill default: on only with >= 3 training frames."""
    return n_train is not None and n_train >= MIN_TRAIN_FOR_SHAPE_PRIORS


def clean(p, px_um, thr=0.5, min_area_um2=6.0, close_um=0.3,
          central=False, fill=False):
    """Steps 1-5.  ``central`` / ``fill`` default OFF (see module caveat)."""
    m = (p > thr).astype(np.uint8)
    k = max(3, int(close_um / px_um) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    H, W = m.shape
    comps = [i for i in range(1, n) if st[i, 4] * px_um ** 2 >= min_area_um2]
    if central and comps:
        inner = [i for i in comps
                 if st[i, 0] > 0 and st[i, 1] > 0
                 and st[i, 0] + st[i, 2] < W and st[i, 1] + st[i, 3] < H]
        comps = inner if inner else comps
        comps = [max(comps, key=lambda i: st[i, 4])]
    out = np.zeros_like(m)
    for i in comps:
        out[lb == i] = 1
    if fill:
        ff = out.copy()
        cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1)
        out = (out | (1 - ff)).astype(np.uint8)
    return out


def snap(mask, img01, px_um, search_um=1.5, smooth_um=0.08, curve_smooth=2.0):
    """Step 6: move each boundary point onto the strongest image gradient
    within +-search_um along the outward normal, then smooth the contour."""
    H, W = mask.shape
    b = cv2.GaussianBlur(np.asarray(img01, np.float32), (0, 0),
                         max(1.0, smooth_um / px_um))
    gx = cv2.Scharr(b, cv2.CV_32F, 1, 0); gy = cv2.Scharr(b, cv2.CV_32F, 0, 1)
    E = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), max(1.0, 0.05 / px_um))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask
    out = np.zeros_like(mask)
    Lp = max(1, int(round(search_um / px_um))); ts = np.arange(-Lp, Lp + 1)
    for c in cnts:
        pts = c.reshape(-1, 2).astype(np.float32)
        if len(pts) < 20:
            continue
        d = np.roll(pts, -2, 0) - np.roll(pts, 2, 0)
        nv = np.stack([d[:, 1], -d[:, 0]], 1)
        nv /= np.linalg.norm(nv, axis=1, keepdims=True) + 1e-9
        pr = np.clip((pts + 3 * nv).astype(int), [0, 0], [W - 1, H - 1])
        nv *= np.where(mask[pr[:, 1], pr[:, 0]] > 0, -1., 1.)[:, None]
        Xs = np.clip(pts[:, None, 0] + nv[:, None, 0] * ts[None, :], 0, W - 1)
        Ys = np.clip(pts[:, None, 1] + nv[:, None, 1] * ts[None, :], 0, H - 1)
        j = E[Ys.round().astype(int), Xs.round().astype(int)].argmax(1)
        q = pts + nv * ts[j][:, None]
        q = cv2.GaussianBlur(q.astype(np.float32).reshape(-1, 1, 2), (0, 0),
                             curve_smooth).reshape(-1, 2)
        cv2.drawContours(out, [q.round().astype(np.int32).reshape(-1, 1, 2)], -1, 1, -1)
    return out


def finish(p, img01, px_um, *, n_train=None, central=None, fill=None,
           thr=0.5, min_area_um2=6.0, close_um=0.3, search_um=1.5):
    """The default output path.  ``central`` / ``fill`` left as None follow
    :func:`shape_priors_default` (on only with >= 3 training frames).
    ``search_um=0`` disables snap."""
    on = shape_priors_default(n_train)
    central = on if central is None else central
    fill = on if fill is None else fill
    m = clean(p, px_um, thr=thr, min_area_um2=min_area_um2, close_um=close_um,
              central=central, fill=fill)
    return snap(m, img01, px_um, search_um=search_um) if search_um else m
