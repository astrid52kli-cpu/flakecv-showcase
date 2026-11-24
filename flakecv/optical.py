"""Optical (20x) image: background removal, gold, overlays.

Segmentation by BACKGROUND REMOVAL, not by thresholding the flake.
The bare SiO2 is one uniform colour across the whole field, so it is the thing
that can be described exactly; the flake is not (its colour runs from strong
cyan on thick regions to almost-substrate on a monolayer).  Thresholding ON the
flake's own hue therefore clips the thin regions, which is what made the old
mask too small and ragged.  Thresholding on the BACKGROUND keeps everything
that is not bare substrate, however faint.

Illumination is not flat (measured: L* spans 186-218 inside the pads on one
frame, ~8%), so the background is fitted as a smooth surface per channel rather
than as a single colour.

The burnt-in scale bars are NOT calibrated; pixel sizes measured against SEM
are in ``flakecv.io.OPTICAL_PX_UM`` (20x 0.2475, 50x 0.090 um/px).
Ported unchanged from ``optical/ombg.py``.
"""
import numpy as np, cv2


def _poly_basis(X, Y, order=2):
    terms = [np.ones_like(X)]
    for i in range(1, order + 1):
        for j in range(i + 1):
            terms.append((X ** (i - j)) * (Y ** j))
    return np.stack(terms, 1)


def background_model(lab, seed, order=2):
    """Fit a smooth surface per Lab channel over the seed (background) pixels."""
    H, W = seed.shape
    yy, xx = np.mgrid[0:H, 0:W]
    X = (xx / W - 0.5).ravel(); Y = (yy / H - 0.5).ravel()
    B = _poly_basis(X, Y, order)
    m = seed.ravel() > 0
    out = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        v = lab[:, :, c].ravel()
        coef, *_ = np.linalg.lstsq(B[m], v[m], rcond=None)
        out[:, :, c] = (B @ coef).reshape(H, W)
    return out


def segment(bgr, order=2, k=4.0, close_um=0.8, px_um=0.2475, min_area_um2=3.0,
            seed_q=0.35, iters=3):
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = lab.shape[:2]

    # --- seed: the most common colour in the frame is the bare substrate ------
    a, b = lab[:, :, 1], lab[:, :, 2]
    hist, xe, ye = np.histogram2d(a.ravel(), b.ravel(), bins=64, range=[[0, 255], [0, 255]])
    hs = cv2.GaussianBlur(hist.astype(np.float32), (0, 0), 1.5)
    i, j = np.unravel_index(int(np.argmax(hs)), hs.shape)
    a0, b0 = 0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])
    d0 = np.hypot(a - a0, b - b0)
    seed = (d0 <= np.quantile(d0, seed_q)).astype(np.uint8)

    for _ in range(iters):
        bg = background_model(lab, seed, order)
        R = lab - bg
        # robust per-channel spread of the background residual
        s = np.array([1.4826 * np.median(np.abs(R[:, :, c][seed > 0])) + 1e-3 for c in range(3)])
        # chroma carries the signal; L is contaminated by illumination and by
        # the metal, so weight it down
        w = np.array([0.5, 1.0, 1.0]) / s
        Dm = np.sqrt(((R * w[None, None, :]) ** 2).sum(2) / 3.0)
        seed = (Dm <= 1.5).astype(np.uint8)

    mat = (Dm > k).astype(np.uint8)
    kk = max(3, int(close_um / px_um) | 1)
    mat = cv2.morphologyEx(mat, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(mat)
    out = np.zeros_like(mat)
    for i in range(1, n):
        if st[i, 4] * px_um ** 2 >= min_area_um2: out[lb == i] = 1
    return out, Dm, bg, seed


def drop_overlays(bgr, mask, px_um=0.2475, sat_max=25, fill_min=0.9, min_area_um2=20.0):
    """Remove burnt-in graphics (the scale bar): grey, and exactly rectangular."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(int)
    n, lb, st, _ = cv2.connectedComponentsWithStats(np.asarray(mask, np.uint8))
    out = np.asarray(mask, np.uint8).copy()
    for i in range(1, n):
        a = st[i, 4]
        if a * px_um ** 2 < min_area_um2: continue
        m = (lb == i)
        fill = a / float(st[i, 2] * st[i, 3])
        if fill >= fill_min and np.median(S[m]) <= sat_max:
            out[m] = 0
    return out


def split_gold(bgr, material, hue=(8, 42), smin=40, vmin=70, px_um=0.2475,
               rim_um=0.6, min_area_um2=3.0, min_width_um=0.8):
    """Separate the metal from the rest of the material mask.

    The gold edge carries a colour fringe one or two pixels wide that is
    neither substrate nor flake, so the metal mask is grown by `rim_um` before
    subtraction and leftover slivers are dropped by their inscribed width."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    g = ((Hh >= hue[0]) & (Hh <= hue[1]) & (S > smin) & (V > vmin)).astype(np.uint8) & material
    k = max(3, int(0.8 / px_um) | 1)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    rim = max(3, int(2 * rim_um / px_um) | 1)
    grow = cv2.dilate(g, np.ones((rim, rim), np.uint8))
    f = (material & (1 - grow)).astype(np.uint8)
    n, lb, st, _ = cv2.connectedComponentsWithStats(f)
    out = np.zeros_like(f)
    for i in range(1, n):
        if st[i, 4] * px_um ** 2 < min_area_um2: continue
        m = (lb == i).astype(np.uint8)
        if cv2.distanceTransform(m, cv2.DIST_L2, 5).max() * 2 * px_um < min_width_um: continue
        out |= m
    # give the rim back only inside a flake's own hull - a flake crossing a
    # lead keeps its continuity, but the colour fringe along a bare lead does
    # not come back
    hull = np.zeros_like(out)
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) * px_um ** 2 < 10 * min_area_um2: continue
        cv2.fillPoly(hull, [cv2.convexHull(c)], 1)
    out = ((out | (hull & material)) & (1 - g)).astype(np.uint8)
    return g, out
