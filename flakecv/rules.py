"""Hand-tuned rules baseline (no training data).

Ported unchanged from ``sem/semseg.py``.  Measured on the six labelled frames:
IoU 0.333 / 0.455 / 0.557 / 0.465 / 0.201 / 0.710, mean 0.453, boundary F 0.352.
"""
import numpy as np, cv2
from skimage.filters import threshold_multiotsu


def _rect_score(cnt):
    a = cv2.contourArea(cnt)
    if a < 1: return 0.0, 99
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    r = a / (w * h + 1e-9)
    p = cv2.arcLength(cnt, True)
    k = len(cv2.approxPolyDP(cnt, 0.02 * p, True))
    return float(r), int(k)


def segment(img01, px_um, min_area_um2=4.0, close_um=0.35,
            rect_thr=0.82, corner_max=8, pad_min_um2=3.0):
    b = cv2.GaussianBlur(img01, (0, 0), max(1.0, 0.08 / px_um))
    th = threshold_multiotsu(b, classes=3)
    cls = np.digitize(b, th)
    fg = (cls >= 1).astype(np.uint8)
    k = max(3, int(close_um / px_um) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((max(3, k // 2) | 1,) * 2, np.uint8))

    # --- candidate contact pads: brightest class, rectangular -------------
    bright = ((cls == 2).astype(np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pads = np.zeros_like(fg)
    pad_list = []
    for c in cnts:
        a = cv2.contourArea(c) * px_um ** 2
        if a < pad_min_um2: continue
        r, nk = _rect_score(c)
        if r >= rect_thr and nk <= corner_max:
            cv2.drawContours(pads, [c], -1, 1, -1)
            pad_list.append(dict(area_um2=a, rect=r, corners=nk))

    # --- flake = foreground minus pads, largest components -----------------
    core = cv2.bitwise_and(fg, 1 - pads)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(core)
    flakes = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, 4] * px_um ** 2 >= min_area_um2:
            flakes[lab == i] = 1

    # --- put back the pads that sit ON a flake ----------------------------
    npad, plab, pst, _ = cv2.connectedComponentsWithStats(pads)
    grow = cv2.dilate(flakes, np.ones((k * 2 + 1,) * 2, np.uint8))
    kept = 0
    for i in range(1, npad):
        m = (plab == i)
        if (m & (grow > 0)).sum() / max(1, m.sum()) > 0.45:
            flakes[m] = 1; kept += 1

    flakes = cv2.morphologyEx(flakes, cv2.MORPH_CLOSE, np.ones((k * 2 + 1,) * 2, np.uint8))
    cnts, _ = cv2.findContours(flakes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(flakes)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) * px_um ** 2 < min_area_um2: continue
        cv2.drawContours(out, [c], -1, 1, -1)
        polys.append(c[:, 0, :].astype(float))
    return out, polys, dict(pads=pad_list, pads_on_flake=kept, thr=th.tolist())


# ---------------------------------------------------------------------------
# Why this is the default
#
# Three thresholding strategies were compared on all six frames:
#
#   this one (3-class Otsu)   contour roughness 2.2 - 8.9
#   substrate mode + MAD      contour roughness 2.1 - 17.7, outline wanders
#                             around the pads through bare substrate
#   hysteresis (seed + grow)  contour roughness 2.3 - 9.6, same problem
#
# The later two put the substrate/flake threshold close to the noise, which
# does keep the thinnest veil - but it also keeps the bright/dark halo that a
# 70 nm metal step throws around every pad, and that halo links the pads into
# the flake.  The Otsu split sits higher: it loses a little of the faintest
# veil and it keeps a pad the flake lies on, but the boundary it draws is the
# cleanest of the three.
#
# The remaining defect - a large pad inside the mask on four of six frames -
# is NOT fixable by subtracting the pads.  At 10 kV a few nm of flake on 70 nm
# of Au gives no separable contrast (no distinct histogram mode in any of the
# four frames that contain a large pad), so 'bare pad' and 'flake-covered pad'
# are the same measurement.  Subtracting them removes real flake: doing so
# takes 261015-tpt1-1 from 2832 um2 to 0 and 261021-tpt1-3 from 652 to 147.
# The information needed is not in a 10 kV image; a low-kV frame has it.
