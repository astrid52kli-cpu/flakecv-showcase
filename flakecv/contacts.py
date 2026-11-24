"""Lithographed contact pads in a SEM frame.

A pad is a large uniform grey level above the substrate whose bare part has an
orthogonal boundary (long straight edges at right angles; exfoliated flakes
cleave at arbitrary angles).  The bare part is completed to its lithographic
rectangle, recovering the piece hidden under a flake.

Used by registration (stages 2-3 and the pad corners).  Ported unchanged from
``sem/subtract.py`` (modes, contacts) and ``flakecv/padgeom.py``.
"""
import numpy as np, cv2

def _segments(cnt, px_um, min_len_um=1.5, eps_um=0.08):
    p = cnt.reshape(-1, 2).astype(np.float32)
    ap = cv2.approxPolyDP(p, eps_um / px_um, True).reshape(-1, 2)
    segs = []
    for i in range(len(ap)):
        a, b = ap[i], ap[(i + 1) % len(ap)]
        L = float(np.hypot(*(b - a))) * px_um
        if L >= min_len_um:
            segs.append((L, float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))) % 180))
    return segs


def right_angle_score(cnt, px_um, tol=7.0):
    """fraction of boundary length lying on a dominant orthogonal pair of directions."""
    segs = _segments(cnt, px_um)
    if not segs: return 0.0, 0.0
    tot = sum(L for L, _ in segs)
    best = 0.0; bestd = 0.0
    for _, a0 in segs:
        acc = 0.0
        for L, a in segs:
            d = abs((a - a0 + 90) % 180 - 90)
            if d <= tol or abs(d - 90) <= tol: acc += L
        if acc > best: best, bestd = acc, a0
    return best / tot, bestd


def modes(img01, px_um, bins=256, min_weight=0.04, prom=0.12):
    """All large uniform grey populations in the frame, dark to bright."""
    sm = cv2.GaussianBlur(img01, (0, 0), max(1.5, 0.10 / px_um))
    h, e = np.histogram(sm, bins=bins, range=(0, 1))
    c = 0.5 * (e[1:] + e[:-1])
    hs = cv2.GaussianBlur(h.astype(np.float32).reshape(-1, 1), (0, 0), 2.0).ravel()
    tot = hs.sum(); out = []
    for i in range(1, bins - 1):
        if not (hs[i] >= hs[i-1] and hs[i] >= hs[i+1] and hs[i] > 0): continue
        if hs[i] < prom * hs.max(): continue
        lo, hi = i, i
        while lo > 0 and hs[lo-1] <= hs[lo]: lo -= 1
        while hi < bins-1 and hs[hi+1] <= hs[hi]: hi += 1
        w = hs[lo:hi+1].sum() / tot
        if w < min_weight: continue
        out.append(dict(level=float(c[i]), sigma=max(float(c[hi]-c[lo])/4.0, 0.004), weight=float(w)))
    out.sort(key=lambda d: d['level'])
    # merge near-duplicates
    merged = []
    for d in out:
        if merged and abs(d['level']-merged[-1]['level']) < 2*max(d['sigma'], merged[-1]['sigma']):
            if d['weight'] > merged[-1]['weight']: merged[-1] = d
        else: merged.append(d)
    return merged, sm


def contacts(img01, px_um, substrate, min_area_um2=8.0, ra_thr=0.72, grow_um=0.3):
    """Pads: a large uniform level above the substrate whose BARE part has an
    orthogonal boundary.  The bare part is then completed to its full
    lithographic rectangle, which recovers the piece hidden under a flake."""

    ms, sm = modes(img01, px_um)
    sub_lvl = ms[0]['level'] if ms else 0.0
    out = np.zeros(img01.shape, np.uint8); found = []
    k = max(3, int(0.3 / px_um) | 1)
    for d in ms[1:]:
        m = (np.abs(sm - d['level']) <= 2.5 * d['sigma']).astype(np.uint8)
        m = cv2.bitwise_and(m, 1 - substrate)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cn in cnts:
            a = cv2.contourArea(cn) * px_um ** 2
            if a < min_area_um2: continue
            ra, _ = right_angle_score(cn, px_um)
            if ra < ra_thr: continue
            box = cv2.boxPoints(cv2.minAreaRect(cn))
            fill = np.zeros_like(out); cv2.fillPoly(fill, [box.astype(np.int32)], 1)
            frac = fill.sum() / float(out.size)
            bare = np.zeros_like(out); cv2.drawContours(bare, [cn], -1, 1, -1)
            # complete the pad to its lithographic rectangle only when that
            # rectangle is a plausible pad; a pad running off the frame would
            # otherwise swallow the whole image
            if frac > 0.45 or cv2.contourArea(cn) / max(1.0, fill.sum()) < 0.35:
                out |= bare
            else:
                out |= fill
            found.append(dict(area_um2=a, ra=ra, level=d['level'], completed=bool(frac <= 0.45)))
    if grow_um > 0 and out.any():
        g = max(3, int(grow_um / px_um) | 1)
        out = cv2.dilate(out, np.ones((g, g), np.uint8))
    return out, found, ms


