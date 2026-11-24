"""Step 3 - cracks and bubbles INSIDE the flake mask.   EXPERIMENTAL.

!!! No crack or bubble has been hand-labelled yet, so nothing here has a
!!! measured accuracy.  These are CANDIDATE detectors for human review; use
!!! :func:`score` once annotations exist (format: :func:`save_annotations`).

Three detectors, ported from the exploratory research code
(``flakecv/crackseg.py`` and ``flakecv/defect.py`` in the old pipeline), with
every pixel parameter converted to micrometres.  The originals were tuned at
~50 nm/px, so ``N px`` became ``N * 0.05 um``.

  cracks    straight CHORDS across one flake region along which the grey level
            STEPS consistently from one side to the other (a crack or fold
            separates regions of different thickness - a step, not a line).
  lines     thin ridge-like structures (Frangi) away from the flake outline;
            catches curved cracks and wrinkles the chord test misses.
  bubbles   small round bright or dark spots (multi-scale top-hat + LoG):
            blisters, bubbles, residue.  In SEM alone these are not reliably
            separable, so they share one class with a polarity.

Everything outside the flake mask (substrate, contacts, graphics) is ignored,
a band along the flake outline is excluded so the outline itself is not
reported as a crack, and a band along the edges of the gold contacts (also
those lying under the flake) is excluded - those edges are lithography.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REF_PX_UM = 0.05      # the pixel size the legacy parameters were tuned at

PARAMS = dict(
    # cracks (chords)
    crack_boundary_step_um=0.5, crack_side_offset_um=0.3, crack_min_len_um=3.0,
    crack_span_frac=0.55, crack_z=1.8, crack_agree_min=0.78, crack_inside_min=0.97,
    crack_depth_frac=0.15, crack_max_n=8, crack_nms_um=1.25, crack_erode_um=0.2,
    crack_min_region_um2=3.75,
    # lines (ridges)
    line_sigmas_um=(0.05, 0.1, 0.15, 0.2), line_percentile=99.3, line_edge_band_um=0.45,
    line_min_len_um=1.0,
    # bubbles (blobs)
    bubble_tophat_radii_um=(0.15, 0.3, 0.6, 1.0), bubble_r_um=(0.08, 1.2), bubble_z_min=6.0,
    bubble_erode_um=0.45, bubble_top_n=30, bubble_nms=3.0,
    # contacts under the flake: their edges are lithography, not defects
    metal_edge_band_um=0.4, crack_max_on_metal_edge=0.3,
)


def metal_edges(img01, px_um, p=PARAMS):
    """Band around the edges of the brightest grey class (the Ti/Au contacts),
    including contacts lying UNDER the flake.  Detections there are dropped."""
    from skimage.filters import threshold_multiotsu
    img = cv2.GaussianBlur(np.asarray(img01, np.float32), (0, 0), max(1.0, 0.08 / px_um))
    try:
        th = threshold_multiotsu(img, classes=3)[1]
    except ValueError:
        return np.zeros(img.shape, np.uint8)
    k = _px(0.3, px_um, 3) | 1
    metal = cv2.morphologyEx((img > th).astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    band = _px(p["metal_edge_band_um"], px_um, 2) * 2 + 1
    return cv2.morphologyEx(metal, cv2.MORPH_GRADIENT, np.ones((band, band), np.uint8))


def _px(um, px_um, lo=1):
    return max(lo, int(round(um / px_um)))


# ---------------------------------------------------------------- cracks
def cracks(img01, flake_mask, px_um, p=PARAMS, exclude=None):
    """Straight crack/fold candidates: list of dicts with p0, p1 (x, y px),
    len_um, step (grey-level jump), snr, agree, score; best first."""
    img = np.clip(np.asarray(img01, np.float32), 0, 1)
    sm = cv2.GaussianBlur(img, (0, 0), max(1.0, 0.075 / px_um))
    noise = float(np.median(np.abs(np.diff(img, axis=1)))) + 1e-6
    e = _px(p["crack_erode_um"], px_um)
    full = cv2.erode((np.asarray(flake_mask) > 0).astype(np.uint8), np.ones((e, e), np.uint8))
    if full.sum() * px_um ** 2 < 1.25:
        return []
    n_cc, cc_lab, cc_st, _ = cv2.connectedComponentsWithStats(full)
    out = []
    for i in range(1, n_cc):
        if cc_st[i, 4] * px_um ** 2 >= p["crack_min_region_um2"]:
            out += _cracks_one(img, sm, (cc_lab == i).astype(np.uint8), px_um, noise, p, exclude)
    out.sort(key=lambda r: -r["score"])
    nms, keep = p["crack_nms_um"] / px_um, []
    for r in out:
        if all(min(max(np.hypot(*(r["p0"] - k["p0"])), np.hypot(*(r["p1"] - k["p1"]))),
                   max(np.hypot(*(r["p0"] - k["p1"])), np.hypot(*(r["p1"] - k["p0"])))) > nms
               for k in keep):
            keep.append(r)
        if len(keep) >= p["crack_max_n"]:
            break
    for r in keep:
        r["p0"], r["p1"] = r["p0"].tolist(), r["p1"].tolist()
    return keep


def _cracks_one(img, sm, reg, px_um, noise, p, exclude=None):
    H, W = img.shape
    step = _px(p["crack_boundary_step_um"], px_um, 2)
    d = _px(p["crack_side_offset_um"], px_um, 2)
    dist = cv2.distanceTransform(reg, cv2.DIST_L2, 5)
    cont, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cont:
        return []
    pts = np.vstack([c.reshape(-1, 2)[::step] for c in cont if len(c) > step] or [np.zeros((0, 2))])
    if len(pts) < 4:
        return []
    if len(pts) > 240:
        pts = pts[np.linspace(0, len(pts) - 1, 240).astype(int)]
    P = pts.astype(np.float32); m = len(P)
    Wc = float(np.sqrt(reg.sum()))
    I, J = np.triu_indices(m, k=1)
    D = np.hypot(P[J, 0] - P[I, 0], P[J, 1] - P[I, 1])
    sel = D >= max(p["crack_min_len_um"] / px_um, p["crack_span_frac"] * Wc)
    if not sel.any():
        return []
    I, J, D = I[sel], J[sel], D[sel]
    t = np.linspace(0.04, 0.96, 40)[None, :]
    dx = (P[J, 0] - P[I, 0])[:, None]; dy = (P[J, 1] - P[I, 1])[:, None]
    L = D[:, None]; nx, ny = -dy / L, dx / L
    cx = P[I, 0][:, None] + dx * t; cy = P[I, 1][:, None] + dy * t
    ci = np.clip(cx.round().astype(np.int32), 0, W - 1)
    cj = np.clip(cy.round().astype(np.int32), 0, H - 1)
    ok = reg[cj, ci].mean(axis=1) >= p["crack_inside_min"]
    if not ok.any():
        return []
    ok &= dist[cj, ci].max(axis=1) >= p["crack_depth_frac"] * Wc
    if exclude is not None:
        ok &= (exclude[cj, ci] > 0).mean(axis=1) <= p["crack_max_on_metal_edge"]
    if not ok.any():
        return []

    def samp(ox, oy):
        xs = np.clip((cx + ox).round().astype(np.int32), 0, W - 1)
        ys = np.clip((cy + oy).round().astype(np.int32), 0, H - 1)
        return sm[ys, xs]
    diff = samp(nx * d, ny * d)[ok] - samp(-nx * d, -ny * d)[ok]
    med = np.median(diff, axis=1)
    agree = (np.sign(diff) == np.sign(med)[:, None]).mean(axis=1)
    amp = np.abs(med)
    good = (agree >= p["crack_agree_min"]) & (amp >= p["crack_z"] * noise)
    out = []
    for k, a_, g_ in zip(np.where(ok)[0][good], amp[good], agree[good]):
        out.append(dict(kind="crack", p0=P[I[k]].astype(float), p1=P[J[k]].astype(float),
                        len_um=round(float(D[k]) * px_um, 2), step=round(float(a_), 4),
                        agree=round(float(g_), 2), snr=round(float(a_ / noise), 1),
                        score=round(float(a_ / noise * g_ * D[k] * px_um), 2)))
    return out


# ---------------------------------------------------------------- lines
def lines(img01, flake_mask, px_um, p=PARAMS, exclude=None):
    """Ridge-like line candidates inside the mask, away from its outline.
    Returns (list of dicts, binary map)."""
    from skimage.filters import frangi
    fm = (np.asarray(flake_mask) > 0).astype(np.uint8)
    if fm.sum() == 0:
        return [], np.zeros_like(fm)
    sig = [max(0.7, s / px_um) for s in p["line_sigmas_um"]]
    r = frangi(np.asarray(img01, np.float32), sigmas=sig, black_ridges=False)
    band = _px(p["line_edge_band_um"], px_um, 2) * 2 + 1
    core = cv2.erode(fm, np.ones((band, band), np.uint8))
    if exclude is not None:
        core[exclude > 0] = 0
    v = r[core > 0]
    if v.size < 100:
        return [], np.zeros_like(fm)
    m = ((r > np.percentile(v, p["line_percentile"])) & (core > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m)
    out, keepmap = [], np.zeros_like(m)
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (cx, cy), (w, h), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        length = max(w, h) * px_um
        if length < p["line_min_len_um"] or max(w, h) / max(min(w, h), 1.0) < 3.0:
            continue
        keepmap |= comp
        out.append(dict(kind="line", x=float(cx), y=float(cy), len_um=round(length, 2),
                        angle_deg=round(float(ang), 1), area_um2=round(float(st[i, 4]) * px_um ** 2, 3)))
    out.sort(key=lambda d: -d["len_um"])
    return out, keepmap


# ---------------------------------------------------------------- bubbles
def bubbles(img01, flake_mask, px_um, p=PARAMS, exclude=None):
    """Round bright/dark spot candidates (bubbles, blisters, residue), best first."""
    from skimage.feature import blob_log
    img = cv2.GaussianBlur(np.asarray(img01, np.float32), (0, 0), max(0.6, 0.04 / px_um))
    e = _px(p["bubble_erode_um"], px_um, 2) * 2 + 1
    core = cv2.erode((np.asarray(flake_mask) > 0).astype(np.uint8), np.ones((e, e), np.uint8))
    if exclude is not None:
        core[exclude > 0] = 0
    if core.sum() < 300:
        return []
    Wt = np.zeros_like(img); Bt = np.zeros_like(img)
    for r_um in p["bubble_tophat_radii_um"]:
        r = _px(r_um, px_um, 1)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
        Wt = np.maximum(Wt, cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se))
        Bt = np.maximum(Bt, cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se))
    rmin, rmax = (max(1.0, r / px_um) for r in p["bubble_r_um"])
    cand = []
    for pol, resp in (("bright", Wt), ("dark", Bt)):
        v = resp[core > 0]
        med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-6
        sc = np.where(core > 0, np.clip((resp - med) / mad, 0, None), 0).astype(np.float32)
        if sc.max() <= 0:
            continue
        for y, x, s in blob_log(sc / sc.max(), min_sigma=rmin / np.sqrt(2), max_sigma=rmax / np.sqrt(2),
                                num_sigma=8, threshold=0.015):
            y, x = int(y), int(x)
            if core[y, x] == 0 or sc[y, x] < p["bubble_z_min"]:
                continue
            r = float(s) * np.sqrt(2)
            cand.append(dict(kind="bubble", x=float(x), y=float(y), r_px=round(float(r), 2),
                             r_um=round(float(r * px_um), 3), polarity=pol, score=round(float(sc[y, x]), 1)))
    cand.sort(key=lambda d: -d["score"])
    keep = []
    for d in cand:
        if all((d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2 > (p["bubble_nms"] * max(d["r_px"], k["r_px"])) ** 2
               for k in keep):
            keep.append(d)
        if len(keep) >= p["bubble_top_n"]:
            break
    return keep


def detect(img01, flake_mask, px_um, p=PARAMS, margin_um=3.0, exclude_metal=True) -> dict:
    """All three detectors, run on the flake's bounding box (plus a margin)
    for speed; coordinates are returned in the full frame.
    -> {'cracks': [...], 'lines': [...], 'bubbles': [...], 'line_map': HxW}"""
    img = np.asarray(img01, np.float32)
    fm = (np.asarray(flake_mask) > 0).astype(np.uint8)
    H, W = fm.shape
    empty = dict(cracks=[], lines=[], bubbles=[], line_map=np.zeros_like(fm))
    ys, xs = np.nonzero(fm)
    if len(xs) == 0:
        return empty
    mg = _px(margin_um, px_um)
    y0, y1 = max(0, ys.min() - mg), min(H, ys.max() + mg + 1)
    x0, x1 = max(0, xs.min() - mg), min(W, xs.max() + mg + 1)
    ci, cm = img[y0:y1, x0:x1], fm[y0:y1, x0:x1]
    ex = metal_edges(img, px_um, p)[y0:y1, x0:x1] if exclude_metal else None
    ln, lmap = lines(ci, cm, px_um, p, ex)
    cr, bb = cracks(ci, cm, px_um, p, ex), bubbles(ci, cm, px_um, p, ex)
    for c in cr:
        c["p0"] = [c["p0"][0] + x0, c["p0"][1] + y0]; c["p1"] = [c["p1"][0] + x0, c["p1"][1] + y0]
    for d in ln + bb:
        d["x"] += x0; d["y"] += y0
    full = np.zeros_like(fm); full[y0:y1, x0:x1] = lmap
    return dict(cracks=cr, lines=ln, bubbles=bb, line_map=full)


# ------------------------------------------------- annotations & scoring
def save_annotations(path, frame, px_um, cracks=(), bubbles=()):
    """Annotation file (also the candidate export format, so candidates can be
    corrected by hand and saved back):
        {"frame": name, "px_um": ..., "cracks": [{"p0": [x, y], "p1": [x, y]}, ...],
         "bubbles": [{"x": .., "y": .., "r_px": ..}, ...]}
    Coordinates are pixels of the prepared frame; px_um makes them physical."""
    doc = dict(frame=frame, px_um=px_um,
               cracks=[dict(p0=list(map(float, c["p0"])), p1=list(map(float, c["p1"]))) for c in cracks],
               bubbles=[dict(x=float(b["x"]), y=float(b["y"]), r_px=float(b.get("r_px", 0))) for b in bubbles])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1))
    return doc


def load_annotations(path) -> dict:
    return json.loads(Path(path).read_text())


def _seg_dist(a0, a1, b0, b1, n=21):
    """Mean distance (px) from points along segment a to segment b."""
    a0, a1, b0, b1 = (np.asarray(v, float) for v in (a0, a1, b0, b1))
    t = np.linspace(0, 1, n)[:, None]
    P = a0 + (a1 - a0) * t
    v = b1 - b0; L2 = max(float(v @ v), 1e-9)
    s = np.clip(((P - b0) @ v) / L2, 0, 1)[:, None]
    return float(np.linalg.norm(P - (b0 + v * s), axis=1).mean())


def _greedy(cost, tol):
    pairs, used_p, used_g = [], set(), set()
    for i, j in sorted(np.ndindex(cost.shape), key=lambda ij: cost[ij]) if cost.size else []:
        if cost[i, j] > tol:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); pairs.append((i, j))
    return pairs


def _prf(n_match, n_pred, n_ref):
    pr = n_match / n_pred if n_pred else float("nan")
    rc = n_match / n_ref if n_ref else float("nan")
    f = 2 * pr * rc / (pr + rc) if n_pred and n_ref and (pr + rc) else (0.0 if n_pred or n_ref else float("nan"))
    return dict(precision=pr, recall=rc, f=f, matched=n_match, n_pred=n_pred, n_ref=n_ref)


def score(pred: dict, ref: dict, px_um, crack_tol_um=0.5, bubble_tol_um=0.5) -> dict:
    """Detection precision / recall / F against annotations, tolerances in um.
    A crack matches when the mean distance between the two segments (both
    directions averaged) is within ``crack_tol_um``; a bubble when the centres
    are within ``bubble_tol_um``.  One-to-one, greedy by distance."""
    pc, rc = pred.get("cracks", []), ref.get("cracks", [])
    C = np.array([[0.5 * (_seg_dist(a["p0"], a["p1"], b["p0"], b["p1"]) +
                          _seg_dist(b["p0"], b["p1"], a["p0"], a["p1"])) for b in rc] for a in pc]).reshape(len(pc), len(rc))
    pb, rb = pred.get("bubbles", []), ref.get("bubbles", [])
    B = np.array([[np.hypot(a["x"] - b["x"], a["y"] - b["y"]) for b in rb] for a in pb]).reshape(len(pb), len(rb))
    return dict(cracks=_prf(len(_greedy(C * px_um, crack_tol_um)), len(pc), len(rc)),
                bubbles=_prf(len(_greedy(B * px_um, bubble_tol_um)), len(pb), len(rb)))
