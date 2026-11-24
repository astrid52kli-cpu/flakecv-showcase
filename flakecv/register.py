"""Optical -> SEM registration: three stages and two guards.

The contact pattern is tiled 36x36 across the chip, so on its own it gives
many equally good wrong answers; the flake outline is unique but soft.  So the
flake finds the right place and the lithography may refine it.

  stage 1  flake-SHAPE match, searching MIRROR x rotation x translation
           (scale is fixed by the two pixel sizes).  The SEM frame is
           mirrored relative to the optical image; no rotation can absorb a
           reflection, so a rotation-only search has no solution at all.
  stage 2  pair gold-pad corners inside a gate around stage 1 and re-fit.
  stage 3  minimise the distance from optical pad edges to SEM pad edges.

Guards (both measured, both required) - :func:`register`:
  * ACCEPTANCE GATE: keep a stage-2/3 refinement only if it IMPROVES overlap
    with the SEM-side flake mask.  With sparse pads the pad stages drag a good
    fit into a bad one (tpt1-2: 0.245 -> 0.048).  On all three test frames
    the final transform stays at stage 1.
  * AREA-CONSISTENT CANDIDATES: reject optical flake candidates whose area
    differs from the SEM flake area by more than 3x (without it tpt1-2 picked
    a 1552 um^2 candidate for a 165 um^2 flake).
The referee for both is the MODEL's SEM mask, never the hand label, so this
works at inference time.

Deliberately NOT included: a local (dx, dy, scale) refinement.  Tested: +0.006
overall, -0.050 on one frame; its ceiling is set by the referee mask's quality.

Measured (OM->SEM IoU vs hand label, three 20 kV frames): unguarded 0.301,
guarded with the rules mask 0.506, guarded with the model mask 0.705.
Resolution ceiling 0.845 (a perfect optical outline off by one 247 nm pixel).

Ported from ``optical/register.py`` and ``optical/register2.py``.
"""
import numpy as np, cv2

from .io import OPTICAL_PX_UM

OMPX_20X = OPTICAL_PX_UM["20x"]
# ---------------------------------------------------------------- primitives
def umeyama(P, Q, fixed_scale=None):
    mp, mq = P.mean(0), Q.mean(0)
    A, B = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, d]) @ Vt
    s = fixed_scale if fixed_scale is not None else (S * np.array([1, d])).sum() / ((A ** 2).sum() + 1e-12)
    return s, R, mq - s * R @ mp


def apply(T, P):
    s, R, t = T
    return (s * (R @ np.atleast_2d(np.asarray(P, float)).T)).T + t


def mat(T):
    s, R, t = T
    M = np.zeros((2, 3)); M[:, :2] = s * R; M[:, 2] = t
    return M


def warp_into(src, T, shape, nearest=True):
    return cv2.warpAffine(np.asarray(src, np.float32), mat(T), (shape[1], shape[0]),
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def _warp_centred(tpl, scale, ang_deg, out_shape):
    Hs, Ws = out_shape; h, w = tpl.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang_deg, scale)
    M[0, 2] += Ws / 2.0 - w / 2.0; M[1, 2] += Hs / 2.0 - h / 2.0
    return cv2.warpAffine(np.asarray(tpl, np.float32), M, (Ws, Hs), flags=cv2.INTER_LINEAR)


def _fft_shift(a, b):
    H, W = a.shape
    az, bz = a - a.mean(), b - b.mean()
    cc = np.fft.irfft2(np.fft.rfft2(bz) * np.conj(np.fft.rfft2(az)), s=(H, W))
    cc /= (np.sqrt((az ** 2).sum() * (bz ** 2).sum()) + 1e-9)
    i = int(np.argmax(cc)); dy, dx = np.unravel_index(i, (H, W))
    if dy > H // 2: dy -= H
    if dx > W // 2: dx -= W
    return int(dx), int(dy), float(cc.flat[i])


# ---------------------------------------------------------------- masks/corners
def optical_gold(bgr, hue=(8, 42), smin=40, vmin=70):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    m = ((H >= hue[0]) & (H <= hue[1]) & (S > smin) & (V > vmin)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def optical_flakes(bgr, px_um=OMPX_20X, hue=(55, 102), vmin=60, min_px=600, n=6):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, V = hsv[:, :, 0].astype(int), hsv[:, :, 2].astype(int)
    m = ((H > hue[0]) & (H < hue[1]) & (V > vmin)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    nn, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in sorted(range(1, nn), key=lambda j: -st[j, 4])[:n]:
        if st[i, 4] < min_px: continue
        c = (lab == i).astype(np.float32); ys, xs = np.nonzero(c)
        y0, x0 = max(0, ys.min() - 6), max(0, xs.min() - 6)
        out.append(dict(id=int(i), area_um2=float(st[i, 4] * px_um ** 2),
                        origin=(int(x0), int(y0)), full=c,
                        crop=c[y0:ys.max() + 7, x0:xs.max() + 7]))
    return out


def corners_of(mask, px_um, gray01=None, eps_um=0.35, ang_tol=28.0, min_edge_um=1.2, win_um=0.9):
    cnts, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = []
    for c in cnts:
        if cv2.contourArea(c) * px_um ** 2 < 4.0: continue
        ap = cv2.approxPolyDP(c, eps_um / px_um, True).reshape(-1, 2).astype(float)
        n = len(ap)
        if n < 3: continue
        for i in range(n):
            p0, p1, p2 = ap[(i - 1) % n], ap[i], ap[(i + 1) % n]
            a, b = p0 - p1, p2 - p1
            la, lb = np.linalg.norm(a), np.linalg.norm(b)
            if la * px_um < min_edge_um or lb * px_um < min_edge_um: continue
            ang = np.degrees(np.arccos(np.clip(a @ b / (la * lb), -1, 1)))
            if abs(ang - 90.0) <= ang_tol: pts.append(p1)
    P = np.array(pts, float).reshape(-1, 2)
    if len(P) and gray01 is not None:
        w = max(3, int(win_um / px_um))
        q = np.ascontiguousarray(P.astype(np.float32).reshape(-1, 1, 2))
        img = (np.clip(gray01, 0, 1) * 255).astype(np.uint8)
        cv2.cornerSubPix(img, q, (w, w), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
        q = q.reshape(-1, 2)
        P = q[np.linalg.norm(q - P, axis=1) * px_um < win_um]
    return P


# ---------------------------------------------------------------- the 3 stages
def stage1_flake(om_flake_crop, sem_flake, om_px, sem_px, step=2.0, fine_step=0.25):
    sc = om_px / sem_px; H, W = sem_flake.shape
    F = np.asarray(sem_flake, np.float32)
    best = None
    for mir in (False, True):
        src = om_flake_crop[:, ::-1].copy() if mir else np.asarray(om_flake_crop, np.float32)
        for ang in np.arange(0, 360, step):
            t = _warp_centred(src, sc, ang, (H, W))
            if t.sum() < 60: continue
            dx, dy, v = _fft_shift(t, F)
            if best is None or v > best[-1]: best = (mir, ang, dx, dy, v)
    mir, a0 = best[0], best[1]
    src = om_flake_crop[:, ::-1].copy() if mir else np.asarray(om_flake_crop, np.float32)
    for ang in np.arange(a0 - 3, a0 + 3 + 1e-9, fine_step):
        t = _warp_centred(src, sc, ang, (H, W))
        if t.sum() < 60: continue
        dx, dy, v = _fft_shift(t, F)
        if v > best[-1]: best = (mir, ang, dx, dy, v)
    mir, ang, dx, dy, v = best
    th = np.radians(ang)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    hs, ws = om_flake_crop.shape
    t = np.array([W / 2.0 + dx, H / 2.0 + dy]) - sc * R @ np.array([ws / 2.0, hs / 2.0])
    return (sc, R, t), bool(mir), float(v)


def stage2_corners(T0, om_corners, sem_corners, om_px, sem_px,
                   gate_um=8.0, min_pairs=4, iters=10):
    A = np.asarray(om_corners, float)[:, :2]
    B = np.asarray(sem_corners, float)[:, :2]
    if len(A) < min_pairs or len(B) < min_pairs: return None
    sc = om_px / sem_px; T = T0; hist = []; sel = None; nn = None
    for g in np.linspace(gate_um, gate_um / 4.0, iters):
        P = apply(T, A)
        d = np.linalg.norm(P[:, None, :] - B[None, :, :], axis=2) * sem_px
        nn = d.argmin(1); dmin = d.min(1)
        used, keep = set(), []
        for i in np.argsort(dmin):
            if dmin[i] > g or nn[i] in used: continue
            used.add(nn[i]); keep.append(i)
        if len(keep) < min_pairs: break
        sel = np.array(keep)
        T = umeyama(A[sel], B[nn[sel]], fixed_scale=sc)
        r = np.linalg.norm(apply(T, A[sel]) - B[nn[sel]], axis=1) * sem_px
        hist.append((len(sel), float(np.sqrt((r ** 2).mean()))))
    if sel is None or not hist: return None
    return dict(T=T, n=hist[-1][0], rms_um=hist[-1][1], idx_om=sel, idx_sem=nn[sel], history=hist)


def stage3_edges(T, om_pad_mask, sem_pad_mask, sem_px, trim=0.7, dth=1.5, dxy=25.0, rounds=4, step=1):
    e = cv2.morphologyEx(np.asarray(om_pad_mask, np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(e)
    if len(xs) > 40000:
        k = np.linspace(0, len(xs) - 1, 40000).astype(int); xs, ys = xs[k], ys[k]
    E = np.stack([xs, ys], 1).astype(float)[::step]
    se = cv2.morphologyEx(np.asarray(sem_pad_mask, np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    DT = cv2.distanceTransform((se == 0).astype(np.uint8), cv2.DIST_L2, 5)
    H, W = DT.shape; s, R, t = T

    def cost(th, dx, dy):
        c, sn = np.cos(th), np.sin(th)
        P = (s * ((np.array([[c, -sn], [sn, c]]) @ R) @ E.T)).T + t + np.array([dx, dy])
        xi = np.rint(P[:, 0]).astype(int); yi = np.rint(P[:, 1]).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)
        if ok.sum() < 200: return 1e9
        d = np.sort(DT[yi[ok], xi[ok]])
        return float(d[:max(50, int(trim * len(d)))].mean())

    best = (0.0, 0.0, 0.0); bc = cost(0, 0, 0)
    for r in range(rounds):
        tr, xr = np.radians(dth) / (3 ** r), dxy / (3 ** r)
        moved = True
        while moved:
            moved = False
            for a in (-tr, 0, tr):
                for b in (-xr, 0, xr):
                    for c_ in (-xr, 0, xr):
                        cand = (best[0] + a, best[1] + b, best[2] + c_)
                        v = cost(*cand)
                        if v < bc - 1e-6: bc, best, moved = v, cand, True
    th, dx, dy = best
    c, sn = np.cos(th), np.sin(th)
    return (s, np.array([[c, -sn], [sn, c]]) @ R, t + np.array([dx, dy])), float(bc * sem_px)


# ---------------------------------------------------------------- driver
def register_unguarded(om_bgr, sem_img01, sem_px, sem_flake_mask, sem_pad_mask,
             om_px=OMPX_20X, om_flake_index=None, gate_um=8.0):
    """Returns the transform mapping optical pixels (possibly mirrored) to SEM
    pixels, plus everything needed to apply it."""
    Wom = om_bgr.shape[1]
    gold = optical_gold(om_bgr)
    gray = cv2.cvtColor(om_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    oc = corners_of(gold, om_px, gray)
    flakes = optical_flakes(om_bgr, om_px)
    cand = range(len(flakes)) if om_flake_index is None else [om_flake_index]

    best = None
    for i in cand:
        T1, mir, v = stage1_flake(flakes[i]['crop'], sem_flake_mask, om_px, sem_px)
        if best is None or v > best[0]: best = (v, i, T1, mir)
    v, i, T1, mir = best
    d = flakes[i]; x0, y0 = d['origin']; hs, ws = d['crop'].shape
    if mir:
        gold_m = gold[:, ::-1].copy()
        om_m = d['full'][:, ::-1].copy()
        oc_m = oc.copy(); oc_m[:, 0] = Wom - 1 - oc_m[:, 0]
        X0 = Wom - ws - x0
    else:
        gold_m, om_m, oc_m, X0 = gold.copy(), d['full'].copy(), oc.copy(), x0
    s, R, t = T1
    T0 = (s, R, t - s * R @ np.array([X0, y0]))

    sc_corners = corners_of(sem_pad_mask, sem_px, sem_img01)
    r2 = stage2_corners(T0, oc_m, sc_corners, om_px, sem_px, gate_um=gate_um)
    T2 = r2['T'] if r2 else T0
    T3, resid = stage3_edges(T2, gold_m, sem_pad_mask, sem_px)
    return dict(T=T3, mirrored=mir, om_flake=i, stage1_ncc=v,
                stage2=r2, residual_um=resid,
                gold_mirrored=gold_m, flake_mirrored=om_m, corners_om=oc_m, corners_sem=sc_corners)


# ---------------------------------------------------------------- guarded
def _iou(a, b):
    a = a > 0; b = b > 0
    u = (a | b).sum()
    return float((a & b).sum()) / max(1, float(u))

def register(om_bgr, sem_img01, sem_px, sem_flake_mask, sem_pad_mask,
              om_flake_mask, om_gold_mask, om_px=OMPX_20X, area_ratio=3.0,
              verbose=False):
    """om_flake_mask / om_gold_mask come from ombg.split_gold()."""
    H, W = sem_img01.shape
    Wom = om_bgr.shape[1]
    gray = cv2.cvtColor(om_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    oc = corners_of(om_gold_mask, om_px, gray)
    sc_corners = corners_of(sem_pad_mask, sem_px, sem_img01)
    sem_area = float(sem_flake_mask.sum()) * sem_px ** 2

    n, lb, st, _ = cv2.connectedComponentsWithStats(
        (om_flake_mask > 0).astype(np.uint8), 8)
    cands = [i for i in sorted(range(1, n), key=lambda j: -st[j, 4])[:8]
             if st[i, 4] * om_px ** 2 > 50]
    if sem_area > 0:                                    # guard 2a
        keep = [i for i in cands
                if 1.0 / area_ratio <= (st[i, 4] * om_px ** 2) / sem_area <= area_ratio]
        if keep: cands = keep

    best = None
    for i in cands:
        c = (lb == i).astype(np.float32)
        ys, xs = np.nonzero(c)
        y0, x0 = max(0, ys.min() - 6), max(0, xs.min() - 6)
        crop = c[y0:ys.max() + 7, x0:xs.max() + 7]
        hs, ws = crop.shape
        T1, mir, ncc = stage1_flake(crop, sem_flake_mask, om_px, sem_px)
        full = (lb == i).astype(np.float32)
        if mir:
            gold_m = om_gold_mask[:, ::-1].copy(); fl_m = full[:, ::-1].copy()
            oc_m = oc.copy(); oc_m[:, 0] = Wom - 1 - oc_m[:, 0]; X0 = Wom - ws - x0
        else:
            gold_m, fl_m, oc_m, X0 = om_gold_mask.copy(), full.copy(), oc.copy(), x0
        s, R, t = T1
        T = (s, R, t - s * R @ np.array([X0, y0]))

        def proj(TT):
            return (warp_into(fl_m, TT, (H, W)) > 0.5).astype(np.uint8)

        cur, cur_s, stage = T, _iou(proj(T), sem_flake_mask), "stage1"
        r2 = stage2_corners(T, oc_m, sc_corners, om_px, sem_px, gate_um=8.0)
        if r2:                                           # guard 1
            v = _iou(proj(r2['T']), sem_flake_mask)
            if v > cur_s: cur, cur_s, stage = r2['T'], v, "stage2"
        T3, resid = stage3_edges(cur, gold_m, sem_pad_mask, sem_px)
        v = _iou(proj(T3), sem_flake_mask)
        if v > cur_s: cur, cur_s, stage = T3, v, "stage3"

        rec = dict(T=cur, score=cur_s, stage=stage, ncc=ncc, mirrored=mir,
                   cand=i, om_area=st[i, 4] * om_px ** 2, resid_um=resid,
                   flake_mirrored=fl_m, gold_mirrored=gold_m)
        if verbose:
            print("    cand %d  area %7.0f  ncc %.3f  -> %s  referee %.3f"
                  % (i, rec['om_area'], ncc, stage, cur_s))
        if best is None or cur_s > best['score']: best = rec
    return best




# ---------------------------------------------------------------- pipeline
def prepare_optical(om_bgr, om_px=OMPX_20X):
    """Optical image -> (gold mask, flake mask), via Lab background removal."""
    from . import optical
    mat, _, _, _ = optical.segment(om_bgr, px_um=om_px)
    mat = optical.drop_overlays(om_bgr, mat, px_um=om_px)
    return optical.split_gold(om_bgr, mat, px_um=om_px)


def sem_pads(sem_img01, sem_px):
    from .contacts import contacts
    from .substrate import substrate_mask
    sub, _, _ = substrate_mask(sem_img01, sem_px)
    pads, _, _ = contacts(sem_img01, sem_px, sub)
    return pads


def om_to_sem(om_images, sem_img01, sem_px, sem_flake_mask, om_px=OMPX_20X):
    """Register every candidate optical image to one SEM frame and keep the
    best by the referee score (overlap with ``sem_flake_mask``, which should
    be the MODEL's mask).

    om_images: {name: BGR uint8 image}.
    Returns dict(om=name, mask=optical flake outline warped into the SEM frame,
    om_warped=the optical image warped into the SEM frame (BGR), plus the
    register() record) or None if nothing could be matched.
    """
    H, W = sem_img01.shape
    pads = sem_pads(sem_img01, sem_px)
    best = None
    for name, bgr in om_images.items():
        gold, fl = prepare_optical(bgr, om_px)
        r = register(bgr, sem_img01, sem_px, sem_flake_mask, pads, fl, gold, om_px=om_px)
        if r and (best is None or r["score"] > best["score"]):
            best = dict(r, om=name, _bgr=bgr)
    if best is None:
        return None
    bgr = best.pop("_bgr")
    img = bgr[:, ::-1].copy() if best["mirrored"] else bgr
    M = mat(best["T"]).astype(np.float32)
    best["om_warped"] = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
    best["mask"] = (warp_into(best["flake_mirrored"], best["T"], (H, W)) > 0.5).astype(np.uint8)
    return best
