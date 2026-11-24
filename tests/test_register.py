"""Registration on synthetic shapes: stage 1 must find a MIRRORED, rotated
copy (the SEM frame is mirrored relative to the optical image)."""
import cv2
import numpy as np

from flakecv import register as R


def _shape(H=200, W=240):
    m = np.zeros((H, W), np.uint8)
    poly = np.array([[60, 40], [170, 55], [190, 120], [120, 160], [70, 130], [90, 90]], np.int32)
    cv2.fillPoly(m, [poly], 1)
    return m


def test_stage1_recovers_mirror_and_rotation():
    om_px, sem_px = 0.25, 0.125                      # 2x scale, fixed by pixel sizes
    om = _shape()
    ys, xs = np.nonzero(om)
    crop = om[ys.min() - 6: ys.max() + 7, xs.min() - 6: xs.max() + 7].astype(np.float32)
    # build the 'SEM' view: mirror, rotate 30 deg, scale x2, place in a bigger frame
    src = crop[:, ::-1]
    H, W = 600, 700
    M = cv2.getRotationMatrix2D((src.shape[1] / 2, src.shape[0] / 2), 30, om_px / sem_px)
    M[0, 2] += W / 2 - src.shape[1] / 2 + 40
    M[1, 2] += H / 2 - src.shape[0] / 2 - 25
    sem = (cv2.warpAffine(src, M, (W, H)) > 0.5).astype(np.uint8)

    T, mirrored, ncc = R.stage1_flake(crop, sem, om_px, sem_px)
    assert mirrored
    proj = (R.warp_into(crop[:, ::-1].copy(), T, sem.shape) > 0.5)
    assert R._iou(proj, sem) > 0.9


def test_umeyama_exact():
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 100, (10, 2))
    th = np.radians(23); Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = (2.0 * (Rm @ P.T)).T + [5, -7]
    s, Rf, t = R.umeyama(P, Q)
    assert np.isclose(s, 2.0) and np.allclose(Rf, Rm) and np.allclose(t, [5, -7])
    assert np.allclose(R.apply((s, Rf, t), P), Q)
