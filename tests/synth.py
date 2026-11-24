"""Deterministic synthetic SEM-like frames and masks for tests.

Not physics - just enough structure (dark substrate, a mid-grey flake, bright
rectangular contacts, Gaussian noise) to exercise every code path with
known geometry.
"""
import cv2
import numpy as np


def frame(seed=0, H=240, W=320, px_um=0.05):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 0.20, np.float32)
    # contacts: bright bars entering from the left and right edges
    img[60:90, :70] = 0.85
    img[150:175, W - 80:] = 0.85
    # flake: an irregular polygon in the middle, touching one contact
    cx, cy = W // 2 + rng.integers(-10, 10), H // 2 + rng.integers(-10, 10)
    ang = np.sort(rng.uniform(0, 2 * np.pi, 9))
    rad = rng.uniform(45, 75, 9)
    poly = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
    flake = np.zeros((H, W), np.uint8)
    cv2.fillPoly(flake, [poly], 1)
    img[flake > 0] = 0.32
    img[(flake > 0) & (img > 0.8)] = 0.90       # flake on top of a contact
    img += rng.normal(0, 0.012, img.shape).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return np.clip(img, 0, 1).astype(np.float32), flake, px_um


def random_label(seed, H=120, W=160, p_unc=0.1):
    """A blobby three-class label (0/1/255) and a perturbed prediction."""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(size=(H, W)).astype(np.float32), (0, 0), 6)
    ref = np.where(noise > 0.02, 1, 0).astype(np.uint8)
    unc_noise = cv2.GaussianBlur(rng.normal(size=(H, W)).astype(np.float32), (0, 0), 3)
    ref[unc_noise > np.quantile(unc_noise, 1 - p_unc)] = 255
    pn = cv2.GaussianBlur(rng.normal(size=(H, W)).astype(np.float32), (0, 0), 4)
    pred = ((noise + 0.4 * pn) > 0.0).astype(np.uint8)
    return pred, ref


def disk(H, W, cx, cy, r):
    yy, xx = np.mgrid[0:H, 0:W]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8)
