"""features / rules / model / postproc on synthetic frames: shapes, units,
switches, determinism.  The real-data regression lives in test_regression.py."""
from collections import Counter

import numpy as np
import pytest

from flakecv import curve, features as F, model, postproc, rules

import synth


@pytest.fixture(scope="module")
def frame():
    img, flake, px = synth.frame(0)
    return img, flake, px


def test_feature_sets(frame):
    img, _, px = frame
    X10, i10 = F.compute(img, px, "photometric")
    X12, i12 = F.compute(img, px, "framing")
    assert X10.shape == img.shape + (10,) and X12.shape == img.shape + (12,)
    assert i12["names"] == F.PHOTOMETRIC + ["dist_border", "geo_border"]
    assert np.array_equal(X12[..., :10], X10)
    assert X12.dtype == np.float32
    with pytest.raises(ValueError):
        F.compute(img, px, "nope")


def test_framing_features_are_in_um(frame):
    img, _, px = frame
    db, geo = F.framing(img, px)
    H, W = img.shape
    # centre pixel is min(H, W)/2 pixels from the edge -> that many * px um
    assert db[H // 2, W // 2] == pytest.approx((min(H, W) // 2 - 1) * px, rel=0.05)
    db2, _ = F.framing(img, px * 2)
    assert np.allclose(db2, 2 * db)


def test_geo_border_separates_centred_flake_from_contact(frame):
    """The whole point of the prior: a contact is a short hop from the edge,
    the centred flake a long one, although the two are connected."""
    img, flake, px = frame
    _, geo = F.framing(img, px)
    contact = np.zeros_like(flake); contact[60:90, 5:60] = 1
    core = (flake > 0) & (img < 0.6)
    assert np.median(geo[core]) > 2 * np.median(geo[contact > 0])


def test_rules_runs(frame):
    img, flake, px = frame
    m, polys, info = rules.segment(img, px)
    assert m.shape == img.shape and set(np.unique(m)) <= {0, 1}


def test_shape_priors_off_below_three_frames():
    assert not postproc.shape_priors_default(None)
    assert not postproc.shape_priors_default(1)
    assert not postproc.shape_priors_default(2)
    assert postproc.shape_priors_default(3)


def test_finish_switches(frame):
    img, flake, px = frame
    p = np.zeros(img.shape, np.float32)
    p[flake > 0] = 0.9
    p[30:95, :70] = 0.9                     # an 11 um^2 "contact" touching the edge
    p[118:122, 158:162] = 0.0               # a hole
    loose = postproc.finish(p, img, px, n_train=1, search_um=0)
    strict = postproc.finish(p, img, px, n_train=5, search_um=0)
    assert loose[75, 2] == 1                # contact kept with 1 training frame
    assert strict[75, 2] == 0               # dropped by `central` with 5
    assert strict.sum() < loose.sum()
    forced = postproc.finish(p, img, px, n_train=1, central=True, fill=True, search_um=0)
    assert np.array_equal(forced, strict)


def test_snap_keeps_region(frame):
    img, flake, px = frame
    m = postproc.clean(flake.astype(np.float32), px)
    s = postproc.snap(m, img, px)
    inter = (s & m).sum() / max(1, (s | m).sum())
    assert inter > 0.9


def test_model_is_deterministic(frame):
    img, flake, px = frame
    X, _ = F.compute(img, px)
    lab = flake.astype(np.uint8)
    p1 = model.predict_proba(model.fit([(X, lab)], seed=0, n_per_class=2000), X)
    p2 = model.predict_proba(model.fit([(X, lab)], seed=0, n_per_class=2000), X)
    assert np.array_equal(p1, p2)
    m1 = postproc.finish(p1, img, px, n_train=1)
    m2 = postproc.finish(p2, img, px, n_train=1)
    assert m1.tobytes() == m2.tobytes()


def test_curve_plan_run_counts():
    c = Counter(k for k, *_ in curve.plan())
    assert [c[k] for k in (1, 2, 3, 4, 5)] == [48, 48, 24, 24, 6]


def test_rank_without_ties_equals_source_definition():
    a = np.random.default_rng(0).permutation(500).astype(np.float32).reshape(20, 25)
    order = np.argsort(a.ravel())
    ref = np.empty(a.size, np.float32); ref[order] = np.linspace(0, 1, a.size, dtype=np.float32)
    assert np.allclose(F.rank01(a), ref.reshape(a.shape), atol=1e-6)


def test_rank_ties_get_one_value_and_ignore_position():
    a = np.array([[3, 1, 1], [2, 1, 3]], np.float32)
    r = F.rank01(a)
    # positions 0..5: 1,1,1 -> mean(0,1,2)=1; 2 -> 3; 3,3 -> 4.5; scaled by 1/5
    assert np.allclose(r, [[0.9, 0.2, 0.2], [0.6, 0.2, 0.9]])
    assert np.array_equal(F.rank01(a[:, ::-1]), r[:, ::-1])   # moving pixels moves ranks only


def test_rank_8bit_frame_is_order_independent(frame):
    img = (np.round(frame[0] * 255) / 255).astype(np.float32)        # heavy ties
    perm = np.random.default_rng(1).permutation(img.size)
    r1 = F.rank01(img).ravel()
    r2 = F.rank01(img.ravel()[perm])
    assert np.array_equal(r1[perm], r2)
