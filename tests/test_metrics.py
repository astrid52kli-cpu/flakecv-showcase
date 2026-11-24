"""Metrics.  Everything downstream is scored by these, so they are tested three
ways: hand-computed cases, invariants (UNCERTAIN really is ignored; lengths
really are physical), and equality with the original research code."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from flakecv import BACKGROUND, FLAKE, UNCERTAIN
from flakecv import metrics as M

import synth

GOLD = json.loads((Path(__file__).parent / "golden" / "legacy_values.json").read_text())


def lab(*rows):
    return np.array(rows, np.uint8)


# ------------------------------------------------------------ hand-computed
def test_iou_dice_hand_computed():
    #           pred                    ref
    pred = lab([1, 1, 0, 0],
               [1, 1, 0, 0])
    ref = lab([0, 1, 1, 0],
              [0, 1, 1, 0])
    # intersection 2, union 6
    assert M.iou(pred, ref) == pytest.approx(2 / 6)
    assert M.dice(pred, ref) == pytest.approx(2 * 2 / (4 + 4))


def test_perfect_and_disjoint():
    ref = lab([0, 1, 1, 0])
    assert M.iou(ref, ref) == 1.0
    assert M.dice(ref, ref) == 1.0
    assert M.iou(lab([1, 0, 0, 1]), ref) == 0.0


def test_pred_any_positive_is_flake():
    ref = lab([0, 1, 1, 0])
    for v in (1, 2, 128, 255):
        pred = lab([0, v, v, 0])
        assert M.iou(pred, ref) == 1.0
    assert M.iou(np.array([[0, .7, .2, 0]], np.float32), ref) == 1.0


def test_empty_union_is_nan_not_zero_or_one():
    z = np.zeros((4, 4), np.uint8)
    assert math.isnan(M.iou(z, z))
    assert math.isnan(M.dice(z, z))
    assert math.isnan(M.area_error(z, z, 0.05)[0])
    assert math.isnan(M.boundary_f(z, z, 0.05))


# --------------------------------------------------------------- UNCERTAIN
def test_uncertain_pixels_count_neither_way():
    ref = lab([1, 1, 255, 255, 0, 0])
    # pred wrong on both uncertain pixels in two different ways: no effect
    a = lab([1, 1, 0, 0, 0, 0])
    b = lab([1, 1, 1, 1, 0, 0])
    c = lab([1, 1, 1, 0, 0, 0])
    for p in (a, b, c):
        assert M.iou(p, ref) == 1.0
        assert M.dice(p, ref) == 1.0
        assert M.area_error(p, ref, 0.1)[0] == 0.0


def test_all_uncertain_is_nan():
    ref = np.full((5, 5), UNCERTAIN, np.uint8)
    pred = np.ones((5, 5), np.uint8)
    assert math.isnan(M.iou(pred, ref))
    assert math.isnan(M.boundary_f(pred, ref, 0.05))


@pytest.mark.parametrize("seed", range(5))
def test_scores_invariant_to_pred_inside_uncertain(seed):
    """Rewrite the prediction arbitrarily inside the uncertain region: every
    metric must be bit-identical (boundary_f included - boundary points that
    fall on uncertain pixels are dropped)."""
    pred, ref = synth.random_label(seed, p_unc=0.2)
    rng = np.random.default_rng(seed + 1000)
    unc = ref == UNCERTAIN
    for fill in (0, 1, "random"):
        p2 = pred.copy()
        p2[unc] = rng.integers(0, 2, unc.sum()) if fill == "random" else fill
        s1, s2 = M.score(pred, ref, 0.05), M.score(p2, ref, 0.05)
        for k in ("iou", "dice", "area_error", "area_pred_um2", "area_ref_um2"):
            assert s1[k] == s2[k], k


def test_raw_disk_label_is_refused():
    """0/128/255 straight from the PNG would score 128 as background."""
    disk = lab([0, 128, 128, 255])
    with pytest.raises(ValueError, match="labels.load"):
        M.iou(lab([0, 1, 1, 0]), disk)


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="shape"):
        M.iou(np.zeros((3, 3)), np.zeros((3, 4), np.uint8))


# ---------------------------------------------------------- physical units
def test_area_is_in_um2_and_relative_error_is_resolution_free():
    ref = np.zeros((10, 10), np.uint8); ref[2:6, 2:6] = FLAKE      # 16 px
    pred = np.zeros((10, 10), np.uint8); pred[2:6, 2:7] = 1        # 20 px
    for px in (0.031, 0.05, 0.122):
        e, pa, ra = M.area_error(pred, ref, px)
        assert ra == pytest.approx(16 * px * px)
        assert pa == pytest.approx(20 * px * px)
        assert e == pytest.approx(0.25)


def test_boundary_f_identity():
    ref = synth.disk(100, 100, 50, 50, 25)
    assert M.boundary_f(ref, ref, 0.05) == 1.0


@pytest.mark.parametrize("px_um", [0.031, 0.05, 0.068, 0.122])
def test_boundary_f_tolerance_is_physical(px_um):
    """The same physical scene at four pixel sizes: an outline displaced by
    0.15 um is inside the 0.25 um tolerance at every resolution; one displaced
    by 1.0 um is outside it at every resolution."""
    H = W = int(round(20.0 / px_um))            # 20 um field
    c, r = W / 2, 5.0 / px_um                    # 5 um radius disc
    ref = synth.disk(H, W, c, c, r)
    near = synth.disk(H, W, c + 0.15 / px_um, c, r)
    far = synth.disk(H, W, c + 1.0 / px_um, c, r)
    assert M.boundary_f(near, ref, px_um) == 1.0
    assert M.boundary_f(far, ref, px_um) < 0.75


def test_boundary_f_tolerance_floor_is_one_pixel():
    """Coarser than the tolerance: max(1, round(tol/px)) keeps 1 px slack."""
    ref = synth.disk(60, 60, 30, 30, 15)
    shifted = synth.disk(60, 60, 31, 30, 15)
    assert M.boundary_f(shifted, ref, px_um=1.0) == 1.0


def test_boundary_f_ignores_pred_boundary_inside_uncertain():
    ref = synth.disk(100, 100, 40, 50, 20).astype(np.uint8)
    pred = ref.copy()
    pred[40:60, 85:95] = 1                                   # spurious blob
    assert M.boundary_f(pred, ref, 0.05) < 1.0
    ref_u = ref.copy(); ref_u[:, 80:] = UNCERTAIN            # ...in uncertain land
    assert M.boundary_f(pred, ref_u, 0.05) == 1.0


def test_boundary_f_flake_uncertain_interface_counts_as_ref_boundary():
    """KNOWN QUIRK of the original metric, kept on purpose (the regression
    numbers depend on it): where a FLAKE region meets an UNCERTAIN region, the
    flake-side pixels of that interface are ref boundary points.  A prediction
    that runs straight through the uncertain zone - arguably correct - is
    charged recall there.  This biases boundary F downwards on frames whose
    labels have uncertain patches touching the flake (flake-on-contact).
    Changing it requires re-measuring every boundary-F number."""
    ref = synth.disk(100, 100, 50, 50, 25).astype(np.uint8)
    pred = ref.copy()
    ref_u = ref.copy(); ref_u[:, 45:] = UNCERTAIN
    assert M.boundary_f(pred, ref, 0.05) == 1.0
    assert M.boundary_f(pred, ref_u, 0.05, tol_um=0.1) < 1.0


# ---------------------------------------------------------------- agreement
def test_agreement_excludes_uncertain_in_either():
    a = lab([1, 1, 255, 0, 1])
    b = lab([1, 1, 0, 255, 0])
    r = M.agreement(a, b, 0.05)
    # committed in both: columns 0,1,4 -> a={0,1,4} b={0,1}: IoU 2/3
    assert r["iou"] == pytest.approx(2 / 3)
    assert r["both_committed"] == pytest.approx(3 / 5)


def test_agreement_with_self_is_perfect():
    _, a = synth.random_label(3)
    r = M.agreement(a, a, 0.05)
    assert r["iou"] == 1.0 and r["boundary_f"] == 1.0


# -------------------------------------------------- equality with legacy code
@pytest.mark.parametrize("row", GOLD["metrics"],
                         ids=lambda r: f"seed{r['seed']}-px{r['px_um']}")
def test_score_matches_original_code(row):
    pred, ref = synth.random_label(row["seed"])
    s = M.score(pred, ref, row["px_um"])
    for k, v in s.items():
        if row[k] is None:
            assert math.isnan(v), k
        else:
            assert v == row[k], k            # exact, not approx


@pytest.mark.parametrize("row", GOLD["agreement"], ids=lambda r: f"seed{r['seed']}")
def test_agreement_matches_original_code(row):
    _, a = synth.random_label(row["seed"])
    _, b = synth.random_label(row["seed"] + 100)
    r = M.agreement(a, b, 0.05)
    for k in ("iou", "both_committed", "boundary_f"):
        assert r[k] == row[k], k


def test_score_keys():
    pred, ref = synth.random_label(0)
    assert set(M.score(pred, ref, 0.05)) == {
        "iou", "dice", "boundary_f", "area_error",
        "area_pred_um2", "area_ref_um2", "uncertain_frac"}
