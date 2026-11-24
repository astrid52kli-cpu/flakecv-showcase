import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flakecv import BACKGROUND, FLAKE, UNCERTAIN
from flakecv import labels as L
from flakecv import substrate as S
from flakecv.io import imread, imwrite

import synth

GOLD = json.loads((Path(__file__).parent / "golden" / "legacy_values.json").read_text())


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_disk_encoding_is_0_128_255(tmp_path):
    lab = np.array([[BACKGROUND, FLAKE, UNCERTAIN]], np.uint8)
    p = tmp_path / "x.label.png"
    L.save(p, lab)
    raw = imread(p)
    assert raw.dtype == np.uint8 and raw.ndim == 2
    assert raw.tolist() == [[0, 128, 255]]


def test_roundtrip(tmp_path):
    _, lab = synth.random_label(0)
    p = tmp_path / "r.label.png"
    L.save(p, lab)
    assert np.array_equal(L.load(p), lab)


def test_load_accepts_prefill_0_1_255_and_rgb_and_antialias(tmp_path):
    a = np.array([[0, 1, 255, 128, 100, 220, 30]], np.uint8)
    p = tmp_path / "a.png"; imwrite(p, a)
    assert L.load(p).tolist() == [[0, 1, 255, 1, 1, 255, 0]]
    rgb = np.dstack([a, a, a]); p2 = tmp_path / "b.png"; imwrite(p2, rgb)
    assert L.load(p2).tolist() == [[0, 1, 255, 1, 1, 255, 0]]


def test_unicode_path(tmp_path):
    d = tmp_path / "Ünïcode data é"
    _, lab = synth.random_label(1)
    L.save(d / "label ü.label.png", lab)
    assert np.array_equal(L.load(d / "label ü.label.png"), lab)


def test_save_is_byte_deterministic(tmp_path):
    _, lab = synth.random_label(2)
    L.save(tmp_path / "1.png", lab); L.save(tmp_path / "2.png", lab)
    assert (tmp_path / "1.png").read_bytes() == (tmp_path / "2.png").read_bytes()


def test_stats_sum_to_one():
    _, lab = synth.random_label(4)
    s = L.stats(lab)
    assert s["flake"] + s["background"] + s["uncertain"] == pytest.approx(1.0)


def test_validate():
    L.validate(np.array([0, 1, 255], np.uint8))
    with pytest.raises(ValueError):
        L.validate(np.array([0, 128], np.uint8))


def test_prefill_finds_the_flake():
    img, flake, px = synth.frame(0)
    lab, info = L.prefill(img, px)
    inner = (flake > 0) & (img < 0.7)
    assert (lab[inner] == FLAKE).mean() > 0.9
    assert info["level"] == pytest.approx(0.20, abs=0.01)


@pytest.mark.parametrize("row", GOLD["prefill"], ids=lambda r: f"seed{r['seed']}-pads{r['pads']}")
def test_prefill_matches_original_code(row):
    img, flake, px = synth.frame(row["seed"])
    pads = np.zeros_like(flake); pads[60:90, :70] = 1
    lab, info = L.prefill(img, px, flake_mask=flake, pads=pads if row["pads"] else None)
    assert sha(lab) == row["sha"]
    for k in ("level", "sigma", "frac_sure", "frac_uncertain"):
        assert info[k] == row[k], k


@pytest.mark.parametrize("row", GOLD["substrate"], ids=lambda r: f"seed{r['seed']}")
def test_substrate_matches_original_code(row):
    img, _, px = synth.frame(row["seed"])
    assert S.substrate_level(img, px) == (row["level"], row["sigma"])
    m, lvl, sig = S.substrate_mask(img, px)
    assert (lvl, sig) == (row["mask_level"], row["mask_sigma"])
    assert sha(m) == row["mask_sha"]
    assert S.pixel_noise(img) == row["noise"]
