"""The in-app labeller's Python side: what the canvas sends back becomes a
label file, frames are added when needed, old labels are backed up."""
import numpy as np
import pytest

from flakecv import FLAKE, UNCERTAIN, io, labels as L, labelui as LU


def _lab(H=60, W=80):
    lab = np.zeros((H, W), np.uint8)
    lab[10:40, 20:60] = FLAKE
    lab[10:40, 58:62] = UNCERTAIN
    return lab


def test_label_url_roundtrip():
    lab = _lab()
    assert np.array_equal(LU.decode_label(LU.label_url(lab), lab.shape), lab)


def test_decode_rejects_wrong_shape_and_garbage():
    with pytest.raises(ValueError):
        LU.decode_label(LU.label_url(_lab()), (61, 80))
    with pytest.raises(ValueError):
        LU.decode_label("data:image/png;base64,AAAA", (60, 80))
    with pytest.raises(ValueError):
        LU.decode_label(None, (60, 80))


def test_save_adds_frame_then_backs_up(tmp_path):
    img = np.linspace(0, 1, 60 * 80, dtype=np.float32).reshape(60, 80)
    sem = io.SemFrame(image=io.quantize01(img), raw=img, px_um=0.05)
    with pytest.raises(FileNotFoundError):
        LU.save_label(tmp_path, "f1", _lab())
    info = LU.save_label(tmp_path, "f1", _lab(), sem)
    assert info["wrote_frame"] and info["backup"] is None
    fr = io.load_frame(tmp_path, "f1")
    assert fr.px_um == 0.05 and fr.image.shape == (60, 80)
    assert np.array_equal(L.load(info["label"]), _lab())
    lab2 = _lab(); lab2[:] = 0
    info2 = LU.save_label(tmp_path, "f1", lab2, sem)
    old = info2["backup"] / "labels" / "f1.label.png"
    assert not info2["wrote_frame"] and old.is_file()
    assert np.array_equal(L.load(old), _lab())
    assert np.array_equal(L.load(info2["label"]), lab2)
    assert io.list_frames(tmp_path) == ["f1"]


def test_from_mask_band():
    m = np.zeros((60, 80), np.uint8); m[10:50, 10:70] = 1
    lab = LU.from_mask(m)
    assert set(np.unique(lab)) == {0, FLAKE}
    lab = LU.from_mask(m, 0.25, 0.05)
    assert (lab == UNCERTAIN).any() and lab[30, 40] == FLAKE and lab[0, 0] == 0


def _sem(val=0.3, shape=(60, 80)):
    img = np.full(shape, val, np.float32)
    return io.SemFrame(image=io.quantize01(img), raw=img, px_um=0.05)


def test_replace_frame_backs_up_and_drops_label(tmp_path):
    r = LU.save_frame(tmp_path, "f1", _sem(0.3), b"RAW1", ".TIF")
    assert r["raw"].name == "f1.tif" and r["backup"] is None
    with pytest.raises(FileExistsError):
        LU.save_frame(tmp_path, "f1", _sem(0.6))
    LU.save_label(tmp_path, "f1", _lab())
    r = LU.save_frame(tmp_path, "f1", _sem(0.6, (50, 70)), b"RAW2", ".tiff", replace=True)
    assert r["removed_label"]
    assert not io.frame_paths(tmp_path, "f1")["label"].is_file()
    assert (r["backup"] / "labels" / "f1.label.png").is_file()
    assert (r["backup"] / "raw" / "f1.tif").read_bytes() == b"RAW1"
    assert [p.name for p in LU.raw_paths(tmp_path, "f1")] == ["f1.tiff"]
    assert io.load_frame(tmp_path, "f1").image.shape == (50, 70)


def test_optical_pairing_and_replacement(tmp_path):
    import cv2
    ok, a = cv2.imencode(".bmp", np.full((30, 40, 3), 90, np.uint8))
    ok, b = cv2.imencode(".png", np.full((20, 25, 3), 200, np.uint8))
    assert LU.optical_for(tmp_path, "f1") is None
    r = LU.save_optical(tmp_path, "f1", a.tobytes(), "my om 50x.bmp", "50x", None)
    assert r["om_px_um"] == 0.090 and r["backup"] is None
    got = LU.optical_for(tmp_path, "f1")
    assert got["path"].name == "f1.om.bmp" and got["source_file"] == "my om 50x.bmp"
    r = LU.save_optical(tmp_path, "f1", b.tobytes(), "new.png", "40x", None)
    got = LU.optical_for(tmp_path, "f1")
    assert got["path"].name == "f1.om.png" and got["om_px_um"] is None and got["shape"] == [20, 25]
    assert not (tmp_path / LU.PAIRED / "f1.om.bmp").exists()
    assert (r["backup"] / LU.PAIRED / "f1.om.bmp").is_file()
    with pytest.raises(ValueError):
        LU.save_optical(tmp_path, "f1", b"not an image", "x.bmp", "20x", None)
