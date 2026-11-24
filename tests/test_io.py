"""SEM TIFF reading.  No real XL30 frame is committed yet, so these use
synthetic TIFFs carrying an FEI-style header in the FEI tags."""
import json

import numpy as np
import pytest
import tifffile

from flakecv import io

INI = """[User]
Date=10/15/2026
[Beam]
HV=20000
[Scan]
PixelWidth=4.2e-08
PixelHeight=4.2e-08
[Stage]
StageX=0.0012345
StageY=-0.0006789
[Image]
ResolutionX={W}
ResolutionY={H}
"""


def write_fei(path, img, ini, tag=34682):
    tifffile.imwrite(str(path), img, extratags=[(tag, "s", 0, ini, True)])


@pytest.mark.parametrize("tag", [34682, 34680])
def test_reads_fei_header_and_strips_databar(tmp_path, tag):
    H, W, bar = 200, 256, 24
    rng = np.random.default_rng(0)
    img = rng.integers(1000, 30000, (H + bar, W)).astype(np.uint16)
    img[H:] = 0; img[H:, ::7] = 65535                     # databar
    p = tmp_path / f"f{tag}.tif"
    write_fei(p, img, INI.format(W=W, H=H), tag)
    f = io.read_sem_tiff(p)
    assert f.px_um == pytest.approx(0.042)
    assert f.hv_kV == pytest.approx(20.0)
    assert f.stage_um == pytest.approx((1234.5, -678.9))
    assert f.databar_rows == bar
    assert f.image.shape == (H, W) and f.raw.shape == (H, W)
    assert f.image.dtype == np.float32
    assert 0.0 <= f.image.min() and f.image.max() <= 1.0
    assert f.bit_depth == 16 and f.raw.dtype == np.uint16


def test_missing_pixel_size_is_refused(tmp_path):
    p = tmp_path / "nohdr.tif"
    tifffile.imwrite(str(p), np.zeros((50, 50), np.uint16))
    with pytest.raises(io.MetadataError, match="pixel size"):
        io.read_sem_tiff(p)


def test_implausible_pixel_size_is_refused(tmp_path):
    p = tmp_path / "bad.tif"
    write_fei(p, np.zeros((50, 50), np.uint16), "[Scan]\nPixelWidth=42\n")
    with pytest.raises(io.MetadataError, match="implausible"):
        io.read_sem_tiff(p)


def test_databar_fallback_detection():
    rng = np.random.default_rng(1)
    a = rng.integers(50, 200, (120, 100)).astype(np.uint8)
    a[100:] = 0; a[100:, ::3] = 255
    assert io.detect_databar(a) == 20
    assert io.detect_databar(rng.integers(50, 200, (120, 100)).astype(np.uint8)) == 0


def test_unicode_path_tiff(tmp_path):
    d = tmp_path / "Ünïcode dir é"; d.mkdir()
    p = d / "frame ü.tif"
    write_fei(p, np.full((40, 30), 500, np.uint16), INI.format(W=30, H=40))
    assert io.read_sem_tiff(p).px_um == pytest.approx(0.042)


def test_prepared_frame_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    raw = rng.integers(0, 4096, (64, 80)).astype(np.uint16)
    sem = io.SemFrame(image=io.normalize(raw), raw=raw, px_um=0.05, hv_kV=20.0,
                      stage_um=(1.0, 2.0), bit_depth=16)
    io.save_prepared(tmp_path, "261015-x-1", sem)
    assert io.list_frames(tmp_path) == ["261015-x-1"]
    fr = io.load_frame(tmp_path, "261015-x-1")
    assert fr.px_um == 0.05 and fr.hv_kV == 20.0
    # load_frame == imread(png, 0) / 255, exactly as the original experiments
    expect = io.quantize01(sem.image)
    assert np.array_equal(fr.image, expect)


def test_load_frame_missing_meta(tmp_path):
    with pytest.raises(FileNotFoundError, match="no metadata"):
        io.load_frame(tmp_path, "nope")


def test_optical_scales_are_the_measured_ones():
    assert io.OPTICAL_PX_UM == {"20x": 0.2475, "50x": 0.090}


def test_normalize_is_the_original_1_99_stretch():
    a = np.arange(1000, dtype=np.uint16).reshape(20, 50)
    n = io.normalize(a)
    p1, p99 = np.percentile(a.astype(np.float32), 1), np.percentile(a.astype(np.float32), 99)
    assert np.array_equal(n, np.clip((a.astype(np.float32) - p1) / (p99 - p1 + 1e-9), 0, 1))


def test_tiff_image_equals_prepared_png_path(tmp_path):
    """Raw TIFF in -> exactly the array a prepared PNG of it would give."""
    H, W = 60, 70
    img = np.random.default_rng(3).integers(0, 60000, (H, W)).astype(np.uint16)
    p = tmp_path / "t.tif"; write_fei(p, img, INI.format(W=W, H=H))
    f = io.read_sem_tiff(p)
    io.save_prepared(tmp_path / "d", "t", f)
    assert np.array_equal(io.load_frame(tmp_path / "d", "t").image, f.image)
