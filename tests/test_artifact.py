"""The model file: self-describing, refuses mismatches, deterministic end to end."""
import json

import joblib
import numpy as np
import pytest

from flakecv import cli, io, labels as L, model as Mdl

import synth


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """A tiny data root: three synthetic labelled frames."""
    d = tmp_path_factory.mktemp("data")
    for i in range(3):
        img, flake, px = synth.frame(i)
        sem = io.SemFrame(image=io.quantize01(img), raw=img, px_um=px, hv_kV=20.0)
        io.save_prepared(d, f"syn-{i}", sem)
        L.save(io.frame_paths(d, f"syn-{i}")["label"], flake)
    return d


@pytest.fixture(scope="module")
def artifact(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("m") / "m.joblib"
    assert cli.main(["--data", str(root), "train", "--out", str(out)]) == 0
    return out


def test_artifact_is_self_describing(artifact, root):
    meta = joblib.load(artifact)["meta"]
    assert meta["format"] == Mdl.FORMAT
    assert meta["feature_set"] == "framing" and len(meta["feature_names"]) == 12
    assert meta["feature_params"]
    assert meta["model"]["params"] == Mdl.HGB_PARAMS and meta["model"]["seed"] == 0
    assert {"numpy", "scikit-learn", "scikit-image", "opencv-python-headless", "python"} <= set(meta["libraries"])
    assert {"flakecv_version", "git", "postproc", "training"} <= set(meta)
    t = meta["training"][0]
    assert t["image_sha256"] == Mdl.sha256(io.frame_paths(root, "syn-0")["image"])
    assert t["label_sha256"] == Mdl.sha256(io.frame_paths(root, "syn-0")["label"])
    json.dumps(meta)                                 # plain data only
    # 3 training frames -> central/fill on
    assert meta["postproc"]["central"] and meta["postproc"]["fill"]


def test_shape_priors_off_with_two_frames(root, tmp_path):
    out = tmp_path / "two.joblib"
    cli.main(["--data", str(root), "train", "--frames", "syn-0", "syn-1", "--out", str(out)])
    pp = joblib.load(out)["meta"]["postproc"]
    assert pp["central"] is False and pp["fill"] is False and pp["search_um"] == 1.5


def _tamper(src, dst, fn):
    blob = joblib.load(src)
    fn(blob["meta"])
    joblib.dump(blob, dst)
    return dst


def test_refuses_feature_mismatch(artifact, tmp_path):
    p = _tamper(artifact, tmp_path / "a.joblib", lambda m: m["feature_names"].pop())
    with pytest.raises(Mdl.ArtifactError, match="feature list differs"):
        Mdl.load(p)
    p = _tamper(artifact, tmp_path / "b.joblib", lambda m: m["feature_params"].update(local_um=0.7))
    with pytest.raises(Mdl.ArtifactError, match="feature parameters differ"):
        Mdl.load(p)


def test_refuses_major_library_mismatch(artifact, tmp_path):
    p = _tamper(artifact, tmp_path / "c.joblib",
                lambda m: m["libraries"].update({"scikit-learn": "0.24.2"}))
    with pytest.raises(Mdl.ArtifactError, match="scikit-learn major version"):
        Mdl.load(p)


def test_minor_library_mismatch_only_warns(artifact, tmp_path):
    def bump(m):
        maj = m["libraries"]["scikit-learn"].split(".")[0]
        m["libraries"]["scikit-learn"] = f"{maj}.0.0"
    p = _tamper(artifact, tmp_path / "d.joblib", bump)
    with pytest.warns(UserWarning, match="scikit-learn"):
        Mdl.load(p)


def test_predict_cli_refuses_with_clear_message(artifact, root, tmp_path, capsys):
    p = _tamper(artifact, tmp_path / "e.joblib", lambda m: m.update(feature_set="nope"))
    rc = cli.main(["--data", str(root), "predict", "syn-0", "--model", str(p),
                   "--out", str(tmp_path / "o")])
    assert rc == 3
    assert "refusing to use model" in capsys.readouterr().err


def test_same_input_same_seed_byte_identical_mask(root, tmp_path):
    """Train twice, predict twice: the mask files are byte-identical."""
    outs = []
    for k in range(2):
        m = tmp_path / f"m{k}.joblib"
        cli.main(["--data", str(root), "train", "--out", str(m)])
        o = tmp_path / f"pred{k}"
        cli.main(["--data", str(root), "predict", "syn-2", "--model", str(m), "--out", str(o)])
        outs.append((o / "syn-2.mask.png").read_bytes())
    assert outs[0] == outs[1]


def test_predict_png_needs_pixel_size(artifact, root, tmp_path):
    png = io.frame_paths(root, "syn-0")["image"]
    lone = tmp_path / "lone.png"; lone.write_bytes(png.read_bytes())
    with pytest.raises(SystemExit, match="px-um"):
        cli.main(["predict", str(lone), "--model", str(artifact), "--out", str(tmp_path / "o")])
    assert cli.main(["predict", str(lone), "--px-um", "0.05", "--model", str(artifact),
                     "--out", str(tmp_path / "o")]) == 0
    mask = L.load(tmp_path / "o" / "lone.mask.png")
    assert set(np.unique(mask)) <= {0, 1}
