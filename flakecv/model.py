"""Per-pixel classifier: fit and probability prediction.

Small on purpose: the accuracy curve asks whether MORE DATA helps, so the
model needs enough capacity to use data but not so much that it memorises.

    HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    trained on up to 40 000 pixels per class per frame, UNCERTAIN excluded.

Ported from ``eval/pixelmodel.py``.  The source default was 20 000 per class;
every published number used 40 000, so that is the default here.

A trained model is saved as ONE self-describing file (:func:`save`): the
classifier plus everything needed to use it correctly - feature set and
parameters, post-processing settings, training frames with checksums,
package version and git SHA, seed, and library versions.  :func:`load`
refuses a file whose feature set or a library's major version does not match
the running code, instead of silently producing wrong masks.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import FLAKE, UNCERTAIN

HGB_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                  l2_regularization=1.0, random_state=0)
N_PER_CLASS = 40000


def sample(X, lab, n_per_class=N_PER_CLASS, rng=None):
    rng = rng or np.random.default_rng(0)
    y = (lab == FLAKE).astype(np.int8)
    valid = (lab != UNCERTAIN)
    out_x, out_y = [], []
    for c in (0, 1):
        idx = np.flatnonzero(valid.ravel() & (y.ravel() == c))
        if len(idx) == 0:
            continue
        if len(idx) > n_per_class:
            idx = rng.choice(idx, n_per_class, replace=False)
        out_x.append(X.reshape(-1, X.shape[-1])[idx]); out_y.append(np.full(len(idx), c, np.int8))
    if not out_x:
        return None, None
    return np.concatenate(out_x), np.concatenate(out_y)


def fit(frames, seed=0, n_per_class=N_PER_CLASS, max_iter=200):
    """frames: list of (X, label).  ``seed`` drives pixel sampling only (the
    booster itself is always random_state=0, as in every measurement)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for X, lab in frames:
        a, b = sample(X, lab, n_per_class, rng)
        if a is not None:
            xs.append(a); ys.append(b)
    if not xs:
        return None
    clf = HistGradientBoostingClassifier(**{**HGB_PARAMS, "max_iter": max_iter})
    clf.fit(np.concatenate(xs), np.concatenate(ys))
    return clf


def predict_proba(clf, X) -> np.ndarray:
    """Flake probability per pixel, HxW float."""
    return clf.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(X.shape[:2])


# ------------------------------------------------------------------ artifact
FORMAT = "flakecv-model/1"
CHECKED_LIBS = ("numpy", "scikit-learn", "scikit-image", "opencv-python-headless", "scipy")


class ArtifactError(RuntimeError):
    pass


def library_versions() -> dict:
    import platform
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for lib in CHECKED_LIBS + ("joblib",):
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = None
    return out


def git_sha() -> dict:
    """{'sha': ..., 'dirty': bool} of the repo this package lives in, if any."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=root,
                           capture_output=True, text=True, timeout=10)
        sha = r.stdout.strip() if r.returncode == 0 else None   # None: no commit yet
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "flakecv"],
                                    cwd=root, capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        sha, dirty = None, None
    return {"sha": sha, "dirty": dirty}


def sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, clf, *, feature_set, training, seed, n_per_class=N_PER_CLASS, postproc=None):
    """Write the single-file artifact.

    training: list of {name, px_um, hv_kV, image_sha256, label_sha256}.
    postproc: settings to use at prediction time (defaults derived from the
    number of training frames - see :mod:`flakecv.postproc`).
    """
    import pickle
    from pathlib import Path

    import joblib

    from . import __version__
    from . import features as F
    from . import postproc as PP

    n = len(training)
    pp = dict(PP.DEFAULTS, central=PP.shape_priors_default(n), fill=PP.shape_priors_default(n))
    pp.update(postproc or {})
    meta = dict(
        format=FORMAT,
        flakecv_version=__version__, git=git_sha(),
        feature_set=feature_set, feature_names=F.names(feature_set),
        feature_params=F.PARAMS,
        model=dict(type="HistGradientBoostingClassifier", params=HGB_PARAMS,
                   n_per_class=n_per_class, seed=seed),
        postproc=pp, training=training, libraries=library_versions(),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # the classifier is stored as bytes so the metadata can be checked
    # BEFORE anything version-dependent is unpickled
    joblib.dump(dict(meta=meta, clf=pickle.dumps(clf, protocol=5)), path, compress=3)
    return meta


def _major(v):
    return None if v is None else str(v).split(".")[0]


def check(meta) -> list[str]:
    """Problems that make the artifact unusable here (empty list = OK)."""
    from . import features as F
    errs = []
    if meta.get("format") != FORMAT:
        errs.append(f"unknown file format {meta.get('format')!r} (expected {FORMAT})")
        return errs
    fs = meta["feature_set"]
    try:
        names = F.names(fs)
    except ValueError:
        errs.append(f"feature set {fs!r} does not exist in this version of flakecv")
        names = None
    if names is not None and names != meta["feature_names"]:
        errs.append(f"feature list differs: model {meta['feature_names']} vs code {names}")
    mp, cp = _jsonish(meta["feature_params"]), _jsonish(F.PARAMS)
    if mp != cp:
        diff = {k for k in set(mp) | set(cp) if mp.get(k) != cp.get(k)}
        errs.append(f"feature parameters differ: {sorted(diff)}")
    here = library_versions()
    for lib in CHECKED_LIBS:
        a, b = _major(meta["libraries"].get(lib)), _major(here.get(lib))
        if a != b:
            errs.append(f"{lib} major version differs: model fitted with "
                        f"{meta['libraries'].get(lib)}, running {here.get(lib)}")
    return errs


def _jsonish(meta):
    """JSON round trip, so tuple/list differences don't count as changes."""
    import json
    return json.loads(json.dumps(meta))


def load(path, strict=True):
    """Returns (clf, meta).  Raises ArtifactError with every problem listed."""
    import pickle
    import warnings

    import joblib

    blob = joblib.load(path)
    meta = blob["meta"]
    errs = check(meta)
    if errs and strict:
        raise ArtifactError(f"refusing to use model {path}:\n  - " + "\n  - ".join(errs) +
                            "\nRetrain with `flakecv train`, or install the versions in "
                            "requirements-lock.txt.")
    here = library_versions()
    sk_model, sk_here = meta["libraries"].get("scikit-learn"), here.get("scikit-learn")
    if sk_model != sk_here:
        warnings.warn(f"model fitted with scikit-learn {sk_model}, running {sk_here}; "
                      "predictions may differ slightly", stacklevel=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # sklearn's own version warning: reported above
        clf = pickle.loads(blob["clf"])
    return clf, meta


def probability_image(clf, meta, img01, px_um):
    """Flake probability map for one frame (features as recorded in ``meta``)."""
    from . import features as F
    X, _ = F.compute(img01, px_um, meta["feature_set"])
    return predict_proba(clf, X)


def postproc_settings(meta, **overrides) -> dict:
    pp = dict(meta["postproc"])
    pp.update({k: v for k, v in overrides.items() if v is not None})
    return pp


def finish_image(meta, p, img01, px_um, **overrides):
    """Probability map -> mask, with the artifact's settings (overridable)."""
    from . import postproc as PP
    pp = postproc_settings(meta, **overrides)
    return PP.finish(p, img01, px_um, central=pp["central"], fill=pp["fill"], thr=pp["thr"],
                     min_area_um2=pp["min_area_um2"], close_um=pp["close_um"],
                     search_um=pp["search_um"])


def predict_image(clf, meta, img01, px_um, **postproc_overrides):
    """(mask, probability) for one frame, using the settings stored in the
    artifact (any of them can be overridden)."""
    p = probability_image(clf, meta, img01, px_um)
    return finish_image(meta, p, img01, px_um, **postproc_overrides), p
