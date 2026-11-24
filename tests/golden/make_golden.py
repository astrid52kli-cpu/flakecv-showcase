"""Record what the ORIGINAL research code returns on deterministic synthetic
inputs, so the port can be checked against it number-for-number.

    python tests/golden/make_golden.py --legacy <path to flakecv-source>

Writes tests/golden/legacy_values.json.  Run once; the JSON is committed and
the tests compare against it without needing the legacy code.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import synth  # noqa: E402


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", required=True, help="folder that contains eval/ and sem/")
    a = ap.parse_args()
    leg = Path(a.legacy).resolve()
    sys.path.insert(0, str(leg / "sem"))      # subtract.py (label.py looks in ../cc)
    sys.path.insert(0, str(leg / "eval"))
    L = importlib.import_module("label")
    M = importlib.import_module("metrics")
    S = importlib.import_module("subtract")

    out = {"metrics": [], "agreement": [], "prefill": [], "substrate": []}
    for seed in range(12):
        pred, ref = synth.random_label(seed)
        for px in (0.031, 0.05, 0.122):
            r = M.score(pred, ref, px)
            out["metrics"].append(dict(seed=seed, px_um=px,
                                       **{k: (None if not np.isfinite(v) else v) for k, v in r.items()}))
    for seed in range(6):
        _, a1 = synth.random_label(seed)
        _, b1 = synth.random_label(seed + 100)
        r = M.agreement(a1, b1, 0.05)
        out["agreement"].append(dict(seed=seed, **r))
    for seed in range(4):
        img, flake, px = synth.frame(seed)
        lvl, sig = S.substrate_level(img, px)
        m, ml, ms = S.substrate_mask(img, px)
        out["substrate"].append(dict(seed=seed, level=lvl, sigma=sig,
                                     mask_level=ml, mask_sigma=ms, mask_sha=sha(m),
                                     noise=S.pixel_noise(img)))
        pads = np.zeros_like(flake)
        pads[60:90, :70] = 1
        for use_pads in (False, True):
            lab, info = L.prefill(img, px, flake_mask=flake, pads=pads if use_pads else None)
            out["prefill"].append(dict(seed=seed, pads=use_pads, sha=sha(lab), **info))
    p = HERE / "legacy_values.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p, {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
