"""Leave-one-flake-out learning curve.

Protocol (exactly the one behind the published table):
  * for n = 1..5 training frames, for each held-out frame, take every
    n-combination of the other five; if there are more than 4, draw 4 with a
    single ``np.random.default_rng(7)`` shared across all n, in order
  * pixel-sampling seeds (0, 1) for n <= 2, (0,) otherwise
  -> 48 / 48 / 24 / 24 / 6 runs.  n = 5 has one possible training set per
     held-out flake, which is why it is the noisiest point.

Each fit is scored under several output pipelines at once (they share the
probability map), so "steps 1-3", "+ snap" and "full" come from identical fits.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import features as F
from . import io, labels as L, metrics as M, model, postproc

REGRESSION_FRAMES = ["261015-tpt1-1", "261015-tpt1-2", "261015-tpt1-3",
                     "261103-ely-A", "261103-ely-B", "261103-ely-C"]

PIPELINES = {
    "base": dict(central=False, fill=False, search_um=0),      # steps 1-3
    "fill": dict(central=False, fill=True, search_um=0),
    "central": dict(central=True, fill=False, search_um=0),
    "snap": dict(central=False, fill=False, search_um=1.5),
    "full": dict(central=True, fill=True, search_um=1.5),
}


def load_items(data_root, names=REGRESSION_FRAMES, feature_sets=("framing",), log=print):
    items = []
    for n in names:
        fr = io.load_frame(data_root, n)
        lab = L.load(io.frame_paths(data_root, n)["label"])
        it = dict(name=n, img=fr.image, px=fr.px_um, lab=lab, hv_kV=fr.hv_kV, X={})
        for fs in feature_sets:
            t = time.time()
            it["X"][fs], _ = F.compute(fr.image, fr.px_um, fs)
            log(f"  features {fs:<11} {n}  ({time.time() - t:.1f}s)")
        items.append(it)
    return items


def plan(n_items=6, sizes=(1, 2, 3, 4, 5), max_combs=4, rng_seed=7):
    """The (n, test, train_tuple, seed) list, in the original order."""
    rng = np.random.default_rng(rng_seed)
    out = []
    for k in sizes:
        seeds = (0, 1) if k <= 2 else (0,)
        for te in range(n_items):
            pool = [j for j in range(n_items) if j != te]
            combs = list(itertools.combinations(pool, k))
            if len(combs) > max_combs:
                combs = [combs[i] for i in rng.choice(len(combs), max_combs, replace=False)]
            for c in combs:
                for s in seeds:
                    out.append((k, te, c, s))
    return out


def _key(feature_set, k, test, train, seed):
    return (feature_set, k, test, tuple(train), seed)


def run(items, feature_set="framing", pipelines=PIPELINES, log=print, sizes=(1, 2, 3, 4, 5),
        rows=None, checkpoint=None):
    """Append results to ``rows``.  Fits already present in ``rows`` are
    skipped, and ``rows`` is written to ``checkpoint`` after every fit, so an
    interrupted run resumes where it stopped."""
    rows = [] if rows is None else rows
    done = {_key(r["feature_set"], r["n_train"], r["test"], r["train"], r["seed"]) for r in rows}
    t0 = time.time()
    runs = plan(len(items), sizes)
    for i, (k, te, c, s) in enumerate(runs):
        if _key(feature_set, k, items[te]["name"], [items[j]["name"] for j in c], s) in done:
            continue
        clf = model.fit([(items[j]["X"][feature_set], items[j]["lab"]) for j in c], seed=s)
        p = model.predict_proba(clf, items[te]["X"][feature_set])
        it = items[te]
        for name, kw in pipelines.items():
            m = postproc.finish(p, it["img"], it["px"], **kw)
            sc = M.score(m, it["lab"], it["px"])
            rows.append(dict(feature_set=feature_set, pipeline=name, n_train=k,
                             test=it["name"], train=[items[j]["name"] for j in c], seed=s,
                             **{kk: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                for kk, v in sc.items()}))
        if checkpoint:
            save(rows, checkpoint)
        if log and (i + 1) % 10 == 0:
            log(f"  [{feature_set}] {i + 1}/{len(runs)} fits  ({time.time() - t0:.0f}s)")
    return rows


def summarise(rows, metric="iou"):
    """{(feature_set, pipeline): {n: (mean, n_runs)}}"""
    out = {}
    for r in rows:
        out.setdefault((r["feature_set"], r["pipeline"]), {}).setdefault(r["n_train"], []).append(r[metric])
    return {key: {n: (float(np.nanmean(np.array(v, float))), len(v)) for n, v in sorted(d.items())}
            for key, d in out.items()}


def save(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(rows=rows), indent=1))


def load(path):
    return json.loads(Path(path).read_text())["rows"]
