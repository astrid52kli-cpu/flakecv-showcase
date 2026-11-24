"""Command line.

    flakecv info                         versions, data root, frames
    flakecv train  [--frames ...]        fit and write one model file
    flakecv predict IMAGE_OR_FRAME ...   masks (+ metrics if a label exists)
    flakecv curve                        leave-one-flake-out learning curve
    flakecv figure panels|threeway|curve every figure the app shows, as PNG
    flakecv app                          open the viewer in the browser
    flakecv demo [--full]                acceptance test on the committed data
    flakecv defects FRAME...             step 3: crack / bubble CANDIDATES (experimental)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _info(args):
    import platform

    import cv2, numpy, sklearn, skimage

    from . import __version__
    from .config import DataRootError, data_root
    from .io import list_frames

    print(f"flakecv {__version__}  python {platform.python_version()}  "
          f"numpy {numpy.__version__}  opencv {cv2.__version__}  "
          f"scikit-image {skimage.__version__}  scikit-learn {sklearn.__version__}")
    try:
        root = data_root(args.data)
    except DataRootError as e:
        print(e, file=sys.stderr)
        return 2
    names = list_frames(root)
    print(f"data root: {root}  ({len(names)} prepared frames)")
    for n in names:
        print("  ", n)
    return 0


def _curve(args):
    import functools, time

    from . import curve as C
    from .config import data_root

    log = functools.partial(print, flush=True)
    t = time.time()
    items = C.load_items(data_root(args.data), feature_sets=("photometric", "framing"), log=log)
    from pathlib import Path
    rows = C.load(args.out) if Path(args.out).is_file() else []
    if rows:
        log(f"resuming: {len(rows)} result rows already in {args.out}")
    C.run(items, "framing", log=log, rows=rows, checkpoint=args.out)
    C.run(items, "photometric", pipelines={"base": C.PIPELINES["base"]}, log=log,
          rows=rows, checkpoint=args.out)
    for metric in ("iou", "boundary_f"):
        for (fs, pl), d in C.summarise(rows, metric).items():
            log(f"{metric:<10} {fs:<11} {pl:<8} " +
                "  ".join(f"n={n}:{m:.3f}" for n, (m, _) in d.items()) +
                "   runs " + "/".join(str(c) for _, c in d.values()))
    log(f"wrote {args.out}  ({time.time() - t:.0f}s)")
    return 0


REPO = __import__("pathlib").Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO / "models" / "default.joblib"


def _train(args):
    import time

    from . import features as F, io, labels as L, model as Mdl
    from .config import data_root

    root = data_root(args.data)
    names = args.frames or [n for n in io.list_frames(root)
                            if io.frame_paths(root, n)["label"].is_file()]
    if not names:
        print(f"no labelled frames in {root}/labels", file=sys.stderr)
        return 2
    frames, training = [], []
    for n in names:
        fr = io.load_frame(root, n)
        p = io.frame_paths(root, n)
        X, _ = F.compute(fr.image, fr.px_um, args.feature_set)
        frames.append((X, L.load(p["label"])))
        training.append(dict(name=n, px_um=fr.px_um, hv_kV=fr.hv_kV,
                             image_sha256=Mdl.sha256(p["image"]),
                             label_sha256=Mdl.sha256(p["label"])))
        print(f"  {n}  ({fr.px_um * 1000:.0f} nm/px, {fr.hv_kV} kV)", flush=True)
    t = time.time()
    clf = Mdl.fit(frames, seed=args.seed)
    meta = Mdl.save(args.out, clf, feature_set=args.feature_set, training=training, seed=args.seed)
    pp = meta["postproc"]
    print(f"trained on {len(names)} frames in {time.time() - t:.0f}s -> {args.out}")
    print(f"  post-processing: central={pp['central']} fill={pp['fill']} snap={pp['search_um']} um"
          + ("   (central/fill OFF: fewer than 3 training frames)" if not pp["central"] else ""))
    return 0


def _resolve_input(spec, root, px_um=None):
    """-> (name, img01, px_um, label or None)"""
    from pathlib import Path

    from . import io, labels as L

    p = Path(spec)
    if p.suffix.lower() in (".tif", ".tiff") and p.is_file():
        s = io.read_sem_tiff(p)
        return p.stem, s.image, px_um or s.px_um, None
    if p.is_file():
        side = p.with_suffix(".json")
        if px_um is None and side.is_file():
            import json
            px_um = json.loads(side.read_text())["px_um"]
        if px_um is None:
            raise SystemExit(f"{p}: a PNG/JPG carries no pixel size; pass --px-um "
                             f"or put a {side.name} with px_um next to it")
        return p.stem, io.read_gray01(p), float(px_um), None
    if root is not None and (Path(root) / "frames" / f"{spec}.json").is_file():
        fr = io.load_frame(root, spec)
        lp = io.frame_paths(root, spec)["label"]
        return spec, fr.image, fr.px_um, (L.load(lp) if lp.is_file() else None)
    raise SystemExit(f"not a file and not a frame in the data root: {spec}")


def _predict(args):
    import csv
    from pathlib import Path

    from . import labels as L, metrics as M, model as Mdl, io
    from .config import DataRootError, data_root

    try:
        root = data_root(args.data)
    except DataRootError:
        root = None
    try:
        clf, meta = Mdl.load(args.model)
    except Mdl.ArtifactError as e:
        print(e, file=sys.stderr)
        return 3
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    over = dict(thr=args.threshold, central=args.central, fill=args.fill, search_um=args.snap_um)
    rows = []
    for spec in args.inputs:
        name, img, px, lab = _resolve_input(spec, root, args.px_um)
        mask, prob = Mdl.predict_image(clf, meta, img, px, **over)
        L.save(out / f"{name}.mask.png", mask.astype(np.uint8))
        row = dict(frame=name, px_um=px, area_um2=float(mask.sum()) * px * px)
        if lab is not None:
            row.update(M.score(mask, lab, px))
        rows.append(row)
        msg = f"{name:<18} area {row['area_um2']:8.1f} um2"
        if lab is not None:
            msg += f"   IoU {row['iou']:.3f}  boundary F {row['boundary_f']:.3f}  area err {row['area_error']:+.3f}"
            if name in [t["name"] for t in meta["training"]]:
                msg += "   (!) this frame was in the training set"
        print(msg, flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"masks and metrics.csv -> {out}")
    return 0


def _figure(args):
    from pathlib import Path

    from . import analysis as A, baseline, curve as C, figures as FG, model as Mdl
    from .config import DataRootError, data_root

    out = Path(args.out)
    if args.kind == "curve":
        rows = C.load(args.results)
        p = FG.save(FG.curve(rows, rules_mean=baseline.CURRENT["rules_iou_mean"]), out / "curve.png")
        print("->", p)
        return 0
    if not args.inputs:
        raise SystemExit(f"figure {args.kind}: give at least one frame or image")
    try:
        root = data_root(args.data)
    except DataRootError:
        root = None
    try:
        clf, meta = Mdl.load(args.model)
    except Mdl.ArtifactError as e:
        print(e, file=sys.stderr)
        return 3
    trained_on = {t["name"] for t in meta["training"]}
    optical = None
    if args.kind == "threeway":
        paths = args.optical or (A.default_optical(root) if root else [])
        if not paths:
            raise SystemExit("figure threeway: no optical image (pass --optical)")
        optical = A.load_optical(paths)
    for spec in args.inputs:
        name, img, px, lab = _resolve_input(spec, root, args.px_um)
        res = A.analyse(clf, meta, img, px, lab)
        warn = "   (!) in the training set: scores are optimistic" if name in trained_on else ""
        if args.kind == "panels":
            p = FG.save(FG.panels(name, img, px, res["rules"], res["model"], lab, res["scores"]),
                        out / f"{name}.panels.png")
            sc = res["scores"].get("model")
            print("->", p, f"   model IoU {sc['iou']:.3f}" if sc else "", warn)
        else:
            r = A.register(img, px, res["model"], optical, om_px=args.om_px, label=lab)
            if r is None:
                print(f"{name}: no optical flake could be matched", file=sys.stderr)
                continue
            p = FG.save(FG.threeway(name, r["om_warped"], r["mask"], img, px, res["model"], lab, r["om"]),
                        out / f"{name}.threeway.png")
            extra = f"   OM->SEM IoU vs label {r['iou_vs_label']:.3f}" if "iou_vs_label" in r else ""
            print("->", p, f"  ({r['om']}, {r['stage']})" + extra, warn)
    return 0


def _app(args):
    import subprocess
    app = REPO / "app" / "app.py"
    if not app.is_file():
        raise SystemExit(f"app not found at {app} (the viewer ships with the git repository)")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    if args.headless:
        cmd += ["--server.headless", "true"]
    return subprocess.call(cmd, cwd=str(REPO))


def _demo(args):
    import functools

    from . import demo
    from .config import data_root
    return demo.run(data_root(args.data), full=args.full, out=args.out,
                    log=functools.partial(print, flush=True))


def _defects(args):
    import csv
    from pathlib import Path

    from . import defects as D, figures as FG, model as Mdl
    from .config import DataRootError, data_root

    try:
        root = data_root(args.data)
    except DataRootError:
        root = None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    clf = meta = None
    rows = []
    for spec in args.inputs:
        name, img, px, lab = _resolve_input(spec, root, args.px_um)
        if lab is not None and args.mask in ("auto", "label"):
            mask, src = (lab == 1).astype(np.uint8), "hand label"
        elif args.mask == "label":
            print(f"{name}: no hand label, skipped (--mask label)", file=sys.stderr)
            continue
        else:
            if clf is None:
                try:
                    clf, meta = Mdl.load(args.model)
                except Mdl.ArtifactError as e:
                    print(e, file=sys.stderr)
                    return 3
            mask, src = Mdl.predict_image(clf, meta, img, px)[0], "model"
        res = D.detect(img, mask, px)
        D.save_annotations(out / f"{name}.defects.json", name, px, res["cracks"], res["bubbles"])
        FG.save(FG.defects(name, img, px, mask, res), out / f"{name}.defects.png")
        row = dict(frame=name, mask=src, px_um=px, cracks=len(res["cracks"]), lines=len(res["lines"]),
                   bubbles=len(res["bubbles"]))
        ann = (Path(root) / "defects" / f"{name}.defects.json") if root else None
        if ann is not None and ann.is_file():
            sc = D.score(dict(cracks=res["cracks"], bubbles=res["bubbles"]), D.load_annotations(ann), px)
            row.update(crack_f=sc["cracks"]["f"], bubble_f=sc["bubbles"]["f"])
        rows.append(row)
        print(f"{name:<16} mask={src:<10} cracks {row['cracks']:2d}  lines {row['lines']:2d}  bubbles {row['bubbles']:2d}"
              + (f"   vs annotations: crack F {row['crack_f']:.2f}  bubble F {row['bubble_f']:.2f}" if "crack_f" in row else ""),
              flush=True)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "defects.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"candidates (figure + .defects.json) -> {out}   [experimental: not validated]")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flakecv")
    ap.add_argument("--data", help="data root (overrides FLAKECV_DATA and flakecv.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="versions and what is in the data root").set_defaults(fn=_info)
    c = sub.add_parser("curve", help="leave-one-flake-out learning curve (~30 min)")
    c.add_argument("--out", default="results/curve.json")
    c.set_defaults(fn=_curve)
    t = sub.add_parser("train", help="fit a model and write one self-describing file")
    t.add_argument("--frames", nargs="*", help="frame names (default: every labelled frame)")
    t.add_argument("--feature-set", default="framing", choices=["framing", "photometric"])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=str(DEFAULT_MODEL))
    t.set_defaults(fn=_train)
    p = sub.add_parser("predict", help="flake masks for SEM frames")
    p.add_argument("inputs", nargs="+", help="raw .tif/.tiff, .png (+ --px-um), or a frame name in the data root")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--out", default="results/predict")
    p.add_argument("--px-um", type=float, help="pixel size for PNG input")
    p.add_argument("--threshold", type=float)
    p.add_argument("--central", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fill", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--snap-um", type=float, help="0 disables snap")
    p.set_defaults(fn=_predict)
    fg = sub.add_parser("figure", help="write the app's figures as PNG (headless)")
    fg.add_argument("kind", choices=["panels", "threeway", "curve"])
    fg.add_argument("inputs", nargs="*", help="frames or images (panels / threeway)")
    fg.add_argument("--model", default=str(DEFAULT_MODEL))
    fg.add_argument("--optical", nargs="*", help="optical images (default: data/optical/*)")
    fg.add_argument("--om-px", type=float, default=0.2475, help="optical um/px (20x: 0.2475)")
    fg.add_argument("--px-um", type=float, help="pixel size for PNG input")
    fg.add_argument("--results", default="results/curve.json")
    fg.add_argument("--out", default="results/figures")
    fg.set_defaults(fn=_figure)
    a = sub.add_parser("app", help="open the viewer in the browser")
    a.add_argument("--port", type=int, default=8501)
    a.add_argument("--headless", action="store_true", help="do not open a browser window")
    a.set_defaults(fn=_app)
    d = sub.add_parser("demo", help="acceptance test: reproduce the regression numbers")
    d.add_argument("--full", action="store_true", help="the whole learning curve too (~45 min)")
    d.add_argument("--out", default="results/demo.json")
    d.set_defaults(fn=_demo)
    df = sub.add_parser("defects", help="step 3: crack / bubble candidates inside the flake (experimental)")
    df.add_argument("inputs", nargs="+")
    df.add_argument("--mask", choices=["auto", "label", "model"], default="auto",
                    help="flake mask to search in: hand label if present (auto), only labels, or the model")
    df.add_argument("--model", default=str(DEFAULT_MODEL))
    df.add_argument("--px-um", type=float)
    df.add_argument("--out", default="results/defects")
    df.set_defaults(fn=_defects)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
