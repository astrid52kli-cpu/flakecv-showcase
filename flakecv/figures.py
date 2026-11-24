"""Every figure, shared by the app and ``flakecv figure`` so results can be
regenerated headlessly.  Each function returns a matplotlib Figure.

Colours: hand label = cyan, model = green, rules baseline = orange,
optical outline = yellow.  Every image panel carries a scale bar in um.
"""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import FLAKE  # noqa: E402

COL = dict(label=(0, 208, 255), model=(57, 211, 83), rules=(255, 140, 0), optical=(255, 210, 31))


def _rgb(img):
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.clip(a * 255 if a.dtype.kind == "f" else a, 0, 255).astype(np.uint8)
        return np.dstack([a, a, a])
    return a.copy()


def outline(img, layers, width=None):
    """RGB uint8 image with each (mask, rgb) drawn as an outline."""
    out = np.ascontiguousarray(_rgb(img))
    w = width or max(2, int(round(out.shape[1] / 400)))
    for m, c in layers:
        if m is None:
            continue
        cnts, _ = cv2.findContours((np.asarray(m) > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, tuple(int(v) for v in c), w)
    return out


def _nice_length(field_um):
    for v in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if v >= field_um / 6:
            return v
    return 1000


def scale_bar(ax, shape, px_um):
    H, W = shape[:2]
    L = _nice_length(W * px_um)
    n = L / px_um
    x0, y0 = W * 0.04, H * 0.94
    ax.plot([x0, x0 + n], [y0, y0], color="white", lw=4, solid_capstyle="butt")
    ax.plot([x0, x0 + n], [y0, y0], color="black", lw=1.2, solid_capstyle="butt")
    ax.text(x0 + n / 2, y0 - H * 0.025, f"{L:g} µm", color="white", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground="black")])


def _show(ax, rgb, title, px_um=None, note=None):
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    if px_um:
        scale_bar(ax, rgb.shape, px_um)
    if note:
        ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))


def _fmt(s):
    if not s:
        return None
    return (f"IoU {s['iou']:.3f}\nboundary F {s['boundary_f']:.3f}\n"
            f"area err {s['area_error']:+.1%}")


def panels(name, img, px_um, rules_mask, model_mask, label=None, scores=None):
    """raw | rules baseline | model | hand label (if any).
    scores: {'rules': score dict, 'model': score dict} when a label exists."""
    scores = scores or {}
    ref = None if label is None else (np.asarray(label) == FLAKE)
    cols = 4 if label is not None else 3
    fig, ax = plt.subplots(1, cols, figsize=(4.6 * cols, 4.6 * img.shape[0] / img.shape[1] + 0.6))
    _show(ax[0], _rgb(img), name, px_um)
    _show(ax[1], outline(img, [(rules_mask, COL["rules"])]), "rules baseline", px_um,
          _fmt(scores.get("rules")))
    _show(ax[2], outline(img, [(model_mask, COL["model"])]), "model", px_um, _fmt(scores.get("model")))
    if label is not None:
        _show(ax[3], outline(img, [(ref, COL["label"])]), "hand label", px_um)
    fig.tight_layout()
    return fig


def threeway(name, om_warped_bgr, om_mask, img, px_um, model_mask, label=None, om_name=""):
    """optical with its own outline | SEM with hand label | SEM with model,
    all in the SEM frame at the same scale."""
    ref = None if label is None else (np.asarray(label) == FLAKE)
    om_rgb = cv2.cvtColor(om_warped_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 3, figsize=(14, 14 * img.shape[0] / img.shape[1] / 3 + 0.8))
    _show(ax[0], outline(om_rgb, [(om_mask, COL["optical"])]), f"optical {om_name}".strip(), px_um)
    _show(ax[1], outline(img, [(ref, COL["label"])]) if ref is not None else _rgb(img),
          "SEM · hand label" if ref is not None else "SEM (no hand label)", px_um)
    _show(ax[2], outline(img, [(model_mask, COL["model"])]), "SEM · model", px_um)
    fig.suptitle(f"{name}: three views, same field, same scale", fontsize=12)
    fig.tight_layout()
    return fig


def curve(rows, rules_mean=None, metric="iou"):
    """Learning curve from stored results (``flakecv curve`` output)."""
    from .curve import summarise
    S = summarise(rows, metric)
    fig, ax = plt.subplots(figsize=(7.4, 5))
    series = [(("photometric", "base"), "10 photometric features", "#8a8a8a"),
              (("framing", "base"), "+ framing prior", "#2f6fdf"),
              (("framing", "snap"), "+ framing prior + snap", "#1a9a4a")]
    for key, lab, c in series:
        if key not in S:
            continue
        ns = sorted(S[key]); mu = [S[key][n][0] for n in ns]
        ax.plot(ns, mu, marker="o", lw=2, color=c, label=lab)
    if ("framing", "full") in S and 5 in S[("framing", "full")]:
        v = S[("framing", "full")][5][0]
        ax.scatter([5], [v], marker="*", s=180, color="#d4380d", zorder=5,
                   label=f"full post-processing, n=5 ({v:.3f})")
    if rules_mean is not None:
        ax.axhline(rules_mean, ls="--", c="#e8a33d", lw=2, label=f"hand-tuned rules ({rules_mean:.3f})")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("flakes used for training (leave-one-flake-out)")
    ax.set_ylabel("mean IoU" if metric == "iou" else metric)
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    ax.set_title("accuracy vs amount of training data")
    fig.tight_layout()
    return fig


def defects(name, img, px_um, flake_mask, res, title_note="candidates - not validated"):
    """SEM with the flake outline (green), crack chords (red), line ridges
    (orange) and bubble candidates (magenta circles, numbered by rank)."""
    rgb = outline(img, [(flake_mask, COL["model"])])
    lm = np.asarray(res.get("line_map", np.zeros(img.shape[:2], np.uint8))) > 0
    rgb[lm] = (0.4 * rgb[lm] + 0.6 * np.array([255, 140, 0])).astype(np.uint8)
    w = max(2, int(round(rgb.shape[1] / 400)))
    for c in res["cracks"]:
        cv2.line(rgb, tuple(int(v) for v in c["p0"]), tuple(int(v) for v in c["p1"]), (230, 30, 30), w + 1)
    for i, b in enumerate(res["bubbles"]):
        cv2.circle(rgb, (int(b["x"]), int(b["y"])), int(max(4, 2 * b["r_px"])), (230, 0, 230), w)
        if i < 10:
            cv2.putText(rgb, str(i + 1), (int(b["x"] + 2 * b["r_px"] + 3), int(b["y"])), 0,
                        0.5 * w, (230, 0, 230), max(1, w - 1))
    fig, ax = plt.subplots(figsize=(9, 9 * rgb.shape[0] / rgb.shape[1] + 0.6))
    note = (f"{len(res['cracks'])} crack chords (red)\n{len(res['lines'])} line ridges (orange)\n"
            f"{len(res['bubbles'])} bubble/residue spots (magenta)")
    _show(ax, rgb, f"{name}: defect {title_note}", px_um, note)
    fig.tight_layout()
    return fig


def save(fig, path, dpi=130):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path
