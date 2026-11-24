"""Reading SEM frames and images.

Two ways in:

* :func:`read_sem_tiff` - a raw FEI / XL30 TIFF straight off the microscope.
  Pixel size, accelerating voltage and stage position come from the FEI
  metadata tag (34680 or 34682); the databar is stripped using the image
  height recorded in that header.
* :func:`load_frame` - a *prepared* frame from the data root:
  ``frames/<name>.png`` (8-bit, databar stripped, contrast stretched) plus
  ``frames/<name>.json`` (``px_um``, ``hv_kV``, ``stage_um``...).  This is the
  exact form the published regression numbers were measured on, so it is the
  canonical input for ``flakecv demo``.

All image files are read and written through numpy buffers rather than
``cv2.imread`` / ``cv2.imwrite``, because OpenCV cannot open non-ASCII paths on
Windows.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# The burnt-in optical scale bars are NOT calibrated.  Measured against SEM.
OPTICAL_PX_UM = {"20x": 0.2475, "50x": 0.090}

_PX_UM_RANGE = (0.001, 10.0)   # anything outside this is a unit error


class MetadataError(ValueError):
    pass


# --------------------------------------------------------------------- images
def imread(path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """cv2.imread that works on any path (unicode, Windows)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    buf = np.fromfile(str(p), dtype=np.uint8)
    a = cv2.imdecode(buf, flags)
    if a is None:
        raise ValueError(f"could not decode image: {p}")
    return a


def imwrite(path, img) -> None:
    """cv2.imwrite that works on any path.  PNG output is deterministic."""
    p = Path(path)
    ok, buf = cv2.imencode(p.suffix or ".png", np.asarray(img))
    if not ok:
        raise ValueError(f"could not encode image for {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(p))


def read_gray01(path) -> np.ndarray:
    """An 8- or 16-bit grey image as float32 in [0, 1] (no stretching)."""
    a = imread(path, cv2.IMREAD_UNCHANGED)
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2GRAY)
    scale = 255.0 if a.dtype == np.uint8 else float(np.iinfo(a.dtype).max) if a.dtype.kind == "u" else 1.0
    return (a.astype(np.float32) / np.float32(scale))


def normalize(raw, lo=1, hi=99) -> np.ndarray:
    """1-99 percentile stretch to [0, 1].  Identical to the original
    ``flakecv.sem.normalize``; verified to reproduce the prepared PNGs
    byte-for-byte from the raw TIFFs."""
    a = np.asarray(raw).astype(np.float32)
    p1, p99 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p1) / (p99 - p1 + 1e-9), 0, 1)


def to_uint8(img01) -> np.ndarray:
    return (np.clip(img01, 0, 1) * 255).astype(np.uint8)


def quantize01(img01) -> np.ndarray:
    """The 8-bit round trip every training frame went through
    (stretch -> uint8 PNG -> /255).  Applied to raw TIFF input too, so a model
    sees the same thing whether it is given a TIFF or a prepared PNG."""
    return to_uint8(img01).astype(np.float32) / 255.


# ------------------------------------------------------------------ SEM TIFF
@dataclass
class SemFrame:
    image: np.ndarray          # float32 [0,1]: stretched, then 8-bit quantized
                               # (== what load_frame returns for the same frame)
    raw: np.ndarray            # databar stripped, original dtype
    px_um: float               # micrometres per pixel
    hv_kV: float | None = None
    stage_um: tuple[float, float] | None = None
    bit_depth: int = 8
    databar_rows: int = 0
    meta: dict = field(default_factory=dict, repr=False)


def _parse_ini(text: str) -> dict:
    out, sec = {}, None
    for line in text.splitlines():
        line = line.strip().strip("\x00")
        if not line or line.startswith(("#", ";")):
            continue
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            sec = out.setdefault(m.group(1).strip(), {})
            continue
        if "=" in line and sec is not None:
            k, v = line.split("=", 1)
            v = v.strip()
            try:
                v = float(v)
            except ValueError:
                pass
            sec[k.strip()] = v
    return out


def _fei_meta(tf) -> dict:
    """The FEI header as {section: {key: value}}, from whichever tag holds it."""
    md = getattr(tf, "fei_metadata", None)
    if md:
        return md
    page = tf.pages[0]
    for code in (34682, 34680, 270):            # FEI_HELIOS, FEI_SFEG, ImageDescription
        tag = page.tags.get(code)
        if tag is None:
            continue
        v = tag.value
        if isinstance(v, dict) and v:
            return v
        if isinstance(v, bytes):
            v = v.decode("latin-1", "replace")
        if isinstance(v, str) and "[" in v and "=" in v:
            d = _parse_ini(v)
            if d:
                return d
    return {}


def _ini_from_bytes(path) -> dict:
    """Original reader's approach: FEI appends the INI text to the file, so
    scan the bytes for it when no tag carries it."""
    txt = Path(path).read_bytes().decode("latin-1", errors="ignore")
    m = re.search(r"\[User\][\s\S]{0,20000}", txt)
    return _parse_ini(m.group(0)) if m else {}


def _find(md: dict, *keys):
    """First value whose key matches any of ``keys`` (case-insensitive), searching
    all sections in order."""
    want = [k.lower() for k in keys]
    for k in want:
        for sec in md.values():
            if not isinstance(sec, dict):
                continue
            for kk, vv in sec.items():
                if kk.lower() == k:
                    return vv
    return None


def _section(md, name):
    for k, v in md.items():
        if k.lower() == name.lower() and isinstance(v, dict):
            return v
    return {}


def read_sem_tiff(path, databar_rows: int | None = None) -> SemFrame:
    """Read an FEI / XL30 SEM TIFF.

    Pixel size is required; a frame without it is refused, because every
    length in this package is physical.  ``databar_rows`` overrides the
    databar height taken from the header.
    """
    import tifffile

    path = Path(path)
    with tifffile.TiffFile(str(path)) as tf:
        a = tf.pages[0].asarray()
        md = _fei_meta(tf)
    if not md:
        md = _ini_from_bytes(path)
    if a.ndim == 3:                               # RGB-saved greyscale
        a = a[..., 0]

    px_m = _find(md, "PixelWidth", "PixelSizeX", "pixelsize")
    if px_m is None:                    # original fallback: HFW / width
        hfw = _find(md, "HFW")
        px_m = float(hfw) / a.shape[1] if hfw else None
    if px_m is None:
        raise MetadataError(
            f"{path.name}: no pixel size in the TIFF header (looked for FEI "
            "PixelWidth). Save frames as raw FEI TIFF; see the acquisition SOP.")
    px_um = float(px_m) * 1e6
    if not (_PX_UM_RANGE[0] <= px_um <= _PX_UM_RANGE[1]):
        raise MetadataError(f"{path.name}: implausible pixel size {px_um} um "
                            "(expected PixelWidth in metres).")

    hv = _find(md, "HV", "HighVoltage", "AcceleratingVoltage")
    hv_kV = float(hv) / 1000.0 if hv is not None else None
    sx, sy = _find(md, "StageX"), _find(md, "StageY")
    stage_um = (float(sx) * 1e6, float(sy) * 1e6) if sx is not None and sy is not None else None

    if databar_rows is None:
        res_y = _section(md, "Image").get("ResolutionY")
        if res_y is not None and 0 < int(res_y) <= a.shape[0]:
            databar_rows = a.shape[0] - int(res_y)
        else:
            databar_rows = detect_databar(a)
    img = a[: a.shape[0] - databar_rows] if databar_rows else a

    bits = 16 if img.dtype.itemsize >= 2 else 8
    return SemFrame(image=quantize01(normalize(img)), raw=img, px_um=px_um, hv_kV=hv_kV,
                    stage_um=stage_um, bit_depth=bits,
                    databar_rows=int(databar_rows), meta=md)


def detect_databar(a, min_rows=8, frac=0.35) -> int:
    """Fallback when the header does not give the image height: the databar is
    a band at the bottom whose rows are dominated by exact min/max grey values
    (rendered text on a flat panel).  Returns the number of rows to strip."""
    a = np.asarray(a)
    lo, hi = a.min(), a.max()
    ext = ((a == lo) | (a == hi)).mean(axis=1)
    n = 0
    for r in range(a.shape[0] - 1, -1, -1):
        if ext[r] >= frac:
            n += 1
        else:
            break
    return n if n >= min_rows else 0


# ----------------------------------------------------------- prepared frames
@dataclass
class Frame:
    name: str
    image: np.ndarray          # float32 [0,1]
    px_um: float
    meta: dict = field(default_factory=dict, repr=False)

    @property
    def hv_kV(self):
        return self.meta.get("hv_kV")


def frame_paths(data_root, name) -> dict:
    d = Path(data_root)
    return dict(image=d / "frames" / f"{name}.png",
                meta=d / "frames" / f"{name}.json",
                label=d / "labels" / f"{name}.label.png")


def list_frames(data_root) -> list[str]:
    d = Path(data_root) / "frames"
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_frame(data_root, name) -> Frame:
    p = frame_paths(data_root, name)
    if not p["meta"].is_file():
        raise FileNotFoundError(f"no metadata for frame '{name}': {p['meta']}")
    with open(p["meta"], encoding="utf-8") as fh:
        meta = json.load(fh)
    if "px_um" not in meta:
        raise MetadataError(f"{p['meta']}: missing px_um")
    a = imread(p["image"], cv2.IMREAD_GRAYSCALE)
    # exactly what the original experiments did: imread(..., 0) / 255
    img = a.astype(np.float32) / 255.
    return Frame(name=name, image=img, px_um=float(meta["px_um"]), meta=meta)


def save_prepared(data_root, name, sem: SemFrame, extra: dict | None = None) -> dict:
    """Write a raw TIFF's content as a prepared frame (png + json)."""
    p = frame_paths(data_root, name)
    imwrite(p["image"], to_uint8(sem.image))
    meta = dict(name=name, px_um=sem.px_um, shape=list(sem.image.shape),
                hv_kV=sem.hv_kV, stage_um=list(sem.stage_um) if sem.stage_um else None,
                bit_depth=sem.bit_depth, **(extra or {}))
    p["meta"].parent.mkdir(parents=True, exist_ok=True)
    with open(p["meta"], "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return meta
