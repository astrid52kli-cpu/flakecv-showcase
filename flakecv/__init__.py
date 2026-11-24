"""flakecv: SEM flake segmentation and optical-to-SEM registration.

Label convention (everywhere in this package):
    BACKGROUND = 0, FLAKE = 1, UNCERTAIN = 255   (in memory)
    0 / 128 / 255                                (on disk, single-channel PNG)
UNCERTAIN pixels are excluded from every metric.

All lengths are in micrometres.  Pixel size comes from the frame, never a constant.
"""
__version__ = "0.1.0"

BACKGROUND, FLAKE, UNCERTAIN = 0, 1, 255
