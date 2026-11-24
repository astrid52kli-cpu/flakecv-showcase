# data/

Expected layout (names are `<date>-<sample>-<flake>`):

    frames/<name>.png         8-bit, databar stripped, contrast stretched
    frames/<name>.json        {"px_um": ..., "hv_kV": ..., "stage_um": [...], ...}
    labels/<name>.label.png   three-class label, 0 / 128 / 255
    optical/<name>.bmp        optical image (20x = 0.2475 um/px)
    raw/<name>.tif            optional: the raw FEI TIFF (git-ignored)
    optical_paired/<name>.om.<ext>   the optical image of THIS frame (added in the app)
    optical_paired/<name>.om.json    {"objective": "50x", "om_px_um": 0.09, "source_file": ...}
    backup/<time>/...         copies of anything the app replaced (git-ignored)

optical/ holds the images the regression demo registers against; the app's
per-frame optical images go to optical_paired/ so they cannot change those numbers.

This public copy contains three of the six labelled frames (261015-tpt1-1/2/3);
the regression check (`flakecv demo`) needs all six and explains that when run here.
