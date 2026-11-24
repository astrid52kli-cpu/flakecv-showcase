"""Regression values.  ``flakecv demo`` must reproduce CURRENT within TOL.

CURRENT was measured 2026-11-21 with the pinned environment (pyproject.toml)
after one deliberate change: the ``rank`` feature became tie-aware (see
``features.rank01``).  The source's ``rank`` gave tied grey levels arbitrary,
platform-dependent order, so its numbers (PUBLISHED) could not be reproduced
on another machine; the drift was up to 0.014 on a mean and 0.04 on a frame.

Everything that does not use the learned model is unchanged and matches
PUBLISHED exactly (rules baseline; registration with the rules mask).
"""
TOL = 0.005

FRAMES = ["261015-tpt1-1", "261015-tpt1-2", "261015-tpt1-3",
          "261103-ely-A", "261103-ely-B", "261103-ely-C"]

CURRENT = {
    "rules_iou": [0.333, 0.455, 0.557, 0.465, 0.201, 0.710],
    "rules_iou_mean": 0.453,
    "rules_boundary_f_mean": 0.352,
    # learning curve, mean IoU at n = 1..5, runs 48/48/24/24/6
    "curve": {
        "photometric": [0.296, 0.333, 0.354, 0.362, 0.347],
        "framing": [0.389, 0.457, 0.500, 0.536, 0.526],
        "framing+snap": [0.409, 0.474, 0.522, 0.550, 0.551],
    },
    "full_n5_iou": 0.552,
    "full_n5_boundary_f": 0.264,
    # OM -> SEM IoU vs hand label, three 20 kV frames
    "register_rules_target": [0.381, 0.299, 0.837],
    "register_rules_target_mean": 0.506,
    "register_model_target": [0.680, 0.604, 0.860],
    "register_model_target_mean": 0.714,
}

PUBLISHED = {   # the research-code numbers, for reference
    "curve": {
        "photometric": [0.293, 0.334, 0.352, 0.363, 0.351],
        "framing": [0.397, 0.455, 0.501, 0.537, 0.524],
        "framing+snap": [0.419, 0.469, 0.525, 0.551, 0.552],
    },
    "full_n5_iou": 0.565,
    "full_n5_boundary_f": 0.289,
    "register_unguarded_mean": 0.301,
    "register_model_target": [0.662, 0.601, 0.853],
    "register_model_target_mean": 0.705,
    "register_resolution_ceiling": 0.845,
}
