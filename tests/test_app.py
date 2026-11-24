"""The viewer and the figure CLI, headless.  Uses the repo's own data and model,
so these are skipped on a checkout without them."""
from pathlib import Path

import pytest

from flakecv import cli

REPO = Path(__file__).resolve().parent.parent
HAVE = (REPO / "models" / "default.joblib").is_file() and (REPO / "data" / "frames").is_dir()
pytestmark = pytest.mark.skipif(not HAVE, reason="needs data/ and models/default.joblib")


def test_figure_cli_panels_and_curve(tmp_path):
    assert cli.main(["--data", str(REPO / "data"), "figure", "panels", "261015-tpt1-3",
                     "--model", str(REPO / "models" / "default.joblib"), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "261015-tpt1-3.panels.png").stat().st_size > 10_000
    res = REPO / "results" / "curve.json"
    if res.is_file():
        assert cli.main(["figure", "curve", "--results", str(res), "--out", str(tmp_path)]) == 0
        assert (tmp_path / "curve.png").is_file()


def test_app_renders_and_reacts_to_controls():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.selectbox[0].set_value("261015-tpt1-3").run()
    assert not at.exception, at.exception
    iou0 = next(m.value for m in at.metric if m.label == "IoU vs hand label")
    # turn the post-processing knobs: the mask (and so the IoU) must update
    at.sidebar.slider[0].set_value(0.9).run()          # threshold
    at.sidebar.slider[1].set_value(0.0).run()          # snap off
    assert not at.exception, at.exception
    iou1 = next(m.value for m in at.metric if m.label == "IoU vs hand label")
    assert iou0 != iou1
    assert any("training set" in w.value for w in at.warning)
    labels = [b.label for b in at.get("download_button")]
    assert {"Model mask (PNG)", "Metrics (CSV)"} <= set(labels)


def test_app_defects_tab():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].set_value("261015-tpt1-2").run()
    btn = next(b for b in at.button if "crack / bubble" in b.label)
    btn.click().run()
    assert not at.exception, at.exception
    assert any("Candidates" in b.label for b in at.get("download_button"))


def test_app_label_tab_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=180)
    at.run()
    at.sidebar.selectbox[0].set_value("261015-tpt1-2").run()
    assert not at.exception, at.exception
    starts = next(r for r in at.radio if r.label == "Start from")
    assert "existing hand label" in starts.options
    starts.set_value("model mask").run()
    assert not at.exception, at.exception
