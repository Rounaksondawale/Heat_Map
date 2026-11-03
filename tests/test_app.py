"""
End-to-end tests of the Streamlit app with Streamlit's own test harness, on a synthetic city (no network, no browser).
The map widget and the satellite download are replaced by fakes: the widget "returns" a drawn rectangle, and the download copies the
synthetic city's grids under the name the app expects.

    python -m unittest discover -s tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "dashboard"))
import streamlit_folium  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import engine as E  # noqa: E402
import live as LIVE  # noqa: E402
import synthetic  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dir = synthetic.make_assets(cls.tmp.name)
        for k, v in (("HEATMAP_RUN", cls.dir), ("HEATMAP_PATCHES", cls.dir), ("HEATMAP_GRIDS", cls.dir / "grids"),
                     ("HEATMAP_CACHE", cls.dir / "cache")):
            os.environ[k] = str(v)
        M = E.load_model(cls.dir, cls.dir)
        st = E.load_city("Synth", M, cls.dir / "grids", block=10)
        s, w, n, e = E.box_to_lonlat(st, (40, 140, 50, 150))
        cls.rect = (w, s, e, n)
        cls.feature = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}
        cls._orig = (streamlit_folium.st_folium, LIVE.build_area_grids)

        def fake_build(rect, out_dir, ghsl_cache, max_scenes=20, progress=None, **kw):
            name = LIVE.area_key(rect, max_scenes)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            for kind in ("features", "temp"):
                shutil.copy(cls.dir / "grids" / f"Synth_{kind}.nc", Path(out_dir) / f"{name}_{kind}.nc")
            info = {"name": name, "n_scenes": 20, "seconds": 1, "dates": ["2022-03-01", "2025-05-01"]}
            (Path(out_dir) / f"{name}_info.json").write_text(json.dumps(info))
            if progress:
                progress("fake download")
            return info

        LIVE.build_area_grids = fake_build

    @classmethod
    def tearDownClass(cls):
        streamlit_folium.st_folium, LIVE.build_area_grids = cls._orig
        cls.tmp.cleanup()

    def setUp(self):
        streamlit_folium.st_folium = lambda *a, **k: {"last_active_drawing": None}

    def metric_labels(self, at):
        return [m.label for m in at.metric]

    def test_prepared_mode_typed_box(self):
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        at.sidebar.radio[0].set_value("Prepared cities")
        at.run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertIn("City mean (satellite)", self.metric_labels(at))
        for w, v in zip(at.number_input, [40, 120, 50, 130]):
            w.set_value(v)
        [b for b in at.button if b.label == "Use this box"][0].click()
        at.run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertEqual(tuple(at.session_state["box"]), (40, 120, 50, 130))
        self.assertIn("Hotter than neighbours (model)", self.metric_labels(at))
        self.assertEqual([t.label for t in at.tabs], ["Why is it hot?", "How much cooling is needed?", "Maps", "Numbers"])

    def test_live_mode_draw_analyse_and_results(self):
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()                                                     # live mode is the default: the India map, nothing drawn yet
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertIsNone(at.session_state["live_rect"])
        self.assertEqual(self.metric_labels(at), [])                 # no results before an area is chosen
        streamlit_folium.st_folium = lambda *a, **k: {"last_active_drawing": self.feature}
        at.run()                                                     # the user draws a rectangle
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertIsNotNone(at.session_state["live_rect"])
        analyse = [b for b in at.button if b.label == "Analyse this area"]
        self.assertEqual(len(analyse), 1)
        analyse[0].click()
        at.run()                                                     # download (faked) and analysis
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertTrue(str(at.session_state["live_name"]).startswith("live_"))
        self.assertIsNotNone(at.session_state["box"])
        self.assertIn("Hotter than neighbours (model)", self.metric_labels(at))
        self.assertIn("Window", self.metric_labels(at))
        # a different place starts again
        [b for b in at.button if b.label == "Analyse a different place"][0].click()
        at.run()
        self.assertIsNone(at.session_state["live_name"])

    def test_live_mode_rejects_a_rectangle_outside_india(self):
        paris = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[2.0, 48.8], [2.1, 48.8], [2.1, 48.9], [2.0, 48.9], [2.0, 48.8]]]}}
        streamlit_folium.st_folium = lambda *a, **k: {"last_active_drawing": paris}
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertIsNone(at.session_state["live_rect"])
        self.assertIn("outside India", " ".join(w.value for w in at.warning))


if __name__ == "__main__":
    unittest.main()
