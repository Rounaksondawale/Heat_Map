"""Tests of the dashboard engine on a synthetic city:  python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import synthetic  # noqa: E402
import engine as E  # noqa: E402


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dir = synthetic.make_assets(cls.tmp.name)
        cls.M = E.load_model(cls.dir, cls.dir)
        cls.state = E.load_city("Synth", cls.M, cls.dir / "grids", block=10, cache_dir=cls.dir / "cache")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_and_cities(self):
        self.assertEqual(len(self.M["ck"]["features"]), 51)
        self.assertEqual(E.list_cities(self.dir / "grids"), ["Synth"])

    def test_prediction_cached_and_shaped(self):
        st = self.state
        self.assertEqual(st["pred"].shape, (synthetic.N, synthetic.N))
        self.assertTrue(np.isfinite(st["pred"][st["P"]["usable"]]).all())
        self.assertTrue((self.dir / "cache" / f"Synth_{self.dir.name}_pred.npy").exists())
        again = E.load_city("Synth", self.M, self.dir / "grids", block=10, cache_dir=self.dir / "cache")
        self.assertTrue(np.allclose(again["pred"], st["pred"]))

    def test_box_round_trip_and_limits(self):
        box = (60, 120, 50, 130)
        s, w, n, e = E.box_to_lonlat(self.state, box)
        r0, r1, c0, c1 = E.box_from_lonlat(self.state, w + 1e-6, s + 1e-6, e - 1e-6, n - 1e-6)
        self.assertLessEqual(abs(r0 - box[0]), 1)
        self.assertLessEqual(abs(c1 - box[3]), 1)
        with self.assertRaises(ValueError):
            E.check_box((0, 3, 0, 3), 200, 200)                 # too small
        with self.assertRaises(ValueError):
            E.check_box((0, 400, 0, 400), 1000, 1000)           # too large
        with self.assertRaises(ValueError):
            E.check_box((300, 400, 300, 400), 200, 200)         # outside the grid

    def test_analyse_area(self):
        res = E.analyse_area(self.state, self.M, (60, 130, 50, 130), ring=10, steps=4, z=1.645)
        self.assertEqual(set(res["phi"]), set(E.GROUPS))
        self.assertEqual(list(res["table"]["dose_%"]), [0, 25, 50, 75, 100])
        for k in ("veg", "roof", "both"):
            self.assertIn(k, res["need"])
            self.assertIn(f"{k}_gap_left", res["table"].columns)
        self.assertTrue(np.isfinite(res["pred_gap"]) and np.isfinite(res["obs_gap"]))
        self.assertAlmostEqual(res["unexplained"], res["obs_gap"] - res["pred_gap"], places=6)
        self.assertGreater(res["ha"], 50)
        self.assertLess(res["drift"], 0.05)                     # recomputing features without edits changes nothing
        cooling, gap = E.dose_now(res["S"], 0.0, 0.0)
        self.assertAlmostEqual(cooling, 0.0, places=5)
        self.assertAlmostEqual(gap, res["pred_gap"], places=5)

    def test_hotspots_and_layers(self):
        lab, px = E.hotspot_mask(self.state, 1.645)
        self.assertEqual(px.shape, (synthetic.N, synthetic.N))
        url, (lo, hi) = E.temperature_layer(self.state["pred"], self.state["P"]["usable"])
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertLess(lo, hi)
        self.assertTrue(E.mask_layer(px).startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
