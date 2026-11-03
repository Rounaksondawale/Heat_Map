"""Tests of the live-pipeline helpers (no network):  python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import live as LIVE  # noqa: E402

PUNE = (73.84, 18.51, 73.87, 18.54)          # about 3 x 3 km


class LiveTests(unittest.TestCase):
    def test_check_rect_accepts_a_valid_rectangle(self):
        self.assertEqual(LIVE.check_rect(*PUNE), PUNE)

    def test_check_rect_rejects_bad_rectangles(self):
        with self.assertRaises(ValueError):
            LIVE.check_rect(73.87, 18.51, 73.84, 18.54)                   # reversed
        with self.assertRaises(ValueError):
            LIVE.check_rect(2.0, 48.8, 2.05, 48.85)                       # Paris: outside India
        with self.assertRaises(ValueError):
            LIVE.check_rect(73.84, 18.51, 73.8405, 18.5105)               # a few tens of metres
        with self.assertRaises(ValueError):
            LIVE.check_rect(73.5, 18.2, 74.5, 19.2)                       # about 100 km

    def test_window_is_centred_and_big_enough(self):
        lon, lat, half = LIVE.window_for(PUNE)
        self.assertAlmostEqual(lon, 73.855, places=4)
        self.assertAlmostEqual(lat, 18.525, places=4)
        self.assertGreaterEqual(half, LIVE.MIN_HALF_KM)
        big = (73.80, 18.50, 73.90, 18.58)                                 # about 10 x 9 km
        self.assertGreater(LIVE.window_for(big)[2], half)

    def test_area_key_is_stable_and_specific(self):
        k = LIVE.area_key(PUNE)
        self.assertEqual(k, LIVE.area_key(PUNE))
        self.assertTrue(k.startswith("live_"))
        self.assertNotEqual(k, LIVE.area_key(PUNE, max_scenes=30))
        self.assertNotEqual(k, LIVE.area_key((73.85, 18.51, 73.88, 18.54)))

    def test_is_built(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            self.assertFalse(LIVE.is_built(d, "live_x"))
            (Path(d) / "live_x_features.nc").write_bytes(b"x")
            self.assertFalse(LIVE.is_built(d, "live_x"))
            (Path(d) / "live_x_temp.nc").write_bytes(b"x")
            self.assertTrue(LIVE.is_built(d, "live_x"))

    def test_only_one_build_at_a_time(self):
        self.assertTrue(LIVE._LOCK.acquire(blocking=False))
        try:
            with self.assertRaises(LIVE.Busy):
                LIVE.build_area_grids(PUNE, "unused", "unused")
        finally:
            LIVE._LOCK.release()


if __name__ == "__main__":
    unittest.main()
