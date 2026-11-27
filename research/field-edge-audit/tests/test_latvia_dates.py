"""Verify source-date grouping without loading any target inference result."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from latvia_local_dates import assign, capture_date


class DateGroupTests(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(capture_date("2020:07:04 12:11:10"), "2020-07-04")
        self.assertEqual(capture_date("2020:02:31 12:11:10"), "")
        self.assertEqual(capture_date(""), "")

    def test_linked_rows_are_excluded_and_days_are_never_split(self):
        def row(i, dy, label, h):
            return dict(
                source_path=str(i),
                source_sha256="sha" + str(i),
                phash=f"{h:016x}",
                exif_datetime_original=f"2020:07:{dy:02d} 12:00:00" if dy else "",
                y=label,
            )

        # Different high bits separate six genuine records. Group four extras with a
        # path through an undated bridge (4 changed bits per step) across camera days.
        rows = [
            row(0, 1, 0, 0xFFFF000000000000),
            row(1, 1, 1, 0x00FFFF0000000000),
            row(2, 2, 0, 0x0000FFFF00000000),
            row(3, 2, 1, 0x000000FFFF000000),
            row(4, 3, 0, 0x0000000000FFFF00),
            row(5, 3, 1, 0x000000000000FFFF),
            row(6, 1, 0, 0x0),
            row(7, 0, 0, 0xF),
            row(8, 3, 0, 0xFF),
            row(9, 0, 0, 0xFFFF0000FFFF0000),
        ]
        info, ds, cut, reason = assign(rows)
        self.assertEqual((len(ds), cut, reason), (3, 1, []))
        self.assertEqual(
            [v["status"] for v in info],
            ["calibration"] * 2
            + ["test"] * 4
            + ["cross_date"] * 3
            + ["missing_original_date"],
        )
        for index in (0, 1):
            rows[index]["y"] = 1
        self.assertIn("calibration_missing_one_or_both_classes", assign(rows)[-1])
