import tempfile
import unittest
from pathlib import Path

import pandas as pd

from shared_data import business_day_lag, load_cor1m_series, read_csv_source


HEADER = '"Date","Price","Open","High","Low","Vol.","Change %"\n'


class SharedDataTests(unittest.TestCase):
    def _write_pair(self, directory: str) -> None:
        base = Path(directory)
        (base / "daily.csv").write_text(
            HEADER
            + '"08/22/2026","9","9","9","9","",""\n'
            + '"08/21/2026","8","8","8","8","",""\n',
            encoding="utf-8-sig",
        )
        (base / "weekly.csv").write_text(
            HEADER + '"03/27/2022","20","20","20","20","",""\n',
            encoding="utf-8-sig",
        )

    def test_remote_pair_is_preferred_and_weekend_daily_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_pair(directory)
            base = Path(directory)
            frame, source = load_cor1m_series(
                directory,
                remote_files=(base / "daily.csv", base / "weekly.csv"),
                local_files=("missing-daily.csv", "missing-weekly.csv"),
                reader=read_csv_source,
            )
            self.assertEqual(source, "IOVIQUANT_DATA")
            self.assertNotIn(pd.Timestamp("2026-08-22"), frame.index)
            self.assertEqual(frame.loc[pd.Timestamp("2026-08-21"), "Close"], 8)

    def test_remote_failure_falls_back_to_complete_local_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_pair(directory)

            def reader(source):
                if str(source).startswith("https://"):
                    raise OSError("offline")
                return read_csv_source(source)

            frame, source = load_cor1m_series(
                directory,
                remote_files=("https://daily", "https://weekly"),
                local_files=("daily.csv", "weekly.csv"),
                reader=reader,
            )
            self.assertEqual(source, "fallback locale")
            self.assertFalse(frame.empty)

    def test_business_day_lag_ignores_weekend(self):
        self.assertEqual(business_day_lag("2026-08-21", "2026-08-23"), 0)
        self.assertEqual(business_day_lag("2026-08-21", "2026-08-24"), 1)


if __name__ == "__main__":
    unittest.main()
