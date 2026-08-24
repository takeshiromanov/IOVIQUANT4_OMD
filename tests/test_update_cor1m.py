import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.update_cor1m import LatestQuote, business_day_lag, upsert_quote


CSV_HEADER = '"Date","Price","Open","High","Low","Vol.","Change %"\n'


class Cor1MUpdaterTests(unittest.TestCase):
    def _csv(self, directory: str) -> Path:
        path = Path(directory) / "cor1m.csv"
        path.write_text(
            CSV_HEADER
            + '"08/20/2026","10.00","10.00","10.00","10.00","","0.00%"\n',
            encoding="utf-8-sig",
        )
        return path

    def test_business_day_lag_ignores_weekend(self):
        friday = date(2026, 8, 21)
        self.assertEqual(business_day_lag(friday, date(2026, 8, 22)), 0)
        self.assertEqual(business_day_lag(friday, date(2026, 8, 23)), 0)
        self.assertEqual(business_day_lag(friday, date(2026, 8, 24)), 1)

    def test_missing_date_is_inserted_with_close_for_all_ohlc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv(directory)
            quote = LatestQuote(date(2026, 8, 21), 12.5)
            result = upsert_quote(path, quote)
            frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            row = frame.loc[frame["Date"].eq("08/21/2026")].iloc[0]
            self.assertTrue(result.changed)
            self.assertEqual(result.action, "inserted")
            self.assertEqual(
                row[["Price", "Open", "High", "Low"]].tolist(),
                ["12.50"] * 4,
            )
            self.assertEqual(row["Change %"], "25.00%")

    def test_existing_date_is_updated_then_becomes_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv(directory)
            quote = LatestQuote(date(2026, 8, 20), 11.25)
            updated = upsert_quote(path, quote)
            unchanged = upsert_quote(path, quote)
            frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            self.assertEqual(updated.action, "updated")
            self.assertTrue(updated.changed)
            self.assertEqual(unchanged.action, "unchanged")
            self.assertFalse(unchanged.changed)
            self.assertEqual(frame.iloc[-1]["Price"], "11.25")

    def test_newest_first_csv_keeps_its_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cor1m_desc.csv"
            path.write_text(
                CSV_HEADER
                + '"08/20/2026","10.00","10.00","10.00","10.00","","11.11%"\n'
                + '"08/19/2026","9.00","9.00","9.00","9.00","","0.00%"\n',
                encoding="utf-8-sig",
            )
            result = upsert_quote(
                path, LatestQuote(date(2026, 8, 21), 12.5)
            )
            frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            self.assertTrue(result.changed)
            self.assertEqual(frame.iloc[0]["Date"], "08/21/2026")
            self.assertEqual(frame.iloc[0]["Change %"], "25.00%")


if __name__ == "__main__":
    unittest.main()
