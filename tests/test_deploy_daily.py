import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.deploy.daily import (COLUMNS, HISTORY_START, LIVE_START, bucket_features, bucket_outcomes, live_files,
                              live_mass_editors, split_by_bucket)
from src.features.activity import PageActivity, day_ordinal
from src.ingest.revert_detect import detect_page_reverts

DAY = date(2026, 9, 15)


def rev(page_id, rev_id, when, sha1, user="Alice", title=None, created=None):
    return {"page_id": page_id, "page_title": title or f"Page_{page_id}", "revision_id": rev_id, "parent_id": rev_id - 1,
            "timestamp": when, "user_text": user, "is_anon": False, "is_bot": False, "byte_size": 100 + rev_id,
            "sha1": sha1, "page_created": created}


DUMP = [
    rev(1, 10, "2026-08-20T10:00:00Z", "a", created="2020-01-01T00:00:00Z"),
    rev(1, 11, "2026-08-25T10:00:00Z", "b", "Bob", created="2020-01-01T00:00:00Z"),
    rev(1, 12, "2026-09-01T00:30:00Z", "c", created="2020-01-01T00:00:00Z"),  # also in the live data
    rev(2, 20, "2026-06-01T10:00:00Z", "x", created="2019-05-05T00:00:00Z"),  # dormant since
]
LIVE = [
    rev(1, 12, "2026-09-01T00:30:00Z", "c"),
    rev(1, 13, "2026-09-10T09:00:00Z", "d", "Carol", title="Page_One_renamed"),
    rev(1, 14, "2026-09-12T09:00:00Z", "a", "Dave", title="Page_One_renamed"),  # restores August's content
    rev(1, 15, "2026-09-15T01:00:00Z", "e", title="Page_One_D"),  # on day D itself
    rev(3, 30, "2026-09-10T12:00:00Z", "n", created="2026-09-10T12:00:00Z"),  # created live
    rev(4, 40, "2026-09-15T03:00:00Z", "z"),  # only active on day D
    rev(5, 50, "2026-09-08T12:00:00Z", "o"),  # first edit in years: no dump rows, no creation in the live days
]


def write(rows, path):
    pq.write_table(pa.Table.from_pylist([{k: r[k] for k in COLUMNS} for r in rows]), path)


class BucketFeaturesTest(unittest.TestCase):
    def test_features_match_the_training_code_on_the_merged_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write(DUMP, tmp / "dump.parquet")
            write(LIVE, tmp / "live.parquet")
            count = bucket_features(DAY, [tmp / "dump.parquet"], tmp / "live.parquet", tmp / "out.parquet")
            rows = {r["page_id"]: r for r in pq.read_table(tmp / "out.parquet").to_pylist()}
        self.assertEqual(count, 3)
        self.assertEqual(set(rows), {1, 3, 5})  # 2 is dormant, 4's only edit is on D
        merged = [r for r in DUMP[:2]] + [r for r in LIVE[:3]]  # dump before LIVE_START, live before D
        labeled = detect_page_reverts(merged)
        self.assertTrue(labeled[1]["is_reverted"] and labeled[1]["reverted_by_revision_id"] == 14)  # across sources
        expected = PageActivity(labeled, frozenset(), day_ordinal("2020-01-01T00:00:00Z")).features(DAY.toordinal())
        self.assertEqual({k: rows[1][k] for k in expected}, expected)
        self.assertEqual(rows[1]["page_title"], "Page_One_renamed")  # latest title before D
        self.assertEqual(rows[3]["page_age_days"], 5)  # creation from the live data
        self.assertEqual(rows[5]["page_age_days"], (DAY - HISTORY_START).days)  # predates the history, not new
        self.assertEqual(rows[1]["date"], DAY)

    def test_split_and_live_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write(LIVE[:3], tmp / "2026-09-01.parquet")
            write(LIVE[3:], tmp / "2026-09-02.parquet")
            files = live_files(tmp, date(2026, 9, 1), date(2026, 9, 2))
            with self.assertRaises(FileNotFoundError):
                live_files(tmp, date(2026, 9, 1), date(2026, 9, 3))
            split_by_bucket(files, tmp / "buckets", n_buckets=2)
            even = pq.read_table(tmp / "buckets" / "bucket=000.parquet")["page_id"].to_pylist()
            odd = pq.read_table(tmp / "buckets" / "bucket=001.parquet")["page_id"].to_pylist()
        self.assertEqual((sorted(even), sorted(odd)), ([4], [1, 1, 1, 1, 3, 5]))
        self.assertEqual(LIVE_START, date(2026, 9, 1))


class OutcomesTest(unittest.TestCase):
    def test_bursts_on_the_day_by_two_or_more_non_mass_editors(self):
        quiet = [rev(p, p * 100 + 1, "2026-09-02T10:00:00Z", f"{p}q", created="2020-01-01T00:00:00Z") for p in (1, 2, 3, 4)]
        on_day = lambda p, users: [rev(p, p * 100 + 10 + i, f"2026-09-15T0{i}:00:00Z", f"{p}s{i}", u)
                                   for i, u in enumerate(users)]
        live = (quiet + on_day(1, ["Alice", "Bob", "Alice", "Bob"]) + on_day(2, ["Alice"] * 4)
                + on_day(3, ["Mass", "Mass", "Mass", "Bob"])
                + [rev(4, 499, "2026-09-16T01:00:00Z", "4x", "Carol"), rev(4, 498, "2026-09-16T02:00:00Z", "4y", "Dan")])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write(live, tmp / "live.parquet")
            out = {r["page_id"]: r for r in bucket_outcomes(DAY, [], tmp / "live.parquet", [1, 2, 3, 4],
                                                             frozenset({("Mass", DAY.toordinal())}))}
        self.assertEqual({p: out[p]["burst"] for p in out}, {1: True, 2: False, 3: False, 4: False})
        self.assertEqual((out[1]["edits"], out[1]["editors"]), (4, 2))
        self.assertEqual(out[2]["editors"], 1)  # one editor's saves are not an event
        self.assertEqual(out[3]["editors"], 1)  # the mass editor is left out
        self.assertEqual(out[4]["edits"], 0)  # the next day's edits don't count

    def test_live_mass_editors(self):
        rows = ([rev(p, p, "2026-09-15T10:00:00Z", "a", "Busy") for p in range(26)]
                + [rev(p, 100 + p, "2026-09-15T10:00:00Z", "a", "Nearly") for p in range(25)]
                + [dict(rev(p, 200 + p, "2026-09-15T10:00:00Z", "a", "SomeBot"), is_bot=True) for p in range(40)])
        with tempfile.TemporaryDirectory() as tmp:
            write(rows, Path(tmp) / "day.parquet")
            mass = live_mass_editors(Path(tmp) / "day.parquet", DAY)
        self.assertEqual(mass, frozenset({("Busy", DAY.toordinal())}))


if __name__ == "__main__":
    unittest.main()
