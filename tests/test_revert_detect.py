import unittest
from datetime import datetime, timedelta

from src.ingest.revert_detect import detect_page_reverts, detect_reverts

START = datetime(2020, 1, 1)


def history(sha1s, page_id=1, hours_apart=1.0):
    """One revision per sha1, `hours_apart` hours apart, revision ids 1..n."""
    return [
        {
            "page_id": page_id,
            "revision_id": i + 1,
            "timestamp": (START + timedelta(hours=i * hours_apart)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha1": sha1,
        }
        for i, sha1 in enumerate(sha1s)
    ]


def flags(labeled):
    return (
        [r["is_reverted"] for r in labeled],
        [r["is_revert"] for r in labeled],
        [r["reverted_by_revision_id"] for r in labeled],
    )


class DetectPageRevertsTest(unittest.TestCase):
    def test_simple_revert(self):
        reverted, revert, by = flags(detect_page_reverts(history("ABA")))
        self.assertEqual(reverted, [False, True, False])
        self.assertEqual(revert, [False, False, True])
        self.assertEqual(by, [None, 3, None])

    def test_repeated_vandalism_cycles_do_not_flag_the_restores(self):
        # The earliest-index version flagged revision 3 (the first restore)
        # as reverted, because revision 5 also restores A.
        reverted, revert, by = flags(detect_page_reverts(history("ABACA")))
        self.assertEqual(reverted, [False, True, False, True, False])
        self.assertEqual(revert, [False, False, True, False, True])
        self.assertEqual(by, [None, 3, None, 5, None])

    def test_revert_undoes_everything_back_to_the_restored_state(self):
        reverted, revert, by = flags(detect_page_reverts(history("ABCDA")))
        self.assertEqual(reverted, [False, True, True, True, False])
        self.assertEqual(by, [None, 5, 5, 5, None])

    def test_first_reverting_revision_is_recorded(self):
        # 3 restores B (undoing 2); 4 restores X (undoing 1..3).
        _, revert, by = flags(detect_page_reverts(history("XYVYX")))
        self.assertEqual(revert, [False, False, False, True, True])
        self.assertEqual(by, [None, 5, 4, 5, None])

    def test_null_revision_is_not_a_revert(self):
        reverted, revert, _ = flags(detect_page_reverts(history("AAB")))
        self.assertEqual(reverted, [False, False, False])
        self.assertEqual(revert, [False, False, False])

    def test_revision_window(self):
        # 16 revisions after the undone one is outside the 15-revision window.
        sha1s = ["A", "B"] + [f"C{i}" for i in range(15)] + ["A"]
        labeled = detect_page_reverts(history(sha1s))
        self.assertFalse(labeled[1]["is_reverted"])
        self.assertTrue(labeled[2]["is_reverted"])
        self.assertTrue(labeled[-1]["is_revert"])  # still a revert

    def test_time_window(self):
        labeled = detect_page_reverts(history("ABA", hours_apart=24 * 91))
        self.assertFalse(labeled[1]["is_reverted"])
        self.assertTrue(labeled[2]["is_revert"])

    def test_missing_sha1_is_skipped(self):
        reverted, revert, _ = flags(detect_page_reverts(history(["A", None, "A"])))
        self.assertEqual(reverted, [False, True, False])
        self.assertEqual(revert, [False, False, True])

    def test_input_order_within_page_does_not_matter(self):
        revisions = history("ABACA")
        self.assertEqual(
            detect_page_reverts(list(reversed(revisions))), detect_page_reverts(revisions)
        )


class DetectRevertsStreamingTest(unittest.TestCase):
    def test_pages_are_labeled_independently(self):
        stream = history("AB", page_id=1) + history("BA", page_id=2)
        labeled = list(detect_reverts(stream))
        self.assertEqual([r["page_id"] for r in labeled], [1, 1, 2, 2])
        self.assertFalse(any(r["is_revert"] or r["is_reverted"] for r in labeled))


if __name__ == "__main__":
    unittest.main()
