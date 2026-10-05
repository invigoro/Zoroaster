import unittest

from scripts.review_live import flags, review


class ReviewTest(unittest.TestCase):
    def test_garbled_and_vague_wording_is_flagged(self):
        self.assertEqual(flags("I predict that Khan Younis will see airstrikes by major armed forces and an actor."),
                         ["an actor"])
        self.assertEqual(flags("I predict that an important team representing a major country will win."),
                         ["an important team", "a major country"])
        self.assertEqual(flags("I predict that the Pivnichnyi Bridge in Kyiv will be struck again by Russian forces."), [])

    def test_a_day_s_review_shows_what_each_rewrite_replaced(self):
        record = {"date": "2026-10-05", "predictions": [
            {"text": "I predict that an armed group retains Mekelle.", "published": True, "topic": "conflict",
             "due": "2026-10-12", "rewritten_from": "I predict that the TPLF retains Mekelle."},
            {"text": "I predict that it rains.", "published": False}]}
        lines, counts = review([record])
        self.assertEqual(counts, [("2026-10-05", 1, 1, 0, 0)])
        self.assertIn("   - Rewritten from: I predict that the TPLF retains Mekelle.", lines)
        self.assertNotIn("rains", "\n".join(lines))  # only what was published


if __name__ == "__main__":
    unittest.main()
