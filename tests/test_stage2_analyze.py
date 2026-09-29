import unittest

from scripts.analyze_stage2 import snippet_overlap, title_in_target


class MechanismTest(unittest.TestCase):
    def test_title_in_target(self):
        row = {"added_text": "''[[Vachellia planifrons]]''", "bursting_neighbors": ["Vachellia_planifrons", "Other"]}
        self.assertTrue(title_in_target(row))
        self.assertFalse(title_in_target(row | {"bursting_neighbors": ["Acacia"]}))
        self.assertFalse(title_in_target(row | {"bursting_neighbors": []}))

    def test_snippet_overlap(self):
        row = {"added_text": "The impeachment hearings resumed in April.", "bursting_neighbors": ["A", "B"],
               "neighbor_changes": {"A": "Impeachment proceedings began", "B": None}}
        # words of 4+ letters: impeachment, hearings, resumed, april; one is shared
        self.assertEqual(snippet_overlap(row), 0.25)
        self.assertIsNone(snippet_overlap(row | {"neighbor_changes": {"A": None, "B": None}}))


if __name__ == "__main__":
    unittest.main()
