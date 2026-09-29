import unittest

from src.stage2.examples import EDIT_MARK, build_prompt, edit_context, trigger_text

ARTICLE = (
    "Rachel Reeves is a British politician.\n"
    "== Career ==\n"
    "She was elected in 2010 for Leeds West.\n"
    "=== Shadow Chancellor ===\n"
    "She became Shadow Chancellor in 2021. Later she served in the cabinet."
)


class EditContextTest(unittest.TestCase):
    def test_section_and_marked_window(self):
        offset = ARTICLE.index(" Later")
        section, context = edit_context(ARTICLE, offset, before=40, after=12)
        self.assertEqual(section, "Shadow Chancellor")
        self.assertIn(EDIT_MARK, context)
        before, after = context.split(EDIT_MARK)
        self.assertTrue(ARTICLE[:offset].endswith(before) and ARTICLE[offset:].startswith(after))
        self.assertFalse(before[0].isalpha() and ARTICLE[offset - len(before) - 1].isalpha())  # no cut word

    def test_lead_section(self):
        self.assertEqual(edit_context(ARTICLE, 5)[0], "(lead)")


class PromptTest(unittest.TestCase):
    def test_prompt_with_and_without_triggers(self):
        features = {"edits_1d": 3, "edits_7d": 5, "edits_30d": 9, "is_burst_1d": True, "burst_z_1d": 4.26}
        triggers = trigger_text(features, ["Keir_Starmer", "2024_United_Kingdom_general_election"])
        self.assertIn("Linked pages bursting yesterday: Keir Starmer; 2024 United Kingdom general election.", triggers)
        self.assertIn("z = 4.3", triggers)
        with_t = build_prompt("Rachel_Reeves", "2024-07-05", "Career", "ctx", triggers)
        without = build_prompt("Rachel_Reeves", "2024-07-05", "Career", "ctx")
        self.assertTrue(with_t.startswith("Wikipedia page: Rachel Reeves\nDate: 2024-07-05\nEdits to this page"))
        self.assertTrue(with_t.endswith("Inserted text:\n") and without.endswith("Inserted text:\n"))
        self.assertEqual(without, with_t.replace(triggers + "\n", ""))
        self.assertIn("Linked pages bursting yesterday: none.", trigger_text(features, []))

    def test_changes_on_bursting_neighbors(self):
        features = {"edits_1d": 3, "edits_7d": 5, "edits_30d": 9, "is_burst_1d": False, "burst_z_1d": 0.2}
        neighbors = ["Wilson_Fisk_(Marvel_Cinematic_Universe)", "Daredevil", "Kingpin", "Elektra"]
        changes = {"Wilson_Fisk_(Marvel_Cinematic_Universe)": "Fisk is elected Mayor of New York City",
                   "Daredevil": None, "Kingpin": "a", "Elektra": "b"}
        text = trigger_text(features, neighbors, changes=changes)
        self.assertTrue(text.endswith(
            'What changed on them yesterday:\n'
            '- Wilson Fisk (Marvel Cinematic Universe): "Fisk is elected Mayor of New York City"\n'
            '- Kingpin: "a"\n- Elektra: "b"'
        ))  # Daredevil has no snippet, so the first three with one are shown
        self.assertEqual(trigger_text(features, neighbors), trigger_text(features, neighbors, changes={}))
        self.assertNotIn("What changed", trigger_text(features, neighbors, changes={}))

    def test_ranked_changes_shown_as_given(self):
        features = {"edits_1d": 0, "edits_7d": 0, "edits_30d": 1, "is_burst_1d": False, "burst_z_1d": 0.0}
        neighbors = ["Daredevil", "Kingpin"]
        text = trigger_text(features, neighbors, ranked_changes=[("Kingpin", "Fisk is elected mayor."), ("Daredevil", "b")])
        self.assertTrue(text.endswith('What changed on them yesterday:\n- Kingpin: "Fisk is elected mayor."\n- Daredevil: "b"'))
        # nothing relevant: the same prompt as titles alone
        self.assertEqual(trigger_text(features, neighbors, ranked_changes=[]), trigger_text(features, neighbors))


if __name__ == "__main__":
    unittest.main()
