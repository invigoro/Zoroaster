import unittest
from datetime import date

from src.forecast.prompts import build_prompt, header, parse_forecast, point_in_time_titles, target_text

EXAMPLE = {
    "page_title": "Jane_Roe", "date": date(2026, 3, 2), "lead": "Jane Roe is a British politician. " * 30,
    "heading_titles": ["Career", "Results", "Personal life"],
    "yesterday_known": True, "yesterday_sections": ["(lead)", "Career"], "yesterday_section_chars": [10, 90],
    "yesterday_kinds": ["prose"], "yesterday_prose": "She was re-elected in 2024.",
    "edits_1d": 3, "edits_7d": 5, "edits_30d": 9, "is_burst_1d": True, "burst_z_1d": 4.26,
    "bursting_neighbors": ["2026_United_Kingdom_general_election"],
    "sections": ["(lead)", "Career", "Results"], "section_chars": [5, 300, 120],
    "kinds": ["prose", "references", "table"], "prose": "In 2026 she was appointed chancellor.",
}


class PromptTest(unittest.TestCase):
    def test_variants_add_information_in_steps(self):
        page, yesterday, full = (build_prompt(EXAMPLE, v) for v in ("page", "yesterday", "full"))
        self.assertTrue(page.startswith("Wikipedia page: Jane Roe\nDate: 2026-03-02\nLead: Jane Roe is"))
        self.assertIn("Sections: Career; Results; Personal life", page)
        self.assertNotIn("Yesterday", page)
        self.assertIn('Yesterday\'s changes: Career; (lead) (prose): "She was re-elected in 2024."', yesterday)
        self.assertNotIn("Linked pages", yesterday)
        self.assertIn("Linked pages bursting yesterday: 2026 United Kingdom general election", full)
        self.assertIn("Page bursting yesterday: yes (z = 4.3)", full)
        self.assertTrue(all(p.endswith("Forecast for today's edits:\n") for p in (page, yesterday, full)))
        self.assertLess(len(build_prompt(EXAMPLE, "page", lead_chars=100)), len(page))

    def test_quiet_and_unknown_yesterday(self):
        quiet = build_prompt(EXAMPLE | {"yesterday_sections": [], "yesterday_section_chars": []}, "yesterday")
        self.assertIn("Yesterday's changes: none", quiet)
        unknown = build_prompt(EXAMPLE | {"yesterday_known": False}, "yesterday")
        self.assertIn("Yesterday's changes: unknown", unknown)


class TitlesTest(unittest.TestCase):
    def test_titles_at_the_end_of_d_minus_1(self):
        rows = [{"page_title": "Barry_Wilmore", "prompt_id": 7, "date": date(2026, 3, 2),
                 "bursting_neighbors": ["Starliner", "ISS"]},
                {"page_title": "Unfound", "prompt_id": 8, "date": date(2026, 3, 2), "bursting_neighbors": []}]
        titles = [
            {"kind": "page", "title": "Barry_Wilmore", "date": date(2026, 3, 2), "revision_id": 7,
             "title_then": "Barry_E._Wilmore"},
            {"kind": "page", "title": "Unfound", "date": date(2026, 3, 2), "revision_id": 8, "title_then": None},
            {"kind": "neighbor", "title": "Starliner", "date": date(2026, 3, 1), "revision_id": 1,
             "title_then": "Boeing_Starliner"},
            {"kind": "neighbor", "title": "ISS", "date": date(2026, 3, 2), "revision_id": 2,  # D itself: too late
             "title_then": "International_Space_Station"},
        ]
        renamed = point_in_time_titles(rows, titles)
        self.assertEqual(rows[0]["page_title"], "Barry_E._Wilmore")
        self.assertEqual(rows[0]["bursting_neighbors"], ["Boeing_Starliner", "ISS"])
        self.assertEqual(rows[1]["page_title"], "Unfound")  # not found: the snapshot's title
        self.assertEqual(renamed, {"pages": 1, "neighbors": 1})


class TargetTest(unittest.TestCase):
    def test_header_first_then_text_and_parsing_back(self):
        self.assertEqual(target_text(EXAMPLE), "Sections: Career; Results; (lead)\nKinds: prose, references, table\n"
                                               "New text: In 2026 she was appointed chancellor.")
        self.assertEqual(parse_forecast(target_text(EXAMPLE)),
                         {"sections": ["Career", "Results", "(lead)"], "kinds": ["prose", "references", "table"]})
        empty = EXAMPLE | {"sections": [], "section_chars": [], "kinds": [], "prose": ""}
        self.assertEqual(target_text(empty), "Sections: none\nKinds: none\nNew text: none")
        self.assertEqual(parse_forecast(header(empty)), {"sections": [], "kinds": []})
        self.assertEqual(parse_forecast("Sections: A; B; C; D\nKinds: prose, gossip, links\nNew text: x"),
                         {"sections": ["A", "B", "C"], "kinds": ["prose", "links"]})  # 3 sections, known kinds
        self.assertEqual(parse_forecast("Sections: (lead); (lead); (lead); Career; (le")["sections"],
                         ["(lead)", "Career", "(le"])  # a repeating generation, cut off by the token cap


if __name__ == "__main__":
    unittest.main()
