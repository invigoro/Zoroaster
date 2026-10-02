import unittest
from datetime import date

from scripts.v2_report import (RANKED, SITE_PROSE_CHARS, WITHHELD_INVENTED, WITHHELD_SENSITIVE, forecast_view,
                               public_row, view_row)

ROW = {"page_title": "Jane_Roe", "date": date(2026, 3, 2), "selection": "top", "living": True,
       "heading_titles": ["Career", "Results"], "sections": ["(lead)", "Career", "Results"],
       "section_chars": [5, 300, 120], "kinds": ["prose", "table"], "prose": "x" * 400,
       "yesterday_sections": [], "yesterday_section_chars": [], "yesterday_kinds": []}


class ReportTest(unittest.TestCase):
    def test_marks_sections_and_kinds(self):
        view = forecast_view({"sections": ["Career", "(lead)", "Arrest", "Results"], "kinds": ["prose", "links"]}, ROW)
        self.assertEqual([(s["name"], s["mark"], s["on_page"]) for s in view["sections"]],
                         [("Career", "main", True), ("(lead)", "hit", True), ("Arrest", "miss", False),
                          ("Results", "hit", True)])
        self.assertEqual(view["kinds"], [{"name": "prose", "hit": True}, {"name": "links", "hit": False}])

    def test_row_shows_main_sections_by_size_and_cuts_the_prose(self):
        row = view_row(ROW, {"yesterday again": {"sections": [], "kinds": []}})
        self.assertEqual(row["actual"]["sections"], [["Career", 300], ["Results", 120], ["(lead)", 5]])
        self.assertEqual((row["title"], row["date"], row["top"], row["changed_yesterday"]),
                         ("Jane Roe", "2026-03-02", True, False))
        self.assertEqual(len(row["actual"]["prose"]), 301)  # 300 characters and an ellipsis


class PublicTest(unittest.TestCase):
    NAMED = {"yesterday again": {"sections": ["Career"], "kinds": ["prose"]},
             RANKED: {"sections": ["Career", "Awards", "Personal life"], "kinds": ["prose", "table"]}}

    def test_made_up_and_sensitive_names_are_withheld(self):
        row = ROW | {"living": False, "heading_titles": ["Career", "Personal life"],
                     "sections": ["Career", "Personal life"], "section_chars": [300, 50]}
        out = public_row(row, self.NAMED)
        self.assertEqual(out["f"][1]["s"], [["Career", "main"], [WITHHELD_INVENTED, "miss withheld"],
                                            [WITHHELD_SENSITIVE, "hit withheld"]])
        self.assertEqual(out["f"][1]["k"], [["prose", True], ["table", True]])
        self.assertEqual(out["a"]["s"], [["Career", "main", 300], [WITHHELD_SENSITIVE, "hit withheld", 50]])
        self.assertEqual(out["a"]["p"], "")  # a sensitive section changed: no quote

    def test_quotes_only_for_pages_not_about_living_people(self):
        row = ROW | {"living": False}
        self.assertEqual(len(public_row(row, self.NAMED)["a"]["p"]), SITE_PROSE_CHARS + 1)  # cut, with an ellipsis
        self.assertEqual(public_row(ROW, self.NAMED)["a"]["p"], "")  # ROW is about a living person


if __name__ == "__main__":
    unittest.main()
