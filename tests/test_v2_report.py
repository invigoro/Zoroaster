import unittest
from datetime import date

from scripts.v2_report import forecast_view, view_row

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


if __name__ == "__main__":
    unittest.main()
