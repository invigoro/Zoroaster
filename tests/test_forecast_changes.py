import unittest

from src.forecast.changes import LEAD, day_change, headings, lead, new_spans, section_at

MOVED = "This older paragraph is long enough to count, and it moves to the end."
START = (
    "{{Infobox officeholder|name=Jane Roe}}\n"
    "'''Jane Roe''' is a British politician.<ref>{{cite web|title=Bio}}</ref> She has served since 2010.\n"
    "== Career ==\n"
    "She was elected in 2010.\n"
    f"{MOVED}\n"
    "== Results ==\n"
    '{| class="wikitable"\n|-\n| 2022 || 51%\n|}\n'
)
END = (
    "{{Infobox officeholder|name=Jane Roe}}\n"
    "'''Jane Roe''' is a British politician.<ref>{{cite web|title=Bio}}</ref> She has served since 2010.\n"
    "== Career ==\n"
    "She was elected in 2010.\n"
    "In 2026 she was appointed Chancellor of the Exchequer.<ref>{{cite news|title=Appointed}}</ref>\n"
    "== Results ==\n"
    '{| class="wikitable"\n|-\n| 2022 || 51%\n|-\n| 2026 || 54%\n|}\n'
    "== Legacy ==\n"
    "Her legacy is still debated by historians today.\n"
    f"{MOVED}\n"
)


class PageTest(unittest.TestCase):
    def test_headings_lead_and_sections(self):
        self.assertEqual(headings("== Career ==\n=== [[2026 season|The 2026 season]] ===\n"),
                         [(2, "Career"), (3, "The 2026 season")])
        self.assertEqual(lead(START), "Jane Roe is a British politician. She has served since 2010.")
        self.assertTrue(lead(START, max_chars=30).endswith(" …"))
        self.assertEqual(section_at(END, END.index("Jane")), LEAD)
        self.assertEqual(section_at(END, END.index("In 2026")), "Career")
        self.assertEqual(section_at(END, END.index("| 2026")), "Results")


class DayChangeTest(unittest.TestCase):
    def test_new_prose_sections_and_kinds(self):
        change = day_change(START, END)
        self.assertEqual(change["prose"], "In 2026 she was appointed Chancellor of the Exchequer. "
                                          "Her legacy is still debated by historians today.")
        self.assertNotIn("moves to the end", change["prose"])  # moved, not new
        self.assertEqual(change["sections"], ["Career", "Results", "Legacy"])
        self.assertEqual(change["kinds"], ["new section", "prose", "references", "table", "template fields"])
        self.assertEqual(len(change["section_chars"]), 3)
        self.assertGreater(change["section_chars"][0], change["section_chars"][1])  # Career's sentence > Results' row
        self.assertTrue(any("Chancellor of the Exchequer" in block for block in change["blocks"]))

    def test_a_wording_fix_is_a_copyedit_not_prose(self):
        change = day_change(START, START.replace("She was elected in 2010.", "She was first elected in 2010."))
        self.assertEqual((change["kinds"], change["prose"], change["sections"]), (["copyedits"], "", ["Career"]))

    def test_page_created_that_day(self):
        change = day_change(None, END)
        self.assertTrue(change["prose"].startswith("Jane Roe is a British politician."))
        self.assertEqual(change["sections"][0], LEAD)
        self.assertIn("new section", change["kinds"])

    def test_no_change(self):
        self.assertEqual(day_change(START, START), {"prose": "", "sections": [], "section_chars": [], "kinds": [],
                                                    "inserted_chars": 0, "removed_chars": 0, "spans": [], "blocks": []})
        self.assertEqual(new_spans(START, START), ([], 0, 0))


if __name__ == "__main__":
    unittest.main()
