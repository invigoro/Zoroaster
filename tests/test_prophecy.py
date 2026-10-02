import unittest
from datetime import date

from src.forecast.guardrails import sensitive_words
from src.prophecy.checks import (NO_ANSWER, already_known, confirmed_people, listed_names, names_a_person,
                                 novelty_messages, person_roles, screen)
from src.prophecy.evidence import eligible, evidence_text, is_biography, page_block
from src.prophecy.prophet import messages, normalize, parse_predictions

ROW = {"page_id": 7, "date": date(2026, 9, 30), "rank": 2, "page_title": "2026_Asian_Games", "living": False,
       "is_burst_1d": True, "edits_1d": 40, "editors_1d": 12, "edits_7d": 90,
       "lead": "The 2026 Asian Games are a multi-sport event in Aichi and Nagoya. " * 10,
       "yesterday_known": True, "yesterday_sections": ["Medal table", "(lead)"], "yesterday_section_chars": [500, 20],
       "yesterday_kinds": ["prose", "table"], "yesterday_prose": "Japan leads the medal table.",
       "bursting_neighbors": ["Japan_at_the_2026_Asian_Games"]}


class EvidenceTest(unittest.TestCase):
    def test_eligible_pages_skip_every_biography_and_keep_rank_order(self):
        rows = [ROW | {"rank": 3, "page_id": 1}, ROW | {"rank": 1, "page_id": 2, "living": True},
                ROW | {"rank": 2, "page_id": 3}, ROW | {"rank": 4, "page_id": 4, "page_text": "[[Category:2026 deaths]]"}]
        self.assertEqual([r["page_id"] for r in eligible(rows)], [3, 1])
        self.assertEqual([r["page_id"] for r in eligible(rows, people=True)], [2, 3, 1, 4])
        self.assertEqual([r["page_id"] for r in eligible(rows, top=1)], [3])

    def test_biographies_by_category(self):
        for text in ("[[Category:Living people]]", "[[Category:1940 births]]", "[[ Category : 2026_deaths|Smith]]",
                     "[[Category:500s BC births]]", "[[Category:Year of birth missing (living people)]]"):
            self.assertTrue(is_biography(text), text)
        for text in ("[[Category:2026 in Japan]]", "[[Category:Births in fiction]]", "no categories"):
            self.assertFalse(is_biography(text), text)

    def test_a_page_block_has_the_evidence_and_the_forecast(self):
        block = page_block(4, ROW, {"sections": ["Medal table"], "kinds": ["table"]})
        self.assertTrue(block.startswith("[4] 2026 Asian Games (ranked 2 for bursting today)"))
        self.assertIn("40 edits by 12 editors, a burst", block)
        self.assertIn('in Medal table; (lead) (prose, table). New text: "Japan leads the medal table."', block)
        self.assertIn("Linked pages that burst yesterday: Japan at the 2026 Asian Games", block)
        self.assertIn("The edit forecaster expects changes in: Medal table (table)", block)
        self.assertLess(len(block.split("About: ")[1].split("\n")[0]), 330)  # the lead is cut short
        text = evidence_text([ROW, ROW | {"page_id": 8}], {(8, "2026-09-30"): {"sections": [], "kinds": []}})
        self.assertIn("[2] 2026 Asian Games", text)
        self.assertEqual(text.count("The edit forecaster"), 1)  # only page 8 has a forecast


class ProphetTest(unittest.TestCase):
    def test_the_rules_name_the_day_and_the_evidence_comes_from_the_day_before(self):
        chat = messages(date(2026, 10, 1), "[1] ...")
        self.assertIn("on Thursday, 1 October 2026 (UTC)", chat[0]["content"])
        self.assertIn("as of the end of Wednesday, 30 September 2026", chat[1]["content"])

    def test_parsing_survives_a_broken_object_and_adds_the_prefix(self):
        answer = ('[\n{"prediction": "I predict that Japan wins.", "evidence": [1]},\n'
                  '{"prediction": "The election will go to a runoff.", "evidence": [2], "confidence": "low"}}\n]')
        got = parse_predictions(answer, ["A", "B"])  # the second object ends in an extra brace, as on 2026-09-20
        self.assertEqual([p["text"] for p in got], ["I predict that Japan wins.", "I predict that the election will go to a runoff."])
        self.assertEqual(normalize("Japan will win."), "I predict that Japan will win.")  # proper nouns keep their capital
        self.assertEqual(normalize("i predict that  it rains."), "I predict that it rains.")

    def test_parsing_tolerates_extra_text_and_skips_bad_entries(self):
        answer = ('Here you go:\n[{"prediction": "I predict that Japan will top the medal table.", "evidence": [2, 9, 2], '
                  '"confidence": "high"}, {"prediction": ""}, "junk", {"prediction": "I predict that a final is played.", '
                  '"evidence": "1", "confidence": "sure"}]\nThat is all.')
        got = parse_predictions(answer, ["A", "B"])
        self.assertEqual(got, [
            {"text": "I predict that Japan will top the medal table.", "question": "", "evidence": ["B"], "confidence": "high"},
            {"text": "I predict that a final is played.", "question": "", "evidence": [], "confidence": None}])
        with_question = parse_predictions('[{"question": " Who wins  Game 1? ", "prediction": "I predict that A wins."}]', [])
        self.assertEqual(with_question[0]["question"], "Who wins Game 1?")
        self.assertEqual(parse_predictions("no list here", ["A"]), [])
        self.assertEqual(parse_predictions("[not json]", ["A"]), [])


class ChecksTest(unittest.TestCase):
    def test_sensitive_topics_and_people_are_dropped(self):
        predictions = [{"text": "I predict that Japan will top the medal table."},
                       {"text": "I predict that the trial of the club's owner will open."},
                       {"text": "I predict that the president of France will visit Aichi."}]
        out = screen(predictions, ["No.", "no", "No"], [[], [], ["the president of France"]], ["no", "No", "no"],
                     ["yes", "Yes", "yes"], "")
        self.assertEqual([p["kept"] for p in out], [True, False, False])
        self.assertEqual(out[1]["dropped_because"], ["sensitive topic (trial)"])
        self.assertEqual(out[2]["dropped_because"],  # "No" missed it; the list and the role words didn't
                         ["names or points to a person (the president of France; president)"])
        one = predictions[:1]
        self.assertEqual(screen(one, ["Yes"], [[]], ["no"], ["yes"], "")[0]["dropped_because"],
                         ["names or points to a person"])
        self.assertEqual(screen(one, ["no"], [[]], ["Yes."], ["yes"], "")[0]["dropped_because"],
                         ["the evidence already settles it"])
        self.assertEqual(screen(one, ["no"], [[]], ["no"], ["No."], "")[0]["dropped_because"],
                         ["not about its cited evidence"])
        self.assertTrue(screen(one, ["no"], [[]], ["no"], ["Unclear"], "")[0]["kept"])  # only a clear no drops

    def test_copies_of_instruction_examples_and_repeats_are_dropped(self):
        instructions = ('A good one: {"question": "Who wins?", "prediction": "I predict that the Houston Astros will '
                        'beat the Chicago White Sox in Game 1 of the Wild Card Series.", "evidence": [2]}')
        predictions = [{"text": "I predict that the Houston Astros will beat the Chicago White Sox in Game 1."},
                       {"text": "I predict that Egypt will win the 2026 Men's African Nations Volleyball Championship."},
                       {"text": "I predict that Egypt will win the 2026 Men's African Nations Volleyball Championship!"}]
        out = screen(predictions, ["no"] * 3, [[]] * 3, ["no"] * 3, ["yes"] * 3, instructions)
        self.assertEqual([p["dropped_because"] for p in out],
                         [["copies an example from the instructions"], [], ["repeats an earlier prediction"]])
        self.assertTrue(all(p["kept"] for p in screen(predictions[1:2], ["no"], [[]], ["no"], ["yes"], instructions)))
        # As on 2026-09-18: the first copy cites the wrong page, so the second one is the one kept.
        twice = screen(predictions[1:], ["no"] * 2, [[]] * 2, ["no"] * 2, ["No", "yes"], instructions)
        self.assertEqual([p["dropped_because"] for p in twice], [["not about its cited evidence"], []])

    def test_the_novelty_check_sees_only_the_cited_evidence(self):
        chat = novelty_messages("I predict that it is held in Okazaki.", ["[3] Volleyball ... in Okazaki"])
        self.assertIn("[3] Volleyball ... in Okazaki", chat[0]["content"])
        self.assertIn("(no pages cited)", novelty_messages("I predict that X.", [])[0]["content"])
        self.assertTrue(already_known("Yes"))
        self.assertFalse(already_known("Probably"))  # a quality filter: only a clear yes drops

    def test_anything_but_a_clear_no_counts_as_a_person(self):
        self.assertFalse(names_a_person(" No"))
        self.assertTrue(names_a_person("Probably not"))
        self.assertTrue(names_a_person(""))
        self.assertEqual(listed_names(" None. "), [])
        self.assertEqual(listed_names("Atlanta Braves; none"), ["Atlanta Braves"])  # names a team, then hedges
        self.assertEqual(listed_names("H.E.R.; Liza Soberano"), ["H.E.R.", "Liza Soberano"])
        self.assertEqual(listed_names(""), [NO_ANSWER])
        self.assertEqual(confirmed_people(["Atlanta Braves", "Luke Hodge", "Israel"], ["team", "person", "Country."]),
                         ["Luke Hodge"])
        self.assertEqual(confirmed_people(["Someone", NO_ANSWER], ["athlete", ""]), ["Someone", NO_ANSWER])  # unclear: a person
        self.assertEqual(person_roles("The defending champion beats the Coach and the coach"), ["defending champion", "coach"])
        self.assertEqual(person_roles("Malta will host the contest; the Lions win."), [])
        dropped = screen([{"text": "I predict that the defending champion will win the darts."}], ["No"], [[]], ["no"],
                         ["yes"], "")
        self.assertEqual(dropped[0]["dropped_because"], ["names or points to a person (defending champion)"])
        self.assertEqual(sensitive_words("Arrest, then a Trial and another trial"), ["arrest", "trial"])


if __name__ == "__main__":
    unittest.main()
