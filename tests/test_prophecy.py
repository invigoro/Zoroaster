import unittest
from datetime import date

from src.forecast.guardrails import sensitive_words
from src.prophecy.checks import (NO_ANSWER, already_known, confirmed_orgs, confirmed_people, listed_names,
                                 names_a_person, novelty_messages, on_a_sensitive_topic, person_roles, screen)
from src.prophecy.evidence import eligible, evidence_text, is_biography, mark_dates, page_block
from src.prophecy.prophet import normalize, parse_prediction, parse_question, prediction_messages, question_messages

ROW = {"page_id": 7, "date": date(2026, 9, 30), "rank": 2, "page_title": "2026_Asian_Games", "living": False,
       "is_burst_1d": True, "edits_1d": 40, "editors_1d": 12, "edits_7d": 90,
       "lead": "The 2026 Asian Games are a multi-sport event in Aichi and Nagoya. " * 20,
       "yesterday_known": True, "yesterday_sections": ["Medal table", "(lead)"], "yesterday_section_chars": [500, 20],
       "yesterday_kinds": ["prose", "table"], "yesterday_prose": "Japan leads the medal table.",
       "bursting_neighbors": ["Japan_at_the_2026_Asian_Games"]}


def clean(n: int) -> list[str]:
    """n "no" answers: the checks finding nothing."""
    return ["no"] * n


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
        self.assertLess(len(block.split("About: ")[1].split("\n")[0]), 910)  # the lead is cut short
        self.assertNotIn("\n\n", block)  # blocks are joined, and split again, on blank lines
        text = evidence_text([ROW, ROW | {"page_id": 8}], {(8, "2026-09-30"): {"sections": [], "kinds": []}})
        self.assertIn("[2] 2026 Asian Games", text)
        self.assertEqual(text.count("The edit forecaster"), 1)  # only page 8 has a forecast

    def test_the_evidence_never_sees_the_day_itself(self):
        # The rows also hold what the day brought, for the judge. None of it may reach the prophet.
        day_side = {"prose": "Fremantle won by 30 points.", "sections": ["Result"], "section_chars": [900],
                    "kinds": ["prose"], "inserted_chars": 900, "removed_chars": 0, "spans": ["won by 30"],
                    "blocks": ["| winner = Fremantle"], "end_id": 99}
        block = page_block(1, ROW, None)
        self.assertEqual(page_block(1, ROW | day_side, None), block)
        for value in ("Fremantle", "30 points", "Result"):
            self.assertNotIn(value, block)

    def test_dates_are_marked_relative_to_the_day_foretold(self):
        day = date(2026, 9, 30)
        cases = {"will be played on October 4 at Accor Stadium": "October 4 [in 4 days] at",
                 "held from 20 to 24 September 2026 at": "20 to 24 September 2026 [ended 6 days ago] at",
                 "from 24 September to 3 October 2026,": "24 September to 3 October 2026 [under way, ends in 3 days],",
                 "on Saturday, 26 September 2026, at": "26 September 2026 [4 days ago], at",
                 "begin on September 30, with Game 2s": "September 30 [today], with",
                 "scheduled for November 3, 2026, in": "November 3, 2026 [in 34 days], in",
                 "from 28 December to 3 January 2027": "28 December to 3 January 2027 [starts in about 3 months]",
                 "(born June 19, 1983)": "June 19, 1983 [about 43 years ago])",
                 "between 1 and 2 October": "1 and 2 October [starts tomorrow]",
                 "began on 30 September and ends on 1 October": "30 September [today] and ends on 1 October [tomorrow]"}
        for text, marked in cases.items():
            self.assertIn(marked, mark_dates(text, day), text)
        unmarked = "in September 2026, on 31 September, or on 29 February"  # no day of the month, or no such day
        self.assertEqual(mark_dates(unmarked, day), unmarked)
        self.assertIn("4 October [in 4 days]", page_block(1, ROW | {"lead": "The final is on 4 October."}, None))


class ProphetTest(unittest.TestCase):
    def test_both_steps_name_the_day_and_the_evidence_comes_from_the_day_before(self):
        for chat in (question_messages(date(2026, 10, 1), "[1] ..."),
                     prediction_messages(date(2026, 10, 1), "[1] ...", "Who wins the final?")):
            self.assertIn("by the end of Wednesday, 30 September 2026", chat[1]["content"])
            self.assertIn("[today] means Thursday, 1 October 2026", chat[1]["content"])
            self.assertIn("[1] ...", chat[1]["content"])
        self.assertIn("The question Thursday, 1 October 2026 will answer: Who wins the final?",
                      prediction_messages(date(2026, 10, 1), "[1] ...", "Who wins the final?")[1]["content"])

    def test_questions_parse_and_none_means_no_question(self):
        for answer in ("none", "None.", "No.", "Nothing is decided that day.", ""):
            self.assertIsNone(parse_question(answer), answer)
        self.assertEqual(parse_question("Who will win the final between A and B?"), "Who will win the final between A and B?")
        self.assertEqual(parse_question('Question: "Who wins  Game 1?"'), "Who wins Game 1?")
        self.assertEqual(parse_question("Yes.\nWhich party will win the most seats?"), "Which party will win the most seats?")

    def test_predictions_parse_from_json_or_a_bare_sentence(self):
        got = parse_prediction('{"prediction": "Japan will win.", "confidence": "high"}}', "A", "Who wins?")
        self.assertEqual(got, {"text": "I predict that Japan will win.", "question": "Who wins?", "evidence": ["A"],
                               "confidence": "high"})  # the extra brace, as on 2026-09-20, doesn't matter
        bare = parse_prediction("Sure! I predict that Japan will score 3.5 goals. Hope that helps.", "A", "How many?")
        self.assertEqual((bare["text"], bare["confidence"]), ("I predict that Japan will score 3.5 goals.", None))
        self.assertIsNone(parse_prediction('{"prediction": ""}', "A", "Who wins?"))
        self.assertEqual(normalize("Japan will win."), "I predict that Japan will win.")  # proper nouns keep their capital
        self.assertEqual(normalize("The final goes to extra time."), "I predict that the final goes to extra time.")
        self.assertEqual(normalize("i predict that  it rains."), "I predict that it rains.")


class ChecksTest(unittest.TestCase):
    def test_wars_disasters_and_crime_only_in_general_terms(self):
        # The user's examples (2026-10-01): kept when general, dropped when about a specific person or organization.
        predictions = [{"text": "I predict that President Trump will rob the Bank of America."},
                       {"text": "I predict that an important politician will rob a major bank."},
                       {"text": "I predict that Vladimir Putin will be killed by a Ukrainian drone attack."},
                       {"text": "I predict that a major Ukrainian drone attack will take place."},
                       {"text": "I predict that someone will rob the Bank of America."}]
        out = screen(predictions, ["Yes", "No", "Yes", "No", "No"], [["President Trump"], [], ["Vladimir Putin"], [], []],
                     ["Yes"] * 5, [["Bank of America"], [], [], [], ["Bank of America"]], clean(5), ["yes"] * 5, "")
        self.assertEqual([p["kept"] for p in out], [False, True, False, True, False])
        self.assertIn("sensitive topic with a specific person or organization (by the topic question: President Trump; "
                      "president; Bank of America)", out[0]["dropped_because"])  # the title counts too
        self.assertEqual(out[4]["dropped_because"],
                         ["sensitive topic with a specific person or organization (by the topic question: Bank of America)"])
        # Outside those topics an organization is fine: parties win elections and teams win finals.
        party = screen([{"text": "I predict that the PAM will win the most seats."}], ["No"], [[]], ["No"], [["PAM"]],
                       clean(1), ["yes"], "")
        self.assertTrue(party[0]["kept"])

    def test_people_and_quality_drops(self):
        one = [{"text": "I predict that Japan will top the medal table."}]
        self.assertEqual(screen(one, ["Yes"], [[]], ["no"], [[]], ["no"], ["yes"], "")[0]["dropped_because"],
                         ["names or points to a person"])
        self.assertEqual(screen(one, ["no"], [["the president of France"]], ["no"], [[]], ["no"], ["yes"],
                                "")[0]["dropped_because"], ["names or points to a person (the president of France)"])
        self.assertEqual(screen(one, ["no"], [[]], ["no"], [[]], ["Yes."], ["yes"], "")[0]["dropped_because"],
                         ["the evidence already settles it"])
        self.assertEqual(screen(one, ["no"], [[]], ["no"], [[]], ["no"], ["No."], "")[0]["dropped_because"],
                         ["not about its cited evidence"])
        self.assertTrue(screen(one, ["no"], [[]], ["no"], [[]], ["no"], ["Unclear"], "")[0]["kept"])  # only a clear no drops

    def test_copies_of_instruction_examples_and_repeats_are_dropped(self):
        instructions = ('A good one: {"question": "Who wins?", "prediction": "I predict that the Houston Astros will '
                        'beat the Chicago White Sox in Game 1 of the Wild Card Series.", "evidence": [2]}')
        predictions = [{"text": "I predict that the Houston Astros will beat the Chicago White Sox in Game 1."},
                       {"text": "I predict that Egypt will win the 2026 Men's African Nations Volleyball Championship."},
                       {"text": "I predict that Egypt will win the 2026 Men's African Nations Volleyball Championship!"}]
        out = screen(predictions, clean(3), [[]] * 3, clean(3), [[]] * 3, clean(3), ["yes"] * 3, instructions)
        self.assertEqual([p["dropped_because"] for p in out],
                         [["copies an example from the instructions"], [], ["repeats an earlier prediction"]])
        # As on 2026-09-18: the first copy cites the wrong page, so the second one is the one kept.
        twice = screen(predictions[1:], clean(2), [[]] * 2, clean(2), [[]] * 2, clean(2), ["No", "yes"], instructions)
        self.assertEqual([p["dropped_because"] for p in twice], [["not about its cited evidence"], []])

    def test_the_novelty_check_sees_only_the_cited_evidence(self):
        chat = novelty_messages("I predict that it is held in Okazaki.", ["[3] Volleyball ... in Okazaki"])
        self.assertIn("[3] Volleyball ... in Okazaki", chat[0]["content"])
        self.assertIn("(no pages cited)", novelty_messages("I predict that X.", [])[0]["content"])
        self.assertTrue(already_known("Yes"))
        self.assertFalse(already_known("Probably"))  # a quality filter: only a clear yes drops

    def test_a_sensitive_topic_by_word_or_by_question(self):
        self.assertEqual(on_a_sensitive_topic("I predict that the trial will open.", "no"), ["trial"])
        self.assertEqual(on_a_sensitive_topic("I predict that a major bank will be robbed.", "Yes."), ["by the topic question"])
        self.assertEqual(on_a_sensitive_topic("I predict that Brazil will win the shootout.", "No"), [])
        self.assertEqual(on_a_sensitive_topic("I predict that Brazil will win.", "Maybe"), [])  # only a clear yes
        self.assertEqual(sensitive_words("Arrest, then a Trial and another trial"), ["arrest", "trial"])

    def test_who_counts_as_a_specific_person_or_organization(self):
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
        self.assertEqual(confirmed_orgs(["Bank of America", "Hamas", "Ukraine", "Moscow Oblast", "Manchester City", NO_ANSWER],
                                        ["organization", "Organization.", "country", "place", "team", ""]),
                         ["Bank of America", "Hamas", "Manchester City", NO_ANSWER])
        self.assertEqual(confirmed_orgs(["Boeing"], ["company"]), ["Boeing"])  # not classed as anything else: counts
        self.assertEqual(person_roles("The defending champion beats the Coach and the coach"), ["defending champion", "coach"])
        self.assertEqual(person_roles("Malta will host the contest; the Lions win."), [])
        self.assertEqual(person_roles("The incumbent wins; a former minister and an elected mayor lose"), ["incumbent"])
        self.assertEqual(person_roles("Players and drivers will strike, and the defending champions will win"), [])
        # Role words that begin a name, as on 2026-09-27; a possessive role still counts.
        self.assertEqual(person_roles("The United States will win the 2026 Presidents Cup, Queens Park Rangers their "
                                      "match, and Mercedes the Drivers' Championship"), [])
        self.assertEqual(person_roles("The president's party wins the President's Cup"), ["president"])
        dropped = screen([{"text": "I predict that the defending champion will win the darts."}], ["No"], [[]], ["no"],
                         [[]], ["no"], ["yes"], "")
        self.assertEqual(dropped[0]["dropped_because"], ["names or points to a person (defending champion)"])


if __name__ == "__main__":
    unittest.main()
