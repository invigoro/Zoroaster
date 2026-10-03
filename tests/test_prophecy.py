import json
import unittest
from datetime import date

from src.forecast.guardrails import sensitive_words
from src.prophecy.checks import (IDENTIFIABLE, NO_ANSWER, already_known, apply_identifies, confirmed_orgs,
                                 confirmed_people, general_mention, guarded_only, harms, listed_names, merge_rewrites,
                                 names_a_person, novelty_messages, one_persons_contest, person_roles, screen)
from src.prophecy.evidence import (clean_line, dated_lines, eligible, evidence_text, is_biography, mark_dates, page_block,
                                   past_year)
from src.prophecy.prophet import (marked_within, normalize, parse_prediction, parse_question, parse_rewrite,
                                  prediction_messages, question_messages)
from src.prophecy.selection import is_sport_page, select, settles_a_title, topic
from src.prophecy.stories import reports_between, stories, story_block

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
        # A calendar day's page lists anniversaries (run 9: an 1884 shipwreck read as 2026's), but a year's doesn't.
        days = [ROW | {"rank": 1, "page_id": 5, "page_title": "September_23"},
                ROW | {"rank": 2, "page_id": 6, "page_title": "2026"},
                ROW | {"rank": 3, "page_id": 7, "page_title": "September_2026_nor'easter"}]
        self.assertEqual([r["page_id"] for r in eligible(days)], [6, 7])

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

    def test_a_page_about_a_past_year_keeps_its_dates_in_that_year(self):
        # Run 8: "2024 East–West Line disruption" said services "resumed on 1 October", read as five days after
        # 2026-09-26, and the prophet predicted a resumption two years late.
        day = date(2026, 9, 26)
        self.assertEqual(past_year("2024_East–West_Line_disruption", day), 2024)
        self.assertEqual(past_year("Great Fire of New York (1776)", day), 1776)
        self.assertEqual(past_year("Second Battle of Kehl (1796)", day), 1796)
        for title in ("2025–26 Premier League", "2026 AFL Grand Final", "2027 Nigerian general election", "September 23",
                      "Starship flight 14"):
            self.assertIsNone(past_year(title, day), title)
        row = ROW | {"date": day, "page_title": "2024_East–West_Line_disruption",
                     "lead": "Full services resumed on 1 October.", "yesterday_known": False}
        self.assertIn("1 October [about 2 years ago]", page_block(1, row, None))
        self.assertIn("1 October [in 5 days]", page_block(1, row | {"page_title": "East–West Line"}, None))
        self.assertIn("31 December [about 2 years ago]", mark_dates("on 31 December", day, 2024))  # not 2023
        self.assertIn("1 January [about 3 years ago]", mark_dates("on 1 January", day, 2024))  # not 2025

    def test_lines_dated_today_come_from_the_page_as_it_stood(self):
        day = date(2026, 9, 24)
        page = "\n".join(["It began on 20 September.", "{| class=wikitable",
                          "| 24 September || Thursday || 15:00 || Gold medal match", "| 25 September || Friday || Closing",
                          "{{Football box", "|date = {{Start date|2026|9|24}}", "|team1 = {{fb|JPN}}", "|score = v",
                          "|team2 = {{fb|KOR}}", "}}", "On 24 September 2022 the last final was played."])
        self.assertEqual(dated_lines(page, day), ["24 September · Thursday · 15:00 · Gold medal match",
                                                  "date = 24 September 2026 · team1 = JPN · score = v · team2 = KOR"])
        block = page_block(1, ROW | {"date": day, "lead": "No dates.", "page_text": page}, None)
        self.assertIn("Dated today on the page: 24 September [today] · Thursday", block)
        self.assertTrue(marked_within(block, 0))
        self.assertEqual(clean_line("| winner = [[Japan national team|Japan]] {{flagicon|JPN}}<ref>x</ref>"),
                         "winner = Japan JPN")
        self.assertEqual(clean_line("| RD2-team01={{flagIOC|CHN|2026 Asian Games}} | RD2-score01= 3"),
                         "RD2-team01=CHN | RD2-score01= 3")  # brackets name their teams with flags


class ProphetTest(unittest.TestCase):
    def test_both_steps_name_the_day_and_the_evidence_comes_from_the_day_before(self):
        day = date(2026, 10, 1)
        for chat in (question_messages(day, "[1] ..."), prediction_messages(day, "[1] ...", "Who wins?", day)):
            self.assertIn("by the end of Wednesday, 30 September 2026", chat[1]["content"])
            self.assertIn("[today] means Thursday, 1 October 2026", chat[1]["content"])
            self.assertIn("[1] ...", chat[1]["content"])
        self.assertIn("between Thursday, 1 October 2026 and Thursday, 8 October 2026",
                      question_messages(day, "[1] ...")[1]["content"])
        self.assertIn("decided on Thursday, 1 October 2026 itself", question_messages(day, "[1] ...", 0)[1]["content"])
        self.assertIn("The question, settled on Sunday, 4 October 2026: Who wins the final?",
                      prediction_messages(day, "[1] ...", "Who wins the final?", date(2026, 10, 4))[1]["content"])

    def test_pages_are_asked_when_something_is_due_within_the_horizon(self):
        day = date(2026, 9, 24)
        for lead, today, week in (("The final is on 24 September.", True, True),
                                  ("It runs from 20 to 24 September.", True, True),
                                  ("It runs from 24 to 30 September.", True, True),
                                  ("It runs from 20 to 26 September.", False, True),  # ends in 2 days
                                  ("The final is on 25 September.", False, True),
                                  ("The vote is on 1 October.", False, True),  # in 7 days
                                  ("The vote is on 2 October.", False, False),  # in 8 days
                                  ("It ended on 22 September.", False, False), ("No dates here.", False, False)):
            block = page_block(1, ROW | {"date": day, "lead": lead}, None)
            self.assertEqual((marked_within(block, 0), marked_within(block, 7)), (today, week), lead)

    def test_questions_parse_with_their_due_date(self):
        day = date(2026, 9, 24)
        self.assertEqual(parse_question("Event: The final is played.\nDate: 24 September 2026\nQuestion: Who  wins?", day),
                         ("Who wins?", day))
        self.assertEqual(parse_question('**Event:** Game 2.\n**Date:** September 30 [in 6 days]\n**Question:** "Who wins '
                                        'Game 2?"', day), ("Who wins Game 2?", date(2026, 9, 30)))
        self.assertEqual(parse_question("Event: The vote.\nDate: tomorrow\nQuestion: Who wins?", day),
                         ("Who wins?", date(2026, 9, 25)))
        # As in run 8's three-line answers: no "?", a bare mark for a date, the prompt's words repeated.
        self.assertEqual(parse_question("Event: Final\nDate: [in 2 days]\nQuestion: who wins the gold medal.", day),
                         ("Who wins the gold medal?", date(2026, 9, 26)))
        self.assertEqual(parse_question("Event: The vote\nDate: [tomorrow]\nQuestion: The main question that day will "
                                        "settle is the outcome of the FIDE presidential election.", day),
                         ("The outcome of the FIDE presidential election", date(2026, 9, 25)))
        # A due date the evidence doesn't mark moves to the nearest one it does: the tournament "ends in 4 days",
        # and the model wrote the horizon's last day (run 8, 2026-09-18), or the day itself (run 9).
        day18, final = date(2026, 9, 18), "Event: The tournament's final\nDate: {}\nQuestion: Who wins?"
        self.assertEqual(parse_question(final.format("Friday, 25 September 2026"), day18, 7, {4}), ("Who wins?", date(2026, 9, 22)))
        self.assertEqual(parse_question(final.format("Friday, 18 September 2026"), day18, 7, {1, 5}), ("Who wins?", date(2026, 9, 19)))
        self.assertEqual(parse_question(final.format("21 September"), day18, 7, {1, 3, 5}), ("Who wins?", date(2026, 9, 21)))
        self.assertEqual(parse_question(final.format("20 September"), day18, 7, {1, 3}), ("Who wins?", date(2026, 9, 21)))  # a tie
        self.assertEqual(parse_question(final.format("Friday, 25 September 2026"), day18, 7), ("Who wins?", date(2026, 9, 25)))
        for answer in ("Event: nothing\nDate: none\nQuestion: none", "Event: The final.\nDate: today\nQuestion: none",
                       "Event: Release\nDate: 25 September 2026\nQuestion: None, as the release date is already set.",
                       "Event: The final.\nDate: next week\nQuestion: Who wins?",  # no date to read
                       "Event: The final.\nDate: 4 October 2026\nQuestion: Who wins?",  # 10 days away
                       # as for the NRL grand final: a question after saying nothing happens
                       "Event: Nothing significant is expected.\nDate: today\nQuestion: Will the final be close?",
                       "Who wins?", ""):
            self.assertIsNone(parse_question(answer, day), answer)
        self.assertIsNone(parse_question("Event: The vote.\nDate: tomorrow\nQuestion: Who wins?", day, horizon=0))

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


class SelectionTest(unittest.TestCase):
    def prediction(self, page: str, kept: bool = True) -> dict:
        return {"text": f"I predict that {page} …", "question": "?", "evidence": [page], "kept": kept}

    def test_world_events_first_then_only_sports_titles(self):
        # The user (2026-10-02): at most ten a day, world events first, only the most interesting sport.
        world = [self.prediction(f"W{n}") for n in range(12)]
        sport = [self.prediction("final"), self.prediction("pool match"), self.prediction("dropped", kept=False)]
        ranks = {f"W{n}": 20 + n for n in range(12)} | {"final": 1, "pool match": 2, "dropped": 3}
        out = select(world + sport, ranks, ["Politics."] * 12 + ["sport", "Sport", ""], ["no"] * 12 + ["Yes", "no", ""])
        published = [p["evidence"][0] for p in out if p["published"]]
        self.assertEqual(published, [f"W{n}" for n in range(10)])  # the best ranked ten world events, sport or not
        self.assertEqual({p["evidence"][0]: p["unpublished_because"] for p in out if p["kept"] and not p["published"]},
                         {"W10": "over the day's limit: 10, at most 3 of them sport",
                          "W11": "over the day's limit: 10, at most 3 of them sport",
                          "final": "over the day's limit: 10, at most 3 of them sport",
                          "pool match": "a sports prediction that settles no title"})
        self.assertIsNone(out[-1]["topic"])  # dropped by the checks: never classed

    def test_titles_fill_the_rest_of_a_quiet_day_up_to_three(self):
        finals = [self.prediction(f"F{n}") for n in range(5)]
        out = select([self.prediction("vote")] + finals, {"vote": 9} | {f"F{n}": n for n in range(5)},
                     ["politics"] + ["sport"] * 5, ["no"] + ["yes"] * 5)
        self.assertEqual([p["evidence"][0] for p in out if p["published"]], ["vote", "F0", "F1", "F2"])
        self.assertEqual(topic(" Sports."), "sport")
        self.assertEqual(topic("weather"), "other")
        self.assertEqual(topic(""), "other")
        self.assertFalse(settles_a_title("Possibly"))  # only a clear yes lets sport through

    def test_a_sports_page_makes_sport_whatever_the_topic_answer(self):
        # Run 8: the topic question called "Who wins the gold medal?" on a badminton page "other", and
        # sport filled the day. And the title question said no to the AFL Grand Final's "Which team wins the match?".
        badminton, grand_final, vote = (self.prediction("Badminton – Women's team"), self.prediction("2026 AFL Grand Final"),
                                        self.prediction("2026 Berlin state election"))
        out = select([badminton, grand_final, vote], {"Badminton – Women's team": 1, "2026 AFL Grand Final": 2,
                                                       "2026 Berlin state election": 3},
                     ["other", "other", "politics"], ["no", "no", "no"], {"Badminton – Women's team", "2026 AFL Grand Final"})
        self.assertEqual([(p["topic"], p["published"]) for p in out], [("sport", False), ("sport", True), ("politics", True)])
        self.assertEqual(out[0]["unpublished_because"], "a sports prediction that settles no title")
        self.assertFalse(select([self.prediction("2026 Men's semi-finals")], {}, ["sport"], ["no"])[0]["settles_a_title"])

    def test_sports_pages_from_their_infobox_templates_and_categories(self):
        sport = ["{{Infobox sports competition event\n| event = Women's team}}",
                 "{{Infobox country at games\n| NOC = PHI}}",
                 "{{Infobox racehorse\n| horsename = Hurricane Fly}}",
                 "{{Short description|Football tournament qualification stage}}\n{{#invoke:Sports table|main}}",
                 "{{Tennis events|2026|Chengdu Open}}\n[[Category:2026 ATP Tour]]",  # no infobox, no sport category
                 "{{Short description|Video game tournament series}}\n{{Infobox recurring event}}",
                 "Text.\n[[Category:Current sports events]]"]
        world = ["{{Infobox election\n| election_name = 2026 Berlin state election}}\n[[Category:2026 elections in Germany]]",
                 "{{Short description|2026 gubernatorial race in Georgia}}",
                 "{{Infobox civilian attack}}\n[[Category:Mass shootings in the United States]]",
                 "{{Infobox film\n| name = Heart of the Beast}}\n[[Category:Sports films]]",
                 "{{Infobox automobile}}\n[[Category:Compact sport utility vehicles]]",
                 "{{Infobox video game}}\n[[Category:PlayStation 5 games]]\n[[Category:Racing video games]]",
                 "{{Infobox summit meeting}}\n[[Category:General debates of the United Nations General Assembly]]"]
        for text in sport:
            self.assertTrue(is_sport_page(text), text)
        for text in world:
            self.assertFalse(is_sport_page(text), text)


class ChecksTest(unittest.TestCase):
    def test_wars_disasters_and_crime_only_in_general_terms(self):
        # The user's examples (2026-10-01): kept when general, dropped when about a specific person or organization.
        predictions = [{"text": "I predict that President Trump will rob the Bank of America."},
                       {"text": "I predict that an important politician will rob a major bank."},
                       {"text": "I predict that Vladimir Putin will be killed by a Ukrainian drone attack."},
                       {"text": "I predict that a major Ukrainian drone attack will take place."},
                       {"text": "I predict that someone will rob the Bank of America."}]
        out = screen(predictions, ["Yes", "No", "Yes", "No", "No"], [["President Trump"], [], ["Vladimir Putin"], [], []],
                     [["Bank of America"], [], [], [], ["Bank of America"]], [[["Yes", "No", "company"]], [], [], [], [["No", "yes.", "company"]]],
                     clean(5), "")
        self.assertEqual([p["kept"] for p in out], [False, True, False, True, False])
        self.assertEqual(out[0]["dropped_because"], ["names or points to a person (President Trump; president)",
                                                     "harm to or by a specific organization (Bank of America)"])
        self.assertEqual(out[4]["dropped_because"], ["harm to or by a specific organization (Bank of America)"])
        # An organization is fine when nothing harmful happens to or by it: parties win elections, teams win finals.
        party = [{"text": "I predict that the PAM will win the most seats."}]
        self.assertTrue(screen(party, ["No"], [[]], [["PAM"]], [[["No.", "no", "party"]]], clean(1), "")[0]["kept"])
        self.assertFalse(screen(party, ["No"], [[]], [["PAM"]], [[["No", "Possibly", "party"]]], clean(1), "")[0]["kept"])

    def test_people_and_quality_drops(self):
        one = [{"text": "I predict that Japan will top the medal table."}]
        self.assertEqual(screen(one, ["Yes"], [[]], [[]], [[]], ["no"], "")[0]["dropped_because"],
                         ["names or points to a person"])
        self.assertEqual(screen(one, ["no"], [["the president of France"]], [[]], [[]], ["no"], "")[0]["dropped_because"],
                         ["names or points to a person (the president of France)"])
        self.assertEqual(screen(one, ["no"], [[]], [[]], [[]], ["Yes."], "")[0]["dropped_because"],
                         ["the evidence already settles it"])
        self.assertTrue(screen(one, ["no"], [[]], [[]], [[]], ["Probably"], "")[0]["kept"])  # only a clear yes drops

    def test_copies_of_instruction_examples_and_repeats_are_dropped(self):
        instructions = ('A good one: {"question": "Who wins?", "prediction": "I predict that the Houston Astros will '
                        'beat the Chicago White Sox in Game 1 of the Wild Card Series.", "evidence": [2]}')
        predictions = [{"text": "I predict that the Houston Astros will beat the Chicago White Sox in Game 1."},
                       {"text": "I predict that Egypt will win the 2026 Men's African Nations Volleyball Championship."},
                       {"text": "I predict that Egypt will win the 2026 Men's African Nations Volleyball Championship!"}]
        out = screen(predictions, clean(3), [[]] * 3, [[]] * 3, [[]] * 3, clean(3), instructions)
        self.assertEqual([p["dropped_because"] for p in out],
                         [["copies an example from the instructions"], [], ["repeats an earlier prediction"]])
        # As on 2026-09-18, when the first copy cited the wrong page: when it's dropped, the second is the one kept.
        twice = screen(predictions[1:], clean(2), [[]] * 2, [[]] * 2, [[]] * 2, ["Yes.", "no"], instructions)
        self.assertEqual([p["dropped_because"] for p in twice], [["the evidence already settles it"], []])

    def test_the_novelty_check_sees_only_the_cited_evidence(self):
        chat = novelty_messages("I predict that it is held in Okazaki.", ["[3] Volleyball ... in Okazaki"])
        self.assertIn("[3] Volleyball ... in Okazaki", chat[0]["content"])
        self.assertIn("(no pages cited)", novelty_messages("I predict that X.", [])[0]["content"])
        self.assertTrue(already_known("Yes"))
        self.assertFalse(already_known("Probably"))  # a quality filter: only a clear yes drops

    def test_harm_and_one_persons_contests(self):
        self.assertFalse(harms([" No.", "no", "Company."]))  # harm done, harm suffered, the kind of organization
        for answer in ("Yes", "Possibly", ""):  # a guardrail: anything but a clear no to both counts
            self.assertTrue(harms(["no", answer, "company"]), answer)
            self.assertTrue(harms([answer, "no", "company"]), answer)
        self.assertTrue(harms(["No", "No", "Military"]))  # as for "the Israel Defense Forces will strike a hospital"
        # By title or wording, whatever the model says: it called a Grand Prix a contest between teams.
        self.assertTrue(one_persons_contest("2026 Azerbaijan Grand Prix", "Who will win the race?", "teams"))
        self.assertTrue(one_persons_contest("Badminton at the 2026 Asian Games – Men's singles", "Which country wins gold?",
                                            "teams"))
        self.assertTrue(one_persons_contest("2026 Presidents Cup", "Who wins?", "individuals"))
        self.assertTrue(one_persons_contest("2026 Presidents Cup", "Who wins?", "unsure"))  # not clearly teams
        for answer in ("teams", "Parties.", "neither"):
            self.assertFalse(one_persons_contest("2026 Presidents Cup", "Who wins the 2026 Presidents Cup?", answer))
        self.assertFalse(one_persons_contest("2026 São Toméan parliamentary election", "Which party wins the most seats?",
                                             "parties"))
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
                         ["Bank of America", "Hamas", "Manchester City"])
        self.assertEqual(confirmed_orgs(["Boeing"], ["company"]), ["Boeing"])  # not classed as anything else: counts
        self.assertEqual(person_roles("The defending champion beats the Coach and the coach"), ["defending champion", "coach"])
        self.assertEqual(person_roles("Malta will host the contest; the Lions win."), [])
        self.assertEqual(person_roles("The incumbent wins; a former minister and an elected mayor lose"), ["incumbent"])
        self.assertEqual(person_roles("Players and drivers will strike, and the defending champions will win"), [])
        # Role words that begin a name, as on 2026-09-27; a possessive role still counts.
        self.assertEqual(person_roles("The United States will win the 2026 Presidents Cup, Queens Park Rangers their "
                                      "match, and Mercedes the Drivers' Championship"), [])
        self.assertEqual(person_roles("The president's party wins the President's Cup"), ["president"])
        dropped = screen([{"text": "I predict that the defending champion will win the darts."}], ["No"], [[]], [[]],
                         [[]], ["no"], "")
        self.assertEqual(dropped[0]["dropped_because"], ["names or points to a person (defending champion)"])


class StoriesTest(unittest.TestCase):
    """Portal:Current events as the prophet's evidence (PLAN.md §2, decided 2026-10-02)."""

    DAY = date(2026, 9, 25)

    def known(self, pages: dict[str, list[str]]) -> dict:
        return {"date": self.DAY.isoformat(), "known_at": "2026-09-24T23:59:59+00:00",
                "days": [{"date": d, "items": items} for d, items in pages.items()]}

    def test_the_prophet_reads_what_was_known_by_the_end_of_the_day_before(self):
        from datetime import datetime, timezone

        from src.prophecy.current_events import known_at
        self.assertEqual(known_at(self.DAY), datetime(2026, 9, 24, 23, 59, 59, tzinfo=timezone.utc))
        # Point in time: a page for the day foretold, or after, is never read, whatever a record holds.
        record = self.known({"2026-09-24": ["Politics and elections › 2026 Moroccan general election › Polls open."],
                             "2026-09-25": ["Politics and elections › 2026 Moroccan general election › The PAM wins."],
                             "2026-09-26": ["Armed conflicts and attacks › Gaza war › A ceasefire is agreed."]})
        found = stories(record, self.DAY)
        self.assertEqual([s["story"] for s in found], ["2026 Moroccan general election"])
        self.assertNotIn("PAM wins", story_block(1, found[0], self.DAY))

    def test_stories_group_reports_by_topic_and_leave_out_sport(self):
        record = self.known({
            "2026-09-18": ["Armed conflicts and attacks › 2026 Iran war › 2026 Strait of Hormuz crisis › Shipping rises.",
                           "Sports › 2026 Asian Games › Japan wins gold.",  # sport comes from the pages
                           "Politics and elections › The United Kingdom sets up a disinformation centre."],  # no topic
            "2026-09-23": ["Armed conflicts and attacks › Middle Eastern crisis › 2026 Iran war › Talks resume on "
                           "October 7 in Doha, years after the October 7 attacks.",  # an umbrella topic
                           "Armed conflicts and attacks › Middle Eastern crisis › Yemeni civil war › A drone strikes."],
            "2026-09-24": ["Disasters and accidents › 2026 Atlantic hurricane season › Tropical Storm Fay forms."]})
        found = stories(record, self.DAY)
        self.assertEqual([(s["story"], s["recent"], s["week"]) for s in found],  # recent reports first, then the week's
                         [("2026 Iran war", 1, 2), ("2026 Atlantic hurricane season", 1, 1),
                          ("Middle Eastern crisis", 1, 1)])  # the Iran war is its own story, listed alone on the 18th
        self.assertEqual([s["topic"] for s in found], ["conflict", "disaster", "conflict"])
        block = story_block(1, found[0], self.DAY)
        self.assertIn("[S1] 2026 Iran war (conflict): 2 reports this week, the latest 2 days ago", block)
        self.assertIn("[7 days ago] 2026 Strait of Hormuz crisis › Shipping rises.", block)
        self.assertIn("Talks resume on October 7 [in 12 days] in Doha, years after the October 7 attacks.", block)
        self.assertNotIn("Japan", "\n".join(story_block(n, s, self.DAY) for n, s in enumerate(found, start=1)))
        # Only stories with a report on either of the two days before are read.
        self.assertEqual(stories(self.known({"2026-09-18": record["days"][0]["items"]}), self.DAY), [])

    def test_a_story_is_graded_on_its_reports_up_to_the_due_day(self):
        records = {"2026-09-25": ["Politics and elections › 2026 Moroccan general election › The PAM wins.",
                                  "Armed conflicts and attacks › Gaza war › Airstrikes continue."],
                   "2026-09-27": ["Politics and elections › 2026 Moroccan general election › A coalition forms."],
                   "2026-10-02": ["Politics and elections › 2026 Moroccan general election › Too late."]}
        self.assertEqual(reports_between(records, "2026 Moroccan general election", self.DAY, date(2026, 10, 1)),
                         ["2026-09-25: The PAM wins.", "2026-09-27: A coalition forms."])
        # A later page may file the story under an umbrella topic (run 10's Houthi–Saudi pack missed two reports).
        nested = {"2026-09-27": ["Armed conflicts and attacks › Middle Eastern crisis › Yemeni civil war › "
                                 "Houthi–Saudi Arabian conflict › A Saudi airstrike hits a market in Taiz."]}
        self.assertEqual(reports_between(nested, "Houthi–Saudi Arabian conflict", self.DAY, date(2026, 10, 1)),
                         ["2026-09-27: A Saudi airstrike hits a market in Taiz."])
        self.assertEqual(reports_between(nested, "Yemeni civil war", self.DAY, date(2026, 10, 1)),
                         ["2026-09-27: Houthi–Saudi Arabian conflict › A Saudi airstrike hits a market in Taiz."])

    def test_a_story_s_question_without_a_date_is_due_at_the_week_s_end(self):
        week_end = date(2026, 10, 2)
        answer = "Event: Ceasefire talks\nDate: {}\nQuestion: Whether a ceasefire is agreed"
        self.assertEqual(parse_question(answer.format("this week"), self.DAY, 7, set(), week_end)[1], week_end)
        self.assertEqual(parse_question(answer.format("28 September 2026"), self.DAY, 7, set(), week_end)[1],
                         week_end)  # no report gives a date: the model's guess
        self.assertEqual(parse_question(answer.format("28 September 2026"), self.DAY, 7, {2}, week_end)[1],
                         date(2026, 9, 27))  # a report marks one: the nearest
        self.assertEqual(parse_question(answer.format("27 September 2026"), self.DAY, 7, {2}, week_end)[1],
                         date(2026, 9, 27))
        self.assertIsNone(parse_question(answer.format("this week"), self.DAY, 7, set()))  # pages: no default
        self.assertIsNone(parse_question(answer.format("15 October 2026"), self.DAY, 7, set(), week_end))


class DetailsTest(unittest.TestCase):
    """Details a story's prediction adds that its reports don't give (the user asked to stop them, 2026-10-02)."""

    def test_numbers_and_names_must_come_from_the_evidence(self):
        from src.prophecy.details import unsupported_details

        reports = "[2 days ago] Clashes in Abyei kill 26 people and injure 82. Tropical Storm Fay forms near the Azores."
        self.assertEqual(unsupported_details("I predict that clashes in Abyei will kill at least 30 people by the end of "
                                             "Monday, 28 September 2026.", reports), ["30"])  # the due date is exempt
        self.assertEqual(unsupported_details("I predict that Tropical Storm Fay will make landfall in the Canary Islands.",
                                             reports), ["Canary", "Islands"])
        self.assertEqual(unsupported_details("I predict that the clashes in Abyei will leave 82 more injured by 2027.",
                                             reports), [])  # numbers in the evidence, and years, are fine
        # Numbers in words, from three up; the trial's revisions wrote "ten air strikes" for a made-up "15".
        self.assertEqual(unsupported_details("I predict that ten air strikes hit Abyei.", reports), ["ten"])
        self.assertEqual(unsupported_details("I predict that the two sides clash again, with at least one more strike, "
                                             "on the 29th of September.", reports), [])
        # Nationalities, abbreviations, possessives and other scripts, as the evidence may write them otherwise.
        evidence = ("Ukraine launches drones at Moscow. The United States and the European Union meet. El Niño grows. "
                    "Iran's economy shrinks 10.1% and twenty ships wait.")
        for allowed in ("I predict that Ukrainian drones will strike Moscow again.",
                        "I predict that the U.S. will meet the EU's envoys.",
                        "I predict that El Niño will strengthen.",
                        "I predict that Iran's GDP shrinks again, with 20 ships still waiting."):
            self.assertEqual(unsupported_details(allowed, evidence), [], allowed)

    def test_only_kept_story_predictions_are_flagged(self):
        from src.prophecy.details import flagged

        by_title = {"Current events: Fay": "Tropical Storm Fay forms near the Azores.",
                    "Basketball at the 2026 Asian Games": "teamA=CHN · teamB=INA"}
        day = [{"text": "I predict that Fay reaches the Canary Islands.", "evidence": ["Current events: Fay"], "kept": True},
               {"text": "I predict that Fay reaches the Canary Islands.", "evidence": ["Current events: Fay"], "kept": False},
               # Grounded, by the codes, but a word check can't tell: page predictions are left alone.
               {"text": "I predict that China beats Indonesia 80-60.", "evidence": ["Basketball at the 2026 Asian Games"],
                "kept": True}]
        self.assertEqual(flagged(day, by_title), {0: ["Canary", "Islands"]})

    def test_the_prophet_revises_in_its_own_conversation(self):
        from src.prophecy.details import revise_messages
        from src.prophecy.prophet import story_prediction_messages

        prediction = {"text": "I predict that Fay reaches the Canary Islands.", "question": "Where does Fay go?",
                      "due": "2026-09-28", "confidence": "low"}
        chat = revise_messages(date(2026, 9, 21), "Tropical Storm Fay forms near the Azores.", prediction,
                               ["Canary", "Islands"])
        self.assertEqual(chat[:2], story_prediction_messages(date(2026, 9, 21), "Tropical Storm Fay forms near the Azores.",
                                                             "Where does Fay go?", date(2026, 9, 28)))
        self.assertEqual(json.loads(chat[2]["content"]), {"prediction": prediction["text"], "confidence": "low"})
        self.assertEqual(chat[2]["role"], "assistant")
        self.assertIn('none of the reports give: "Canary", "Islands"', chat[3]["content"])

    def test_a_revision_replaces_the_prediction_only_if_it_passes(self):
        from src.prophecy.details import merge_revisions

        evidence = {0: "Clashes in Abyei kill 26 people.", 1: "Fay forms near the Azores.", 3: "Strikes in Nyala."}
        day = [{"text": "I predict that clashes in Abyei kill at least 30 people.", "evidence": ["A"], "kept": True,
                "dropped_because": [], "rewritten_from": "I predict that a named militia kills 30 people in Abyei."},
               {"text": "I predict that Fay reaches the Canary Islands.", "evidence": ["B"], "kept": True, "dropped_because": []},
               {"text": "I predict that talks resume in Doha.", "evidence": ["C"], "kept": True, "dropped_because": []},
               {"text": "I predict that strikes hit Nyala and Kassala.", "evidence": ["D"], "kept": True, "dropped_because": []}]
        found = {0: ["30"], 1: ["Canary", "Islands"], 3: ["Kassala"]}
        revisions = {0: {"text": "I predict that clashes in Abyei kill at least 26 more people.", "evidence": ["A"],
                         "kept": True, "dropped_because": []},
                     1: {"text": "I predict that Fay reaches the Canary Islands soon.", "evidence": ["B"], "kept": True,
                         "dropped_because": []},  # the revision kept the made-up place
                     3: {"text": "I predict that strikes hit Nyala.", "evidence": ["D"], "kept": False,
                         "dropped_because": ["the evidence already settles it"]}}
        out = merge_revisions(day, revisions, found, evidence)
        self.assertEqual([p["kept"] for p in out], [True, False, True, False])
        self.assertEqual((out[0]["text"], out[0]["revised_from"], out[0]["unsupported"], out[0]["rewritten_from"]),
                         ("I predict that clashes in Abyei kill at least 26 more people.",
                          "I predict that clashes in Abyei kill at least 30 people.", ["30"],
                          "I predict that a named militia kills 30 people in Abyei."))
        self.assertEqual(out[1]["dropped_because"], ["adds details its evidence doesn't give (Canary; Islands)"])
        self.assertEqual(out[1]["revision_dropped_because"], ["adds details its evidence doesn't give (Canary; Islands)"])
        self.assertEqual(out[3]["revision_dropped_because"], ["the evidence already settles it"])
        from src.prophecy.judge import gradable
        self.assertTrue(gradable(out[1]))  # a quality drop: still graded, to test the check


class RewriteTest(unittest.TestCase):
    """A prediction the guardrails drop, rewritten in general terms (PLAN.md §2, decided 2026-10-02)."""

    def screened(self, text: str, reasons: list[str]) -> dict:
        return {"text": text, "evidence": ["A"], "kept": not reasons, "dropped_because": reasons}

    def test_only_guardrail_drops_are_rewritten(self):
        self.assertTrue(guarded_only(self.screened("x", ["names or points to a person (Kevin Bacon)"])))
        self.assertTrue(guarded_only(self.screened("x", ["names or points to a person", "harm to or by a specific "
                                                                                        "organization (Hamas)"])))
        self.assertFalse(guarded_only(self.screened("x", ["names or points to a person", "the evidence already "
                                                                                         "settles it"])))
        self.assertFalse(guarded_only(self.screened("x", ["repeats an earlier prediction"])))
        self.assertFalse(guarded_only(self.screened("x", [])))

    def test_a_rewrite_replaces_the_original_only_if_every_check_passes(self):
        day = [self.screened("I predict that Kevin Bacon will die tomorrow.", ["names or points to a person (Kevin Bacon)"]),
               self.screened("I predict that a major Ukrainian drone attack will take place.", []),
               self.screened("I predict that Hamas will fire rockets at Tel Aviv.", ["harm to or by a specific organization"]),
               self.screened("I predict that the Pope will visit Lebanon.", ["names or points to a person (the Pope)"])]
        rewrites = {0: self.screened("I predict that a prominent actor will die tomorrow.", []),
                    2: self.screened("I predict that a major Ukrainian drone attack will take place.", []),  # a repeat
                    3: self.screened("I predict that the head of the Catholic Church will visit Lebanon.",
                                     ["names or points to a person"])}
        out = merge_rewrites(day, rewrites)
        self.assertEqual([p["kept"] for p in out], [True, True, False, False])
        self.assertEqual((out[0]["text"], out[0]["rewritten_from"]),
                         ("I predict that a prominent actor will die tomorrow.", "I predict that Kevin Bacon will die tomorrow."))
        self.assertEqual(out[2]["rewrite_dropped_because"], ["repeats an earlier prediction"])
        self.assertEqual(out[3]["text"], "I predict that the Pope will visit Lebanon.")  # still dropped, as it was
        self.assertEqual(out[3]["rewrite_dropped_because"], ["names or points to a person"])
        self.assertEqual(parse_rewrite('Rewritten: "I predict that a prominent actor will die tomorrow."\nNote: …'),
                         "I predict that a prominent actor will die tomorrow.")
        self.assertEqual(parse_rewrite("A prominent actor will die tomorrow."),
                         "I predict that a prominent actor will die tomorrow.")

    def test_a_rewrite_must_not_let_a_reader_tell_whom_it_means(self):
        rewrites = [self.screened("I predict that a prominent political leader in the Philippines will be removed.", []),
                    self.screened("I predict that a prominent actor will die tomorrow.", []),
                    self.screened("I predict that an armed group will attack.", ["harm to or by a specific organization"]),
                    # From the rewrite trial: the named institution still points to the one meant.
                    self.screened("I predict that an important political leader will be removed from office by the "
                                  "Philippine Senate.", []) | {"orgs_named": ["Philippine Senate"]}]
        out = apply_identifies(rewrites, ["Yes.", "No", "", "No"])
        self.assertEqual([(p["kept"], p["dropped_because"]) for p in out],
                         [(False, [IDENTIFIABLE]), (True, []), (False, ["harm to or by a specific organization"]),
                          (False, ["the rewrite still names a specific organization (Philippine Senate)"])])
        self.assertTrue(general_mention("armed group", "I predict that an armed group will agree to negotiate."))
        self.assertFalse(general_mention("armed group", "I predict that the armed group will agree, as an armed group."))
        self.assertFalse(general_mention("Hamas", "I predict that Hamas will fire rockets."))
        self.assertEqual(normalize("Yemen will beat Bahrain in their match on 27 September [in 3 days]."),
                         "I predict that Yemen will beat Bahrain in their match on 27 September.")


if __name__ == "__main__":
    unittest.main()
