import unittest

from src.stage2 import relevance
from src.stage2.relevance import BUDGET_CHARS, MAX_SENTENCE_CHARS, Idf, names_page, rank, select, sentences

FILLER = [f"Unrelated sentence number {i} about gardening and weather." for i in range(200)]


class SentencesTest(unittest.TestCase):
    def test_prose_sentences_only(self):
        spans = ["The rally was halted when shots were fired.<ref>{{cite news|title=X}}</ref> Crowds fled. "
                 "Officials later confirmed one death at the scene.", "{{Infobox event|name=Rally}}"]
        self.assertEqual(sentences(spans), ["The rally was halted when shots were fired.",
                                            "Officials later confirmed one death at the scene."])  # "Crowds fled." is too short


class NamesPageTest(unittest.TestCase):
    def test_full_and_short_titles(self):
        self.assertTrue(names_page("She met Rachel Reeves in Leeds.", "Rachel_Reeves"))
        self.assertTrue(names_page("He was appointed by Mayor Wilson Fisk.", "Wilson_Fisk_(Marvel_Cinematic_Universe)"))
        self.assertFalse(names_page("Reevesville is a town.", "Reeves"))  # whole words only
        self.assertFalse(names_page("The UK voted.", "UK"))  # titles under 4 characters are too ambiguous


class RankTest(unittest.TestCase):
    def setUp(self):
        self.idf = Idf(FILLER + ["The chancellor spoke.", "The election result was declared."])

    def test_naming_first_then_shared_rare_words(self):
        candidates = {
            "Election": ["The election result was declared for the chancellor seat in Leeds.",
                         "Turnout was the highest since records began in the county."],
            "Cabinet": ["Rachel Reeves was named in the new cabinet list today."],
        }
        ranked = rank("Rachel_Reeves", "Career", "She became chancellor after the election in Leeds.", candidates, self.idf)
        self.assertEqual(ranked, [("Cabinet", "Rachel Reeves was named in the new cabinet list today."),
                                  ("Election", "The election result was declared for the chancellor seat in Leeds.")])
        # the turnout sentence shares no content word with the page, so it doesn't qualify

    def test_common_words_do_not_count(self):
        idf = Idf(FILLER)  # "about" and "weather" are in every sentence
        candidates = {"N": ["A long talk about the weather followed the match."]}
        self.assertEqual(rank("Page", "", "Nothing about the weather here.", candidates, idf), [])
        self.assertTrue(idf.is_content("chancellor") and not idf.is_content("weather"))

    def test_one_shared_word_is_not_enough(self):
        candidates = {"N": ["The election was held on a Tuesday afternoon."]}
        self.assertEqual(rank("Page", "", "An election story.", candidates, self.idf), [])
        self.assertEqual(relevance.MIN_SHARED, 2)


class SelectTest(unittest.TestCase):
    def test_budget_cut_and_grouping(self):
        long = ("word " * 80).strip()  # cut to 201 characters
        ranked = [("A", "First sentence of A."), ("B", long), ("A", "Second sentence of A."),
                  ("C", "c" * 190), ("D", "d" * 100), ("E", "e" * 15)]
        chosen = select(ranked)
        self.assertEqual([n for n, _ in chosen], ["A", "B", "C", "E"])  # D doesn't fit, the shorter E still does
        self.assertEqual(chosen[0][1], "First sentence of A. Second sentence of A.")
        self.assertTrue(chosen[1][1].endswith(" …") and len(chosen[1][1]) <= MAX_SENTENCE_CHARS + 2)
        self.assertLessEqual(sum(len(t) for _, t in chosen), BUDGET_CHARS)
        self.assertEqual(select([]), [])


if __name__ == "__main__":
    unittest.main()
