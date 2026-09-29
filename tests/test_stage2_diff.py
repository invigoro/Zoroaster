import unittest

from src.stage2.diff import SPAN_SEPARATOR, tokenize, word_diff

PARAGRAPH = "Keir Starmer is a British politician who has been the leader of the Labour Party since 2020."


class TokenizeTest(unittest.TestCase):
    def test_tokens_join_back_exactly(self):
        text = "== Early life ==\nHe was born in [[London]], 1962.\n\n{{cite web|url=x}}"
        self.assertEqual("".join(tokenize(text)), text)

    def test_urls_and_wikilinks_stay_whole(self):
        self.assertIn("|url=https://a.org/x?id=42", tokenize("<ref>{{cite web |url=https://a.org/x?id=42 }}"))
        self.assertIn("[[hydrolysis]]", tokenize("the [[hydrolysis]] of"))


class WordDiffTest(unittest.TestCase):
    def test_extending_a_sentence(self):
        new = PARAGRAPH.replace("2020.", "2020, and Prime Minister since 2024.")
        d = word_diff(PARAGRAPH + "\nNext line.\n", new + "\nNext line.\n")
        self.assertEqual((d.inserted, d.removed), (["2020, and Prime Minister since 2024."], ["2020."]))
        self.assertEqual(d.first_offset, PARAGRAPH.index("2020."))
        self.assertEqual((d.old_blocks, d.new_blocks), ([PARAGRAPH + "\n"], [new + "\n"]))

    def test_nearby_changes_merge_into_one_span(self):
        d = word_diff("The population was 1,000 in 2010.\n", "The population was 1,250 in 2020.\n")
        self.assertEqual((d.inserted, d.removed), (["1,250 in 2020."], ["1,000 in 2010."]))

    def test_a_changed_citation_is_one_span_not_fragments(self):
        old = "Census data (2015)<ref>{{cite web |url=https://a.org/x?id=42 |date=2015}}</ref> shows growth.\n"
        new = "Census data (2025)<ref>{{cite web |url=https://a.org/y?id=86 |date=2025}}</ref> shows growth.\n"
        d = word_diff(old, new)
        self.assertEqual(d.inserted, ["(2025)<ref>{{cite web |url=https://a.org/y?id=86 |date=2025}}</ref>"])

    def test_distant_changes_are_separate_spans(self):
        old = "Alpha beta gamma delta epsilon zeta eta theta.\n"
        new = "Alpha BETA gamma delta epsilon zeta eta THETA.\n"
        d = word_diff(old, new)
        self.assertEqual(d.inserted, ["BETA", "THETA."])
        self.assertEqual(d.added_text, "BETA" + SPAN_SEPARATOR + "THETA.")

    def test_added_line_and_page_creation(self):
        d = word_diff("Intro.\n", "Intro.\n== Death ==\nHe died on 5 July 2024.\n")
        self.assertEqual(d.added_text, "== Death ==\nHe died on 5 July 2024.")
        self.assertEqual(d.first_offset, len("Intro.\n"))
        created = word_diff(None, "New article text.")
        self.assertEqual((created.added_text, created.first_offset), ("New article text.", 0))

    def test_no_change_and_whitespace_only(self):
        self.assertEqual(word_diff("Same.\n", "Same.\n").inserted, [])
        whitespace = word_diff("A  b.\n", "A b.\n")
        self.assertEqual((whitespace.inserted, whitespace.old_blocks), ([], []))


if __name__ == "__main__":
    unittest.main()
