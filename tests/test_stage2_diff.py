import unittest

from src.stage2.diff import tokenize, word_diff

PARAGRAPH = "Keir Starmer is a British politician who has been the leader of the Labour Party since 2020."


class TokenizeTest(unittest.TestCase):
    def test_tokens_join_back_exactly(self):
        text = "== Early life ==\nHe was born in [[London]], 1962.\n\n{{cite web|url=x}}"
        self.assertEqual("".join(tokenize(text)), text)


class WordDiffTest(unittest.TestCase):
    def test_one_word_fix_inside_a_paragraph(self):
        d = word_diff(PARAGRAPH + "\nNext line.\n", PARAGRAPH.replace("2020", "2020, and Prime Minister since 2024") + "\nNext line.\n")
        self.assertEqual(d.inserted, [", and Prime Minister since 2024"])
        self.assertEqual(d.removed, [])
        self.assertEqual(d.first_offset, PARAGRAPH.index("."))  # the change starts where the old "." was

    def test_replacement(self):
        d = word_diff("The population was 1,000 in 2010.\n", "The population was 1,250 in 2020.\n")
        self.assertEqual((d.inserted, d.removed), (["250", "2020"], ["000", "2010"]))

    def test_added_line_and_page_creation(self):
        d = word_diff("Intro.\n", "Intro.\n== Death ==\nHe died on 5 July 2024.\n")
        self.assertEqual(d.added_text, "== Death ==\nHe died on 5 July 2024.")
        self.assertEqual(d.first_offset, len("Intro.\n"))
        created = word_diff(None, "New article text.")
        self.assertEqual((created.added_text, created.first_offset), ("New article text.", 0))

    def test_no_change_and_whitespace_only(self):
        self.assertEqual(word_diff("Same.\n", "Same.\n").inserted, [])
        self.assertEqual(word_diff("A  b.\n", "A b.\n").inserted, [])


if __name__ == "__main__":
    unittest.main()
