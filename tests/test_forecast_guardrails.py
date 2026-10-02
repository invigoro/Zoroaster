import unittest

from src.forecast.guardrails import is_sensitive


class GuardrailsTest(unittest.TestCase):
    def test_sensitive_section_names(self):
        for name in ("Personal life", "Death and legacy", "Arrest and trial", "Health", "2026: Dementia diagnosis",
                     "Legal issues", "Controversies", "Early life and family", "Illness and death"):
            self.assertTrue(is_sensitive(name), name)

    def test_ordinary_section_names(self):
        for name in ("Career", "(lead)", "Results", "Environmental impact", "Discography", "Group A",
                     "2026 season", "Squad", "Industrial history"):
            self.assertFalse(is_sensitive(name), name)  # "mental" and "trial" only count at a word's start


if __name__ == "__main__":
    unittest.main()
