import unittest
from datetime import date

from src.prophecy.current_events import items, page_title

PAGE = """{{Current events|year=2026|month=09|day=20|top=yes}}
<!-- All news items below this line -->
'''Armed conflicts and attacks'''
*[[Russo-Ukrainian war (2022–present)|Russo-Ukrainian war]]
**[[Attacks in Russia during the Russo-Ukrainian war (2022–present)|Attacks in Russia]]
***[[Armed Forces of Ukraine|Ukrainian forces]] launch a drone attack on [[Moscow Oblast]]. [https://example.org/a (''Al Jazeera'')]
***A second item under the same topic. [https://example.org/b (AP)]
*A top-level item with no topic. [https://example.org/c (Reuters)]

'''Sports'''
*[[2026 Asian Games]]
**[[Japan]] wins the '''men's''' 3x3 basketball gold. [https://example.org/d (Kyodo)]
"""


class CurrentEventsTest(unittest.TestCase):
    def test_title(self):
        self.assertEqual(page_title(date(2026, 9, 5)), "Portal:Current events/2026 September 5")

    def test_items_carry_their_category_and_topics(self):
        self.assertEqual(items(PAGE), [
            "Armed conflicts and attacks › Russo-Ukrainian war › Attacks in Russia › Ukrainian forces launch a drone "
            "attack on Moscow Oblast.",
            "Armed conflicts and attacks › Russo-Ukrainian war › Attacks in Russia › A second item under the same topic.",
            "Armed conflicts and attacks › A top-level item with no topic.",
            "Sports › 2026 Asian Games › Japan wins the men's 3x3 basketball gold.",
        ])


if __name__ == "__main__":
    unittest.main()
