import unittest
from datetime import date, datetime, timezone
from unittest import mock

from scripts.fetch_current_events import fetch
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

    def test_a_revision_with_its_text_hidden_is_passed_over(self):
        # On 2026-10-06 the 10-05 page's last revisions by the cutoff were revision-deleted, and the prophecy crashed.
        def responses(*batches):  # the API's pages of revisions, newest first, each continuing to the next
            return [{"query": {"pages": [{"title": page_title(date(2026, 10, 5)), "revisions": batch}]},
                     **({"continue": {"rvcontinue": f"batch{n + 1}"}} if n < len(batches) - 1 else {})}
                    for n, batch in enumerate(batches)]
        hidden = [{"revid": 4, "timestamp": "2026-10-05T23:01:58Z", "slots": {"main": {"texthidden": True}}},
                  {"revid": 3, "timestamp": "2026-10-05T23:01:33Z", "slots": {"main": {"texthidden": True}}}]
        shown = [{"revid": 2, "timestamp": "2026-10-05T22:40:00Z", "slots": {"main": {"content": PAGE}}},
                 {"revid": 1, "timestamp": "2026-10-05T20:00:00Z", "slots": {"main": {"content": ""}}}]
        with mock.patch("scripts.fetch_current_events.api_get", side_effect=responses(hidden, shown)) as api:
            page = fetch(None, date(2026, 10, 5), datetime(2026, 10, 6, tzinfo=timezone.utc))
        self.assertEqual((page["revision_id"], page["edited"], len(page["items"])), (2, "2026-10-05T22:40:00Z", 4))
        self.assertEqual(api.call_args.args[2]["rvstart"], "2026-10-06T00:00:00Z")  # still as of the cutoff
        self.assertEqual(api.call_args.args[2]["rvcontinue"], "batch1")
        with mock.patch("scripts.fetch_current_events.api_get", side_effect=responses(hidden, hidden[1:])):
            self.assertTrue(fetch(None, date(2026, 10, 5))["missing"])  # nothing shown at all


if __name__ == "__main__":
    unittest.main()
