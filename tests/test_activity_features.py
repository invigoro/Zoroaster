import random
import unittest
from datetime import datetime, timedelta

from src.features.activity import (
    FEATURE_NAMES,
    MASS_EDITOR_PAGES_PER_DAY,
    PageActivity,
    day_ordinal,
    editor_day_page_counts,
    mass_editor_days,
    page_editor_days,
)
from src.features.bursts import CoBurstCounter
from src.ingest.revert_detect import detect_page_reverts

START = datetime(2020, 1, 1)
USERS = ["Alice", "Bob", "Carol", "Dave", "10.0.0.1", "10.0.0.2", "FooBot"]


def revision(rev_id, when, sha1, user):
    return {
        "page_id": 1,
        "revision_id": rev_id,
        "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha1": sha1,
        "user_text": user,
    }


def page(revisions):
    return PageActivity(detect_page_reverts(revisions))


def random_history(rng, days=150):
    """A page with quiet spells, busy days, bots, and reverts of recent states."""
    revisions, states = [], []
    for day in range(days):
        roll = rng.random()
        n = rng.randint(4, 15) if roll < 0.04 else rng.randint(1, 3) if roll < 0.35 else 0
        if day == 0:
            n = max(n, 1)
        for when in sorted(START + timedelta(days=day, seconds=rng.randrange(86400)) for _ in range(n)):
            rev_id = len(revisions) + 1
            if states and rng.random() < 0.25:
                sha1 = rng.choice(states[-4:])
            elif rng.random() < 0.02:
                sha1 = None
            else:
                sha1 = f"s{rev_id}"
                states.append(sha1)
            revisions.append(revision(rev_id, when, sha1, rng.choice(USERS)))
            revisions[-1]["byte_size"] = rng.choice([None, rng.randint(0, 50_000)])
    return revisions


class PointInTimeTest(unittest.TestCase):
    def test_features_do_not_depend_on_the_future(self):
        """Features for day D must equal those computed from a history that
        stops at the start of D, with revert detection re-run on it, i.e.
        as if the dump had been taken at midnight before D."""
        rng = random.Random(0)
        checked = 0
        for _ in range(40):
            revisions = random_history(rng)
            full = page(revisions)
            for day in range(full.first_day + 1, full.days[-1] + 3):
                truncated = [r for r in revisions if day_ordinal(r["timestamp"]) < day]
                self.assertEqual(full.features(day), page(truncated).features(day), day)
                checked += 1
        self.assertGreater(checked, 5000)

    def test_mass_editor_bursts_do_not_depend_on_the_future(self):
        rng = random.Random(1)
        for _ in range(20):
            revisions = random_history(rng)
            days = sorted({day_ordinal(r["timestamp"]) for r in revisions})
            mass = frozenset((user, d) for d in rng.sample(days, len(days) // 3) for user in USERS[:2])
            full = PageActivity(detect_page_reverts(revisions), mass)
            for day in range(full.first_day + 1, full.days[-1] + 3):
                cut = PageActivity(detect_page_reverts([r for r in revisions if day_ordinal(r["timestamp"]) < day]), mass)
                self.assertEqual({d for d in full.burst_days_excl_mass if d < day}, cut.burst_days_excl_mass, day)
                self.assertEqual(full.editor_count_excl_mass(day - 1), cut.editor_count_excl_mass(day - 1))

    def test_label_channel_does_depend_on_the_future(self):
        # Sanity check that the invariance test above could fail: rev 2 looks
        # kept until a revert two days later.
        revisions = [
            revision(1, START, "A", "Alice"),
            revision(2, START + timedelta(days=1), "B", "Bob"),
            revision(3, START + timedelta(days=3), "A", "Carol"),
        ]
        day1 = day_ordinal(revisions[1]["timestamp"])
        self.assertEqual(page(revisions[:2]).counts["kept_edits"][day1], 1)
        self.assertEqual(page(revisions).counts["kept_edits"][day1], 0)


class ChannelTest(unittest.TestCase):
    def test_same_day_reverts_are_known_later_reverts_are_not(self):
        t = START + timedelta(hours=10)
        p = page([
            revision(1, t, "A", "Alice"),
            revision(2, t + timedelta(days=1), "B", "10.0.0.1"),  # vandalism...
            revision(3, t + timedelta(days=1, hours=1), "A", "Bob"),  # ...reverted same day
            revision(4, t + timedelta(days=2), "C", "Carol"),  # looks kept on day 2...
            revision(5, t + timedelta(days=5), "A", "Bob"),  # ...until reverted on day 5
            revision(6, t + timedelta(days=5, hours=1), "D", "FooBot"),
        ])
        d0 = p.first_day
        self.assertEqual(dict(p.counts["edits"]), {d0: 1, d0 + 2: 1})
        self.assertEqual(dict(p.counts["reverted_edits"]), {d0 + 1: 1})
        self.assertEqual(dict(p.counts["reverts"]), {d0 + 1: 1, d0 + 5: 1})
        self.assertEqual(dict(p.counts["bot_edits"]), {d0 + 5: 1})
        self.assertEqual(dict(p.counts["kept_edits"]), {d0: 1})
        f = p.features(d0 + 3)
        self.assertEqual((f["edits_1d"], f["edits_7d"], f["reverted_edits_7d"], f["reverts_7d"]), (1, 2, 1, 1))
        self.assertEqual((f["days_since_last_edit"], f["page_age_days"]), (1, 3))


class HistorySourceTest(unittest.TestCase):
    """Behavior specific to the MediaWiki history dumps (English Wikipedia)."""

    def test_the_sources_bot_flag_overrides_the_name_heuristic(self):
        revisions = [
            revision(1, START, "a", "FooBot") | {"is_bot": False},  # a human despite the name
            revision(2, START + timedelta(hours=1), "b", "Alice") | {"is_bot": True},  # a bot account
        ]
        p = page(revisions)
        self.assertEqual((p.counts["edits"][p.first_day], p.counts["bot_edits"][p.first_day]), (1, 1))
        self.assertEqual(page_editor_days(detect_page_reverts(revisions)), {("FooBot", p.first_day)})

    def test_page_age_uses_the_creation_day_when_given(self):
        revisions = [revision(1, START + timedelta(days=500), "a", "Alice")]
        first = day_ordinal(revisions[0]["timestamp"])
        self.assertEqual(page(revisions).features(first + 10)["page_age_days"], 10)
        created = PageActivity(detect_page_reverts(revisions), created_day=first - 400)
        self.assertEqual(created.features(first + 10)["page_age_days"], 410)
        self.assertEqual(created.first_day, first)  # existence still starts with the data


class PageBytesTest(unittest.TestCase):
    def test_size_is_the_last_known_size_before_the_day(self):
        revisions = [
            revision(1, START, "a", "Alice") | {"byte_size": 100},
            revision(2, START + timedelta(hours=5), "b", "Bob") | {"byte_size": 250},  # same day: last wins
            revision(3, START + timedelta(days=2), "c", "Carol") | {"byte_size": None},  # unknown: keep 250
            revision(4, START + timedelta(days=3), "d", "Dave") | {"byte_size": 90},
        ]
        p = page(revisions)
        d0 = p.first_day
        self.assertEqual([p.features(d)["page_bytes"] for d in (d0, d0 + 1, d0 + 3, d0 + 4)], [None, 250, 250, 90])


class BurstTest(unittest.TestCase):
    def test_busy_new_page_bursts_on_day_one(self):
        # Impossible before: a page needed >= 10 active days to reach z >= 3.
        t = START
        p = page([revision(i, t + timedelta(minutes=10 * i), f"s{i}", USERS[i % 2]) for i in range(1, 6)])
        self.assertIn(p.first_day, p.burst_days)
        f = p.features(p.first_day + 1)
        self.assertTrue(f["is_burst_1d"])
        self.assertEqual(f["editors_1d"], 2)

    def test_dormant_page_spike_bursts_but_routine_activity_does_not(self):
        revisions = [revision(i + 1, START + timedelta(days=7 * i), f"s{i}", "Alice") for i in range(20)]
        spike_day = START + timedelta(days=7 * 20 + 3)
        revisions += [revision(21 + i, spike_day + timedelta(minutes=i), f"t{i}", USERS[i % 3]) for i in range(4)]
        p = page(revisions)
        self.assertEqual(p.burst_days, {day_ordinal(spike_day.strftime("%Y-%m-%dT%H:%M:%SZ"))})

    def test_mass_editor_edits_burst_only_in_the_regular_definition(self):
        t = START + timedelta(days=100)
        revisions = [revision(1, START, "s0", "Alice")]
        revisions += [revision(2 + i, t + timedelta(minutes=i), f"m{i}", "Mass") for i in range(5)]
        revisions += [revision(10 + i, t + timedelta(days=1, minutes=i), f"h{i}", USERS[i % 2]) for i in range(4)]
        mass_day = day_ordinal(revisions[1]["timestamp"])
        p = PageActivity(detect_page_reverts(revisions), frozenset({("Mass", mass_day)}))
        self.assertEqual(p.burst_days, {mass_day, mass_day + 1})
        self.assertEqual(p.burst_days_excl_mass, {mass_day + 1})
        self.assertEqual((p.editor_count(mass_day), p.editor_count_excl_mass(mass_day)), (1, 0))
        self.assertEqual(page(revisions).burst_days_excl_mass, p.burst_days)  # no mass set: identical

    def test_mass_editor_days_counts_distinct_pages_of_human_non_revert_edits(self):
        day = START.strftime("%Y-%m-%dT%H:%M:%SZ")
        edit = {"timestamp": day, "is_revert": False}
        pages = [[{**edit, "user_text": "Mass"}, {**edit, "user_text": "Mass"}] for _ in range(MASS_EDITOR_PAGES_PER_DAY + 1)]
        pages += [[{**edit, "user_text": "Alice"}]]
        pages += [[{**edit, "user_text": "FooBot"}]] * 40 + [[{**edit, "user_text": "Carol", "is_revert": True}]] * 40
        counts = editor_day_page_counts(pages)
        self.assertEqual(counts[("Mass", day_ordinal(day))], MASS_EDITOR_PAGES_PER_DAY + 1)  # distinct pages
        self.assertEqual(counts[("Alice", day_ordinal(day))], 1)
        self.assertNotIn(("FooBot", day_ordinal(day)), counts)
        self.assertNotIn(("Carol", day_ordinal(day)), counts)
        self.assertEqual(mass_editor_days(counts), frozenset({("Mass", day_ordinal(day))}))
        self.assertEqual(mass_editor_days(counts, max_pages=100), frozenset())

    def test_co_burst_excludes_the_page_itself(self):
        counter = CoBurstCounter()
        for editors in (1, 2, 3):
            counter.add(100, editors)
        self.assertEqual(counter.others(100, self_bursting=True, self_editors=2), (2, 1))
        self.assertEqual(counter.others(100, self_bursting=True, self_editors=1), (2, 2))
        self.assertEqual(counter.others(100, self_bursting=False, self_editors=5), (3, 2))
        self.assertEqual(counter.others(101, self_bursting=False, self_editors=0), (0, 0))

    def test_feature_names_match(self):
        p = page([revision(1, START, "A", "Alice")])
        self.assertEqual(tuple(p.features(p.first_day + 1)), FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
