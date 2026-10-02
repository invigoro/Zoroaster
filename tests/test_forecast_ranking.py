import math
import unittest

from src.forecast.changes import KINDS
from src.forecast.ranking import (build_trie, choose_thresholds, first_section_probs, kind_marginals,
                                  kinds_continuation, kinds_forecast, kinds_lines, rank_sections, section_candidates,
                                  section_continuations)


class SectionsTest(unittest.TestCase):
    def test_candidates_continuations_and_ranking(self):
        candidates = section_candidates({"heading_titles": ["Career", "(lead)", "Results", "Career"]})
        self.assertEqual(candidates, ["(lead)", "Career", "Results"])  # the lead first, no repeats
        self.assertEqual(section_continuations(["Career"]), [" Career;", " Career\n"])
        logprobs = [math.log(p) for p in (0.1, 0.05, 0.3, 0.2, 0.25, 0.0001)]
        probs = first_section_probs(logprobs, candidates)
        self.assertEqual([round(p, 4) for p in probs], [0.15, 0.5, 0.2501])  # either ending counts
        self.assertEqual(rank_sections(candidates, probs, 2), ["Career", "Results"])
        self.assertEqual(rank_sections(["A", "B", "C"], [0.2, 0.2, 0.1], 2), ["A", "B"])  # ties keep page order


class TrieTest(unittest.TestCase):
    def test_shared_prefixes_become_one_node(self):
        tokens, parents, depths, paths = build_trie([[5, 6, 7], [5, 6, 8], [9], [5]])
        self.assertEqual(tokens, [5, 6, 7, 8, 9])
        self.assertEqual(parents, [-1, 0, 1, 1, -1])
        self.assertEqual(depths, [0, 1, 2, 2, 0])
        self.assertEqual(paths, [[0, 1, 2], [0, 1, 3], [4], [0]])
        self.assertTrue(all(p < i for i, p in enumerate(parents)))


class KindsTest(unittest.TestCase):
    def test_lines_cover_every_set_in_order(self):
        lines = kinds_lines()
        self.assertEqual(len(lines), 2 ** len(KINDS))
        self.assertEqual(lines[0], ())
        self.assertTrue(all(list(line) == [k for k in KINDS if k in line] for line in lines))
        self.assertEqual(kinds_continuation(()), " none\n")
        self.assertEqual(kinds_continuation(("prose", "table")), " prose, table\n")

    def test_marginals_are_exact_for_independent_kinds(self):
        # If each kind is listed independently with probability q, each line's
        # probability is a product, and the marginals must come back as q.
        q = {k: 0.1 + 0.08 * i for i, k in enumerate(KINDS)}
        lines = kinds_lines()
        logprobs = [sum(math.log(q[k] if k in line else 1 - q[k]) for k in KINDS) + math.log(0.8) for line in lines]
        marginals, total = kind_marginals(lines, logprobs)
        self.assertAlmostEqual(total, 0.8)  # the rest went to invalid lines
        for k in KINDS:
            self.assertAlmostEqual(marginals[k], q[k])

    def test_thresholds_maximize_mean_jaccard(self):
        def m(prose):
            return {k: (prose if k == "prose" else 0.0) for k in KINDS}

        marginals = [m(0.2), m(0.35), m(0.6), m(0.1)]
        actual = [[], ["prose"], ["prose"], []]
        self.assertEqual(sum(kinds_forecast(x, {k: 0.5 for k in KINDS}) == a for x, a in zip(marginals, actual)), 3)
        thresholds = choose_thresholds(marginals, actual)
        self.assertEqual(thresholds["prose"], 0.25)  # the first grid value that gets all four right
        self.assertEqual(thresholds["table"], 0.5)  # never present, never predicted: left alone
        self.assertEqual([kinds_forecast(x, thresholds) for x in marginals], actual)


if __name__ == "__main__":
    unittest.main()
