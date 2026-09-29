import unittest

from src.ingest.sampling import hash_sample_mask, hash_sampled


class HashSampledTest(unittest.TestCase):
    def test_rate_overall_and_within_buckets(self):
        ids = range(1, 400_001)
        chosen = [i for i in ids if hash_sampled(i, 0.2)]
        self.assertAlmostEqual(len(chosen) / len(ids), 0.2, delta=0.005)
        per_bucket = [sum(1 for i in chosen if i % 128 == b) / (len(ids) / 128) for b in range(128)]
        self.assertLess(max(abs(r - 0.2) for r in per_bucket), 0.03)

    def test_mask_matches_the_per_page_function(self):
        mask = hash_sample_mask(200_000, 0.2)
        self.assertEqual(mask.tolist(), [hash_sampled(i, 0.2) for i in range(200_000)])

    def test_deterministic_and_nested(self):
        self.assertEqual([hash_sampled(i, 0.2) for i in range(1000)], [hash_sampled(i, 0.2) for i in range(1000)])
        # A smaller sample is a subset of a larger one.
        self.assertTrue(all(hash_sampled(i, 0.2) for i in range(10_000) if hash_sampled(i, 0.05)))


if __name__ == "__main__":
    unittest.main()
