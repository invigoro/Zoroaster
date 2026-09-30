import unittest

from scripts.backtest_daily import summarize


def day(model, baseline):
    return {"precision": {"model": {"100": model}, "edits yesterday": {"100": baseline}}}


class SummarizeTest(unittest.TestCase):
    def test_mean_precision_and_paired_gain(self):
        s = summarize([day(0.2, 0.1), day(0.1, 0.1), day(0.3, 0.4)])
        self.assertEqual(s["days"], 3)
        self.assertAlmostEqual(s["mean_precision"]["model"]["100"], 0.2)
        gain = s["model_minus_baseline"]["100"]
        self.assertAlmostEqual(gain["mean"], 0.0)
        self.assertEqual((gain["days_better"], gain["days_worse"]), (1, 1))
        self.assertAlmostEqual(gain["se"], 0.1 / 3 ** 0.5 * 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
