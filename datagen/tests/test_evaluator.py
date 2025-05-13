from unittest import TestCase
import numpy as np

from datagen.src.evaluator import calculate_entropy, eval_entropy


class Test(TestCase):
    def test_calculate_entropy(self):
        frequency_pairs = [
            ([0.5, 0.5], np.log(2)),       # entropy = -0.5*ln(0.5) -0.5*ln(0.5) = ln(2)
            ([1.0, 0.0], 0.0),             # entropy = 0 (fully stable)
            ([0.0, 1.0], 0.0),             # entropy = 0 (fully unstable)
        ]

        for freqs, expected_entropy in frequency_pairs:
            entropy = calculate_entropy(freqs)
            self.assertAlmostEqual(entropy, expected_entropy, places=6)

    def test_eval_entropy(self):
        cases = [
            ([1, 1, 1, 1], None),
            ([0, 0, 0, 0], None),
            ([1, 0, 1, 0], None),
            ([1, 1, 0, 0], 0.5),
            ([1, 1, 1, 0], 0.2)
        ]

        for stabilities, entropy_parent in cases:
            freqs = [stabilities.count(1) / len(stabilities),
                     stabilities.count(0) / len(stabilities)]
            expected_entropy = calculate_entropy(freqs)
            entropy, delta = eval_entropy(stabilities, entropy_parent)
            self.assertAlmostEqual(entropy, expected_entropy, places=6)
            if entropy_parent is None:
                self.assertEqual(delta, 1)
            else:
                self.assertAlmostEqual(delta, expected_entropy - entropy_parent, places=6)
