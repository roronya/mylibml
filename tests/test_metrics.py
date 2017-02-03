import mylibml
import numpy as np
import unittest

class TestMetrics(unittest.TestCase):
    def test_propensity_scored_mse(self):
        true = np.array([1,2,3,4,5])
        pred = np.array([1,2,3,4,5])
        propensity_score = np.array([1,1,1,1,1])
        result = mylibml.metrics.propensity_scored_mse(true, pred, propensity_score)
        self.assertEqual(result, 0)
        true = np.array([1,0,1,0,1])
        pred = np.array([0,1,0,1,0])
        propensity_score = np.array([0.5,0.5,0.5,0.5,0.5])
        result = mylibml.metrics.propensity_scored_mse(true, pred, propensity_score)
        self.assertEqual(result, 2)

    def test_dcg(self):
        true = np.array([1,2,3,4,5])
        pred = np.array([1,2,3,4,5])
        result = mylibml.metrics.dcg(true, pred)
        self.assertAlmostEqual(result, 12.3234658188)
        pred = np.array([5,4,3,2,1])
        result = mylibml.metrics.dcg(true, pred)
        self.assertAlmostEqual(result, 9.04617205108)

    def test_ndcg(self):
        true = np.array([1,2,3,4,5])
        pred = np.array([1,2,3,4,5])
        result = mylibml.metrics.ndcg(true, pred)
        self.assertEqual(result, 1)
        pred = np.array([5,4,3,2,1])
        result = mylibml.metrics.ndcg(true, pred)
        self.assertAlmostEqual(result, 0.734060708578)

if __name__ == '__main__':
    unittest.main()
