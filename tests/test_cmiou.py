import unittest
import numpy as np
from cmiou_metric.modeling.cmiou import CMIoU
from cmiou_metric.modeling.semantic_similarity import word_embedding_similarity


class TestCMIoU(unittest.TestCase):
    def setUp(self):
        self.cmiou_calculator = CMIoU(word_embedding_similarity)

    def test_calculate_ciou(self):
        ground_truth = np.array([[1, 1], [1, 0]])
        prediction = np.array([[1, 1], [0, 1]])
        class_name = "cat"

        ciou = self.cmiou_calculator.calculate_ciou(
            ground_truth, prediction, class_name
        )
        self.assertIsInstance(ciou, float)
        self.assertTrue(0 <= ciou <= 1)

    def test_calculate_cmiou(self):
        ground_truth = [np.array([[1, 1], [1, 0]]), np.array([[1, 0], [0, 1]])]
        prediction = [np.array([[1, 1], [0, 1]]), np.array([[1, 1], [0, 0]])]
        class_names = ["cat", "dog"]

        cmiou = self.cmiou_calculator.calculate_cmiou(
            ground_truth, prediction, class_names
        )
        self.assertIsInstance(cmiou, float)
        self.assertTrue(0 <= cmiou <= 1)


if __name__ == "__main__":
    unittest.main()
