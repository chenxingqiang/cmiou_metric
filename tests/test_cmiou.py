import unittest
import numpy as np
from cmiou_metric.modeling.cmiou import CMIoU
from cmiou_metric.modeling.semantic_similarity import word_embedding_similarity
from cmiou_metric.utils.evaluation import calculate_miou, calculate_similarity_matrix


class TestCMIoU(unittest.TestCase):

    def setUp(self):
        # 创建一个简单的词嵌入字典用于测试
        self.embeddings = {
            "cat": np.array([0.1, 0.2, 0.3]),
            "dog": np.array([0.15, 0.25, 0.35]),
            "car": np.array([0.5, 0.6, 0.7]),
        }
        self.cmiou_calculator = CMIoU(word_embedding_similarity)

    def test_word_embedding_similarity(self):
        similarity = word_embedding_similarity("cat", "dog", self.embeddings)
        self.assertAlmostEqual(similarity, 0.9998477, places=6)

    def test_calculate_ciou(self):
        ground_truth = np.array([[1, 1], [1, 0]])
        prediction = np.array([[1, 1], [0, 1]])
        class_name = "cat"
        all_classes = ["cat", "dog", "car"]

        ciou = self.cmiou_calculator.calculate_ciou(
            ground_truth, prediction, class_name, all_classes, self.embeddings
        )
        self.assertAlmostEqual(ciou, 0.7499369, places=6)

    def test_calculate_cmiou(self):
        ground_truth = [np.array([[1, 1], [1, 0]]), np.array([[1, 0], [0, 1]])]
        predictions = [np.array([[1, 1], [0, 1]]), np.array([[1, 1], [0, 0]])]
        class_names = ["cat", "dog"]

        cmiou = self.cmiou_calculator.calculate_cmiou(
            ground_truth, predictions, class_names, self.embeddings
        )
        self.assertAlmostEqual(cmiou, 0.7499684, places=6)

    def test_calculate_miou(self):
        ground_truth = [np.array([[1, 1], [1, 0]]), np.array([[1, 0], [0, 1]])]
        predictions = [np.array([[1, 1], [0, 1]]), np.array([[1, 1], [0, 0]])]

        miou = calculate_miou(ground_truth, predictions)
        self.assertAlmostEqual(miou, 0.5833333, places=6)

    def test_calculate_similarity_matrix(self):
        class_names = ["cat", "dog", "car"]
        similarity_matrix = calculate_similarity_matrix(
            class_names, word_embedding_similarity, self.embeddings
        )

        expected_matrix = np.array(
            [
                [1.0, 0.9998477, 0.9486833],
                [0.9998477, 1.0, 0.9535150],
                [0.9486833, 0.9535150, 1.0],
            ]
        )

        np.testing.assert_array_almost_equal(
            similarity_matrix, expected_matrix, decimal=6
        )


if __name__ == "__main__":
    unittest.main()
