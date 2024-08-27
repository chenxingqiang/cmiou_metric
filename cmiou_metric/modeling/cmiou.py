import numpy as np
from typing import List, Callable, Dict


class CMIoU:
    def __init__(self, semantic_similarity_func: Callable[[str, str, Dict], float]):
        self.semantic_similarity_func = semantic_similarity_func

    def calculate_ciou(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        class_name: str,
        all_classes: List[str],
        similarity_data: Dict,
    ) -> float:
        """
        计算单个类别的概念校准IoU (CIoU)

        参数:
        ground_truth: 真实标签的二值化掩码
        prediction: 预测的二值化掩码
        class_name: 当前计算的类别名称
        all_classes: 所有类别的列表
        similarity_data: 用于计算语义相似度的数据（如词嵌入或知识图谱）

        返回:
        float: CIoU值
        """
        intersection = np.logical_and(ground_truth, prediction)
        union = np.logical_or(ground_truth, prediction)

        if union.sum() == 0:
            return 0.0

        numerator = 0
        denominator = 0

        for i in range(union.shape[0]):
            for j in range(union.shape[1]):
                if union[i, j]:
                    gt_class = class_name if ground_truth[i, j] else "background"
                    pred_class = class_name if prediction[i, j] else "background"
                    similarity = self.semantic_similarity_func(
                        gt_class, pred_class, similarity_data
                    )

                    if ground_truth[i, j]:
                        numerator += similarity
                    denominator += min(1, similarity)

        return numerator / denominator if denominator > 0 else 0.0

    def calculate_cmiou(
        self,
        ground_truth: List[np.ndarray],
        prediction: List[np.ndarray],
        class_names: List[str],
        similarity_data: Dict,
    ) -> float:
        """
        计算概念校准平均IoU (CMIoU)

        参数:
        ground_truth: 真实标签掩码列表
        prediction: 预测掩码列表
        class_names: 类别名称列表
        similarity_data: 用于计算语义相似度的数据

        返回:
        float: CMIoU值
        """
        cious = [
            self.calculate_ciou(gt, pred, name, class_names, similarity_data)
            for gt, pred, name in zip(ground_truth, prediction, class_names)
        ]
        return np.mean(cious)
