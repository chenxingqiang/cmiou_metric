import numpy as np
from typing import List, Dict, Callable
from cmiou_metric.modeling.cmiou import CMIoU
from cmiou_metric.data.dataset import SegmentationDataset
from cmiou_metric.modeling.models import load_pretrained_model, predict
from scripts.calculate_cmiou import load_knowledge_graph, load_word_embeddings

from cmiou_metric.modeling.semantic_similarity import knowledge_graph_similarity, ontology_similarity, word_embedding_similarity


def calculate_miou(
    ground_truth: List[np.ndarray], predictions: List[np.ndarray]
) -> float:
    """
    计算平均交并比（Mean Intersection over Union，mIoU）

    参数:
    ground_truth: 真实标签列表
    predictions: 预测标签列表

    返回:
    float: mIoU 得分
    """
    ious = []
    for gt, pred in zip(ground_truth, predictions):
        intersection = np.logical_and(gt, pred)
        union = np.logical_or(gt, pred)
        iou = np.sum(intersection) / np.sum(union)
        ious.append(iou)
    return np.mean(ious)


def evaluate_open_vocab(
    model,
    dataset: SegmentationDataset,
    seen_percent: int,
    cmiou_calculator: CMIoU,
    similarity_data: Dict,
) -> float:
    """
    在不同的开放词汇场景下评估模型

    参数:
    model: 要评估的模型
    dataset: 数据集
    seen_percent: 已知类别的百分比
    cmiou_calculator: CMIoU 计算器
    similarity_data: 用于计算语义相似度的数据

    返回:
    float: CMIoU 得分
    """
    num_classes = len(dataset.class_names)
    num_seen = int(num_classes * seen_percent / 100)
    seen_classes = np.random.choice(dataset.class_names, num_seen, replace=False)

    ground_truth = []
    predictions = []

    for image, mask, class_name in dataset:
        if class_name in seen_classes:
            ground_truth.append(mask.numpy())
            pred = predict(model, image)
            predictions.append(pred)

    return cmiou_calculator.calculate_cmiou(
        ground_truth, predictions, seen_classes, similarity_data
    )


def calculate_similarity_matrix(
    class_names: List[str], similarity_func: Callable, similarity_data: Dict
) -> np.ndarray:
    """
    计算类别之间的语义相似度矩阵

    参数:
    class_names: 类别名称列表
    similarity_func: 计算语义相似度的函数
    similarity_data: 用于计算语义相似度的数据

    返回:
    np.ndarray: 语义相似度矩阵
    """
    num_classes = len(class_names)
    similarity_matrix = np.zeros((num_classes, num_classes))

    for i, class1 in enumerate(class_names):
        for j, class2 in enumerate(class_names):
            similarity_matrix[i, j] = similarity_func(class1, class2, similarity_data)

    return similarity_matrix


def evaluate_similarity_methods(
    dataset: SegmentationDataset, model_names: List[str], similarity_methods: List[str]
) -> List[List[float]]:
    """
    评估不同语义相似度方法的性能

    参数:
    dataset: 数据集
    model_names: 模型名称列表
    similarity_methods: 语义相似度方法列表

    返回:
    List[List[float]]: 每种方法对应每个模型的 CMIoU 得分
    """
    scores = []

    for method in similarity_methods:
        method_scores = []

        # 根据方法加载相应的语义相似度数据和函数
        if method == "Word Embedding":
            similarity_data = load_word_embeddings("path/to/embeddings.txt")
            similarity_func = word_embedding_similarity
        elif method == "Knowledge Graph":
            similarity_data = load_knowledge_graph("path/to/knowledge_graph.json")
            similarity_func = knowledge_graph_similarity
        elif method == "Ontology":
            similarity_data = None
            similarity_func = ontology_similarity

        cmiou_calculator = CMIoU(similarity_func)

        for model_name in model_names:
            model = load_pretrained_model(
                f"path/to/models/{model_name}.pth", len(dataset.class_names)
            )

            ground_truth = []
            predictions = []

            for image, mask, _ in dataset:
                ground_truth.append(mask.numpy())
                pred = predict(model, image)
                predictions.append(pred)

            cmiou_score = cmiou_calculator.calculate_cmiou(
                ground_truth, predictions, dataset.class_names, similarity_data
            )
            method_scores.append(cmiou_score)

        scores.append(method_scores)

    return scores
