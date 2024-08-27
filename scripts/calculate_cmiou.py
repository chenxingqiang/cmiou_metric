import argparse
import torch
import numpy as np
from cmiou_metric.modeling.cmiou import CMIoU
from cmiou_metric.modeling.semantic_similarity import (
    word_embedding_similarity,
   knowledge_graph_similarity,
   ontology_similarity,
)
from cmiou_metric.data.dataset import load_pascal_voc, load_cityscapes, load_ade20k
from cmiou_metric.modeling.models import load_pretrained_model, predict
from cmiou_metric.utils.visualizer import (
    plot_segmentation_result,
   plot_semantic_similarity_heatmap,
)
from sklearn.metrics import confusion_matrix


def load_word_embeddings(path):
    embeddings = {}
    with open(path, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def load_knowledge_graph(path):
    with open(path, "r") as f:
        return json.load(f)


def main(args):
    # 加载数据集
    if args.dataset == "pascal_voc":
        dataset = load_pascal_voc(args.data_dir)
    elif args.dataset == "cityscapes":
        dataset = load_cityscapes(args.data_dir)
    elif args.dataset == "ade20k":
        dataset = load_ade20k(args.data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # 加载语义相似度资源
    if args.similarity_method == "word_embedding":
        similarity_data = load_word_embeddings(args.similarity_data_path)
        similarity_func = word_embedding_similarity
    elif args.similarity_method == "knowledge_graph":
        similarity_data = load_knowledge_graph(args.similarity_data_path)
        similarity_func = knowledge_graph_similarity
    elif args.similarity_method == "ontology":
        similarity_data = None  # WordNet is used directly in the function
        similarity_func = ontology_similarity
    else:
        raise ValueError(f"Unsupported similarity method: {args.similarity_method}")

    # 加载预训练模型
    model = load_pretrained_model(args.model_path, len(dataset.class_names))

    # 初始化CMIoU计算器
    cmiou_calculator = CMIoU(similarity_func)

    # 计算CMIoU
    ground_truth = []
    predictions = []
    class_names = dataset.class_names

    for image, mask, _ in dataset:
        ground_truth.append(mask.numpy())
        pred = predict(model, image)
        predictions.append(pred)

    cmiou_score = cmiou_calculator.calculate_cmiou(
        ground_truth, predictions, class_names, similarity_data
    )

    # 计算混淆矩阵
    y_true = np.concatenate(ground_truth).flatten()
    y_pred = np.concatenate(predictions).flatten()
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算语义相似度矩阵
    similarity_matrix = np.zeros((len(class_names), len(class_names)))
    for i, class1 in enumerate(class_names):
        for j, class2 in enumerate(class_names):
            similarity_matrix[i, j] = similarity_func(class1, class2, similarity_data)

    # 输出结果
    print(f"CMIoU for {args.dataset} using {args.similarity_method}: {cmiou_score}")

    # 可视化结果
    plot_segmentation_result(image, mask, pred, class_names)
    plot_semantic_similarity_heatmap(similarity_matrix, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CMIoU for a dataset")
    parser.add_argument(
        "--dataset",
      type=str,
       required=True,
       choices=["pascal_voc", "cityscapes", "ade20k"],
    help="Dataset name",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--similarity_method",
      type=str,
       required=True,
       choices=["word_embedding", "knowledge_graph", "ontology"],
    help="Semantic similarity method",
    )
    parser.add_argument(
        "--similarity_data_path",
type=str,
       help="Path to similarity data (not required for ontology method)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to pretrained model"
    )
    args = parser.parse_args()
    main(args)
