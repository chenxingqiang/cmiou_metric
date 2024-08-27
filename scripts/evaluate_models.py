import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
from cmiou_metric.modeling.cmiou import CMIoU
from cmiou_metric.modeling.semantic_similarity import (
    word_embedding_similarity,
    knowledge_graph_similarity,
    ontology_similarity,
)
from cmiou_metric.data.dataset import load_pascal_voc, load_cityscapes, load_ade20k
from cmiou_metric.modeling.models import load_pretrained_model, predict
from cmiou_metric.utils.evaluation import (
    calculate_miou,
    calculate_similarity_matrix,
    evaluate_open_vocab,
    evaluate_similarity_methods,
)
from cmiou_metric.utils.visualizer import (
    plot_cmiou_comparison,
    plot_confusion_matrix,
    plot_cmiou_vs_miou,
    plot_open_vocab_performance,
    plot_semantic_similarity_heatmap,
    plot_segmentation_result,
    plot_similarity_method_comparison,
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

    # 初始化CMIoU计算器
    cmiou_calculator = CMIoU(similarity_func)

    # 评估模型
    model_names = ["SPNet", "ZS3Net", "LSeg"]
    cmiou_scores = []
    miou_scores = []
    confusion_matrices = []

    # 不同开放词汇场景的评估
    seen_percentages = [25, 50, 75]
    open_vocab_scores = {model: [] for model in model_names}

    for model_name in model_names:
        model_path = f"{args.model_dir}/{model_name}.pth"
        model = load_pretrained_model(model_path, len(dataset.class_names))

        ground_truth = []
        predictions = []

        for image, mask, _ in dataset:
            ground_truth.append(mask.numpy())
            pred = predict(model, image)
            predictions.append(pred)

        cmiou_score = cmiou_calculator.calculate_cmiou(
            ground_truth, predictions, dataset.class_names, similarity_data
        )
        cmiou_scores.append(cmiou_score)

        # 计算mIoU
        miou_score = calculate_miou(ground_truth, predictions)
        miou_scores.append(miou_score)

        # 评估不同开放词汇场景
        for seen_percent in seen_percentages:
            score = evaluate_open_vocab(
                model, dataset, seen_percent, cmiou_calculator, similarity_data
            )
            open_vocab_scores[model_name].append(score)

        # 绘制语义相似度热图
        similarity_matrix = calculate_similarity_matrix(
            dataset.class_names, similarity_func, similarity_data
        )
        plot_semantic_similarity_heatmap(similarity_matrix, dataset.class_names)

        # 绘制不同语义相似度方法的对比图
        similarity_methods = ["Word Embedding", "Knowledge Graph", "Ontology"]
        similarity_scores = evaluate_similarity_methods(
            dataset, model_names, similarity_methods
        )
        plot_similarity_method_comparison(
            similarity_methods, similarity_scores, model_names
        )

        # 计算混淆矩阵
        y_true = np.concatenate(ground_truth).flatten()
        y_pred = np.concatenate(predictions).flatten()
        conf_matrix = confusion_matrix(y_true, y_pred)
        confusion_matrices.append(conf_matrix)

        # 评估不同开放词汇场景
        for seen_percent in seen_percentages:
            score = evaluate_open_vocab(
                model, dataset, seen_percent, cmiou_calculator, similarity_data
            )
            open_vocab_scores[model_name].append(score)

    # 绘制CMIoU与mIoU的比较图
    plot_cmiou_vs_miou(cmiou_scores, miou_scores, model_names)

    # 绘制不同开放词汇场景下的性能图
    plot_open_vocab_performance(
        seen_percentages,
        [open_vocab_scores[model] for model in model_names],
        model_names,
    )

    # 绘制语义相似度热图
    similarity_matrix = calculate_similarity_matrix(
        dataset.class_names, similarity_func, similarity_data
    )
    plot_semantic_similarity_heatmap(similarity_matrix, dataset.class_names)

    # 绘制分割结果示例
    sample_image, sample_mask, _ = dataset[0]
    sample_pred = predict(model, sample_image)
    plot_segmentation_result(
        sample_image, sample_mask, sample_pred, dataset.class_names
    )

    # 绘制不同语义相似度方法的对比图
    similarity_methods = ["Word Embedding", "Knowledge Graph", "Ontology"]
    similarity_scores = evaluate_similarity_methods(
        dataset, model_names, similarity_methods
    )
    plot_similarity_method_comparison(
        similarity_methods, similarity_scores, model_names
    )

    # 绘制混淆矩阵
    for i, model_name in enumerate(model_names):
        plot_confusion_matrix(confusion_matrices[i], dataset.class_names)
        plt.title(f"Confusion Matrix for {model_name}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models using CMIoU")
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
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing pretrained models",
    )
    args = parser.parse_args()
    main(args)
