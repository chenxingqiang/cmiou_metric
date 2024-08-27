import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_segmentation_result(image, ground_truth, prediction, class_names):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(ground_truth, cmap="tab20")
    ax2.set_title("Ground Truth")
    ax2.axis("off")

    ax3.imshow(prediction, cmap="tab20")
    ax3.set_title("Prediction")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def plot_cmiou_comparison(model_names, cmiou_scores):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=cmiou_scores)
    plt.title("CMIoU Comparison Across Models")
    plt.xlabel("Models")
    plt.ylabel("CMIoU Score")
    plt.ylim(0, 1)
    plt.show()


def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_semantic_similarity_heatmap(similarity_matrix, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Semantic Similarity Heatmap")
    plt.xlabel("Classes")
    plt.ylabel("Classes")
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_cmiou_vs_miou(cmiou_scores, miou_scores, model_names):
    """
    绘制CMIoU与mIoU的比较图
    """
    plt.figure(figsize=(10, 6))
    x = range(len(model_names))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        cmiou_scores,
        width,
        label="CMIoU",
        color="b",
        alpha=0.7,
    )
    plt.bar(
        [i + width / 2 for i in x],
        miou_scores,
        width,
        label="mIoU",
        color="r",
        alpha=0.7,
    )

    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("CMIoU vs mIoU Comparison")
    plt.xticks(x, model_names)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_open_vocab_performance(seen_percentages, cmiou_scores, model_names):
    """
    绘制不同开放词汇场景下的性能图
    """
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(model_names):
        plt.plot(seen_percentages, cmiou_scores[i], marker="o", label=model)

    plt.xlabel("Percentage of Seen Classes")
    plt.ylabel("CMIoU Score")
    plt.title("Performance in Different Open-Vocabulary Scenarios")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_semantic_similarity_heatmap(similarity_matrix, class_names):
    """
    绘制语义相似度热图
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Semantic Similarity Heatmap")
    plt.xlabel("Classes")
    plt.ylabel("Classes")
    plt.tight_layout()
    plt.show()


def plot_segmentation_result(image, ground_truth, prediction, class_names):
    """
    可视化分割结果
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(ground_truth, cmap="tab20")
    ax2.set_title("Ground Truth")
    ax2.axis("off")

    ax3.imshow(prediction, cmap="tab20")
    ax3.set_title("Prediction")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def plot_similarity_method_comparison(similarity_methods, cmiou_scores, model_names):
    """
    绘制不同语义相似度方法的对比图
    """
    plt.figure(figsize=(12, 6))
    x = range(len(model_names))
    width = 0.25

    for i, method in enumerate(similarity_methods):
        plt.bar(
            [xi + i * width for xi in x],
            cmiou_scores[i],
            width,
            label=method,
            alpha=0.7,
        )

    plt.xlabel("Models")
    plt.ylabel("CMIoU Score")
    plt.title("Comparison of Different Semantic Similarity Methods")
    plt.xticks([xi + width for xi in x], model_names)
    plt.legend()
    plt.tight_layout()
    plt.show()
