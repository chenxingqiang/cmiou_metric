import argparse
from cmiou_metric.modeling.cmiou import CMIoU
from cmiou_metric.modeling.semantic_similarity import word_embedding_similarity
from cmiou_metric.data.dataset import load_pascal_voc, load_cityscapes, load_ade20k
from cmiou_metric.utils.visualizer import plot_cmiou_comparison


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

    # 初始化CMIoU计算器
    cmiou_calculator = CMIoU(word_embedding_similarity)

    # 评估模型
    model_names = ["SPNet", "ZS3Net", "LSeg"]
    cmiou_scores = []

    for model_name in model_names:
        # TODO: 加载模型
        # TODO: 在数据集上运行模型
        # TODO: 计算CMIoU
        cmiou_scores.append(cmiou_score)

    # 可视化结果
    plot_cmiou_comparison(model_names, cmiou_scores)


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
    args = parser.parse_args()
    main(args)
