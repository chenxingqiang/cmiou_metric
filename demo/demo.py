import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmiou_metric.data.dataset import load_pascal_voc
from cmiou_metric.modeling.cmiou import CMIoU
from cmiou_metric.modeling.semantic_similarity import word_embedding_similarity
from cmiou_metric.modeling.models import load_pretrained_model, predict
from cmiou_metric.utils.visualizer import plot_segmentation_result, plot_cmiou_comparison

def main():
    # 加载数据集
    dataset = load_pascal_voc("/path/to/pascal_voc")

    # 加载词嵌入
    embeddings = load_word_embeddings("/path/to/word_embeddings.txt")

    # 初始化CMIoU计算器
    cmiou_calculator = CMIoU(word_embedding_similarity)

    # 加载预训练模型
    model = load_pretrained_model("/path/to/pretrained_model.pth", len(dataset.class_names))

    # 选择一个样本进行演示
    image, mask, class_name = dataset[0]

    # 模型预测
    prediction = predict(model, image)

    # 计算CMIoU
    cmiou = cmiou_calculator.calculate_ciou(mask.numpy(), prediction, class_name, dataset.class_names, embeddings)

    # 可视化结果
    plot_segmentation_result(image, mask, prediction, dataset.class_names)
    print(f"CMIoU for {class_name}: {cmiou}")

    # 比较不同模型的CMIoU（这里使用虚拟数据作为示例）
    model_names = ['SPNet', 'ZS3Net', 'LSeg']
    cmiou_scores = [0.75, 0.78, 0.82]
    plot_cmiou_comparison(model_names, cmiou_scores)

if __name__ == "__main__":
    main()