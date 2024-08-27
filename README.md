# CMIoU Metric

## Overview

CMIoU (Concept-calibrated Mean Intersection over Union) Metric is a novel evaluation metric for open-vocabulary semantic segmentation models. This project provides tools to calculate CMIoU, compare it with traditional mIoU, and evaluate different semantic similarity methods.

## Features

- Support for multiple datasets: PASCAL VOC, Cityscapes, ADE20K
- Implementation of various semantic similarity calculation methods: word embeddings, knowledge graphs, ontologies
- Model evaluation and result visualization tools
- Performance assessment in different open-vocabulary scenarios

## Installation

1. Clone the repository:

```bash
git clone https://github.com/chenxingqiang/cmiou_metric.git
cd cmiou_metric
```

2. Create and activate a virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the project:

```bash
pip install -e .
```

## Usage

### Evaluating Models

To evaluate models and generate results, use the following command:

```bash
python scripts/evaluate_models.py --dataset pascal_voc --data_dir /path/to/pascal_voc --similarity_method word_embedding --similarity_data_path /path/to/word_embeddings.txt --model_dir /path/to/pretrained_models
```

Parameters:
- `--dataset`: Choose dataset (pascal_voc, cityscapes, ade20k)
- `--data_dir`: Path to the dataset directory
- `--similarity_method`: Choose semantic similarity method (word_embedding, knowledge_graph, ontology)
- `--similarity_data_path`: Path to semantic similarity data file (not required for ontology method)
- `--model_dir`: Directory containing pretrained models

### Calculating CMIoU

To calculate CMIoU for a single dataset, use:

```bash
python scripts/calculate_cmiou.py --dataset cityscapes --data_dir /path/to/cityscapes --similarity_method knowledge_graph --similarity_data_path /path/to/knowledge_graph.json --model_path /path/to/model.pth
```

## Project Structure

```
cmiou_metric/
│
├── cmiou_metric/
│   ├── modeling/
│   │   ├── cmiou.py
│   │   ├── semantic_similarity.py
│   │   └── models.py
│   ├── data/
│   │   └── dataset.py
│   └── utils/
│       ├── evaluation.py
│       ├── transforms.py
│       └── visualizer.py
│
├── scripts/
│   ├── calculate_cmiou.py
│   └── evaluate_models.py
│
├── tests/
│   └── ...
│
├── examples/
│   └── example_usage.py
│
├── requirements.txt
├── setup.py
└── README.md
```

## Contributing

Contributions are welcome! Feel free to submit a Pull Request or open an issue to discuss new features and improvements.

## Citation

If you use this project in your research, please cite our paper:

```
@article{author2023cmiou,
  title={Towards Unified Open-Vocabulary Segmentation Metric via Concept Calibration},
  author={Author, xingqiang,chen.},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.