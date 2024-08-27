import numpy as np
from typing import List, Dict
from nltk.corpus import wordnet as wn


def word_embedding_similarity(
    word1: str, word2: str, embeddings: Dict[str, np.ndarray]
) -> float:
    """
    使用词嵌入计算语义相似度

    参数:
    word1, word2: 需要计算相似度的两个词
    embeddings: 词嵌入字典，键为词，值为嵌入向量

    返回:
    float: 相似度得分
    """
    if word1 not in embeddings or word2 not in embeddings:
        return 0.0
    vec1, vec2 = embeddings[word1], embeddings[word2]
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def knowledge_graph_similarity(
    word1: str, word2: str, graph: Dict[str, List[str]]
) -> float:
    """
    使用知识图谱计算语义相似度

    参数:
    word1, word2: 需要计算相似度的两个词
    graph: 知识图谱，键为词，值为相关词列表

    返回:
    float: 相似度得分
    """
    if word1 not in graph or word2 not in graph:
        return 0.0

    related_words1 = set(graph[word1])
    related_words2 = set(graph[word2])

    jaccard_similarity = len(related_words1.intersection(related_words2)) / len(
        related_words1.union(related_words2)
    )
    return jaccard_similarity


def ontology_similarity(word1: str, word2: str, ontology: Dict = None) -> float:
    """
    使用WordNet本体计算语义相似度

    参数:
    word1, word2: 需要计算相似度的两个词
    ontology: 未使用，保留参数以保持一致性

    返回:
    float: 相似度得分
    """
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    if not synsets1 or not synsets2:
        return 0.0

    max_similarity = 0.0
    for s1 in synsets1:
        for s2 in synsets2:
            similarity = s1.path_similarity(s2)
            if similarity and similarity > max_similarity:
                max_similarity = similarity

    return max_similarity
