from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np


def compute_similarity(normalized_vector: torch.tensor, to_normalize_vector: torch.tensor) -> list:
    """
    Computes the cosine similarity between two texts.
    :param normalized_vector: vector representation of the normalized text.
    :param to_normalize_vector: vector representation of the text to normalize.
    :return: list contains indexes of the most similar sentences of the 'to_normalize_vector' in the 'normalized_vector' list.
    """
    a, b = normalized_vector.numpy(), to_normalize_vector.numpy()
    similarity_matrix = cosine_similarity(a, b)
    indices = np.flip(np.argsort(similarity_matrix, axis=-1), axis=1)[:, :1].flatten()
    return indices.tolist()
