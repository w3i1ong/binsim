import matplotlib.pyplot as plt
from typing import List
import numpy as np


def visualize_token_embeddings(embeddings: np.ndarray,
                               labels: List[str],
                               figure_size=(10, 10),
                               title: str = None,
                               save_path: str = None,
                               reduction='tsne'):
    if reduction == 'tsne':
        from sklearn.manifold import TSNE
        embeddings = TSNE(n_components=2).fit_transform(embeddings)
    elif reduction == 'pca':
        from sklearn.decomposition import PCA
        embeddings = PCA(n_components=2).fit_transform(embeddings)
    else:
        raise ValueError(f'Unknown reduction method {reduction}')
    plt.figure(figsize=figure_size)
    for label, coord in zip(labels, embeddings):
        plt.text(coord[0], coord[1], label)
    plt.xlim((np.min(embeddings[:, 0]), np.max(embeddings[:, 0])))
    plt.ylim((np.min(embeddings[:, 1]), np.max(embeddings[:, 1])))
    if save_path:
        plt.savefig(save_path)
    plt.show()
