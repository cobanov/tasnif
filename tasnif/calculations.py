import numpy as np
from img2vec_pytorch import Img2Vec
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import PCA


def get_embeddings(use_gpu=False, images=None):
    """
    This Python function initializes an Img2Vec object, runs it on either GPU or CPU, and retrieves
    image embeddings.
    """

    print(f"Img2Vec is running on {'GPU' if use_gpu else 'CPU'}...")
    img2vec = Img2Vec(cuda=use_gpu)

    embeddings = img2vec.get_vec(images, tensor=False)
    return embeddings


def calculate_pca(embeddings, pca_dim):
    print("Calculating PCA")
    n_samples, n_features = embeddings.shape
    if n_samples < pca_dim:
        n_components = min(n_samples, pca_dim)
        print(
            f"Number of samples is less than the desired dimension. Setting n_components to {n_components}"
        )

    else:
        n_components = pca_dim

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings.squeeze())
    print("PCA calculating done!")
    return pca_embeddings


def calculate_kmeans(pca_embeddings, num_classes):
    """
    The function `calculate_kmeans` performs KMeans clustering on PCA embeddings data to assign
    labels and centroids.
    """
    print("KMeans processing...")
    if not isinstance(pca_embeddings, np.ndarray):
        raise ValueError("pca_embeddings must be a numpy array")

    if num_classes > len(pca_embeddings):
        raise ValueError(
            "num_classes must be less than or equal to the number of samples in pca_embeddings"
        )

    try:
        centroid, labels = kmeans2(data=pca_embeddings, k=num_classes, minit="points")
        counts = np.bincount(labels)
        return centroid, labels, counts
    except Exception as e:
        raise RuntimeError(f"An error occurred during KMeans processing: {e}")
