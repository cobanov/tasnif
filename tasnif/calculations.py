import numpy as np
from img2vec_pytorch import Img2Vec
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import PCA

from .logger import info


def get_embeddings(use_gpu=False, images=None, model='resnet-18'):
    """
    This Python function initializes an Img2Vec object, runs it on either GPU or CPU, and retrieves
    image embeddings.
    :param use_gpu: The `use_gpu` parameter is a boolean that specifies whether to use GPU or CPU.
    :param images: The `images` parameter is a list of image paths to be used for generating embeddings.
    :param model: The `model` parameter is a string that specifies the model to use for generating.
        For available models, see https://github.com/christiansafka/img2vec
    :return: The function `get_embeddings` returns the embeddings of the images as np.ndarray.
    """

    info(f"Img2Vec is running on {'GPU' if use_gpu else 'CPU'}...")
    img2vec = Img2Vec(cuda=use_gpu, model=model)
    print(f"Using model: {model}")
    embeddings = img2vec.get_vec(images, tensor=False)
    return embeddings


def calculate_pca(embeddings, pca_dim):
    """
    The function `calculate_pca` takes embeddings and a desired PCA dimension as input, performs PCA
    transformation, and returns the transformed embeddings.

    :param embeddings: The `embeddings` parameter is a NumPy array containing the data points to be used
    for PCA. Each row in the array represents a data point, and the columns represent the features of
    that data point
    :param pca_dim: The `pca_dim` parameter in the `calculate_pca` function represents the desired
    dimensionality of the PCA (Principal Component Analysis) transformation. It specifies the number of
    principal components to retain after the dimensionality reduction process
    :return: The function `calculate_pca` returns the embeddings transformed using PCA with the
    specified dimensionality reduction (`pca_dim`).
    """

    n_samples, _ = embeddings.shape
    if n_samples < pca_dim:
        n_components = min(n_samples, pca_dim)
        info(
            f"Number of samples is less than the desired dimension. Setting n_components to {n_components}"
        )

    else:
        n_components = pca_dim

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings.squeeze())
    info("PCA calculated.")
    return pca_embeddings


def calculate_kmeans(pca_embeddings, num_classes, iter=10):
    """
    The function `calculate_kmeans` performs KMeans clustering on PCA embeddings data to assign
    labels and centroids.
    :param pca_embeddings: The `pca_embeddings` parameter is a NumPy array containing the data points.
    :param num_classes: The `num_classes` parameter is an integer that specifies the number of clusters.
    :param iter: The `iter` parameter is an integer that specifies the number of iterations for the KMeans algorithm. Default is 10. Should be a positive integer.
    """

    if not isinstance(pca_embeddings, np.ndarray):
        raise ValueError("pca_embeddings must be a numpy array")

    if num_classes > len(pca_embeddings):
        raise ValueError(
            "num_classes must be less than or equal to the number of samples in pca_embeddings"
        )

    try:
        centroid, labels = kmeans2(
            data=pca_embeddings, k=num_classes, minit="points", iter=iter
        )
        counts = np.bincount(labels)
        info("KMeans calculated.")
        return centroid, labels, counts

    except Exception as e:
        raise RuntimeError(f"An error occurred during KMeans processing: {e}")
