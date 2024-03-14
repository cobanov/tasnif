import numpy as np

from tasnif.calculations import calculate_kmeans, calculate_pca, get_embeddings
from tasnif.utils import read_images_from_directory, read_with_pil


def test_get_embeddings():
    image_paths = read_images_from_directory("tests/test_images")
    images = read_with_pil(image_paths)
    embeddings = get_embeddings(images=images, use_gpu=False, model="resnet-18")
    assert embeddings is not None, "Embeddings were not generated."


def test_calculate_pca():
    embeddings = np.random.rand(20, 2048)
    pca_embeddings = calculate_pca(embeddings, pca_dim=16)
    assert pca_embeddings.shape[1] == 16, "PCA did not reduce to the correct dimension."


def test_calculate_kmeans():
    pca_embeddings = np.random.rand(10, 16)
    num_classes = 2
    centroid, labels, counts = calculate_kmeans(pca_embeddings, num_classes, iter=10)
    assert (
        len(set(labels)) == num_classes
    ), "K-means did not cluster into the expected number of classes."
