import numpy as np
from tasnif.calculations import get_embeddings, calculate_pca, calculate_kmeans
from tasnif.utils import read_with_pil, read_images_from_directory


def test_get_embeddings():
    # Assuming "test_images" contains at least one image for embedding generation
    image_paths = read_images_from_directory("tests/test_images")
    images = read_with_pil(image_paths)
    embeddings = get_embeddings(images=images, use_gpu=False)
    assert embeddings is not None, "Embeddings were not generated."


def test_calculate_pca():
    # Generate some random data for PCA
    embeddings = np.random.rand(20, 2048)  # Simulate 10 embeddings of dimension 2048
    pca_embeddings = calculate_pca(embeddings, pca_dim=16)
    assert pca_embeddings.shape[1] == 16, "PCA did not reduce to the correct dimension."


def test_calculate_kmeans():
    # Use PCA embeddings from the previous test or generate new random data
    pca_embeddings = np.random.rand(10, 16)  # Simulate 10 samples in 16 dimensions
    num_classes = 2
    centroid, labels, counts = calculate_kmeans(pca_embeddings, num_classes)
    assert (
        len(set(labels)) == num_classes
    ), "K-means did not cluster into the expected number of classes."
