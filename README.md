
<p align="center">
  <img src="assets/asdd.png" width="350" alt="accessibility text">
</p>


Tasnif is a Python package designed for clustering images into user-defined classes based on their visual content. It utilizes deep learning to generate image embeddings, Principal Component Analysis (PCA) for dimensionality reduction, and K-means for clustering. Tasnif supports processing on both GPU and CPU, making it versatile for different computational environments.

## Features

- Generate embeddings for images using a pre-trained model.
- Dimensionality reduction using PCA to enhance clustering performance.
- Clustering of images into user-specified classes with K-means.
- Visualization support by creating image grids for each cluster.
- Efficient image reading and preprocessing utilities.

## Installation

To install Tasnif, you need Python 3.6 or later. Clone this repository to your local machine and install the required dependencies:

```bash
pip install tasnif
```

## Usage

Import `Tasnif` and initialize it with the desired number of classes, PCA dimensions, and whether to use GPU:

```python
from tasnif import Tasnif

# Initialize Tasnif with 5 classes, PCA dimensions set to 16, and GPU usage
classifier = Tasnif(num_classes=5, pca_dim=16, use_gpu=False)
```

Read the images from a directory, calculate the embeddings, PCA, and perform K-means clustering:

```python
# Read images from a specified directory
classifier.read('path/to/your/images')

# Calculate embeddings, PCA, and perform clustering
classifier.calculate()
```

Finally, export the clustered images and visualization grids to a specified directory:

```python
# Export clustered images and grids
classifier.export('path/to/output')
```

## To-Do

- [x] Prevent calculation if there is no image read (PCA & k-means)
- [x] Export embeddings
- [ ] Make model independent from img2vec
- [ ] Separate cpu and gpu installation and catch gpu errors


## Contributing

Contributions to `Tasnif` are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

Tasnif is released under the MIT License. See the [LICENSE](LICENSE) file for more details.
