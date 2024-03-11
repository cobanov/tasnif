"""
Tasnif is a simple library for image classification and clustering.
"""

__author__ = "Mert Cobanov"
__author_email__ = "mertcobanov@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/cobanov/tasnif"


from tasnif.calculations import calculate_kmeans, calculate_pca, get_embeddings
from tasnif.tasnif import Tasnif
from tasnif.utils import (
    create_dir,
    create_image_grid,
    read_images_from_directory,
    read_with_pil,
)
