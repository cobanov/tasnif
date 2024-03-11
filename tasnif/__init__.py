"""
Tasnif is a simple library for image classification and clustering.
"""

__version__ = "0.1.0"
__author__ = "Mert Cobanov"
__author_email__ = "mertcobanov@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/cobanov/tasnif"


from tasnif.tasnif import Tasnif
from tasnif.calculations import get_embeddings, calculate_pca, calculate_kmeans
from tasnif.utils import (
    read_images_from_directory,
    read_with_pil,
    create_dir,
    create_image_grid,
)
