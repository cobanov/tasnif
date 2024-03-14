import os
import shutil
import warnings
from itertools import compress

import numpy as np
from tqdm import tqdm

from .calculations import calculate_kmeans, calculate_pca, get_embeddings
from .logger import error, info
from .utils import (
    create_dir,
    create_image_grid,
    read_images_from_directory,
    read_with_pil,
)

warnings.filterwarnings("ignore")


class Tasnif:
    def __init__(self, num_classes, pca_dim=16, use_gpu=False):
        self.num_classes = num_classes
        self.pca_dim = pca_dim
        self.use_gpu = use_gpu
        self.embeddings = None
        self.pca_embeddings = None
        self.centroid = None
        self.labels = None
        self.image_paths = None
        self.images = []
        self.project_name = None
        self.counts = None

    def read(self, folder_path):
        """
        This function reads images from a specified folder path using the PIL library.

        :param folder_path: The `folder_path` parameter is a string that represents the path to a
        directory containing images that you want to read and process
        """

        self.project_name = os.path.split(folder_path)[-1]
        self.image_paths = read_images_from_directory(folder_path)
        self.images = read_with_pil(self.image_paths)

    def calculate(self, pca=True, iter=10, model="resnet-18"):
        """
        The function calculates embeddings, performs PCA, and applies K-means clustering to the
        embeddings. It will not perform these operations if no images have been read.

        :param pca: The `pca` parameter is a boolean that specifies whether to perform PCA or not. Default is True
        :param iter: The `iter` parameter is an integer that specifies the number of iterations for the KMeans algorithm. Default is 10.
        :param model: The `model` parameter is a string that specifies the model to use for generating embeddings. Default is 'resnet-18'.
            For available models, see https://github.com/christiansafka/img2vec
        """

        if not self.images:
            raise ValueError(
                "The images list can not be empty. Please call the read method before calculating."
            )

        self.embeddings = get_embeddings(use_gpu=self.use_gpu, images=self.images, model=model)
        if pca:
            self.pca_embeddings = calculate_pca(self.embeddings, self.pca_dim)
            self.centroid, self.labels, self.counts = calculate_kmeans(
                self.pca_embeddings, self.num_classes, iter=iter
            )
        else:
            self.centroid, self.labels, self.counts = calculate_kmeans(
                self.embeddings, self.num_classes, iter=iter
            )

    def export(self, output_folder="./"):
        """
        Export images into separate directories based on their labels and create an image grid for each label.

        Parameters:
        - output_folder (str): The base directory to export the images and grids into.
        """

        # Create the main project directory
        project_path = os.path.join(output_folder, self.project_name)
        create_dir(project_path)

        for label_number in tqdm(range(self.num_classes)):
            label_mask = self.labels == label_number
            path_images = list(compress(self.image_paths, label_mask))
            target_directory = os.path.join(project_path, f"cluster_{label_number}")

            # Create a directory for the current label
            create_dir(target_directory)

            # Copy images to their respective label directory
            for img_path in path_images:
                try:
                    shutil.copy2(img_path, target_directory)
                except Exception as e:
                    error(f"Error copying {img_path} to {target_directory}: {e}")

            # Create an image grid for the current label
            label_images = list(compress(self.images, label_mask))
            create_image_grid(label_images, project_path, label_number)

        info(f"Exported images and grids to {project_path}")

    def export_embeddings(self, output_folder="./"):
        """
        Export the calculated embeddings to a specified output folder.

        Parameters:
        - output_folder (str): The directory to export the embeddings into.
        """

        if self.embeddings is None:
            raise ValueError(
                "Embeddings can not be empty. Please call the calculate method first."
            )

        embeddings_path = os.path.join(
            output_folder, f"{self.project_name}_embeddings.npy"
        )
        np.save(embeddings_path, self.embeddings)
        info(f"Embeddings have been saved to {embeddings_path}")
