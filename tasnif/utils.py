import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from .logger import info


def read_images_from_directory(image_directory: str) -> list:
    """
    Reads all images from the given directory and returns a list of image paths.
    :param image_directory: The directory where the images are stored.
    :return: A list of image paths.
    """
    list_of_images = []
    image_extensions = ("*.gif", "*.png", "*.jpg", "*.jpeg")
    for ext in image_extensions:
        list_of_images.extend(glob.glob(os.path.join(image_directory, ext)))
    info(f"Images found: {len(list_of_images)}")
    return list_of_images


def read_with_pil(list_of_images: list, resize=True) -> list:
    """
    Reads a list of images using PIL and returns a list of PIL images.
    :param list_of_images: List of image paths.
    :param resize: If True, resize the image to 512x512. Defaults to True.
    :return: A list of PIL images.
    """
    pil_images = []
    for img_path in tqdm(list_of_images):
        img = Image.open(img_path).convert("RGB")
        if resize:
            img.thumbnail((512, 512))
        pil_images.append(img)
    info("Image reading done!")
    return pil_images


def create_image_grid(label_images, project_path, label_number):
    """
    Creates a grid of images with labels and saves it to a file.
    :param label_images: List of labeled images.
    :param label_number: The label number.
    """

    for i, image in enumerate(label_images):
        if i >= 9:
            break
        plt.subplot(3, 3, i + 1)
        plt.imshow(image, cmap="gray", interpolation="none")
        plt.title(f"Class: {label_number}")
        plt.axis("off")
        # Save the figure after creating all subplots
        plt.savefig(f"{project_path}/grid_{label_number}.png", dpi=300)


def create_dir(directory_path):
    """
    Creates a directory if it doesn't exist.
    :param directory_path: The path of the directory.
    :return: The stem of the directory path.
    """
    directory = Path(directory_path)
    if not directory.is_dir():
        directory.mkdir(exist_ok=True, parents=True)
    return directory.stem
