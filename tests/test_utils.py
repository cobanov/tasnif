import os
from tasnif.utils import read_images_from_directory, read_with_pil


def test_read_images_from_directory():
    # Assuming you have a directory named "test_images" with some image files
    images = read_images_from_directory("tests/test_images")
    assert len(images) > 0, "No images found in the directory."


def test_read_with_pil():
    # Use the same directory as above or make sure it contains at least one image
    image_paths = read_images_from_directory("tests/test_images")
    images = read_with_pil(image_paths)
    assert len(images) == len(image_paths), "Not all images were read successfully."
