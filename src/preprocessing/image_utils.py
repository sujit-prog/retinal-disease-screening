"""
Image loading utilities for the retinal disease screening project.
Reads image files from disk, resizes them, and scales pixel values to 0..1.
This is the only module allowed to directly read image files from the data folder.
"""

from pathlib import Path
import cv2
import numpy as np

import config


def load_image(relative_path):
    """
    Load an image from a path relative to RAW_DIR, resize, scale, and return as RGB.
    
    Why resize and scaling must be exactly the same:
    The neural network learns spatial patterns at a specific resolution and pixel intensity range.
    Training, validation, test images, and any future images must undergo the exact same
    preprocessing so that the model receives consistent inputs.
    """
    full_path = config.RAW_DIR / relative_path

    # Read the image from disk using OpenCV (loads in BGR channel order)
    bgr_img = cv2.imread(str(full_path))
    if bgr_img is None:
        raise FileNotFoundError(f"Could not read image file at: {full_path}")

    # Convert color channels from OpenCV BGR order to standard RGB order
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Note: A plain resize squeezes the image slightly when the original is not square.
    # We accept this slight distortion because it is applied identically to every image
    # across all classes without introducing artificial border padding.
    resized_img = cv2.resize(rgb_img, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Scale pixel intensities from integer [0, 255] to float32 [0.0, 1.0]
    scaled_img = resized_img.astype(np.float32) / 255.0

    return scaled_img
