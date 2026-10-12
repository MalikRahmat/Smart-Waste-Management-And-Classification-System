import os
import cv2
import numpy as np
import imghdr


# =====================================================
# Base Paths Configuration
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "..", "dataset_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "dataset_refined")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# Configuration Constants
# =====================================================

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
VALID_HEADER_TYPES = ['jpeg', 'png', 'bmp', 'tiff', 'webp']

MIN_WIDTH = 64
MIN_HEIGHT = 64
BLUR_THRESHOLD = 70
TARGET_SIZE = (224, 224)


# =====================================================
# Validation Utilities
# =====================================================

def validate_image_format(file_path):
    """
    Validate image using both file extension and file header.
    """
    extension = os.path.splitext(file_path)[1].lower().replace('.', '')
    header_type = imghdr.what(file_path)

    if header_type not in VALID_HEADER_TYPES and extension not in ALLOWED_EXTENSIONS:
        return False

    return True


def is_blurry(image):
    """
    Detect blur using variance of Laplacian method.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
    return focus_measure < BLUR_THRESHOLD


# =====================================================
# Image Preprocessing Pipeline
# =====================================================

def preprocess_and_save(source_path, destination_path):
    """
    Validate, preprocess and save image into refined dataset_split.
    """

    # 1. Validate format
    if not validate_image_format(source_path):
        return False

    # 2. Safe image loading
    image = cv2.imread(source_path)
    if image is None:
        return False

    height, width = image.shape[:2]

    # 3. Resolution check
    if height < MIN_HEIGHT or width < MIN_WIDTH:
        return False

    # 4. Blur detection
    if is_blurry(image):
        return False

    # 5. Resize image
    image = cv2.resize(image, TARGET_SIZE)

    # 6. Normalize to range [0, 1]
    image = image.astype('float32') / 255.0

    # 7. Convert back to uint8 and save
    cv2.imwrite(destination_path, (image * 255).astype(np.uint8))

    return True


# =====================================================
# Dataset Refinement Execution
# =====================================================

for category in os.listdir(SOURCE_DIR):
    category_path = os.path.join(SOURCE_DIR, category)

    # Skip non-directory files
    if not os.path.isdir(category_path):
        continue

    refined_category_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(refined_category_path, exist_ok=True)

    for image_name in os.listdir(category_path):
        source_image_path = os.path.join(category_path, image_name)
        destination_image_path = os.path.join(refined_category_path, image_name)

        success = preprocess_and_save(source_image_path, destination_image_path)

        if not success:
            print(f"Skipped: {source_image_path}")


print("Refined dataset_split created successfully in 'dataset_refined/'")