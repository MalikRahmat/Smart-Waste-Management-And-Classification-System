import cv2
import numpy as np
import imghdr
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


ALLOWED_EXT = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']


def validate_image_format(file_path):
    """
    Checks if the file is a valid image using both extension and file header.
    """
    # Check extension first
    ext_from_path = file_path.split('.')[-1].lower()

    # Check magic bytes using imghdr
    header_ext = imghdr.what(file_path)

    # Note: imghdr identifies .jpg as 'jpeg'
    valid_header_types = ['jpeg', 'png', 'bmp', 'tiff', 'webp']

    if header_ext not in valid_header_types and ext_from_path not in ALLOWED_EXT:
        return False, f"Format {ext_from_path} is not supported. Use JPG, PNG, BMP, TIFF, or WEBP."

    return True, None



def is_blurry(image):
    """
    Detects blurry images using variance of Laplacian.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < 30  # Threshold for blurriness


def preprocess_image(file_path):
    """
    Full preprocessing pipeline:
    - Validate file format
    - Load image safely
    - Check resolution
    - Check blurriness
    - Resize to 224x224
    - Normalize values
    - Return processed image array
    """

    # -------- 1. Validate format --------
    is_valid, msg = validate_image_format(file_path)
    if not is_valid:
        return None, msg

    # -------- 2. Load image safely --------
    try:
        image = cv2.imread(file_path)
        if image is None:
            return None, "Unable to read image. File may be corrupted."
    except:
        return None, "Corrupted image or unsupported format."

    h, w = image.shape[:2]

    # -------- 3. Check resolution --------
    if h < 64 or w < 64:
        return None, "Image is too small. Please upload a clearer image."

    # -------- 4. Blurriness check --------
    if is_blurry(image):
        return None, "Image is blurry. Please upload a clearer picture."

    # -------- 5. Resize image --------
    image = cv2.resize(image, (224, 224))

    # -------- 6. Normalize pixel values --------
    # -------- 6. MobileNetV2 preprocessing --------
    image = image.astype('float32')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)

    # Add batch dimension: (224,224,3) → (1,224,224,3)
    image = np.expand_dims(image, axis=0)

    return image, None
