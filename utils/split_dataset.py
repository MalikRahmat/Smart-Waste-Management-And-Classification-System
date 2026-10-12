import os
import shutil
from sklearn.model_selection import train_test_split


# =====================================================
# Base Paths Configuration
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "..", "dataset_refined")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "dataset_split")

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

SPLIT_RATIO = 0.3  # 70% training, 30% testing
RANDOM_STATE = 42


# =====================================================
# Create Output Directories
# =====================================================

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


# =====================================================
# Dataset Splitting Logic
# =====================================================

for category in os.listdir(SOURCE_DIR):
    category_path = os.path.join(SOURCE_DIR, category)

    # Skip non-directory files
    if not os.path.isdir(category_path):
        continue

    images = os.listdir(category_path)

    # Skip empty categories
    if not images:
        continue

    # Perform train-test split
    train_images, test_images = train_test_split(
        images,
        test_size=SPLIT_RATIO,
        random_state=RANDOM_STATE
    )

    # Create category folders inside train and test
    train_category_dir = os.path.join(TRAIN_DIR, category)
    test_category_dir = os.path.join(TEST_DIR, category)

    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(test_category_dir, exist_ok=True)

    # Copy training images
    for image in train_images:
        shutil.copy(
            os.path.join(category_path, image),
            os.path.join(train_category_dir, image)
        )

    # Copy testing images
    for image in test_images:
        shutil.copy(
            os.path.join(category_path, image),
            os.path.join(test_category_dir, image)
        )


print("Dataset split completed successfully (70% Train / 30% Test).")