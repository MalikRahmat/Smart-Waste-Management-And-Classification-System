import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report


# =====================================================
# Base Paths Configuration
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset_split", "test")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "trained_model.h5")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
HISTORY_PATH = os.path.join(BASE_DIR, "..", "logs", "training_history.json")
GRAPH_PATH = os.path.join(BASE_DIR, "..", "logs", "training_accuracy_report.png")

CONF_MATRIX_PATH = os.path.join(LOG_DIR, "confusion_matrix.png")
METRICS_PATH = os.path.join(LOG_DIR, "evaluation_metrics.json")

os.makedirs(LOG_DIR, exist_ok=True)


# =====================================================
# Evaluation Configuration
# =====================================================

IMG_SIZE = 224
BATCH_SIZE = 16


# =====================================================
# Prepare Test Dataset
# =====================================================

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # Important for correct confusion matrix
)


# =====================================================
# Load Trained Model
# =====================================================

model = load_model(MODEL_PATH)


# =====================================================
# Generate Predictions
# =====================================================

predictions = model.predict(val_data)     # Return class probability for every image
y_pred = np.argmax(predictions, axis=1)   # To take the Maximum Probability in row
y_true = val_data.classes                 # Actual labels of test dataset
class_labels = list(val_data.class_indices.keys()) # Assigns folder names a number and make list of labels(keys)


# =====================================================
# Confusion Matrix Visualization
# =====================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.close()


# =====================================================
# Classification Report
# =====================================================

# Print report to console
print(classification_report(y_true, y_pred, target_names=class_labels))

# Save report as JSON
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    output_dict=True
)

with open(METRICS_PATH, "w") as f:
    json.dump(report_dict, f, indent=4)


print("Evaluation metrics saved to:", METRICS_PATH)
print("Confusion matrix saved to:", CONF_MATRIX_PATH)



print("Graph saved:", GRAPH_PATH)


