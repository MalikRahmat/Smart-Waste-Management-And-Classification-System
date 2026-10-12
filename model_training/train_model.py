import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

# ============================
# Paths (MATCH YOUR PROJECT)
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset_split")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "trained_model.h5")
LABELS_SAVE_PATH = os.path.join(MODEL_DIR, "labels.json")
LOGS_PATH = os.path.join(LOG_DIR, "train_progress.json")
HISTORY_PATH = os.path.join(LOG_DIR, "training_history.json")
GRAPH_PATH = os.path.join(LOG_DIR, "training_accuracy_report.png")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================
# Config (CPU Friendly)
# ============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10   # stable for CPU

# ============================
# Data Generators
# ============================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ============================
# Model (Stable + Accurate)
# ============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # SAFE (no overfitting)

x = base_model.output                               # Extracts Features from images
x = GlobalAveragePooling2D()(x)                     # Calculate the feature map average
x = Dense(128, activation="relu")(x)          # 128 hidden neurons that extracts new features, relu convert negative numbers to zero
x = Dropout(0.5)(x)                                # randomely off the 50 percent neurons, so that model will not only learn the selected images
outputs = Dense(train_data.num_classes, activation="softmax")(x) # softmax for probabilities, numclasses for no. of classes

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.0007),           # adam adjust the weights if prediction is wrong
    loss="categorical_crossentropy",                # howmany predictions are wrong{calculate loss_}
    metrics=["accuracy"]
)

# ===============
# Class Weights
# ===============
classes = np.unique(train_data.classes)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_data.classes
)

class_weights = dict(zip(classes, class_weights_array))

# ============================
# Progress Logger (FRONTEND SAFE)
# ============================
class ProgressLogger(Callback):
    def __init__(self):
        super().__init__()
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1

        progress = {
            "status": "training",
            "epoch": self.current_epoch,
            "total_epochs": self.params.get("epochs"),
            "total": self.params.get("epochs")
        }

        with open(LOGS_PATH, "w") as f:
            json.dump(progress, f)

    def on_batch_end(self, batch, logs=None):
        total_batches = self.params.get("steps") or self.params.get("steps_per_epoch")

        progress = {
            "status": "batch_end",
            "epoch": self.current_epoch,
            "batch": batch + 1,
            "total_batches": total_batches,
            "total": self.params.get("epochs")
        }

        with open(LOGS_PATH, "w") as f:
            json.dump(progress, f)

    def on_epoch_end(self, epoch, logs=None):
        progress = {
            "status": "epoch_end",
            "epoch": self.current_epoch,
            "total": self.params.get("epochs"),
            "loss": float(logs.get("loss", 0)),
            "accuracy": float(logs.get("accuracy", 0)),
            "val_loss": float(logs.get("val_loss", 0)),
            "val_accuracy": float(logs.get("val_accuracy", 0))
        }

        with open(LOGS_PATH, "w") as f:
            json.dump(progress, f)

        # stop flag support
        if os.path.exists(os.path.join(LOG_DIR, "stop.flag")):
            print("Training stopped by user.")
            self.model.stop_training = True

# ============================
# Start Training
# ============================
print("\nTraining Started...\n")

initial_progress = {
    "status": "training",
    "epoch": 0,
    "total": EPOCHS,
    "message": "Training started"
}

with open(LOGS_PATH, "w") as f:
    json.dump(initial_progress, f)

early_stop = EarlyStopping(
    monitor="val_loss",   # FIXED (important)
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[ProgressLogger(), early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# ============================
# Save Model
# ============================
model.save(MODEL_SAVE_PATH)

labels = {str(v): k for k, v in train_data.class_indices.items()}    #dictionary comprehension
with open(LABELS_SAVE_PATH, "w") as f:
    json.dump(labels, f)

# ============================
# Save Training History
# ============================
history_data = {
    "accuracy": [float(x) for x in history.history.get("accuracy", [])],         #List comprehension
    "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
    "loss": [float(x) for x in history.history.get("loss", [])],
    "val_loss": [float(x) for x in history.history.get("val_loss", [])]
}

with open(HISTORY_PATH, "w") as f:
    json.dump(history_data, f)

# ============================
# Save Final Metrics (Frontend)
# ============================
final_metrics = {
    "status": "completed",
    "message": "Training completed successfully",
    "progress": 100,
    "total_epochs": EPOCHS,
    "loss": float(history.history["loss"][-1]),
    "accuracy": float(history.history["accuracy"][-1]),
    "val_loss": float(history.history["val_loss"][-1]),
    "val_accuracy": float(history.history["val_accuracy"][-1])
}

with open(LOGS_PATH, "w") as f:
    json.dump(final_metrics, f)

# SAVE BACKUP
backup_path = os.path.join(LOG_DIR, "last_completed_training.json")

with open(backup_path, "w") as f:
    json.dump(final_metrics, f)

# ============================
# Plot Graph (for report)
# ============================
plt.figure(figsize=(10, 5))

plt.plot(history_data["accuracy"], label="Train Accuracy")
plt.plot(history_data["val_accuracy"], label="Validation Accuracy")
plt.plot(history_data["loss"], label="Train Loss")
plt.plot(history_data["val_loss"], label="Validation Loss")

plt.legend()
plt.tight_layout()
plt.savefig(GRAPH_PATH)
plt.close()

print("\nTraining Completed Successfully")
print("Graph saved at:", GRAPH_PATH)