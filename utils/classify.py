import os
import json
import numpy as np
import tensorflow as tf
import time
from utils.admin_tools import log_system_event
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_preprocess

# Path variables
DEFAULT_MODEL_PATH = os.path.join("model", "trained_model.h5")
DEFAULT_LABELS_PATH = os.path.join("model", "labels.json")

FALLBACK_LABELS = ["organic", "plastic", "glass", "metal", "paper", "other"]

_MODEL = None
_LABELS = None


def load_labels(labels_path=DEFAULT_LABELS_PATH):
    global _LABELS
    if _LABELS is not None:
        return _LABELS

    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            _LABELS = json.load(f)
            if isinstance(_LABELS, dict):
                try:
                    ordered = [_LABELS[str(i)] for i in range(len(_LABELS))]
                    _LABELS = ordered
                except Exception:
                    pass
    else:
        _LABELS = FALLBACK_LABELS
    return _LABELS


def build_default_model(num_classes=None):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=(224, 224, 3), pooling="avg", weights="imagenet"
    )
    x = base.output
    x = tf.keras.layers.Dropout(0.3)(x)
    if num_classes is None:
        num_classes = len(load_labels())
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=base.input, outputs=output)
    return model


def load_model(model_path=DEFAULT_MODEL_PATH):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if os.path.exists(model_path):
        try:
            _MODEL = tf.keras.models.load_model(model_path)
            return _MODEL
        except Exception as ex:
            raise RuntimeError(f"Failed to load model from {model_path}: {ex}")
    else:
        labels = load_labels()
        _MODEL = build_default_model(num_classes=len(labels))
        return _MODEL


def predict_image(
    preprocessed_image,
    filename="uploaded_image",
    model=None,
    labels=None,
    model_path=DEFAULT_MODEL_PATH,
    labels_path=DEFAULT_LABELS_PATH
):
    """
    Optimized: Use preloaded model if provided, else fallback to loading
    """

    # ✅ USE PRELOADED MODEL (FAST)
    if model is not None and labels is not None:
        pass
    else:
        labels = load_labels(labels_path)
        model = load_model(model_path)

    # Convert dict → ordered list
    if isinstance(labels, dict):
        labels = [labels[str(i)] for i in range(len(labels))]

    try:
        # =========================
        # TIME START
        # =========================
        start = time.time()

        preds = model.predict(preprocessed_image)

        # =========================
        # TIME END
        # =========================
        end = time.time()
        processing_time = round(end - start, 2)

        probs = preds[0].tolist()

        idx = int(np.argmax(probs))
        best_confidence = float(probs[idx])

        CONFIDENCE_THRESHOLD = 0.60
        if best_confidence < CONFIDENCE_THRESHOLD:
            best_label = "other"
        else:
            best_label = labels[idx]

        # =========================
        # LOG SYSTEM EVENT
        # =========================
        log_system_event(
            filename=filename,
            prediction=best_label,
            confidence=best_confidence,
            status="Success",
            processing_time=processing_time
        )

        return best_label, best_confidence, probs

    except Exception:
        log_system_event(
            filename=filename,
            prediction="Unknown",
            confidence=0,
            status="Failed",
            processing_time=0
        )
        raise

