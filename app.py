# =========================================================
# AI Smart Waste Management System - Flask Application
# Part 1: Core Setup, Public Routes, Classification APIs
# =========================================================

# ---------------------------
# Standard Library Imports
# ---------------------------
import os
import base64
import subprocess
import shutil
import time
import json
import numpy as np
from datetime import datetime

# ---------------------------
# Flask Imports
# ---------------------------
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
)
from subprocess import Popen
training_process: Popen | None = None


# ---------------------------
# Project Utility Imports
# ---------------------------
from utils.preprocess import preprocess_image, validate_image_format
from utils.classify import load_model, load_labels, predict_image
from utils.recommender import get_recommendation
from utils.admin_tools import (
    history_data,
    get_history,
    get_system_logs
)

# =========================================================
# Flask App Configuration
# =========================================================

app = Flask(__name__, template_folder="templates")

app.config["UPLOAD_FOLDER"] = os.path.join("static", "samples")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024



# =========================================================
# Load Model and Labels (Loaded Once at Startup)
# =========================================================

MODEL_PATH = os.path.join("model", "trained_model.h5")
LABELS_PATH = os.path.join("model", "labels.json")

model = load_model(MODEL_PATH)
labels = load_labels(LABELS_PATH)

# =============================
# Warm-Up Prediction
# =============================

try:
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("Model warm-up completed.")
except Exception as e:
    print("Warm-up failed:", e)

# =========================================================
# Model Reload Utility
# =========================================================

def reload_model():
    global model, labels
    try:
        model = load_model(MODEL_PATH)
        labels = load_labels(LABELS_PATH)
        print("Model reloaded successfully.")
    except Exception as e:
        print("Model reload failed:", e)



# =========================================================
# Helper Functions
# =========================================================

def save_uploaded_file(fileobj, dest_folder=None):
    """
    Save uploaded file into destination folder.
    Returns absolute file path.
    """
    dest_folder = dest_folder or app.config["UPLOAD_FOLDER"]
    os.makedirs(dest_folder, exist_ok=True)

    filename = fileobj.filename.replace(" ", "_")
    save_path = os.path.join(dest_folder, filename)

    fileobj.save(save_path)
    return save_path


def save_base64_image(data_uri, filename="captured.jpg", dest_folder=None):
    """
    Save base64 encoded image into destination folder.
    Returns absolute file path.
    """
    dest_folder = dest_folder or app.config["UPLOAD_FOLDER"]
    os.makedirs(dest_folder, exist_ok=True)

    header, encoded = data_uri.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    save_path = os.path.join(dest_folder, filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    return save_path


# =========================================================
# Public UI Routes
# =========================================================

@app.route("/", methods=["GET"])
def index():
    """Render homepage."""
    return render_template("index.html")


@app.route("/result", methods=["GET"])
def result_page():
    """Render classification result page."""

    predicted_class = request.args.get("class") or request.args.get("prediction")
    confidence = request.args.get("confidence")
    image_url = request.args.get("image") or request.args.get("image_url")

    confidence = float(confidence) if confidence else 0
    confidence = round(confidence * 100, 2)

    recommendation = (
        get_recommendation(predicted_class) if predicted_class else ""
    )

    return render_template(
        "result.html",
        predicted_class=predicted_class,
        confidence=confidence,
        image_url=image_url,
        recommendation=recommendation,
    )


@app.route("/history", methods=["GET"])
def history_page():
    """Render classification history page."""
    try:
        items = sorted(get_history(), key=lambda x: x["time"], reverse=True)
    except Exception:
        items = []

    return render_template("history.html", history=items)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:

        log_file = "logs/classification_history.json"

        # overwrite with empty list
        with open(log_file, "w") as f:
            json.dump([], f)

        return {"status": "success", "message": "History cleared"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
# =========================================================
# Classification API
# =========================================================

@app.route("/classify", methods=["POST"])
def api_classify():

    # ---------------------------
    # Case A: File Upload
    # ---------------------------
    if "file" in request.files:

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        saved_path = save_uploaded_file(file)


        try:
            img_arr, err = preprocess_image(saved_path)
        except Exception as e:
            return jsonify(
                {"error": "Preprocessing failed", "detail": str(e)}
            ), 500

        predicted_class, confidence, _ = predict_image(
            preprocessed_image=img_arr,
            filename=os.path.basename(saved_path),
            model=model,
            labels=labels
        )

        history_data(
            os.path.basename(saved_path),
            predicted_class,
            float(confidence),
        )

        return jsonify(
            {
                "file": os.path.basename(saved_path),
                "class": predicted_class,
                "confidence": float(confidence),
                "recommendation": get_recommendation(predicted_class),
                "image_url": saved_path,
            }
        )

    # ---------------------------
    # Case B: JSON File Path
    # ---------------------------
    data = request.get_json(silent=True)

    if data and data.get("filepath"):

        filepath = data["filepath"]

        if not os.path.exists(filepath):
            return jsonify(
                {"error": "File not found", "filepath": filepath}
            ), 400

        try:
            img_arr, err = preprocess_image(filepath)

            if err:
                return jsonify({"error": err}), 400

            predicted_class, confidence, _ = predict_image(
                preprocessed_image=img_arr,
                filename=os.path.basename(filepath),
                model=model,
                labels=labels
            )

            history_data(
                os.path.basename(filepath),
                predicted_class,
                float(confidence),
            )

            return jsonify(
                {
                    "file": os.path.basename(filepath),
                    "class": predicted_class,
                    "confidence": float(confidence),
                    "recommendation": get_recommendation(predicted_class),
                    "image_url": filepath,
                }
            )

        except Exception as e:
            return jsonify(
                {"error": "Processing failed", "detail": str(e)}
            ), 500


# =========================================================
# Upload API
# =========================================================

@app.route("/upload", methods=["POST"])
def api_upload():
    """Upload image and validate format."""

    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    saved_path = save_uploaded_file(file)

    is_valid, msg = validate_image_format(saved_path)
    if not is_valid:
        return jsonify({"error": msg}), 400

    return jsonify({"message": "Uploaded", "filepath": saved_path})


# =========================================================
# Capture API
# =========================================================

@app.route("/capture", methods=["POST"])
def api_capture():
    """Receive base64 image from camera."""

    data = request.json.get("image")

    if not data:
        return jsonify({"error": "No image received"}), 400

    saved = save_base64_image(
        data,
        filename=f"captured_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
    )

    return jsonify({"message": "Captured", "filepath": saved})

# =========================================================
# Part 2: Admin UI Routes and Dataset Management
# =========================================================


# =========================================================
# Admin UI Routes
# =========================================================

@app.route("/admin", methods=["GET"])
def admin_page():
    """
    Render admin dashboard.
    (Old stop/progress reset logic intentionally unchanged)
    """


    return render_template("admin.html")


@app.route("/admin/eval", methods=["GET"])
def admin_eval_page():
    """Render evaluation metrics page."""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(BASE_DIR, "logs")

    # =========================
    # LOAD EVALUATION METRICS
    # =========================
    try:
        with open(os.path.join(logs_dir, "evaluation_metrics.json")) as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

    # =========================
    # SYSTEM LOGS
    # =========================
    try:
        logs = get_system_logs()
    except:
        logs = []

    total = len(logs)
    success = len([l for l in logs if l["status"] == "Success"])
    failed = len([l for l in logs if l["status"] == "Failed"])

    avg_time = round(
        sum([l.get("processing_time", 0) for l in logs]) / total, 2
    ) if total > 0 else 0

    # =========================
    # MODEL DATA
    # =========================
    progress_file = os.path.join(logs_dir, "train_progress.json")
    backup_file = os.path.join(logs_dir, "last_completed_training.json")
    classification_file = os.path.join(logs_dir, "classification_history.json")

    total_classifications = 0
    current_accuracy = None
    validation_accuracy = None
    training_loss = None
    validation_loss = None
    total_epochs = None
    training_status = "Not Started"

    # Total Classified
    if os.path.exists(classification_file):
        with open(classification_file, "r") as f:
            total_classifications = len(json.load(f))

    progress = {}

    # Load current progress
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)

    # Load backup if needed
    if not progress or progress.get("status") in ["stopped", None]:
        if os.path.exists(backup_file):
            with open(backup_file, "r") as f:
                progress = json.load(f)

    # Extract values
    if progress:
        training_status = progress.get("status", "completed")

        acc = progress.get("accuracy")
        val_acc = progress.get("val_accuracy")
        loss = progress.get("loss")
        val_loss = progress.get("val_loss")

        if acc is not None:
            current_accuracy = round(acc * 100, 2)

        if val_acc is not None:
            validation_accuracy = round(val_acc * 100, 2)

        if loss is not None:
            training_loss = round(loss, 4)

        if val_loss is not None:
            validation_loss = round(val_loss, 4)

        total_epochs = progress.get("total_epochs") or progress.get("total", "N/A")

    # =========================
    # LAST TRAINING DATE
    # =========================
    last_training_date = None
    if os.path.exists(progress_file):
        timestamp = os.path.getmtime(progress_file)
        last_training_date = datetime.fromtimestamp(timestamp).strftime("%d %B %Y | %I:%M %p")

    # =========================
    # RENDER TEMPLATE
    # =========================
    return render_template(
        "admin_eval.html",

        # evaluation
        metrics=metrics,
        last_training_date=last_training_date,

        # system summary
        total=total,
        success=success,
        failed=failed,
        avg_time=avg_time,

        # model overview
        total_classifications=total_classifications,
        current_accuracy=current_accuracy,
        validation_accuracy=validation_accuracy,
        training_loss=training_loss,
        validation_loss=validation_loss,
        total_epochs=total_epochs,
        training_status=training_status,
    )
@app.route("/admin/confusion_matrix")
def confusion_matrix():
    """Return confusion matrix image."""

    file_path = os.path.join("logs", "confusion_matrix.png")

    if not os.path.exists(file_path):
        return "Confusion matrix not found. Generate report first.", 404

    return send_file(file_path, mimetype="image/png")

@app.route("/admin/training_report")
def training_report():
    path = os.path.join("logs", "training_accuracy_report.png")

    if not os.path.exists(path):
        return "File not found", 404

    return send_file(path, mimetype="image/png")

@app.route("/admin/report", methods=["GET"])
def admin_generate_report():
    """
    Generate evaluation report manually from admin panel.
    """

    try:
        subprocess.run(
            ["python", "evaluate_model.py"],
            cwd="model_training"
        )

        return jsonify({"message": "Report generated!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/logs")
def admin_logs():
        # =========================
        # SYSTEM LOGS (CLASSIFICATION)
        # =========================
    logs = sorted(
            get_system_logs(),
            key=lambda x: x["timestamp"],
            reverse=True
    )

        # =========================
        # RENDER TEMPLATE
        # =========================
    return render_template("admin_logs.html",logs=logs)

# =========================
# Clear Logs
# =========================
@app.route("/clear_logs", methods=["POST"])
def clear_logs():

    try:
        with open("logs/system_logs.json", "w") as f:
            json.dump([], f)

        return {"status": "success"}

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
# =========================================================
# Admin Dataset Management
# =========================================================

@app.route("/admin/dataset/add", methods=["POST"])
def admin_add_dataset():

    files = request.files.getlist("files")

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    temp_dir = os.path.join("dataset_split", f"temp_add_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)

    ALLOWED_CATEGORIES = {"metal", "plastic", "paper", "organic", "glass"}
    accepted = []
    ignored = []

    try:
        # ----------------------------------------
        # Save uploaded files temporarily
        # ----------------------------------------
        for f in files:
            rel_path = os.path.normpath(f.filename.replace("\\", "/"))

            if rel_path.startswith(".."):
                return jsonify({"error": "Invalid file path"}), 400

            abs_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            f.save(abs_path)

        # ----------------------------------------
        # Process files (MERGE mode)
        # ----------------------------------------
        for f in files:
            rel_path = os.path.normpath(f.filename.replace("\\", "/"))
            parts = rel_path.replace("\\", "/").split("/")

            # Find first allowed category
            while parts and parts[0].lower() not in ALLOWED_CATEGORIES:
                parts = parts[1:]

            if not parts:
                ignored.append(rel_path)
                continue

            category = parts[0].lower()
            if category not in ALLOWED_CATEGORIES:
                ignored.append(rel_path)
                continue

            rel_path = "/".join(parts)
            src = os.path.join(temp_dir, rel_path)
            dst = os.path.join("dataset_raw", rel_path)

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # MERGE mode → overwrite if exists, else add
            if os.path.exists(dst):
                os.remove(dst)

            shutil.move(src, dst)
            accepted.append(category)

        shutil.rmtree(temp_dir, ignore_errors=True)

        if not accepted:
            return jsonify({
                "error": "No allowed categories found. Allowed: "
                + ", ".join(ALLOWED_CATEGORIES)
            }), 400

        msg = "Dataset added successfully."
        msg += f" Accepted categories: {', '.join(set(accepted))}."
        if ignored:
            msg += f" Ignored: {', '.join(set(ignored))}."

        # ----------------------------------------
        # Refine dataset_split
        # ----------------------------------------
        refine_task = subprocess.run(
            ["python", "create_refined_dataset.py"],
            cwd="utils",
            capture_output=True,
            text=True,
        )
        if refine_task.returncode != 0:
            return jsonify({
                "error": "Dataset added but refining failed",
                "detail": refine_task.stderr
            }), 500

        # ----------------------------------------
        # Split dataset_split
        # ----------------------------------------
        split_task = subprocess.run(
            ["python", "split_dataset.py"],
            cwd="utils",
            capture_output=True,
            text=True,
        )
        if split_task.returncode != 0:
            return jsonify({
                "error": "Dataset added but splitting failed",
                "detail": split_task.stderr
            }), 500

        return jsonify({"message": msg})

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({
            "error": "Dataset add failed",
            "detail": str(e)
        }), 500


@app.route("/admin/dataset/update", methods=["POST"])
def admin_update_dataset():

    files = request.files.getlist("files")

    if not files:
        return jsonify({"error": "No files selected"}), 400

    DATASET_ROOT = "dataset_raw"
    temp_dir = os.path.join("dataset_split", f"temp_update_{int(time.time())}")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(DATASET_ROOT, exist_ok=True)

    ALLOWED_CATEGORIES = {"metal", "plastic", "paper", "organic", "glass"}
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    accepted = []
    ignored = []

    try:
        # ----------------------------------------
        # Save uploaded files temporarily
        # ----------------------------------------
        for f in files:
            rel_path = os.path.normpath(f.filename.replace("\\", "/"))

            if rel_path.startswith(".."):
                return jsonify({"error": "Invalid file path"}), 400

            abs_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            f.save(abs_path)

        # ----------------------------------------
        # Process files (UPDATE mode)
        # ----------------------------------------
        for f in files:
            rel_path = os.path.normpath(f.filename.replace("\\", "/"))
            parts = rel_path.replace("\\", "/").split("/")

            # Find first allowed category
            while parts and parts[0].lower() not in ALLOWED_CATEGORIES:
                parts = parts[1:]

            if not parts:
                ignored.append(rel_path)
                continue

            category = parts[0].lower()

            if category not in ALLOWED_CATEGORIES:
                ignored.append(rel_path)
                continue

            ext = os.path.splitext(parts[-1])[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                ignored.append(rel_path)
                continue

            rel_path = "/".join(parts)
            src = os.path.join(temp_dir, rel_path)
            dst = os.path.join(DATASET_ROOT, rel_path)

            if not os.path.exists(src):
                ignored.append(rel_path)
                continue

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # UPDATE mode → overwrite existing file, add if missing
            if os.path.exists(dst):
                os.remove(dst)

            shutil.move(src, dst)
            accepted.append(category)

        shutil.rmtree(temp_dir, ignore_errors=True)

        if not accepted:
            return jsonify({
                "error": "No allowed categories found. Allowed: "
                + ", ".join(ALLOWED_CATEGORIES)
            }), 400

        msg = "Dataset updated successfully."
        msg += f" Accepted categories: {', '.join(sorted(set(accepted)))}."
        if ignored:
            msg += f" Ignored: {', '.join(sorted(set(ignored)))}."

        # -------------------------------------------------
        # Step 4: Refine dataset_split
        # -------------------------------------------------
        refine_task = subprocess.run(
            ["python", "create_refined_dataset.py"],
            cwd="utils",
            capture_output=True,
            text=True,
        )

        if refine_task.returncode != 0:
            return jsonify({
                "error": "Dataset updated but refining failed",
                "detail": refine_task.stderr
            }), 500

        # -------------------------------------------------
        # Step 5: Split dataset_split
        # -------------------------------------------------
        split_task = subprocess.run(
            ["python", "split_dataset.py"],
            cwd="utils",
            capture_output=True,
            text=True,
        )

        if split_task.returncode != 0:
            return jsonify({
                "error": "Dataset updated but splitting failed",
                "detail": split_task.stderr
            }), 500

        return jsonify({
            "message": msg,
            "ignored": list(set(ignored))
        }), 200

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({
            "error": "Failed to update dataset_split",
            "detail": str(e)
        }), 500
# =========================================================
# Part 3: Training Management, Monitoring & System Control
# =========================================================


# =========================================================
# Training Management
# =========================================================

@app.route("/admin/retrain", methods=["POST"])
def retrain():
    """
    Start model retraining process.
    Cleans previous flags and launches training in background.
    """

    global training_process

    try:
        # ----------------------------------------
        # Clean old stop flag and progress file
        # ----------------------------------------
        if os.path.exists("logs/stop.flag"):
            os.remove("logs/stop.flag")

        open("logs/train_progress.json", "w").write("{}")

        # ----------------------------------------
        # Terminate previous training process (if any)
        # ----------------------------------------
        if training_process:
            training_process.terminate()
            training_process.wait(timeout=5)

            if training_process.poll() is None:
                training_process.kill()

        # ----------------------------------------
        # Start new training process
        # ----------------------------------------
        print("Starting Model Training...")
        training_process = subprocess.Popen(
            ["python", "train_model.py"],
            cwd="model_training"
        )

        return jsonify({"message": "Training started"}), 200

    except Exception as e:
        return jsonify({
            "error": "System failure",
            "detail": str(e)
        }), 500


@app.route("/admin/stop_training", methods=["POST"])
def stop_training():
    """
    Stop ongoing training process gracefully.
    """

    global training_process

    if training_process:

        training_process.terminate()
        training_process.kill()
        training_process = None

        with open("logs/stop.flag", "w") as f:
            f.write("stop")

        progress_file = "logs/train_progress.json"
        backup_file = "logs/last_completed_training.json"

        # load backup (last completed training)
        if os.path.exists(backup_file):
            with open(backup_file, "r") as f:
                backup = json.load(f)

            # restore old data + mark stopped
            backup["status"] = "stopped"
            backup["message"] = "Training stopped by user"

            with open(progress_file, "w") as f:
                json.dump(backup, f)

        return jsonify({"message": "Training stopped"})

    else:
        return jsonify({"error": "No training process running"}), 400


# =========================================================
# Health & Monitoring APIs
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    """
    System health check endpoint.
    """

    return jsonify({
        "status": "ok",
        "model_loaded": bool(model is not None)
    })


@app.route("/progress", methods=["GET"])
def progress():
    """
    Return current training progress from JSON file.
    """

    try:
        data = json.load(open("logs/train_progress.json"))
        if data.get("status") == "completed":
            reload_model()
    except Exception:
        data = {}

    return jsonify(data)


# =========================================================
# Application Entry Point
# =========================================================

if __name__ == "__main__":
    app.run(debug=True , use_reloader=False)