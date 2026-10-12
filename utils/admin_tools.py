import json, os, datetime

# =========================
# History
# =========================

HISTORY = "logs/classification_history.json"

def history_data(filename, prediction, confidence):
    entry = {
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100,2),
        "time": datetime.datetime.now().strftime("%m/%d/%y %I:%M %p")
    }

    data = []
    if os.path.exists(HISTORY):
        with open(HISTORY, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []

    data.append(entry)

    with open(HISTORY, "w") as f:
        json.dump(data, f, indent=4)


def get_history():
    if not os.path.exists(HISTORY):
        return []
    with open(HISTORY, "r") as f:
        return json.load(f)

# =========================
# SYSTEM LOGS (Show Logs)
# =========================

SYSTEM_LOG_FILE = "logs/system_logs.json"

def log_system_event(filename, prediction, confidence, status="Success", processing_time=0):
    entry = {
        "timestamp": datetime.datetime.now().strftime("%m/%d/%y %I:%M %p"),
        "image": filename,
        "predicted_class": prediction,
        "confidence": confidence,
        "status": status,
        "processing_time": processing_time
    }

    data = []
    if os.path.exists(SYSTEM_LOG_FILE):
        with open(SYSTEM_LOG_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []

    data.append(entry)

    with open(SYSTEM_LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)


def get_system_logs():
    if not os.path.exists(SYSTEM_LOG_FILE):
        return []
    with open(SYSTEM_LOG_FILE, "r") as f:
        return json.load(f)


