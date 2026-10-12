
/* =====================================================
   DOM REFERENCES
===================================================== */

// ===== Training Elements =====
const trainStats = document.getElementById("trainStats");
const progressBox = document.getElementById("trainProgressBox");
const progressBar = document.getElementById("trainProgress");
const percentText = document.getElementById("trainPercent");
const adminMsg = document.getElementById("adminMsg");
const retrainBtn = document.getElementById("retrainBtn");
const stopTrainingBtn = document.getElementById("stopTrainingBtn");

// ===== File Upload =====
const fileInput = document.getElementById("fileInput");
const uploadForm = document.getElementById("uploadForm");

// ===== Camera Elements =====
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const cameraStream = document.getElementById("cameraStream");
const cameraCanvas = document.getElementById("cameraCanvas");
const cameraStatus = document.getElementById("cameraStatus");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const removeCaptureBtn = document.getElementById("removeCaptureBtn");

// ===== Dataset Elements =====
const addDatasetBtn = document.getElementById("addDatasetBtn");
const updateDatasetBtn = document.getElementById("updateDatasetBtn");
const datasetInput = document.getElementById("datasetInput");
const addLoader = document.getElementById("addLoader");
const updateLoader = document.getElementById("updateLoader");


/* =====================================================
   HELPER FUNCTIONS
===================================================== */

/**
 * Update UI status messages safely
 */
function setStatus(element, text, className = "") {
  if (!element) return;

  element.classList.remove("error", "warn", "success");

  if (className) {
    element.classList.add(className);
  }

  element.textContent = text || "";
}

/**
 * Remove captured image preview
 */
function removeCapturedImage() {
  const img = document.getElementById("capturedPreview");

  if (img) img.style.display = "none";
  if (removeCaptureBtn) removeCaptureBtn.style.display = "none";

  if (cameraStatus) {
    cameraStatus.textContent = "Image removed. Capture new image.";
  }
}


/* =====================================================
   TRAINING START
===================================================== */

if (retrainBtn) {
  retrainBtn.onclick = () => {

    if (progressBox) progressBox.style.display = "block";
    if (progressBar) progressBar.value = 1;
    if (percentText) percentText.textContent = "Starting Training...";
    if (stopTrainingBtn) stopTrainingBtn.style.display = "inline-block";

    setStatus(adminMsg, "Training started...", "warn");

    fetch("/admin/retrain", { method: "POST" })
      .then(res => res.json())
      .then(json => {

        if (json.error) {
          setStatus(adminMsg, json.error, "error");
        } else {
          checkTrainingProgress();
        }

      })
      .catch(() => {
        setStatus(adminMsg, "Failed to start training.", "error");
      });
  };
}


/* =====================================================
   FILE UPLOAD PREVIEW
===================================================== */

if (fileInput) {
  fileInput.addEventListener("change", () => {

    const file = fileInput.files?.[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = (e) => {
      const preview = document.getElementById("previewImage");

      if (preview) {
        preview.src = e.target.result;
        preview.style.display = "block";
      }
    };

    reader.readAsDataURL(file);
  });
}


/* =====================================================
   UPLOAD & CLASSIFICATION FLOW
===================================================== */

if (uploadForm) {
  uploadForm.addEventListener("submit", async (e) => {

    e.preventDefault();

    const resultBox = document.getElementById("resultBox");

    // Validate file selection
    if (!fileInput.files || fileInput.files.length === 0) {
      setStatus(resultBox, "Please select an image.", "error");
      return;
    }

    const file = fileInput.files[0];

    // Validate file type
    const validTypes = [
      "image/jpeg",
      "image/png",
      "image/jpg",
      "image/bmp",
      "image/tiff",
      "image/webp"
    ];

    if (!validTypes.includes(file.type)) {
      setStatus(
        resultBox,
        "Invalid image format. Only JPG, JPEG, PNG, BMP, TIFF, WEBP are allowed.",
        "error"
      );
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Upload image
      const uploadRes = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      const uploadJson = await uploadRes.json();

      if (!uploadRes.ok || uploadJson.error) {
        setStatus(resultBox, uploadJson?.error || "Upload failed.", "error");
        return;
      }

      resultBox.innerHTML = "<div class='loader'></div><p>Classifying...</p>";

      // Classify image
      const classifyRes = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath: uploadJson.filepath })
      });

      const classifyJson = await classifyRes.json();

      if (!classifyRes.ok || classifyJson.error) {
        setStatus(resultBox, classifyJson?.error || "Classification failed.", "error");
        return;
      }

      // Redirect to result page
      const query = new URLSearchParams({
        class: classifyJson.class,
        confidence: classifyJson.confidence,
        image: classifyJson.image_url
      }).toString();

      window.location.href = `/result?${query}`;

    } catch {
      setStatus(resultBox, "Network error occurred.", "error");
    }

  });
}

/* =====================================================
   CAMERA CAPTURE & CLASSIFICATION
===================================================== */

if (startCameraBtn && captureBtn && cameraStream && cameraCanvas) {

  // Start camera
  startCameraBtn.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false
      });

      cameraStream.srcObject = stream;

      captureBtn.disabled = false;
      startCameraBtn.disabled = true;
      startCameraBtn.textContent = "Camera Active";

      setStatus(cameraStatus, "Camera ready.");

      if (stopCameraBtn) {
        stopCameraBtn.style.display = "inline-block";
      }

    } catch {
      setStatus(cameraStatus, "Unable to access camera. Check permissions.", "error");
    }
  });

  // Stop camera
  if (stopCameraBtn) {
    stopCameraBtn.addEventListener("click", () => {

      const stream = cameraStream.srcObject;

      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        cameraStream.srcObject = null;
        setStatus(cameraStatus, "Camera stopped.", "warn");
      }

      stopCameraBtn.style.display = "none";
      startCameraBtn.disabled = false;
      startCameraBtn.textContent = "Start camera";
      captureBtn.disabled = true;
    });
  }

  // Capture and classify image
  captureBtn.addEventListener("click", async () => {

    try {
      cameraStatus.innerHTML = "<div class='loader'></div><p>Capturing...</p>";

      const width = cameraStream.videoWidth || 640;
      const height = cameraStream.videoHeight || 480;

      cameraCanvas.width = width;
      cameraCanvas.height = height;

      const ctx = cameraCanvas.getContext("2d");
      ctx.drawImage(cameraStream, 0, 0, width, height);

      const dataUrl = cameraCanvas.toDataURL("image/jpeg", 0.9);

      // Preview captured image
      const preview = document.getElementById("capturedPreview");

      if (preview) {
        preview.src = dataUrl;
        preview.style.display = "block";

        if (removeCaptureBtn) {
          removeCaptureBtn.style.display = "inline-block";
        }
      }

      // Send captured image to backend
      const captureRes = await fetch("/capture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl })
      });

      const captureJson = await captureRes.json();

      if (!captureRes.ok || captureJson.error) {
        setStatus(cameraStatus, captureJson?.error || "Capture failed.", "error");
        return;
      }

      cameraStatus.innerHTML =
        "<div class='loader'></div><p>Classifying captured image...</p>";

      // Classification
      const classifyRes = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath: captureJson.filepath })
      });

      const classifyJson = await classifyRes.json();

      if (!classifyRes.ok || classifyJson.error) {
        setStatus(cameraStatus, classifyJson?.error || "Classification failed.", "error");
        return;
      }

      // Redirect to result page
      const query = new URLSearchParams({
        class: classifyJson.class,
        confidence: classifyJson.confidence,
        image: classifyJson.image_url
      }).toString();

      window.location.href = `/result?${query}`;

    } catch {
      setStatus(cameraStatus, "Unexpected error. Try again.", "error");
    }

  });
}


/* =====================================================
   STOP TRAINING
===================================================== */

if (stopTrainingBtn) {
  stopTrainingBtn.addEventListener("click", async () => {

    setStatus(adminMsg, "Stopping training...");

    try {
      const res = await fetch("/admin/stop_training", { method: "POST" });
      const json = await res.json();

      if (!res.ok || json.error) {
        setStatus(adminMsg, json?.error || "Stop failed.", "error");
      } else {

        setStatus(adminMsg, json.message || "Training stopped.", "warn");

        if (percentText) percentText.textContent = "Stopped";
        if (progressBar) progressBar.value = 0;

        stopTrainingBtn.style.display = "none";
        if (progressBox) progressBox.style.display = "none";
      }

    } catch {
      setStatus(adminMsg, "Network error during stop.", "error");
    }

  });
}


/* =====================================================
   DATASET HANDLING
===================================================== */

function handleDataset(type) {

  const datasetMsg = document.getElementById("datasetMsg");

  datasetInput.onchange = async () => {

    // Validate selection
    if (!datasetInput.files || datasetInput.files.length === 0) {
      setStatus(datasetMsg, "No files selected.", "warn");
      return;
    }

    // Validate folder structure
    const categories = ["metal", "plastic", "paper", "organic", "glass"];
    const paths = Array.from(datasetInput.files).map(
      f => f.webkitRelativePath || f.name
    );

    const topLevels = new Set(
      paths.map(p => p.split("/")[0].toLowerCase())
    );

    if (topLevels.size === 1) {
      const innerFolders = paths
        .map(p => p.split("/")[1]?.toLowerCase())
        .filter(Boolean);

      if (innerFolders.some(f => categories.includes(f))) {
        setStatus(
          datasetMsg,
          "Invalid upload structure. Upload category folders directly (e.g., 'metal/', 'plastic/').",
          "error"
        );
        datasetInput.value = "";
        return;
      }
    }

    const formData = new FormData();

    for (const file of datasetInput.files) {
      const relativePath = file.webkitRelativePath || file.name;
      formData.append("files", file, relativePath);
    }

    const loader = type === "add" ? addLoader : updateLoader;

    if (loader) loader.style.display = "inline-block";

    try {
      const res = await fetch(`/admin/dataset/${type}`, {
        method: "POST",
        body: formData
      });

      const json = await res.json();

      if (!res.ok || json.error) {
        setStatus(datasetMsg, json?.error || "Dataset operation failed.", "error");
      } else {
        setStatus(datasetMsg, json.message || "Dataset updated successfully ✔", "success");
      }

    } catch {
      setStatus(datasetMsg, "Network error during dataset upload.", "error");

    } finally {
      datasetInput.value = "";
      if (loader) loader.style.display = "none";
    }

  };

  datasetInput.click();
}

// Bind dataset buttons
if (addDatasetBtn && datasetInput) {
  addDatasetBtn.addEventListener("click", () => handleDataset("add"));
}

if (updateDatasetBtn && datasetInput) {
  updateDatasetBtn.addEventListener("click", () => handleDataset("update"));
}


/* =====================================================
   TRAINING PROGRESS POLLING
===================================================== */

function checkTrainingProgress() {

  fetch("/progress")
    .then(res => res.json())
    .then(data => {

      // Status handling
      if (data.status === "training") {

        setStatus(adminMsg, "Training started...", "warn");

      } else if (data.status === "completed") {

        setStatus(adminMsg, data.message || "", "success");

        if (percentText) {
          percentText.textContent = "100% Completed";
        }

        // Final stats display
        if (trainStats) {

          const loss = data.loss !== undefined
            ? data.loss.toFixed(4)
            : "N/A";

          const acc = data.accuracy !== undefined
            ? (data.accuracy * 100).toFixed(2) + "%"
            : "N/A";

          const vloss = data.val_loss !== undefined
            ? data.val_loss.toFixed(4)
            : "N/A";

          const vacc = data.val_accuracy !== undefined
            ? (data.val_accuracy * 100).toFixed(2) + "%"
            : "N/A";

          trainStats.innerHTML = `
            <b>Loss:</b> ${loss} &nbsp; | &nbsp;
            <b>Accuracy:</b> ${acc}<br>
            <b>Val Loss:</b> ${vloss} &nbsp; | &nbsp;
            <b>Val Accuracy:</b> ${vacc}
          `;
        }

        if (progressBar) {
          progressBar.value = 100;
        }

        if (stopTrainingBtn) {
          stopTrainingBtn.style.display = "none";
        }

        if (progressBox) {
          progressBox.style.display = "none";
        }

        return;

      } else if (data.status === "stopped") {

        setStatus(adminMsg, data.message || "Training stopped by user.", "warn");

        if (percentText) {
          percentText.textContent = "Stopped";
        }

        if (progressBar) {
          progressBar.value = 0;
        }

        if (stopTrainingBtn) {
          stopTrainingBtn.style.display = "none";
        }

        if (progressBox) {
          progressBox.style.display = "none";
        }

        return;
      }

      // Progress calculation
      if (!data.total) {
        setTimeout(checkTrainingProgress, 1000);
        return;
      }

      let percent;

      if (data.batch && data.total_batches) {

        percent = Math.floor(
          (
            (data.epoch - 1) +
            (data.batch / data.total_batches)
          ) / data.total * 100
        );

      } else {

        percent = Math.floor(
          (data.epoch / data.total) * 100
        );

      }

      if (progressBar) {
        progressBar.value = percent;
      }

      if (percentText) {
        percentText.textContent = percent + "%";
      }

      if (data.status !== "completed" && data.status !== "stopped") {
        setTimeout(checkTrainingProgress, 1000);
      }

    })
    .catch(() => {
      setTimeout(checkTrainingProgress, 1500);
    });
}

/* =====================================================
   GENERATE EVALUATION REPORT
===================================================== */

document.addEventListener("DOMContentLoaded", () => {

  const generateBtn = document.getElementById("generateReportBtn");
  const reportBox = document.getElementById("reportBox");

  if (generateBtn) {
    generateBtn.addEventListener("click", () => {

      reportBox.innerHTML =
        "<div class='loader'></div><p>Generating report, please wait...</p>";

      fetch("/admin/report")
        .then(res => res.json())
        .then(data => {

          reportBox.textContent = data.message || "Report generated!";
          window.location.href = "/admin/eval";

        })
        .catch(() => {
          reportBox.textContent = "Error generating report";
        });

    });
  }

});


/* =====================================================
   RESPONSIVE MENU TOGGLE
===================================================== */

document.addEventListener("DOMContentLoaded", () => {

  const toggle = document.getElementById("menuToggle");
  const nav = document.getElementById("navMenu");

  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      nav.classList.toggle("active");
    });
  }

});


/* =====================================================
   CLEAR HISTORY
===================================================== */

document.addEventListener("DOMContentLoaded", function () {

  const btn = document.getElementById("clearHistoryBtn");

  if (btn) {
    btn.addEventListener("click", async function () {

      // Confirmation before clearing history
      if (!confirm("Are you sure you want to clear history?")) return;

      try {
        const res = await fetch("/clear_history", {
          method: "POST"
        });

        const data = await res.json();

        if (data.status === "success") {
          alert("History cleared");
          location.reload();
        } else {
          alert("Error clearing history");
        }

      } catch {
        alert("Request failed");
      }

    });
  }

});
/* =====================================================
   CLEAR LOGS
===================================================== */

document.addEventListener("DOMContentLoaded", function () {

  const btn = document.getElementById("clearLogsBtn");

  if (btn) {

    btn.addEventListener("click", async function () {

      if (!confirm("Clear all logs?")) return;

      try {

        const res = await fetch("/clear_logs", {
          method: "POST"
        });

        const data = await res.json();

        if (data.status === "success") {

          alert("Logs cleared");
          location.reload();

        } else {

          alert("Failed");

        }

      } catch {

        alert("Request failed");

      }

    });

  }

});