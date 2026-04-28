function parseFeatureVector(text, expectedLength = 128) {
  const parts = text
    .split(/[\s,]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map(Number);

  if (parts.length !== expectedLength || parts.some((x) => Number.isNaN(x))) {
    throw new Error(`Expected ${expectedLength} numeric values, got ${parts.length}.`);
  }
  return parts;
}

function randomVector(n = 128) {
  return Array.from({ length: n }, () => (Math.random() * 2 - 1).toFixed(4));
}

function renderBars(confidences) {
  const bars = document.getElementById("bars");
  bars.innerHTML = "";

  Object.entries(confidences).forEach(([label, prob]) => {
    const pct = Math.round(prob * 1000) / 10;
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `
      <span>${label}</span>
      <div class="track"><div class="fill" style="width:${pct}%"></div></div>
      <strong>${pct}%</strong>
    `;
    bars.appendChild(row);
  });
}

document.getElementById("fillBtn").addEventListener("click", () => {
  document.getElementById("eegInput").value = randomVector().join(", ");
  document.getElementById("speechInput").value = randomVector().join(", ");
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const statusText = document.getElementById("statusText");
  try {
    const apiBase = document.getElementById("apiUrl").value.trim().replace(/\/$/, "") || window.location.origin;
    const eegFeatures = parseFeatureVector(document.getElementById("eegInput").value, 128);
    const speechFeatures = parseFeatureVector(document.getElementById("speechInput").value, 128);

    statusText.textContent = "Running inference...";

    const res = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ eeg_features: eegFeatures, speech_features: speechFeatures }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    statusText.textContent = `Model: ${data.model_type} | Predicted: ${data.label} (${(data.confidence * 100).toFixed(1)}%)`;
    renderBars(data.confidences);
  } catch (err) {
    statusText.textContent = `Error: ${err.message}`;
  }
});
