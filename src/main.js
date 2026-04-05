import "./styles.css";

const app = document.querySelector("#app");

app.innerHTML = `
  <main class="page">
    <section class="card hero">
      <h1>Tox-Agent API Local Tester</h1>
      <p>
        Chay tren localhost, goi API qua route tuong doi (/predict, /analyze, /health).
        Neu chay bang Vite dev server, cac route nay se duoc proxy sang backend.
      </p>
      <div class="row">
        <button id="btn-health" class="btn ghost" type="button">Check Health</button>
        <button id="btn-predict" class="btn" type="button">Test Predict</button>
        <button id="btn-analyze" class="btn primary" type="button">Test Analyze (3 outputs)</button>
      </div>
    </section>

    <section class="card">
      <label for="smiles-input" class="label">SMILES Input</label>
      <textarea id="smiles-input" rows="3" placeholder="Vi du: CCO">CC(=O)Oc1ccccc1C(=O)O</textarea>
      <div class="hint">Nhap 1 chuoi SMILES, sau do bam Predict hoac Analyze.</div>
    </section>

    <section class="card" id="status-card">
      <h2>Trang Thai / Loi API</h2>
      <pre id="status-box">Chua goi request.</pre>
    </section>

    <section class="card" id="analyze-card">
      <h2>Analyze Output (3 khoi)</h2>
      <div class="grid">
        <article>
          <h3>1) Clinical</h3>
          <pre id="clinical-box">Chua co du lieu.</pre>
        </article>
        <article>
          <h3>2) Mechanism</h3>
          <pre id="mechanism-box">Chua co du lieu.</pre>
        </article>
        <article>
          <h3>3) Explanation</h3>
          <pre id="explanation-box">Chua co du lieu.</pre>
        </article>
      </div>
      <div class="verdict">Final Verdict: <strong id="verdict-box">N/A</strong></div>
    </section>
  </main>
`;

const smilesInput = document.querySelector("#smiles-input");
const statusBox = document.querySelector("#status-box");
const clinicalBox = document.querySelector("#clinical-box");
const mechanismBox = document.querySelector("#mechanism-box");
const explanationBox = document.querySelector("#explanation-box");
const verdictBox = document.querySelector("#verdict-box");

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function setStatus(title, payload) {
  statusBox.textContent = `${title}\n\n${pretty(payload)}`;
}

function resetAnalyzeBoxes() {
  clinicalBox.textContent = "Chua co du lieu.";
  mechanismBox.textContent = "Chua co du lieu.";
  explanationBox.textContent = "Chua co du lieu.";
  verdictBox.textContent = "N/A";
}

async function callApi(path, body) {
  const options = {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
  };
  if (body) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({ raw: "Non-JSON response" }));

  if (!response.ok) {
    const error = new Error("API request failed");
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  return payload;
}

async function onHealth() {
  resetAnalyzeBoxes();
  try {
    const payload = await callApi("/health");
    setStatus("GET /health OK", payload);
  } catch (error) {
    setStatus(`GET /health FAILED (${error.status || "unknown"})`, error.payload || { message: error.message });
  }
}

async function onPredict() {
  resetAnalyzeBoxes();
  const smiles = smilesInput.value.trim();
  if (!smiles) {
    setStatus("Input invalid", { message: "Vui long nhap SMILES." });
    return;
  }

  try {
    const payload = await callApi("/predict", {
      smiles,
      threshold: 0.35,
    });
    setStatus("POST /predict OK", payload);
    clinicalBox.textContent = pretty(payload);
    verdictBox.textContent = payload.label || "N/A";
  } catch (error) {
    setStatus(`POST /predict FAILED (${error.status || "unknown"})`, error.payload || { message: error.message });
  }
}

async function onAnalyze() {
  const smiles = smilesInput.value.trim();
  if (!smiles) {
    setStatus("Input invalid", { message: "Vui long nhap SMILES." });
    return;
  }

  try {
    const payload = await callApi("/analyze", {
      smiles,
      clinical_threshold: 0.35,
      mechanism_threshold: 0.5,
      return_all_scores: true,
      explain_only_if_alert: true,
      explainer_epochs: 80,
      explainer_timeout_ms: 30000,
      target_task: null,
    });

    setStatus("POST /analyze OK", payload);
    clinicalBox.textContent = pretty(payload.clinical || {});
    mechanismBox.textContent = pretty(payload.mechanism || {});
    explanationBox.textContent = pretty(payload.explanation || { message: "Explanation = null (co the do explain_only_if_alert=true)." });
    verdictBox.textContent = payload.final_verdict || "N/A";
  } catch (error) {
    resetAnalyzeBoxes();
    setStatus(`POST /analyze FAILED (${error.status || "unknown"})`, error.payload || { message: error.message });
  }
}

document.querySelector("#btn-health").addEventListener("click", onHealth);
document.querySelector("#btn-predict").addEventListener("click", onPredict);
document.querySelector("#btn-analyze").addEventListener("click", onAnalyze);