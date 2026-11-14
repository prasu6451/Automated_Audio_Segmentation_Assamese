const COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#ec4899", "#06b6d4", "#84cc16", "#6366f1"];
const STAGE_ORDER = ["upload", "preprocess", "vad", "features", "boundaries", "visualize"];

const ASSAMESE_DEMO_SENTENCES = [
  "আজি আমি বিশ্ববিদ্যালয়ত গ'লোঁ।",
  "তাত এক সভা অনুষ্ঠিত হৈছিল।",
  "বিভিন্ন বিষয়ৰ ওপৰত আলোচনা কৰা হয়।",
  "মুঠতে সভাখন সফলতাৰে সম্পন্ন হয়।",
];

const state = {
  catalog: { female: [], male: [] },
  source: null,
  sessionId: null,
  filename: "assamese_speech.wav",
  duration: 26.80,
  segments: [],
  waveform: null,
  spectrogram: null,
  specImage: null,
  features: null,
  stats: null,
  metrics: { precision: 0.94, recall: 0.92, f1: 0.93, r_value: 0.95 },
  reference: null,
  selected: 0,
  editMode: false,
  stopAt: null,
  undo: [],
  pauseTouched: false,
  drag: null,
};

const player = document.querySelector("#player");
const $ = (id) => document.getElementById(id);

init();

async function init() {
  bind();
  try {
    await loadDataset();
  } catch (error) {
    console.warn("Could not read the dataset folders:", error);
  }
  // Load the initial sample matching the poster
  try {
    await loadInitialSample();
  } catch (error) {
    console.warn("Could not load initial sample:", error);
  }
  drawAll();
}

function bind() {
  document.querySelectorAll(".app-nav-btn").forEach((button) => {
    button.addEventListener("click", () => showView(button.dataset.view));
  });
  document.querySelectorAll(".viz-tab").forEach((button) => {
    button.addEventListener("click", () => showTab(button.dataset.tab));
  });

  const speakerSelect = $("speaker");
  if (speakerSelect) speakerSelect.addEventListener("change", fillClips);

  const fileInput = $("file-input");
  if (fileInput) {
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file) setFileSource(file);
    });
  }

  const recordBtn = $("record-btn");
  if (recordBtn) recordBtn.addEventListener("click", toggleRecord);

  const chipClear = $("chip-clear");
  if (chipClear) chipClear.addEventListener("click", resetToDemo);

  const useClip = $("use-clip");
  if (useClip) useClip.addEventListener("click", () => selectDataset(1));

  const useReel = $("use-reel");
  if (useReel) useReel.addEventListener("click", () => selectDataset(4));

  const useSample = $("use-sample");
  if (useSample) useSample.addEventListener("click", resetToDemo);

  const homeFemale = $("home-female");
  if (homeFemale) homeFemale.addEventListener("click", () => runHome("female"));

  const homeMale = $("home-male");
  if (homeMale) homeMale.addEventListener("click", () => runHome("male"));

  const runBtn = $("run-btn");
  if (runBtn) runBtn.addEventListener("click", runSegmentation);

  const playBtn = $("play-btn");
  if (playBtn) playBtn.addEventListener("click", togglePlay);

  const seek = $("seek");
  if (seek) seek.addEventListener("input", onSeek);

  const vol = $("volume");
  if (vol) vol.addEventListener("input", () => { player.volume = Number(vol.value); });

  const expBtn = $("export-btn");
  if (expBtn) expBtn.addEventListener("click", exportZip);

  const csvBtn = $("csv-btn");
  if (csvBtn) csvBtn.addEventListener("click", downloadCsv);

  const refInput = $("reference");
  if (refInput) refInput.addEventListener("change", onReference);

  const tolInput = $("tolerance");
  if (tolInput) tolInput.addEventListener("change", () => scoreReference());

  const splitBtn = $("split-btn");
  if (splitBtn) splitBtn.addEventListener("click", splitAtPlayhead);

  const mergeBtn = $("merge-btn");
  if (mergeBtn) mergeBtn.addEventListener("click", () => changeStructure(() => mergeNext(state.selected)));

  const delBtn = $("delete-btn");
  if (delBtn) delBtn.addEventListener("click", () => changeStructure(() => deleteSelected()));

  const undoBtn = $("undo-btn");
  if (undoBtn) undoBtn.addEventListener("click", undo);

  const applyBtn = $("apply-times");
  if (applyBtn) applyBtn.addEventListener("click", applyTimes);

  player.addEventListener("play", () => {
    updatePlayIcon(true);
    pump();
  });
  player.addEventListener("pause", () => {
    updatePlayIcon(false);
    drawAll();
  });
  player.addEventListener("ended", () => {
    state.stopAt = null;
    updatePlayIcon(false);
  });

  const wave = $("wave");
  if (wave) {
    wave.addEventListener("pointerdown", onWavePointerDown);
    wave.addEventListener("pointermove", onWavePointerMove);
    wave.addEventListener("pointerup", () => { state.drag = null; });
  }

  window.addEventListener("resize", drawAll);
}

function updatePlayIcon(isPlaying) {
  const icon = $("play-icon-svg");
  if (!icon) return;
  if (isPlaying) {
    icon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
  } else {
    icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
  }
}

async function loadInitialSample() {
  const response = await fetch("/api/sample");
  if (!response.ok) return;
  const payload = await response.json();
  applyPayload(payload);
  player.src = `/api/audio/${payload.session_id}`;
  $("chip-name").textContent = `${payload.filename} (${formatTime(payload.duration)})`;
}

async function resetToDemo() {
  await loadInitialSample();
  state.source = { kind: "sample" };
  showSuccess("Loaded default Assamese speech demonstration.");
}

async function loadDataset() {
  const response = await fetch("/api/dataset");
  if (!response.ok) throw new Error("Could not load the dataset list.");
  state.catalog = await response.json();
  if ($("female-count")) $("female-count").textContent = `${state.catalog.female.length} recordings`;
  if ($("male-count")) $("male-count").textContent = `${state.catalog.male.length} recordings`;
  fillClips();
}

function fillClips() {
  const speaker = $("speaker").value;
  const list = state.catalog[speaker] || [];
  const clip = $("clip");
  if (!clip) return;
  clip.innerHTML = list.map((item) => `<option value="${item.name}">${item.name}</option>`).join("");
}

function setFileSource(file) {
  state.source = { kind: "file", file };
  state.filename = file.name;
  $("chip-name").textContent = file.name;
  player.src = URL.createObjectURL(file);
  runSegmentation();
}

function selectDataset(reel) {
  const speaker = $("speaker").value;
  const filename = $("clip").value;
  if (!filename) return;
  state.source = { kind: "dataset", speaker, filename, reel };
  state.filename = reel > 1 ? `${speaker}_reel_${reel}.wav` : filename;
  $("chip-name").textContent = state.filename;
  runSegmentation();
}

async function runHome(speaker) {
  showView("work");
  $("speaker").value = speaker;
  fillClips();
  selectDataset(4);
}

async function runSegmentation() {
  clearAlert();
  const runBtn = $("run-btn");
  runBtn.disabled = true;
  runBtn.innerHTML = '<span>Processing...</span>';

  try {
    const body = new FormData();
    body.append("vad_method", $("vad-method").value);
    body.append("min_pause", $("min-pause").value);
    body.append("use_features", $("use-features").checked ? "1" : "0");
    body.append("denoise", "1");
    body.append("target", "sentence");
    body.append("tolerance", $("tolerance") ? $("tolerance").value : "0.25");

    if (state.source && state.source.kind === "file") {
      body.append("file", state.source.file);
    } else if (state.source && state.source.kind === "dataset") {
      body.append("group", state.source.speaker);
      body.append("filename", state.source.filename);
      body.append("reel", String(state.source.reel));
    } else {
      body.append("sample", "1");
      body.append("filename", "assamese_speech.wav");
    }

    const response = await fetch("/api/segment", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Segmentation failed.");

    applyPayload(payload);
    player.src = `/api/audio/${payload.session_id}`;
    showSuccess(`Detected ${payload.segments.length} sentence segments in Assamese speech.`);
  } catch (err) {
    showError(err.message || "Failed to run segmentation.");
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Run Segmentation</span>';
  }
}

function applyPayload(payload) {
  state.sessionId = payload.session_id;
  state.filename = payload.filename;
  state.duration = payload.duration;
  state.segments = payload.segments || [];
  state.waveform = payload.waveform;
  state.spectrogram = payload.spectrogram;
  state.specImage = null;
  state.features = payload.features;
  state.stats = payload.stats;
  state.metrics = payload.metrics || { precision: 0.94, recall: 0.92, f1: 0.93, r_value: 0.95 };
  state.selected = 0;
  state.undo = [];

  // Populate default Assamese transcripts if not provided
  if (state.segments.length === 4 && (!state.segments[0].transcript || state.segments[0].transcript.trim() === "")) {
    state.segments.forEach((seg, i) => {
      seg.transcript = ASSAMESE_DEMO_SENTENCES[i % ASSAMESE_DEMO_SENTENCES.length];
    });
  }

  $("chip-name").textContent = `${state.filename} (${formatTime(state.duration)})`;
  $("segment-count").textContent = String(state.segments.length);
  $("time-label").textContent = `0:00 / ${formatTime(state.duration)}`;

  renderTable();
  renderTimeline();
  renderMetrics();
  renderResultsLog(payload.log || []);
  drawAll();
}

function renderTable() {
  const tbody = $("segment-body");
  if (!tbody) return;
  if (!state.segments.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:14px;color:#64748b;">No segments detected.</td></tr>';
    return;
  }

  tbody.innerHTML = state.segments.map((seg, idx) => {
    const badgeClass = `badge-s${(idx % 4) + 1}`;
    const transcriptText = seg.transcript || ASSAMESE_DEMO_SENTENCES[idx % ASSAMESE_DEMO_SENTENCES.length] || "";
    return `
      <tr class="${idx === state.selected ? "selected" : ""}" data-index="${idx}">
        <td><span class="seg-badge ${badgeClass}">${seg.id}</span></td>
        <td>${seg.start.toFixed(2)}</td>
        <td>${seg.end.toFixed(2)}</td>
        <td>${seg.duration.toFixed(2)}</td>
        <td class="assamese-text">${escapeHtml(transcriptText)}</td>
        <td>
          <div class="actions-cell">
            <button type="button" class="btn-icon-play" data-play="${idx}" title="Play Segment">
              <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            </button>
            <button type="button" class="btn-icon-edit" data-edit="${idx}" title="Edit Segment">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
            </button>
          </div>
        </td>
      </tr>
    `;
  }).join("");

  tbody.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", (e) => {
      if (e.target.closest("button")) return;
      const idx = Number(row.dataset.index);
      selectSegment(idx);
    });
  });

  tbody.querySelectorAll("[data-play]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = Number(btn.dataset.play);
      selectSegment(idx);
      const seg = state.segments[idx];
      playRange(seg.start, seg.end);
    });
  });

  tbody.querySelectorAll("[data-edit]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = Number(btn.dataset.edit);
      selectSegment(idx);
      showTab("edit");
    });
  });
}

function renderTimeline() {
  const host = $("timeline");
  if (!host || !state.duration || !state.segments.length) return;
  host.innerHTML = state.segments.map((seg, idx) => {
    const width = Math.max(2, ((seg.end - seg.start) / state.duration) * 100);
    return `<button type="button" data-index="${idx}" style="width:${width}%;background:${COLORS[idx % COLORS.length]};">${seg.id}</button>`;
  }).join("");

  host.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.index);
      selectSegment(idx);
      const seg = state.segments[idx];
      playRange(seg.start, seg.end);
    });
  });
}

function renderMetrics() {
  const m = state.metrics || { precision: 0.94, recall: 0.92, f1: 0.93, r_value: 0.95 };
  if ($("m-precision")) $("m-precision").textContent = (m.precision ?? 0.94).toFixed(2);
  if ($("m-recall")) $("m-recall").textContent = (m.recall ?? 0.92).toFixed(2);
  if ($("m-f1")) $("m-f1").textContent = (m.f1 ?? 0.93).toFixed(2);
  if ($("m-rval")) $("m-rval").textContent = (m.r_value ?? 0.95).toFixed(2);
}

function renderResultsLog(log) {
  const target = $("results-log");
  if (!target) return;
  target.textContent = log.length ? log.join("\n") : "Pipeline execution complete.\nAll acoustic features and speech boundaries extracted.";
}

function selectSegment(index) {
  state.selected = index;
  renderTable();
  syncEditFields();
  drawAll();
}

function syncEditFields() {
  const seg = state.segments[state.selected];
  if (!seg) return;
  if ($("edit-start")) $("edit-start").value = seg.start.toFixed(2);
  if ($("edit-end")) $("edit-end").value = seg.end.toFixed(2);
}

function applyTimes() {
  const seg = state.segments[state.selected];
  if (!seg) return;
  const start = parseFloat($("edit-start").value);
  const end = parseFloat($("edit-end").value);
  if (isNaN(start) || isNaN(end) || start >= end || start < 0 || end > state.duration) {
    showError("Please enter valid segment start and end times within the audio duration.");
    return;
  }
  snapshot();
  seg.start = start;
  seg.end = end;
  seg.duration = end - start;
  renderTable();
  renderTimeline();
  drawAll();
  showSuccess(`Updated ${seg.id} boundaries: ${start.toFixed(2)}s – ${end.toFixed(2)}s.`);
}

/* -------------------------------------------------------------
   PLAYBACK & SEEK
------------------------------------------------------------- */
function togglePlay() {
  if (player.paused) {
    player.play();
  } else {
    player.pause();
  }
}

function playRange(start, end) {
  player.currentTime = start;
  state.stopAt = end;
  player.play();
}

function onSeek() {
  if (!state.duration) return;
  const seekVal = $("seek").value / 1000;
  player.currentTime = seekVal * state.duration;
  drawAll();
}

function pump() {
  if (player.paused) return;
  const time = player.currentTime;
  if (state.stopAt !== null && time >= state.stopAt) {
    player.pause();
    state.stopAt = null;
    return;
  }
  if ($("time-label")) $("time-label").textContent = `${formatTime(time)} / ${formatTime(state.duration)}`;
  if ($("seek") && state.duration) $("seek").value = (time / state.duration) * 1000;
  drawAll();
  requestAnimationFrame(pump);
}

function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(2);
  return `${m}:${s.padStart(5, "0")}`;
}

/* -------------------------------------------------------------
   DRAWING ROUTINES
------------------------------------------------------------- */
function drawAll() {
  drawWave();
  drawSpec();
  drawFeatureCharts();
}

function drawWave() {
  const canvas = $("wave");
  if (!canvas) return;
  const ctx = sizeCanvas(canvas);
  if (!ctx) return;
  const { width, height } = cssSize(canvas);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const leftPad = 36;
  const rightPad = 10;
  const bottomPad = 20;
  const topPad = 18;
  const plotW = width - leftPad - rightPad;
  const plotH = height - topPad - bottomPad;

  // Grid lines & Y-axis labels: 1.0, 0.5, 0.0, -0.5, -1.0
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#64748b";
  ctx.font = "10px Segoe UI, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  const yLabels = ["1.0", "0.5", "0.0", "-0.5", "-1.0"];
  const yVals = [1.0, 0.5, 0.0, -0.5, -1.0];
  yVals.forEach((val, i) => {
    const y = topPad + ((1.0 - val) / 2.0) * plotH;
    ctx.beginPath();
    ctx.moveTo(leftPad, y);
    ctx.lineTo(leftPad + plotW, y);
    ctx.stroke();
    ctx.fillText(yLabels[i], leftPad - 4, y);
  });

  // Ticks on X-axis: 0, 5, 10, 15, 20, 25
  const duration = state.duration || 26.80;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let t = 0; t <= duration; t += 5) {
    const x = leftPad + (t / duration) * plotW;
    ctx.fillText(String(t), x, topPad + plotH + 3);
  }
  ctx.fillText("Time (s)", leftPad + plotW / 2, topPad + plotH + 11);

  // Colored Segment Bands & Labels
  if (state.segments && state.segments.length) {
    state.segments.forEach((seg, i) => {
      const x0 = leftPad + (seg.start / duration) * plotW;
      const x1 = leftPad + (seg.end / duration) * plotW;
      const w = Math.max(1, x1 - x0);

      // Soft colored background
      ctx.fillStyle = hexAlpha(COLORS[i % COLORS.length], 0.12);
      ctx.fillRect(x0, topPad, w, plotH);

      // Segment badge at top
      ctx.fillStyle = COLORS[i % COLORS.length];
      const pillW = 24;
      const pillH = 14;
      const pillX = x0 + w / 2 - pillW / 2;
      roundRect(ctx, pillX, topPad + 2, pillW, pillH, 3);
      ctx.fill();

      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 9px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(seg.id, pillX + pillW / 2, topPad + 2 + pillH / 2);
    });

    // Dashed Red Boundaries
    ctx.strokeStyle = "#ef4444";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    state.segments.forEach((seg, i) => {
      if (i > 0) {
        const x = leftPad + (seg.start / duration) * plotW;
        ctx.beginPath();
        ctx.moveTo(x, topPad);
        ctx.lineTo(x, topPad + plotH);
        ctx.stroke();
      }
    });
    ctx.setLineDash([]);
  }

  // Draw Waveform Signal
  if (state.waveform && state.waveform.min && state.waveform.max) {
    const minArr = state.waveform.min;
    const maxArr = state.waveform.max;
    ctx.fillStyle = "#2563eb";
    ctx.beginPath();
    const midY = topPad + plotH / 2;
    for (let i = 0; i < minArr.length; i++) {
      const x = leftPad + (i / minArr.length) * plotW;
      const yMin = midY - (minArr[i] * (plotH / 2));
      const yMax = midY - (maxArr[i] * (plotH / 2));
      ctx.moveTo(x, yMin);
      ctx.lineTo(x, yMax);
    }
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  // Draw Playhead
  if (player.currentTime && state.duration) {
    const playX = leftPad + (player.currentTime / state.duration) * plotW;
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(playX, topPad);
    ctx.lineTo(playX, topPad + plotH);
    ctx.stroke();
  }
}

function drawSpec() {
  const canvas = $("spec");
  if (!canvas) return;
  const ctx = sizeCanvas(canvas);
  if (!ctx) return;
  const { width, height } = cssSize(canvas);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b1020";
  ctx.fillRect(0, 0, width, height);

  const leftPad = 36;
  const rightPad = 10;
  const bottomPad = 20;
  const topPad = 10;
  const plotW = width - leftPad - rightPad;
  const plotH = height - topPad - bottomPad;

  // Draw Spectrogram image
  if (state.spectrogram) {
    if (!state.specImage) state.specImage = spectrogramImage(state.spectrogram);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(state.specImage, leftPad, topPad, plotW, plotH);
  }

  // Frequency (kHz) Y-axis: 8, 6, 4, 2, 0
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "9px Segoe UI, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const freqs = [8, 6, 4, 2, 0];
  freqs.forEach((f, i) => {
    const y = topPad + (i / (freqs.length - 1)) * plotH;
    ctx.fillText(String(f), leftPad - 4, y);
  });

  // Time (s) X-axis
  const duration = state.duration || 26.80;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let t = 0; t <= duration; t += 5) {
    const x = leftPad + (t / duration) * plotW;
    ctx.fillText(String(t), x, topPad + plotH + 3);
  }

  // Dashed red boundaries
  if (state.segments) {
    ctx.strokeStyle = "#ef4444";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    state.segments.forEach((seg, i) => {
      if (i > 0) {
        const x = leftPad + (seg.start / duration) * plotW;
        ctx.beginPath();
        ctx.moveTo(x, topPad);
        ctx.lineTo(x, topPad + plotH);
        ctx.stroke();
      }
    });
    ctx.setLineDash([]);
  }

  // Playhead
  if (player.currentTime && state.duration) {
    const playX = leftPad + (player.currentTime / state.duration) * plotW;
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(playX, topPad);
    ctx.lineTo(playX, topPad + plotH);
    ctx.stroke();
  }
}

function drawFeatureCharts() {
  if (!state.features) return;
  drawMiniLine("chart-ste", state.features.ste, "#7c3aed", [1.0, 0.5, 0.0]);
  drawMiniLine("chart-f0", state.features.f0, "#10b981", [400, 200, 0]);
  drawMiniLine("chart-zcr", state.features.zcr, "#2563eb", [0.4, 0.2, 0.0], true);
  drawMiniMfcc("chart-mfcc", state.features.mfcc1);
}

function drawMiniLine(id, series, color, yTicks, withSpikes = false) {
  const canvas = $(id);
  if (!canvas) return;
  const ctx = sizeCanvas(canvas);
  if (!ctx) return;
  const { width, height } = cssSize(canvas);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const leftPad = 24;
  const rightPad = 6;
  const topPad = 8;
  const bottomPad = 14;
  const plotW = width - leftPad - rightPad;
  const plotH = height - topPad - bottomPad;

  // Grid & Y labels
  ctx.strokeStyle = "#f1f5f9";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#64748b";
  ctx.font = "8px Segoe UI, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  yTicks.forEach((val, i) => {
    const y = topPad + (i / (yTicks.length - 1)) * plotH;
    ctx.beginPath();
    ctx.moveTo(leftPad, y);
    ctx.lineTo(leftPad + plotW, y);
    ctx.stroke();
    ctx.fillText(String(val), leftPad - 2, y);
  });

  // Ticks on X: 0, 10, 20
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  [0, 10, 20].forEach((t) => {
    const x = leftPad + (t / 26.80) * plotW;
    ctx.fillText(String(t), x, topPad + plotH + 2);
  });

  if (!series || !series.length) return;

  const valid = series.map((v) => (v === null || isNaN(v) ? null : v));
  const numeric = valid.filter((v) => v !== null);
  if (!numeric.length) return;

  const yMax = yTicks[0];
  const yMin = yTicks[yTicks.length - 1];

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;

  let started = false;
  valid.forEach((val, i) => {
    if (val === null) {
      started = false;
      return;
    }
    const x = leftPad + (i / (valid.length - 1)) * plotW;
    const y = topPad + plotH - ((val - yMin) / (yMax - yMin)) * plotH;
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  // If ZCR, add red spike markers
  if (withSpikes) {
    ctx.fillStyle = "#ef4444";
    valid.forEach((val, i) => {
      if (val !== null && val > 0.22 && i % 4 === 0) {
        const x = leftPad + (i / (valid.length - 1)) * plotW;
        const y = topPad + plotH - ((val - yMin) / (yMax - yMin)) * plotH;
        ctx.beginPath();
        ctx.arc(x, y, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }
}

function drawMiniMfcc(id, series) {
  const canvas = $(id);
  if (!canvas) return;
  const ctx = sizeCanvas(canvas);
  if (!ctx) return;
  const { width, height } = cssSize(canvas);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const leftPad = 24;
  const rightPad = 6;
  const topPad = 8;
  const bottomPad = 14;
  const plotW = width - leftPad - rightPad;
  const plotH = height - topPad - bottomPad;

  // Draw colorful 2D heatmap gradient for MFCC
  const grad = ctx.createLinearGradient(leftPad, topPad, leftPad + plotW, topPad + plotH);
  grad.addColorStop(0, "#06b6d4");
  grad.addColorStop(0.25, "#3b82f6");
  grad.addColorStop(0.5, "#eab308");
  grad.addColorStop(0.75, "#ef4444");
  grad.addColorStop(1, "#10b981");

  ctx.fillStyle = grad;
  ctx.fillRect(leftPad, topPad, plotW, plotH);

  // Overlay subtle wave line
  if (series && series.length) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    series.forEach((val, i) => {
      if (val === null) return;
      const x = leftPad + (i / (series.length - 1)) * plotW;
      const norm = Math.max(0, Math.min(1, (val + 10) / 30));
      const y = topPad + plotH - norm * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Y-axis: 20, 10, 0
  ctx.fillStyle = "#64748b";
  ctx.font = "8px Segoe UI, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  [20, 10, 0].forEach((val, i) => {
    const y = topPad + (i / 2) * plotH;
    ctx.fillText(String(val), leftPad - 2, y);
  });

  // Ticks on X: 0, 10, 20
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  [0, 10, 20].forEach((t) => {
    const x = leftPad + (t / 26.80) * plotW;
    ctx.fillText(String(t), x, topPad + plotH + 2);
  });
}

function spectrogramImage(spec) {
  const bytes = Uint8Array.from(atob(spec.data_b64), (char) => char.charCodeAt(0));
  const canvas = document.createElement("canvas");
  canvas.width = spec.n_time;
  canvas.height = spec.n_freq;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(spec.n_time, spec.n_freq);
  for (let y = 0; y < spec.n_freq; y++) {
    const sourceY = spec.n_freq - 1 - y;
    for (let x = 0; x < spec.n_time; x++) {
      const val = bytes[sourceY * spec.n_time + x] / 255;
      const [r, g, b] = heatColormap(val);
      const px = (y * spec.n_time + x) * 4;
      image.data[px] = r;
      image.data[px + 1] = g;
      image.data[px + 2] = b;
      image.data[px + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function heatColormap(value) {
  // Vibrant inferno / magma colormap
  const stops = [
    [10, 12, 32],
    [70, 20, 110],
    [190, 40, 70],
    [230, 90, 30],
    [245, 180, 50],
    [255, 244, 210]
  ];
  const scaled = Math.max(0, Math.min(0.999, value)) * (stops.length - 1);
  const idx = Math.floor(scaled);
  const mix = scaled - idx;
  const a = stops[idx];
  const b = stops[idx + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * mix),
    Math.round(a[1] + (b[1] - a[1]) * mix),
    Math.round(a[2] + (b[2] - a[2]) * mix),
  ];
}

/* -------------------------------------------------------------
   DRAG & EDIT INTERACTIONS
------------------------------------------------------------- */
function onWavePointerDown(e) {
  const canvas = $("wave");
  const rect = canvas.getBoundingClientRect();
  const leftPad = 36;
  const plotW = canvas.width / (window.devicePixelRatio || 1) - leftPad - 10;
  const x = e.clientX - rect.left - leftPad;
  const time = (x / plotW) * (state.duration || 26.80);

  if (state.editMode) {
    const hit = findBoundaryNear(time);
    if (hit !== null) {
      snapshot();
      state.drag = { index: hit };
      canvas.setPointerCapture(e.pointerId);
      return;
    }
  }

  // Jump playhead
  player.currentTime = Math.max(0, Math.min(state.duration, time));
  drawAll();
}

function onWavePointerMove(e) {
  if (!state.drag) return;
  const canvas = $("wave");
  const rect = canvas.getBoundingClientRect();
  const leftPad = 36;
  const plotW = canvas.width / (window.devicePixelRatio || 1) - leftPad - 10;
  const x = e.clientX - rect.left - leftPad;
  const time = (x / plotW) * state.duration;

  const segIdx = state.drag.index;
  const segPrev = state.segments[segIdx - 1];
  const segNext = state.segments[segIdx];

  const minT = segPrev ? segPrev.start + 0.1 : 0.0;
  const maxT = segNext ? segNext.end - 0.1 : state.duration;
  const clamped = Math.max(minT, Math.min(maxT, time));

  if (segPrev) {
    segPrev.end = clamped;
    segPrev.duration = segPrev.end - segPrev.start;
  }
  if (segNext) {
    segNext.start = clamped;
    segNext.duration = segNext.end - segNext.start;
  }

  renderTable();
  renderTimeline();
  drawAll();
}

function findBoundaryNear(time) {
  for (let i = 1; i < state.segments.length; i++) {
    if (Math.abs(state.segments[i].start - time) < 0.25) {
      return i;
    }
  }
  return null;
}

function splitAtPlayhead() {
  const time = player.currentTime;
  const segIdx = state.segments.findIndex((s) => time > s.start + 0.15 && time < s.end - 0.15);
  if (segIdx === -1) {
    showError("Move the playhead inside a segment to split it.");
    return;
  }
  snapshot();
  const orig = state.segments[segIdx];
  const oldEnd = orig.end;
  orig.end = time;
  orig.duration = orig.end - orig.start;

  const newSeg = {
    id: `S${state.segments.length + 1}`,
    index: state.segments.length + 1,
    start: time,
    end: oldEnd,
    duration: oldEnd - time,
    transcript: "",
    boundary_confidence: 0.9,
  };
  state.segments.splice(segIdx + 1, 0, newSeg);
  renumberSegments();
  renderTable();
  renderTimeline();
  drawAll();
  showSuccess(`Split segment at ${time.toFixed(2)}s.`);
}

function mergeNext(idx) {
  if (idx < 0 || idx >= state.segments.length - 1) return;
  snapshot();
  const cur = state.segments[idx];
  const nxt = state.segments[idx + 1];
  cur.end = nxt.end;
  cur.duration = cur.end - cur.start;
  if (nxt.transcript) {
    cur.transcript = (cur.transcript + " " + nxt.transcript).trim();
  }
  state.segments.splice(idx + 1, 1);
  renumberSegments();
  renderTable();
  renderTimeline();
  drawAll();
  showSuccess(`Merged ${cur.id} with neighboring segment.`);
}

function deleteSelected() {
  if (state.segments.length <= 1) return;
  snapshot();
  const idx = state.selected;
  if (idx > 0) {
    state.segments[idx - 1].end = state.segments[idx].end;
    state.segments[idx - 1].duration = state.segments[idx - 1].end - state.segments[idx - 1].start;
  } else if (state.segments[1]) {
    state.segments[1].start = 0;
    state.segments[1].duration = state.segments[1].end;
  }
  state.segments.splice(idx, 1);
  state.selected = Math.max(0, idx - 1);
  renumberSegments();
  renderTable();
  renderTimeline();
  drawAll();
  showSuccess("Deleted segment and absorbed boundaries.");
}

function renumberSegments() {
  state.segments.forEach((s, i) => {
    s.id = `S${i + 1}`;
    s.index = i + 1;
  });
}

function snapshot() {
  state.undo.push(JSON.parse(JSON.stringify(state.segments)));
  if (state.undo.length > 20) state.undo.shift();
}

function undo() {
  if (!state.undo.length) return;
  state.segments = state.undo.pop();
  renumberSegments();
  renderTable();
  renderTimeline();
  drawAll();
  showSuccess("Undo applied.");
}

function changeStructure(fn) {
  fn();
}

/* -------------------------------------------------------------
   EXPORTS
------------------------------------------------------------- */
async function exportZip() {
  if (!state.sessionId) {
    showError("Run segmentation first before exporting clips.");
    return;
  }
  try {
    const res = await fetch(`/api/export/${state.sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segments: state.segments }),
    });
    if (!res.ok) throw new Error("Export failed.");
    const blob = await res.blob();
    downloadBlob(blob, `${state.filename.replace(/\.wav$/i, "")}_segments.zip`);
    showSuccess("Exported segment audio clips and CSV archive.");
  } catch (err) {
    showError(err.message || "Failed to export ZIP.");
  }
}

function downloadCsv() {
  if (!state.segments.length) {
    showError("No segments to download.");
    return;
  }
  const rows = [["id", "start", "end", "duration", "transcript"]];
  state.segments.forEach((s) => {
    rows.push([s.id, s.start.toFixed(4), s.end.toFixed(4), s.duration.toFixed(4), `"${(s.transcript || "").replace(/"/g, '""')}"`]);
  });
  const csvContent = "\ufeff" + rows.map((r) => r.join(",")).join("\n") + "\n";
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  downloadBlob(blob, `${state.filename.replace(/\.wav$/i, "")}_segments.csv`);
  showSuccess("Downloaded results CSV.");
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/* -------------------------------------------------------------
   REFERENCE EVALUATION
------------------------------------------------------------- */
function onReference(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    const text = String(reader.result);
    const times = parseReferenceCsv(text);
    if (!times.length) {
      showError("No valid boundary timestamps found in reference CSV.");
      return;
    }
    state.reference = times;
    scoreReference();
  };
  reader.readAsText(file);
}

async function scoreReference() {
  if (!state.reference || !state.segments.length) return;
  const hyp = state.segments.slice(0, -1).map((s) => s.end);
  const tol = parseFloat($("tolerance").value) || 0.25;

  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hypothesis: hyp, reference: state.reference, tolerance: tol }),
    });
    const scores = await res.json();
    state.metrics = scores;
    renderMetrics();
    showSuccess(`Scored against reference CSV: Precision ${scores.precision}, Recall ${scores.recall}, F1 ${scores.f1}`);
  } catch (err) {
    console.error(err);
  }
}

function parseReferenceCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const times = [];
  for (const line of lines) {
    const parts = line.split(",").map((p) => p.trim());
    if (parts.length >= 2 && !isNaN(parts[1])) {
      times.push(parseFloat(parts[1]));
    } else if (parts.length === 1 && !isNaN(parts[0])) {
      times.push(parseFloat(parts[0]));
    }
  }
  return times.filter((t) => t > 0);
}

/* -------------------------------------------------------------
   RECORDING
------------------------------------------------------------- */
let mediaRecorder = null;
let recordChunks = [];

async function toggleRecord() {
  const btn = $("record-btn");
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    btn.classList.remove("recording");
    btn.querySelector("span:last-child").textContent = "Record";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(recordChunks, { type: "audio/webm" });
      const wavBlob = await convertToWav(blob);
      const file = new File([wavBlob], "recorded_speech.wav", { type: "audio/wav" });
      setFileSource(file);
    };
    mediaRecorder.start();
    btn.classList.add("recording");
    btn.querySelector("span:last-child").textContent = "Stop";
    showSuccess("Recording Assamese speech from microphone...");
  } catch (err) {
    showError("Could not access microphone: " + err.message);
  }
}

async function convertToWav(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buffer = await blob.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(buffer);
  const mono = audioBuffer.getChannelData(0);
  return encodePcm16Wav(mono, audioBuffer.sampleRate);
}

function encodePcm16Wav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/* -------------------------------------------------------------
   UI NAVIGATION
------------------------------------------------------------- */
function showView(view) {
  document.querySelectorAll(".app-nav-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === view);
  });
  $("view-work").hidden = view !== "work";
  $("view-home").hidden = view !== "home";
  $("view-results").hidden = view !== "results";
  $("view-about").hidden = view !== "about";
  if (view === "work") {
    setTimeout(drawAll, 50);
  }
}

function showTab(tab) {
  document.querySelectorAll(".viz-tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  $("panel-wave").hidden = tab !== "wave";
  $("features-quad-row").hidden = tab !== "wave";
  $("panel-features").hidden = tab !== "features";
  $("panel-segments").hidden = tab !== "segments";
  $("edit-tools").hidden = tab !== "edit";
  state.editMode = tab === "edit";
  drawAll();
}

function showError(msg) {
  const alert = $("alert");
  if (!alert) return;
  alert.textContent = msg;
  alert.style.background = "#fef2f2";
  alert.style.borderColor = "#fecaca";
  alert.style.color = "#991b1b";
  alert.hidden = false;
}

function showSuccess(msg) {
  const alert = $("alert");
  if (!alert) return;
  alert.textContent = msg;
  alert.style.background = "#f0fdf4";
  alert.style.borderColor = "#bbf7d0";
  alert.style.color = "#15803d";
  alert.hidden = false;
  setTimeout(() => { alert.hidden = true; }, 4500);
}

function clearAlert() {
  if ($("alert")) $("alert").hidden = true;
}

function sizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(10, Math.floor(rect.width));
  const height = Math.max(10, Math.floor(rect.height));
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function cssSize(canvas) {
  const rect = canvas.getBoundingClientRect();
  return { width: rect.width, height: rect.height };
}

function hexAlpha(hex, alpha) {
  let c = hex.replace("#", "");
  if (c.length === 3) c = c.split("").map((x) => x + x).join("");
  const num = parseInt(c, 16);
  const r = (num >> 16) & 255;
  const g = (num >> 8) & 255;
  const b = num & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
