// PIPELINE tab: pick a run, see which stages can start (and why not), run one, follow its log and outputs.
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";
import { openBrief } from "./brief.js";

const ps = { runId: null, stages: [], current: null, jobId: null, offset: 0, timer: null, runs: [], engine: null };
const GATES = { preview: "scene", check_expert: "expert", generate: "data_method", convert: "convert",
  eval: "eval", rollout: "rollout", train: "train", register: "rollout", promote: "rollout" };
const ACTIVE = ["queued", "running"];

export function stageGate(stageId) { return GATES[stageId] || null; }

export async function initPipeline() {
  $("#run-select").addEventListener("change", (e) => selectRun(e.target.value));
  $("#btn-run").addEventListener("click", runCurrent);
  $("#btn-stop").addEventListener("click", stopCurrent);
  await refreshRuns();
  setInterval(refreshStages, 3000);
}

export async function refreshRuns(selectId) {
  const runs = await api.get("/api/runs");
  ps.runs = runs;
  const sel = $("#run-select");
  sel.innerHTML = "";
  if (!runs.length) sel.append(new Option("No runs yet: CREATE RUN in PROJECT", ""));
  for (const r of runs) sel.append(new Option(`${r.id}  ·  ${r.selection.simulator}  ${r.note || ""}`, r.id));
  const id = selectId || ps.runId || (runs[0] ? runs[0].id : "");
  sel.value = id;
  await selectRun(id);
}

async function selectRun(id) {
  ps.runId = id || null;
  ps.current = null;
  ps.engine = ((ps.runs.find((r) => r.id === id) || {}).selection || {}).simulator || null;
  stopFollow();
  await refreshStages();
  renderDetail();
}

async function refreshStages() {
  if (!ps.runId) { ps.stages = []; renderStrip(); updateFooter(); return; }
  try {
    ps.stages = await api.get(`/api/runs/${ps.runId}/stages`);
  } catch (e) { setStatus(`STAGES: ${e.message}`); return; }
  renderStrip();
  updateFooter();
}

function current() { return ps.stages.find((s) => s.id === ps.current) || null; }

function lamp(s) {
  const st = s.job ? s.job.state : null;
  if (ACTIVE.includes(st)) return "running";
  if (st === "done") return "done";
  if (st) return "failed";
  return s.enabled ? "ready" : "blocked";
}

function renderStrip() {
  const strip = $("#strip");
  strip.innerHTML = "";
  for (const s of ps.stages) {
    const t = el("div", `stage ${lamp(s)}${ps.current === s.id ? " active" : ""}`);
    t.append(el("div", "", s.label.toUpperCase()),
      el("div", "sub", s.job ? s.job.state : (s.enabled ? "ready" : "blocked")));
    t.title = s.reason || "";
    t.addEventListener("click", () => { ps.current = s.id; renderStrip(); renderDetail(); });
    strip.append(t);
  }
}

function renderDetail() {
  const s = current();
  const box = $("#stage-detail");
  box.innerHTML = "";
  $("#stage-title").textContent = s ? s.label.toUpperCase() : "STAGE";
  if (!s) {
    box.append(el("div", "muted", ps.runId ? "Select a stage." : "Create a run in PROJECT first."));
    updateFooter();
    return;
  }
  box.append(el("div", s.reason ? "p" : "ok", s.reason ? `BLOCKED: ${s.reason}`
    : (s.worker ? `Runs on worker ${s.worker}.` : "Runs in the control plane (model registry).")));
  const gate = stageGate(s.id);
  if (gate) {
    const b = el("button", "btn agent", "AGENT BRIEF FOR THIS STAGE");
    b.type = "button";
    b.addEventListener("click", () => openBrief({ stages: [gate], engines: ps.engine ? [ps.engine] : [] }));
    box.append(b);
  }
  const form = el("form", "form");
  form.id = "stage-form";
  for (const p of s.params) {
    const input = el("input", "field");
    input.name = p.name;
    input.value = p.default ?? "";
    if (p.type === "int") input.type = "number";
    form.append(el("label", "", p.name), input);
    if (p.help) form.append(el("div", "help", p.help));
  }
  form.addEventListener("submit", (e) => { e.preventDefault(); runCurrent(); });
  const meta = el("div", "kv mono", "no job yet");
  meta.id = "stage-job";
  const con = el("pre", "console");
  con.id = "stage-console";
  const files = el("div", "files");
  files.id = "stage-files";
  box.append(form, meta, con, files);
  if (s.job) followJob(s.job.id); else stopFollow();
  updateFooter();
}

function updateFooter() {
  const s = current();
  const running = !!(s && s.job && ACTIVE.includes(s.job.state));
  $("#btn-run").disabled = !(s && s.enabled);
  $("#btn-stop").disabled = !running;
}

function stopFollow() {
  if (ps.timer) clearInterval(ps.timer);
  ps.timer = null;
  ps.jobId = null;
}

function followJob(id) {
  stopFollow();
  ps.jobId = id;
  ps.offset = 0;
  $("#stage-console").textContent = "";
  $("#stage-files").innerHTML = "";
  const tick = async () => {
    if (ps.jobId !== id) return;
    try {
      const r = await api.get(`/api/jobs/${id}/log?offset=${ps.offset}`);
      if (ps.jobId !== id) return;
      const con = $("#stage-console");
      if (r.text) { con.textContent += r.text; con.scrollTop = con.scrollHeight; }
      ps.offset = r.offset;
      $("#stage-job").textContent = `JOB ${id} · ${r.state.toUpperCase()}`;
      if (!ACTIVE.includes(r.state)) {
        if (ps.timer) clearInterval(ps.timer);
        ps.timer = null;
        showFiles(id);
        refreshStages();
      }
    } catch (e) { $("#stage-job").textContent = `JOB ${id} · ${e.message}`; }
  };
  ps.timer = setInterval(tick, 1000);
  tick();
}

async function showFiles(id) {
  const box = $("#stage-files");
  box.innerHTML = "";
  const files = await api.get(`/api/jobs/${id}/files`).catch(() => []);
  if (!files.length) return;
  box.append(el("div", "panel-title", "OUTPUTS"));
  let media = 0;
  for (const f of files) {
    const url = `/api/jobs/${id}/file?path=${encodeURIComponent(f.path)}`;
    const a = el("a", "mono", `${f.path}  (${(f.size / 1024).toFixed(1)} kB)`);
    a.href = url;
    a.target = "_blank";
    box.append(a);
    if (media < 4 && /\.(png|gif|jpg)$/i.test(f.path)) { const img = el("img"); img.src = url; box.append(img); media++; }
    if (media < 4 && /\.mp4$/i.test(f.path)) { const v = el("video"); v.src = url; v.controls = true; box.append(v); media++; }
  }
}

async function runCurrent() {
  const s = current();
  if (!s || !s.enabled) return;
  const params = Object.fromEntries(new FormData($("#stage-form")).entries());
  $("#btn-run").disabled = true;
  try {
    const job = await api.post(`/api/runs/${ps.runId}/stages/${s.id}`, { params });
    setStatus(`STARTED ${s.label.toUpperCase()} (${job.id})`);
    await refreshStages();
    followJob(job.id);
  } catch (e) { setStatus(`STAGE REJECTED: ${e.message}`); }
}

async function stopCurrent() {
  const s = current();
  if (!s || !s.job) return;
  try {
    await api.post(`/api/jobs/${s.job.id}/cancel`);
    setStatus(`STOP SENT (${s.job.id})`);
  } catch (e) { setStatus(`STOP FAILED: ${e.message}`); }
  refreshStages();
}
