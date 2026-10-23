import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";
import { initAgent, openTemplate } from "./agent.js";

// ---- tabs -------------------------------------------------------------------------------
export function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== `tab-${name}`));
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---- PROJECT tab --------------------------------------------------------------------------
const state = { catalog: null, selection: {}, section: null, run: null };

function valueText(section, value) {
  if (Array.isArray(value)) return value.length ? `${value.length} selected` : "none";
  const item = section.items.find((i) => i.id === value);
  return item ? item.label : String(value);
}

function renderTree() {
  const ul = $("#tree");
  ul.innerHTML = "";
  for (const [key, section] of Object.entries(state.catalog.sections)) {
    const li = el("li", key === state.section ? "active" : "");
    li.append(el("span", "", section.title), el("span", "val", valueText(section, state.selection[key])));
    li.addEventListener("click", () => { state.section = key; renderSection(); });
    ul.append(li);
  }
}

function toggleItem(key, section, id) {
  if (section.multi) {
    const cur = new Set(state.selection[key] || []);
    cur.has(id) ? cur.delete(id) : cur.add(id);
    state.selection[key] = section.items.map((i) => i.id).filter((x) => cur.has(x));
  } else {
    state.selection[key] = id;
  }
  renderTree();
  renderSection();
  checkSelection();
}

function renderSection() {
  const key = state.section;
  const section = state.catalog.sections[key];
  $("#section-title").textContent = section.title + (section.multi ? " (multi-select)" : "");
  const box = $("#section-items");
  box.innerHTML = "";
  const chosen = section.multi ? new Set(state.selection[key] || []) : new Set([state.selection[key]]);
  for (const item of section.items) {
    const card = el("div", "item" + (chosen.has(item.id) ? " selected" : "") + (item.status === "planned" ? " disabled" : ""));
    card.append(el("div", "label", item.label));
    if (item.notes) card.append(el("div", "notes", item.notes));
    const st = el("div", `state ${item.status}`, "● " + item.status.replace("_", " ").toUpperCase());
    card.append(st);
    if (item.status !== "ready" && item.harness_task) {
      const b = el("button", "btn agent", "Build with agent");
      b.addEventListener("click", (ev) => { ev.stopPropagation(); buildWithAgent(item.harness_task, key, item.id); });
      card.append(b);
    }
    card.addEventListener("click", () => toggleItem(key, section, item.id));
    box.append(card);
  }
  const add = $("#section-add");
  add.innerHTML = "";
  if (section.add_task) {
    const b = el("button", "btn agent", `+ Add ${section.title.toLowerCase()} with agent`);
    b.addEventListener("click", () => buildWithAgent(section.add_task, key, null));
    add.append(b);
  }
}

function buildWithAgent(taskId, sectionKey, itemId) {
  const prefill = {};
  if (taskId === "add_sim_backend" && itemId) prefill.sim = itemId;
  if (taskId === "add_policy_adapter" && itemId) prefill.model_id = itemId;
  if (taskId === "add_expert" && itemId) prefill.expert_id = itemId;
  if (taskId === "add_variation_axis" && itemId) prefill.axis_id = itemId;
  showTab("agent");
  openTemplate(taskId, prefill);
}

async function checkSelection() {
  const { problems } = await api.post("/api/selection/check", { selection: state.selection });
  const box = $("#problems");
  box.innerHTML = "";
  if (!problems.length) {
    box.append(el("div", "ok", "✓ Selection is runnable."));
  }
  for (const p of problems) {
    const row = el("div", "p");
    row.append(el("span", "", p.message));
    if (p.harness_task) {
      const b = el("button", "btn agent", "Build with agent");
      b.addEventListener("click", () => buildWithAgent(p.harness_task, p.section, p.item));
      row.append(b);
    }
    box.append(row);
  }
  $("#btn-create-run").disabled = problems.length > 0;
}

async function createRun() {
  try {
    const run = await api.post("/api/runs", { selection: state.selection, note: $("#run-note").value });
    state.run = run;
    $("#current-run").textContent = `${run.id}\n${Object.entries(run.selection)
      .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : v}`).join("\n")}`;
    setStatus(`RUN ${run.id} CREATED`);
  } catch (e) {
    setStatus(`RUN REJECTED: ${e.message}`);
  }
}

// ---- PIPELINE tab (stage strip; stages are wired in Plan B) -----------------------------------
const STAGES = ["SCENE", "GENERATE", "INTEL", "TRAIN", "EVAL", "PROMOTE", "ROLLOUT"];
function renderStrip() {
  const strip = $("#strip");
  strip.innerHTML = "";
  STAGES.forEach((s) => {
    const t = el("div", "stage blocked");
    t.append(el("div", "", s), el("div", "sub", "not wired"));
    t.addEventListener("click", () => {
      document.querySelectorAll(".stage").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      $("#stage-title").textContent = s;
      $("#stage-detail").textContent = "Stage execution arrives with Plan B (stages, eval, registry, rollout).";
    });
    strip.append(t);
  });
}

// ---- workers + LOG --------------------------------------------------------------------------
async function refreshWorkers() {
  try {
    const ws = await api.get("/api/workers");
    const box = $("#workers");
    box.innerHTML = "";
    for (const w of ws) {
      const span = el("span", "");
      span.title = w.healthy ? `${w.url}\n${(w.capabilities || []).join(", ")}` : `${w.url}\n${w.error}`;
      span.append(el("span", `lamp ${w.healthy ? "ok" : "off"}`), document.createTextNode(w.name.toUpperCase()));
      box.append(span);
    }
  } catch (e) { /* control plane restarting */ }
}

async function refreshJobs() {
  const [jobs, agentJobs] = await Promise.all([
    api.get("/api/jobs"),
    api.get("/api/agent/jobs").catch(() => []),
  ]);
  const rows = [
    ...jobs.map((j) => ({ id: j.id, kind: j.kind, name: j.name, state: j.state, updated: j.updated })),
    ...agentJobs.map((j) => ({ id: j.id, kind: "agent", name: j.task, state: j.state, updated: j.updated })),
  ].sort((a, b) => b.updated - a.updated);
  const tbody = $("#jobs-table tbody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = el("tr");
    const st = el("td");
    st.append(el("span", `badge ${r.state}`, r.state));
    tr.append(el("td", "mono", r.id), el("td", "", r.kind), el("td", "", r.name), st,
      el("td", "mono", new Date(r.updated * 1000).toLocaleTimeString()));
    tbody.append(tr);
  }
}

// ---- boot -----------------------------------------------------------------------------------
async function boot() {
  state.catalog = await api.get("/api/catalog");
  state.selection = JSON.parse(JSON.stringify(state.catalog.defaults));
  state.section = Object.keys(state.catalog.sections)[0];
  renderTree();
  renderSection();
  renderStrip();
  await checkSelection();
  $("#btn-create-run").addEventListener("click", createRun);
  await initAgent();
  refreshWorkers();
  refreshJobs();
  setInterval(refreshWorkers, 5000);
  setInterval(refreshJobs, 4000);
}
boot().catch((e) => setStatus(`ERROR: ${e.message}`));
