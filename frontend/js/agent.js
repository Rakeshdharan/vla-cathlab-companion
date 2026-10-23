import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";

const st = { templates: {}, agents: null, current: null, logOffset: 0, timer: null };
const ACTIVE = new Set(["created", "worktree", "agent_running", "guard", "checks"]);

export async function initAgent() {
  try {
    const [tasks, agents] = await Promise.all([api.get("/api/agent/tasks"), api.get("/api/agent/agents")]);
    tasks.forEach((t) => { st.templates[t.id] = t; });
    st.agents = agents;
  } catch (e) {
    $("#agent-form-desc").textContent = "Agent harness is not mounted on this control plane (start it with the agent enabled).";
    return;
  }
  renderTemplates();
  refreshAgentJobs();
  setInterval(refreshAgentJobs, 3000);
}

function renderTemplates() {
  const ul = $("#templates");
  ul.innerHTML = "";
  for (const t of Object.values(st.templates)) {
    const li = el("li", "");
    li.append(el("span", "", t.title));
    li.addEventListener("click", () => openTemplate(t.id, {}));
    ul.append(li);
  }
}

function inputFor(p, value) {
  let input;
  if (p.type === "choice") {
    input = el("select", "field");
    p.choices.forEach((c) => { const o = el("option", "", c); o.value = c; input.append(o); });
  } else if (p.type === "text") {
    input = el("textarea", "field");
  } else if (p.type === "bool") {
    input = el("select", "field");
    ["true", "false"].forEach((c) => { const o = el("option", "", c); o.value = c; input.append(o); });
  } else {
    input = el("input", "field");
    if (p.type === "int" || p.type === "float") input.type = "number";
    if (p.type === "float") input.step = "any";
  }
  input.name = p.name;
  const v = value !== undefined ? value : p.default;
  if (v !== null && v !== undefined) input.value = String(v);
  return input;
}

export function openTemplate(taskId, prefill) {
  const t = st.templates[taskId];
  $("#agent-job-wrap").classList.add("hidden");
  $("#agent-form-wrap").classList.remove("hidden");
  if (!t) {
    $("#agent-form-title").textContent = "UNKNOWN TASK";
    $("#agent-form-desc").textContent = `Template '${taskId}' not found.`;
    $("#agent-form").innerHTML = "";
    return;
  }
  $("#agent-form-title").textContent = t.title;
  $("#agent-form-desc").textContent = t.description;
  const form = $("#agent-form");
  form.innerHTML = "";
  for (const p of t.params) {
    form.append(el("label", "", p.name + (p.required ? " *" : "")), inputFor(p, prefill[p.name]));
    if (p.help) form.append(el("div", "help", p.help));
  }
  form.append(el("label", "", "extra instructions"));
  const extra = el("textarea", "field");
  extra.name = "__extra";
  form.append(extra);
  form.append(el("label", "", "agent"));
  const sel = el("select", "field");
  sel.name = "__agent";
  (st.agents?.agents || []).forEach((a) => {
    const o = el("option", "", `${a.name}: ${a.description}`);
    o.value = a.name;
    if (a.name === st.agents.default) o.selected = true;
    sel.append(o);
  });
  form.append(sel);
  const paths = el("div", "help", "Allowed paths: " + t.allowed_paths.join(", ") +
    " · Checks: " + t.acceptance.map((c) => c.name).join(", "));
  form.append(el("div", ""), paths);
  const actions = el("div", "actions row");
  const go = el("button", "btn primary", "Start agent");
  go.type = "submit";
  actions.append(go);
  form.append(actions);
  form.onsubmit = async (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    const params = {};
    for (const p of t.params) {
      const v = fd.get(p.name);
      if (v !== null && v !== "") params[p.name] = v;
    }
    try {
      const job = await api.post("/api/agent/jobs", { task: t.id, params, extra: fd.get("__extra") || "", agent: fd.get("__agent") });
      setStatus(`AGENT JOB ${job.id} STARTED`);
      openJob(job.id);
      refreshAgentJobs();
    } catch (e) {
      setStatus(`AGENT REJECTED: ${e.message}`);
    }
  };
}

async function refreshAgentJobs() {
  const jobs = await api.get("/api/agent/jobs").catch(() => []);
  const ul = $("#agent-jobs");
  ul.innerHTML = "";
  for (const j of jobs) {
    const li = el("li", j.id === st.current ? "active" : "");
    li.append(el("span", "mono", `${j.task}`), el("span", `badge ${j.state}`, j.state));
    li.addEventListener("click", () => openJob(j.id));
    ul.append(li);
  }
}

async function openJob(id) {
  st.current = id;
  st.logOffset = 0;
  $("#agent-console").textContent = "";
  $("#agent-form-wrap").classList.add("hidden");
  $("#agent-job-wrap").classList.remove("hidden");
  if (st.timer) clearInterval(st.timer);
  await pollJob();
  st.timer = setInterval(pollJob, 1500);
}

async function pollJob() {
  const id = st.current;
  if (!id) return;
  const job = await api.get(`/api/agent/jobs/${id}`);
  $("#agent-job-title").textContent = `${job.title} · ${job.id}`;
  const checks = job.checks.map((c) => `${c.ok ? "✓" : "✕"} ${c.name}`).join("  ") || "-";
  $("#agent-job-meta").textContent =
    `STATE    ${job.state}${job.error ? "\nERROR    " + job.error : ""}\nAGENT    ${job.agent}\nBRANCH   ${job.branch}` +
    `\nFILES    ${job.changed_files.join(", ") || "-"}\nCHECKS   ${checks}`;
  $("#btn-push").disabled = job.state !== "ready_for_review";
  $("#btn-discard").disabled = ACTIVE.has(job.state) || job.state === "discarded" || job.state === "pushed";
  const log = await api.get(`/api/agent/jobs/${id}/log?offset=${st.logOffset}`);
  if (log.text) {
    const c = $("#agent-console");
    c.textContent += log.text;
    c.scrollTop = c.scrollHeight;
  }
  st.logOffset = log.offset;
  if (!ACTIVE.has(job.state) && st.timer) { clearInterval(st.timer); st.timer = null; }
}

async function showText(path) {
  const text = await api.get(path);
  $("#agent-console").textContent = text || "(empty)";
  if (st.timer) { clearInterval(st.timer); st.timer = null; }
}

$("#btn-view-prompt").addEventListener("click", () => st.current && showText(`/api/agent/jobs/${st.current}/prompt`));
$("#btn-view-diff").addEventListener("click", () => st.current && showText(`/api/agent/jobs/${st.current}/diff`));
$("#btn-push").addEventListener("click", async () => {
  try { const j = await api.post(`/api/agent/jobs/${st.current}/push`); setStatus(`PUSHED ${j.branch}`); pollJob(); }
  catch (e) { setStatus(`PUSH FAILED: ${e.message}`); }
});
$("#btn-discard").addEventListener("click", async () => {
  try { await api.post(`/api/agent/jobs/${st.current}/discard`); setStatus("DISCARDED"); pollJob(); refreshAgentJobs(); }
  catch (e) { setStatus(`DISCARD FAILED: ${e.message}`); }
});
