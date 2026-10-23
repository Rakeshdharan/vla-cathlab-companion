// MODELS tab: register evaluated checkpoints as versions, promote by the rule, roll back.
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";
import { prefillReport } from "./report.js";

const ms = { evals: [], registry: null };
const pct = (x) => `${Math.round(x * 100)}%`;

export async function loadEvals() {
  ms.evals = await api.get("/api/evals").catch(() => []);
  return ms.evals;
}

export function pickedEvals(box) {
  return Object.fromEntries([...box.querySelectorAll("select")].filter((s) => s.value).map((s) => [s.name, s.value]));
}

export function evalPicker(box, evals, prefix) {
  const keep = pickedEvals(box);
  box.innerHTML = "";
  const engines = [...new Set(evals.map((e) => e.engine))].sort();
  if (!engines.length) box.append(el("div", "muted", "No eval reports yet: run EVALUATE in PIPELINE."));
  for (const engine of engines) {
    const sel = el("select", "field");
    sel.name = engine;
    sel.id = `${prefix}-${engine}`;
    sel.append(new Option("—", ""));
    for (const e of evals.filter((x) => x.engine === engine)) {
      sel.append(new Option(`${e.model} · ${pct(e.rate)} (n=${e.n}) · ${e.path}`, e.path));
    }
    if (keep[engine]) sel.value = keep[engine];
    const label = el("label", "", engine);
    label.htmlFor = sel.id;
    box.append(label, sel);
  }
}

export async function initModels() {
  $("#btn-register").addEventListener("click", register);
  $("#btn-rollback").addEventListener("click", rollback);
  $("#btn-models-refresh").addEventListener("click", refreshModels);
  document.querySelector('#tabs button[data-tab="models"]').addEventListener("click", refreshModels);
  await refreshModels();
}

export async function refreshModels() {
  evalPicker($("#register-evals"), await loadEvals(), "reg");
  try {
    ms.registry = await api.get("/api/registry");
  } catch (e) { setStatus(`REGISTRY: ${e.message}`); return; }
  renderRegistry();
}

function cell(s) {
  return s ? `${pct(s.rate)} (${pct(s.ci[0])}–${pct(s.ci[1])}) · p95 ${Math.round(s.p95_ms)} ms` : "—";
}

function renderRegistry() {
  const r = ms.registry;
  $("#production").textContent = r.production ? `${r.name}@${r.production}` : "none";
  $("#btn-rollback").disabled = !r.production;
  $("#promotion-rule").textContent = `Promote if the 95% CI low on ${r.rule.primary_engine} beats production, `
    + `with ≥ ${r.rule.min_episodes} episodes and p95 latency ≤ ${r.rule.latency_budget_ms_p95} ms. `
    + "FORCE skips the success check, never the latency check.";
  const engines = [...new Set(r.versions.flatMap((v) => Object.keys(v.eval_reports)))].sort();
  const head = el("tr");
  ["VERSION", "MODEL", ...engines.map((e) => e.toUpperCase()), "STATE", ""].forEach((h) => head.append(el("th", "", h)));
  $("#versions thead").replaceChildren(head);
  const rows = [...r.versions].reverse().map((v) => {
    const tr = el("tr");
    tr.append(el("td", "mono", v.card.version),
      el("td", "mono", v.card.adapter === "expert" ? "expert" : v.card.ckpt_uri));
    for (const e of engines) tr.append(el("td", "mono", cell(v.summary[e])));
    const st = el("td");
    st.append(el("span", `badge ${v.card.promoted ? "done" : ""}`, v.card.promoted ? "PRODUCTION" : "candidate"));
    const act = el("td");
    const promoteBtn = el("button", "btn ok", "PROMOTE");
    promoteBtn.disabled = v.card.promoted;
    promoteBtn.addEventListener("click", () => promote(v.card.version, false));
    const forceBtn = el("button", "btn", "FORCE");
    forceBtn.title = "Skip the success check (latency is still checked)";
    forceBtn.disabled = v.card.promoted;
    forceBtn.addEventListener("click", () => promote(v.card.version, true));
    const reportBtn = el("button", "btn", "REPORT");
    reportBtn.addEventListener("click", () => openReportFor(v));
    act.append(promoteBtn, forceBtn, reportBtn);
    tr.append(st, act);
    return tr;
  });
  $("#versions tbody").replaceChildren(...rows);
  $("#history").replaceChildren(...[...r.history].reverse().map((h) => el("div", "",
    `${new Date(h.at * 1000).toLocaleTimeString()}  ${h.action.toUpperCase()}  ${h.version ?? "none"}  ${h.reason}`)));
}

async function register() {
  const body = {
    adapter: $("#reg-adapter").value,
    ckpt_uri: $("#reg-ckpt").value.trim(),
    dataset_versions: $("#reg-datasets").value.split(",").map((s) => s.trim()).filter(Boolean),
    eval_reports: pickedEvals($("#register-evals")),
  };
  try {
    const v = await api.post("/api/registry/register", body);
    setStatus(`REGISTERED ${v.card.model_name}@${v.card.version}`);
    await refreshModels();
  } catch (e) { setStatus(`REGISTER REJECTED: ${e.message}`); }
}

async function promote(version, force) {
  try {
    const d = await api.post("/api/registry/promote", { version, force });
    setStatus(`${d.promoted ? "PROMOTED" : "NOT PROMOTED"} ${version}: ${d.reason}`);
    await refreshModels();
  } catch (e) { setStatus(`PROMOTE FAILED: ${e.message}`); }
}

async function rollback() {
  if (!window.confirm(`Roll back production ${ms.registry.production}?`)) return;
  try {
    const r = await api.post("/api/registry/rollback");
    setStatus(`ROLLED BACK: production is ${r.production ?? "none"}`);
    await refreshModels();
  } catch (e) { setStatus(`ROLLBACK FAILED: ${e.message}`); }
}

function openReportFor(v) {
  document.querySelector('#tabs button[data-tab="report"]').click();
  prefillReport(`${v.card.model_name}@${v.card.version}`, v.eval_reports);
}
