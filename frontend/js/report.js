// REPORT tab: failure report from eval reports -> failing cells -> CREATE NEXT RUN (targeted generate).
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";
import { evalPicker, loadEvals, pickedEvals } from "./models.js";
import { refreshRuns } from "./pipeline.js";

const rs = { current: null };

export async function initReport() {
  $("#btn-build-report").addEventListener("click", build);
  $("#btn-next-run").addEventListener("click", createNextRun);
  document.querySelector('#tabs button[data-tab="report"]').addEventListener("click", refreshPickers);
  await refreshPickers();
  await refreshReports();
}

async function refreshPickers() {
  evalPicker($("#report-evals"), await loadEvals(), "rep");
}

export async function prefillReport(model, evalReports) {
  await refreshPickers();
  $("#rep-model").value = model;
  for (const [engine, path] of Object.entries(evalReports)) {
    const sel = $(`#rep-${engine}`);
    if (sel) sel.value = path;
  }
}

async function refreshReports() {
  const list = await api.get("/api/reports").catch(() => []);
  const items = list.map((r) => {
    const li = el("li", rs.current && rs.current.id === r.id ? "active" : "");
    const rates = Object.entries(r.overall).map(([e, o]) => `${e} ${Math.round(o.rate * 100)}%`).join(" · ");
    li.append(el("span", "", r.model), el("span", "val", `${rates} · ${r.cells} cells`));
    li.addEventListener("click", () => openReport(r.id));
    return li;
  });
  $("#reports").replaceChildren(...items);
}

async function openReport(id) {
  try {
    rs.current = await api.get(`/api/reports/${id}`);
  } catch (e) { setStatus(`REPORT: ${e.message}`); return; }
  render();
  refreshReports();
}

function render() {
  const r = rs.current;
  $("#report-title").textContent = `REPORT ${r.id}`;
  $("#report-md").textContent = r.markdown;
  const cells = r.next_run.cells;
  const episodes = cells.reduce((a, c) => a + c.k, 0);
  $("#next-run-info").textContent = cells.length
    ? `Next run: ${cells.length} train-split failing cells, ${episodes} targeted episodes (GENERATE DATA in PIPELINE).`
    : "No train-split failing cells: follow the recommendations (widen the spec) instead.";
  $("#btn-next-run").disabled = !cells.length;
}

async function build() {
  const body = { model: $("#rep-model").value.trim() || "unnamed", eval_reports: pickedEvals($("#report-evals")),
    k: Number($("#rep-k").value) || 8 };
  try {
    rs.current = await api.post("/api/reports", body);
    render();
    setStatus(`REPORT ${rs.current.id} READY`);
    refreshReports();
  } catch (e) { setStatus(`REPORT REJECTED: ${e.message}`); }
}

async function createNextRun() {
  if (!rs.current) return;
  try {
    const run = await api.post(`/api/reports/${rs.current.id}/next-run`, {});
    setStatus(`NEXT RUN ${run.id} CREATED: run GENERATE DATA in PIPELINE`);
    await refreshRuns(run.id);
    document.querySelector('#tabs button[data-tab="pipeline"]').click();
  } catch (e) { setStatus(`NEXT RUN REJECTED: ${e.message}`); }
}
