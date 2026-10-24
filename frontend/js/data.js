// DATA tab: published datasets (configs/datasets.yaml) and local generate outputs (var/runs/**/manifest.yaml).
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";

const num = (n) => Number(n).toLocaleString("en-US");

function table(sel, heads, rows) {
  const tr = el("tr");
  heads.forEach((h) => tr.append(el("th", "", h)));
  $(`${sel} thead`).replaceChildren(tr);
  $(`${sel} tbody`).replaceChildren(...rows);
}

function link(text, href) {
  if (!href) return el("span", "mono", text);
  const a = el("a", "mono", text);
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener";
  return a;
}

function publishedRow(d) {
  const tr = el("tr");
  const name = el("td");
  name.append(link(d.revision ? `${d.repo_id}@${d.revision}` : d.repo_id, d.url));
  tr.append(name, el("td", "mono", d.embodiment), el("td", "mono", d.engine), el("td", "", d.physics),
    el("td", "mono", num(d.episodes)), el("td", "mono", num(d.frames)), el("td", "mono", String(d.fps)));
  const vis = el("td");
  vis.append(el("span", `badge ${d.visibility === "public" ? "done" : ""}`, d.visibility.toUpperCase()));
  tr.append(vis, el("td", "muted", d.local_path ? `${d.notes} · ${d.local_path}` : d.notes));
  return tr;
}

function localRow(m) {
  const tr = el("tr");
  tr.append(el("td", "mono", `${m.dataset_id}@${m.version}`), el("td", "mono", m.scene_id), el("td", "mono", m.source),
    el("td", "mono", String(m.n_episodes)), el("td", "mono", m.run_id), el("td", "mono", m.path),
    el("td", "muted", m.notes));
  return tr;
}

export async function refreshData() {
  let body;
  try {
    body = await api.get("/api/datasets");
  } catch (e) { setStatus(`DATASETS: ${e.message}`); return; }
  table("#datasets-published", ["DATASET", "ROBOT", "ENGINE", "PHYSICS", "EPISODES", "FRAMES", "FPS", "", "NOTES"],
    body.published.map(publishedRow));
  table("#datasets-local", ["DATASET", "SCENE", "SOURCE", "EPISODES", "RUN", "PATH (var/)", "NOTES"],
    body.local.map(localRow));
  $("#datasets-local-empty").classList.toggle("hidden", body.local.length > 0);
}

export async function initData() {
  $("#btn-data-refresh").addEventListener("click", refreshData);
  document.querySelector('#tabs button[data-tab="data"]').addEventListener("click", refreshData);
  await refreshData();
}
