// AGENT BRIEF drawer: request + stage/engine ticks -> deterministic markdown brief -> copy / download / run.
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";
import { openTemplate } from "./agent.js";

const bs = { markdown: "", paths: [] };

function checkbox(value, label, title) {
  const wrap = el("label", "check");
  const box = el("input");
  box.type = "checkbox";
  box.value = value;
  wrap.title = title || "";
  wrap.append(box, document.createTextNode(` ${label}`));
  return wrap;
}

function ticked(sel) { return [...document.querySelectorAll(`${sel} input:checked`)].map((b) => b.value); }

export async function initBrief() {
  const [gates, engines] = await Promise.all([api.get("/api/agent/gates").catch(() => []),
    api.get("/api/engines").catch(() => [])]);
  const stages = $("#brief-stages");
  for (const g of gates) stages.append(checkbox(g.id, g.title, `${g.status} · ${g.owner}`));
  const eng = $("#brief-engines");
  for (const e of engines) eng.append(checkbox(e.id, e.id, e.label));
  $("#btn-brief").addEventListener("click", () => openBrief());
  $("#brief-close").addEventListener("click", () => $("#brief").classList.add("hidden"));
  $("#brief-build").addEventListener("click", build);
  $("#brief-copy").addEventListener("click", async () => {
    await navigator.clipboard.writeText(bs.markdown);
    setStatus("BRIEF COPIED");
  });
  $("#brief-download").addEventListener("click", () => {
    const a = el("a");
    a.href = URL.createObjectURL(new Blob([bs.markdown], { type: "text/markdown" }));
    a.download = "agent-brief.md";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  $("#brief-run").addEventListener("click", () => {
    $("#brief").classList.add("hidden");
    document.querySelector('#tabs button[data-tab="agent"]').click();
    openTemplate("free_task", { goal: bs.markdown, paths: bs.paths.join(", "), check: "uv run pytest -q" });
  });
}

export function openBrief({ stages = [], engines = [], request = "" } = {}) {
  document.querySelectorAll("#brief-stages input").forEach((b) => { b.checked = stages.includes(b.value); });
  document.querySelectorAll("#brief-engines input").forEach((b) => { b.checked = engines.includes(b.value); });
  if (request) $("#brief-request").value = request;
  $("#brief").classList.remove("hidden");
  $("#brief-request").focus();
}

async function build() {
  const engines = ticked("#brief-engines");
  const extra = $("#brief-new-engine").value.trim();
  if (extra) engines.push(extra);
  try {
    const r = await api.post("/api/agent/brief", { request: $("#brief-request").value, stages: ticked("#brief-stages"),
      engines });
    bs.markdown = r.markdown;
    bs.paths = r.paths;
    $("#brief-out").textContent = r.markdown;
    ["#brief-copy", "#brief-download", "#brief-run"].forEach((s) => { $(s).disabled = false; });
    setStatus("BRIEF READY");
  } catch (e) { setStatus(`BRIEF REJECTED: ${e.message}`); }
}
