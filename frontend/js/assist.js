// ASSIST tab: procedure step -> proposed object -> YES -> rollout of the production model with that instruction.
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";

const as = { procedures: [], proposal: null, timer: null };

export async function initAssist() {
  as.procedures = await api.get("/api/assist/procedures").catch(() => []);
  const sel = $("#assist-procedure");
  sel.replaceChildren(...as.procedures.map((p) => new Option(p.id, p.id)));
  sel.addEventListener("change", renderSteps);
  $("#assist-yes").addEventListener("click", handover);
  $("#assist-no").addEventListener("click", () => { as.proposal = null; renderProposal(); });
  document.querySelector('#tabs button[data-tab="assist"]').addEventListener("click", refreshRuns);
  renderSteps();
  await refreshRuns();
}

async function refreshRuns() {
  const runs = await api.get("/api/runs").catch(() => []);
  const sel = $("#assist-run");
  const keep = sel.value;
  sel.replaceChildren(...runs.map((r) => new Option(`${r.id} · ${r.selection.simulator}`, r.id)));
  if (keep) sel.value = keep;
}

function renderSteps() {
  const proc = as.procedures.find((p) => p.id === $("#assist-procedure").value);
  const items = (proc ? proc.steps : []).map((s, i) => {
    const li = el("li", "");
    li.append(el("span", "", `${i + 1}. ${s.step_id}`), el("span", "val", s.tools.join(", ") || "—"));
    li.title = s.description;
    li.addEventListener("click", () => {
      document.querySelectorAll("#assist-steps li").forEach((x) => x.classList.remove("active"));
      li.classList.add("active");
      propose(proc.id, s.step_id);
    });
    return li;
  });
  $("#assist-steps").replaceChildren(...items);
}

async function propose(procedure, stepId) {
  try {
    as.proposal = await api.post("/api/assist/propose", { procedure, step_id: stepId });
  } catch (e) { setStatus(`ASSIST: ${e.message}`); return; }
  renderProposal();
}

function renderProposal() {
  const p = as.proposal;
  const box = $("#assist-proposal");
  if (!p) { box.textContent = "Click the step the clinician is in."; }
  else if (!p.object) { box.textContent = `Step ${p.step_id}: ${p.reason}.`; }
  else { box.textContent = `Step ${p.step_id}: ${p.question}  (${p.reason})\nInstruction: "${p.instruction}"`; }
  const ready = !!(p && p.object);
  $("#assist-yes").disabled = !ready;
  $("#assist-no").disabled = !ready;
}

async function handover() {
  const p = as.proposal;
  if (!p || !p.object) return;
  try {
    const job = await api.post("/api/assist/handover", { run_id: $("#assist-run").value, object: p.object,
      instruction: p.instruction, policy: $("#assist-policy").value });
    setStatus(`HANDOVER ${p.object.toUpperCase()} STARTED (${job.id})`);
    follow(job.id);
  } catch (e) { setStatus(`HANDOVER REJECTED: ${e.message}`); }
}

function follow(id) {
  if (as.timer) clearInterval(as.timer);
  let offset = 0;
  $("#assist-console").textContent = "";
  $("#assist-files").replaceChildren();
  as.timer = setInterval(async () => {
    try {
      const r = await api.get(`/api/jobs/${id}/log?offset=${offset}`);
      if (r.text) $("#assist-console").textContent += r.text;
      offset = r.offset;
      $("#assist-job").textContent = `JOB ${id} · ${r.state.toUpperCase()}`;
      if (!["queued", "running"].includes(r.state)) {
        clearInterval(as.timer);
        as.timer = null;
        const files = await api.get(`/api/jobs/${id}/files`).catch(() => []);
        for (const f of files.filter((x) => /\.(gif|mp4)$/i.test(x.path))) {
          const url = `/api/jobs/${id}/file?path=${encodeURIComponent(f.path)}`;
          if (f.path.endsWith(".mp4")) { const v = el("video"); v.src = url; v.controls = true; v.autoplay = true; $("#assist-files").append(v); }
          else { const img = el("img"); img.src = url; $("#assist-files").append(img); }
        }
      }
    } catch (e) { $("#assist-job").textContent = `JOB ${id} · ${e.message}`; }
  }, 1000);
}
