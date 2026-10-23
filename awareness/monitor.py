"""Local monitoring page: pick a clip, run it through the engine, see the frames, each VLM reading with its
rationale, the predicted object, the spoken question (with audio) and the rollout params. Not for production."""
import argparse
import json
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from awareness.assistant import Assistant
from awareness.clips import LabeledClip, load_clip
from awareness.engine import AwarenessEngine
from awareness.procedure import ProcedureSpec
from awareness.vlm import BudgetExceeded, ScriptedVLM, VLMReading

ROOT = Path(__file__).resolve().parents[1]


class _Task:
    def instructions(self, obj):
        return [f"Hand me the {obj}"]


class _Recorder:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def classify(self, frames, procedure):
        n = len(self.calls) + 1
        t = time.perf_counter()
        r = self.inner.classify(frames, procedure)
        self.calls.append({"frame": n - 1, "window": list(range(n - len(frames), n)), "reading": asdict(r),
                           "latency_ms": round((time.perf_counter() - t) * 1000)})
        return r


class _RecEngine(AwarenessEngine):
    states: list

    def observe(self, frame, t):
        s = super().observe(frame, t)
        self.states = getattr(self, "states", []) + [asdict(s)]
        return s


def run_clip(clip: LabeledClip, procedure: ProcedureSpec, vlm, speaker=None, fps: float = 1.0) -> dict:
    rec = _Recorder(vlm)
    engine = _RecEngine(procedure, rec)
    questions: list[dict] = []

    def speak(q: str) -> None:
        path = speaker.speak(q) if speaker else None
        questions.append({"text": q, "audio": path.name if path else None, "frame": len(engine.states) - 1})

    assistant = Assistant(engine, _Task(), confirm=lambda q: True, speak=speak, cooldown_s=0.0)
    handovers, error = [], None
    try:
        handovers = [h.rollout_params() for h in assistant.run(clip.stream(fps))]
    except BudgetExceeded as e:
        error = str(e)
    steps = [{**c, "state": engine.states[c["frame"]] if c["frame"] < len(getattr(engine, "states", [])) else None}
             for c in rec.calls]
    final = getattr(engine, "states", [{}])[-1].get("step") if getattr(engine, "states", None) else None
    budget = getattr(vlm, "budget", None)
    return {"clip_id": clip.clip_id, "expected_step": clip.step_id, "final_step": final, "correct": final == clip.step_id,
            "steps": steps, "questions": questions, "handovers": handovers, "error": error,
            "budget": asdict(budget) if budget else None}


PAGE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Awareness engine</title>
<style>
:root{--bg:#0f1419;--panel:#171e26;--line:#26303b;--fg:#e6edf3;--mute:#8b98a5;--acc:#4cc2ff;--ok:#3fb950;--bad:#f85149;--warn:#d29922}
@media (prefers-color-scheme:light){:root{--bg:#f4f6f8;--panel:#fff;--line:#d7dde3;--fg:#15202b;--mute:#5b6b7a;--acc:#0b6bcb;--ok:#1a7f37;--bad:#cf222e;--warn:#9a6700}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:18px;max-width:1180px;margin-inline:auto}
h1{font-size:20px;margin:0}.sub{color:var(--mute);margin:2px 0 14px}.pipe{display:flex;gap:6px;flex-wrap:wrap;color:var(--mute);font-size:12px;margin-bottom:14px}
.pipe span{border:1px solid var(--line);border-radius:99px;padding:2px 10px}.pipe span.on{color:var(--fg);border-color:var(--acc)}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;margin-bottom:14px}
select,button,label{font:inherit}select,button{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:6px 10px}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600}button:disabled{opacity:.5}
.tl{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-bottom:14px}
.st{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px;position:relative}
.st b{display:block;font-size:12px}.st small{color:var(--mute)}.st.exp{outline:1px dashed var(--mute)}.st.eng{border-color:var(--ok);box-shadow:0 0 0 2px color-mix(in srgb,var(--ok) 35%,transparent)}
.st .tag{position:absolute;top:6px;right:8px;font-size:11px;color:var(--mute)}.st.vlm .tag{color:var(--acc)}
.grid{display:grid;grid-template-columns:minmax(260px,380px) 1fr;gap:14px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
.card h3{margin:0 0 8px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute)}
#big{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;border:1px solid var(--line);image-rendering:auto;background:#000}
.strip{display:flex;gap:6px;margin-top:8px}.strip img{width:25%;flex:1;border-radius:6px;border:3px solid var(--line);cursor:pointer;opacity:.55}
.strip img.win{opacity:1;border-color:var(--acc)}.strip img.cur{border-color:var(--ok)}
.kv{display:grid;grid-template-columns:110px 1fr;gap:4px 10px}.kv div:nth-child(odd){color:var(--mute)}
.meter{height:7px;background:var(--line);border-radius:4px;overflow:hidden;margin-top:3px}.meter i{display:block;height:100%;background:var(--acc)}
.big{font-size:22px;font-weight:650}.ok{color:var(--ok)}.bad{color:var(--bad)}.q{font-size:20px;margin:4px 0 8px}
code,pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:2px 6px;font-size:12px}pre{padding:8px;overflow:auto;margin:6px 0 0}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid var(--line);padding:5px;text-align:left;vertical-align:top}tr.sel{background:color-mix(in srgb,var(--acc) 14%,transparent)}tr{cursor:pointer}
.wrap{overflow-x:auto}.mute{color:var(--mute)}
</style>
<h1>Awareness engine</h1><div class=sub>What the clinician is doing → which tool they will need next → ask by voice → parameters for the robot rollout</div>
<div class=pipe><span class=on>1 frames</span><span class=on>2 VLM reading</span><span class=on>3 procedure state machine</span><span class=on>4 proposal + voice</span><span class=on>5 robot rollout params</span></div>
<div class=bar><label>Clip <select id=clip></select></label>
<label>VLM <select id=mode><option value=demo>demo (canned, free)</option><option value=live>live (Claude, capped)</option></select></label>
<label><input type=checkbox id=speak> speak (ElevenLabs, capped)</label>
<button class=primary id=go>Run</button><button id=play disabled>▶ Replay</button><span id=status class=mute></span></div>
<div class=tl id=tl></div>
<div class=grid><div>
 <div class=card><h3>1 · Camera frames</h3><img id=big alt=""><div class=strip id=strip></div><div class="mute" id=cap style="margin-top:6px"></div></div>
 <div class=card id=resultcard hidden><h3>Result</h3><div id=result></div></div></div>
<div>
 <div class=card><h3>2 · VLM reading (last frames in the window)</h3><div class=kv id=vlm><div>—</div><div></div></div></div>
 <div class=card><h3>3 · Procedure state machine</h3><div class=kv id=eng><div>—</div><div></div></div></div>
 <div class=card><h3>4 · Proposal and voice</h3><div id=prop class=mute>Nothing proposed yet</div></div>
 <div class=card><h3>5 · Robot rollout parameters</h3><div id=roll class=mute>—</div></div></div></div>
<div class=card><h3>Every step of the run</h3><div class=wrap id=tbl class=mute>Run a clip to see the log.</div></div>
<script>
const $=id=>document.getElementById(id);let P=[],R=null,cur=0,timer=null,N=0;
const pct=x=>Math.round((x||0)*100);const meter=x=>`<div class=meter><i style="width:${pct(x)}%"></i></div>`;
Promise.all([fetch('/api/procedure').then(r=>r.json()),fetch('/api/clips').then(r=>r.json())]).then(([p,cs])=>{P=p;
 cs.forEach(c=>{const o=document.createElement('option');o.value=c.clip_id;o.textContent=`${c.clip_id} · ${c.step_id}`;o.n=c.n;o.step=c.step_id;$('clip').append(o)});pick()});
$('clip').onchange=pick;
function pick(){const o=$('clip').selectedOptions[0];if(!o)return;N=o.n;R=null;clearInterval(timer);$('play').disabled=true;$('resultcard').hidden=true;
 $('strip').innerHTML=[...Array(N).keys()].map(i=>`<img src=/clip/${o.value}/${i}.png onclick="show(${i})" id=s${i}>`).join('');
 $('big').src=`/clip/${o.value}/${N-1}.png`;$('cap').textContent=`${N} frames · expected step: ${o.step}`;tl(null,null,o.step);
 for(const k of['vlm','eng'])$(k).innerHTML='<div>—</div><div></div>';$('prop').innerHTML='<span class=mute>Nothing proposed yet</span>';$('roll').textContent='—';$('tbl').textContent='Run a clip to see the log.'}
function tl(eng,vlm,exp){$('tl').innerHTML=P.map((s,i)=>`<div class="st ${s.step_id==exp?'exp':''} ${s.step_id==eng?'eng':''} ${s.step_id==vlm?'vlm':''}"><span class=tag>${s.step_id==eng?'engine ✔':''}${s.step_id==vlm&&s.step_id!=eng?'VLM':''}</span><small>${i+1}</small><b>${s.step_id.replace(/_/g,' ')}</b><small>${s.tools.length?'tools: '+s.tools.join(', '):'no tool'}${s.step_id==exp?' · expected':''}</small></div>`).join('')}
function show(i,auto=false){cur=i;const o=$('clip').selectedOptions[0];$('big').src=`/clip/${o.value}/${i}.png`;
 const s=R&&R.steps[i];document.querySelectorAll('.strip img').forEach((e,k)=>{e.classList.toggle('cur',k==i);e.classList.toggle('win',!!s&&s.window.includes(k))});
 document.querySelectorAll('#tbl tr').forEach(e=>e.classList.toggle('sel',e.dataset.i==i));
 if(!s)return;const r=s.reading,st=s.state||{};tl(st.step,r.step_id,R.expected_step);
 $('cap').textContent=`frame ${i+1}/${N} · window of ${s.window.length} frame(s) sent to the VLM · ${s.latency_ms} ms`;
 $('vlm').innerHTML=`<div>step</div><div><b>${r.step_id}</b></div><div>confidence</div><div>${pct(r.confidence)}%${meter(r.confidence)}</div><div>tool in hand</div><div>${r.tool||'none seen'}</div><div>rationale</div><div>${r.rationale}</div>`;
 $('eng').innerHTML=`<div>current step</div><div><b>${st.step||'unknown yet'}</b></div><div>confidence</div><div>${pct(st.confidence)}%${meter(st.confidence)}</div><div>tool check</div><div>${st.corroborated?'<span class=ok>tool matches the step (+confidence)</span>':'no corroboration'}</div><div>next tool</div><div><b>${st.next_tool||'—'}</b> <span class=mute>(from the procedure, not the VLM)</span></div>`;
 const qi=R.questions.reduce((acc,q,k)=>q.frame<=i?k:acc,-1);
 if(qi!==shownQ){shownQ=qi;renderProp(qi)}
 if(i===R.peak){$('cap').textContent+=' · ★ peak confidence: voice plays here';if(auto&&qi>=0)playQ(qi)}}
let A=null,shownQ=-2;
function playQ(qi){const q=R.questions[qi];if(!q||!q.audio)return;if(A)A.pause();A=new Audio('/audio/'+q.audio);
 A.play().then(()=>{$('pnote').textContent='playing…';A.onended=()=>$('pnote').textContent='done'}).catch(()=>{$('pnote').textContent='blocked by the browser: click ▶ Play'})}
function renderProp(qi){if(qi<0){$('prop').innerHTML='<span class=mute>Waiting for a confident step…</span>';$('roll').textContent='—';return}
 const q=R.questions[qi];$('prop').innerHTML=`<div class=q>“${q.text}”</div>${q.audio?`<button id=pl>▶ Play</button> <span id=pnote class=mute></span>`:'<span class=mute>voice off: tick “speak” and run again to hear it</span>'}<div class=mute style="margin-top:6px">asked at frame ${q.frame+1} · auto-confirmed for this test</div>`;
 $('roll').innerHTML=`<pre>${JSON.stringify(R.handovers[qi]||{},null,1)}</pre>`;
 if(q.audio)$('pl').onclick=()=>playQ(qi)}
$('go').onclick=async()=>{$('status').textContent='running…';$('go').disabled=true;clearInterval(timer);
 try{R=await(await fetch('/api/run',{method:'POST',body:JSON.stringify({clip:$('clip').value,mode:$('mode').value,speak:$('speak').checked})})).json()}catch(e){$('status').textContent='failed: '+e;$('go').disabled=false;return}
 $('go').disabled=false;$('status').textContent='';$('play').disabled=false;
 const qf=R.questions.length?Math.min(...R.questions.map(q=>q.frame)):null;let pk=null,best=-1;
 R.steps.forEach(s=>{const c=s.state?s.state.confidence||0:0;if(qf!==null&&s.frame>=qf&&c>best){best=c;pk=s.frame}});R.peak=pk;
 R.proposedAt=R.questions.length?R.steps.findIndex(s=>s.state&&s.state.next_tool):null;if(R.proposedAt<0)R.proposedAt=null;
 $('tbl').innerHTML=`<table><tr><th>#</th><th>window</th><th>VLM step</th><th>conf</th><th>tool</th><th>engine step</th><th>next tool</th><th>ms</th></tr>${R.steps.map(s=>`<tr data-i=${s.frame} onclick="show(${s.frame})"><td>${s.frame}${s.frame===R.peak?' ★':''}</td><td>${s.window.join(',')}</td><td>${s.reading.step_id}</td><td>${pct(s.reading.confidence)}%</td><td>${s.reading.tool||''}</td><td>${s.state?s.state.step||'':''}</td><td>${s.state?s.state.next_tool||'':''}</td><td>${s.latency_ms}</td></tr>`).join('')}</table>`;
 const ok=R.correct;$('resultcard').hidden=false;$('result').innerHTML=`<div class=big>${R.final_step||'unknown'} <span class="${ok?'ok':'bad'}">${ok?'✔ matches':'✘ expected '+R.expected_step}</span></div><div>Proposed object: <b>${R.handovers.map(h=>h.object).join(', ')||'none'}</b></div>${R.error?`<div class=bad>${R.error}</div>`:''}<div class=mute>VLM budget: ${R.budget?`${R.budget.calls}/${R.budget.max_calls} calls, ${R.budget.tokens}/${R.budget.max_tokens} tokens`:'canned (no API)'}</div>`;
 replay()};
$('play').onclick=replay;
function replay(){clearInterval(timer);if(A)A.pause();shownQ=-2;let i=0;show(0,true);timer=setInterval(()=>{i++;if(!R||i>=R.steps.length){clearInterval(timer);return}show(i,true)},900)}
</script>
"""


def make_handler(clips_dir: Path, procedure: ProcedureSpec, make_vlm, make_speaker):
    audio_dir = make_speaker().cache_dir

    class H(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if self.path == "/":
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/procedure":
                return self._send(json.dumps([s.model_dump() for s in procedure.steps]).encode(), "application/json")
            if self.path == "/api/clips":
                import yaml

                rows = [{"clip_id": p.parent.name, "step_id": (l := yaml.safe_load(p.read_text()))["step_id"], "n": len(l["frames"])}
                        for p in sorted(clips_dir.glob("*/labels.yaml"))]
                return self._send(json.dumps(rows).encode(), "application/json")
            if parts[0] == "clip" and len(parts) == 3 and ".." not in self.path:
                f = clips_dir / parts[1] / f"{int(parts[2].removesuffix('.png')):02d}.png"
                return self._send(f.read_bytes(), "image/png") if f.exists() else self._send(b"", "text/plain", 404)
            if parts[0] == "audio" and len(parts) == 2 and "/" not in parts[1] and ".." not in parts[1]:
                f = audio_dir / parts[1]
                if not f.exists():
                    return self._send(b"", "text/plain", 404)
                data, rng = f.read_bytes(), self.headers.get("Range", "")
                if rng.startswith("bytes="):
                    a, _, b = rng[6:].partition("-")
                    start = int(a or 0)
                    end = min(int(b) if b else len(data) - 1, len(data) - 1)
                    self.send_response(206)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return self.wfile.write(data[start : end + 1])
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return self.wfile.write(data)
            self._send(b"", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/run":
                return self._send(b"", "text/plain", 404)
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            clip_dir = clips_dir / Path(req["clip"]).name
            clip = load_clip(clip_dir)
            vlm = make_vlm() if req.get("mode") == "live" else ScriptedVLM(
                [VLMReading(clip.step_id, 0.7, "demo reading (canned, no API call)", tool=clip.tools[0] if clip.tools else None)])
            out = run_clip(clip, procedure, vlm, speaker=make_speaker() if req.get("speak") else None)
            self._send(json.dumps(out).encode(), "application/json")

        def log_message(self, *a):
            pass

    return H


def main() -> None:
    from awareness.vlm import AnthropicVLM, VLMBudget
    from awareness.voice import Speaker, VoiceBudget

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default="var/awareness/clips")
    ap.add_argument("--procedure", default="configs/procedures/coronary_angiography.yaml")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-calls", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=300)
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(".env")
    except ImportError:
        pass
    voice_budget = VoiceBudget(max_chars=a.max_chars)
    speaker = Speaker(budget=voice_budget)
    handler = make_handler(Path(a.clips), ProcedureSpec.from_yaml(a.procedure),
                           lambda: AnthropicVLM(budget=VLMBudget(max_calls=a.max_calls, max_tokens=20_000), cache_dir="var/awareness/vlm_cache"), lambda: speaker)
    print(f"http://127.0.0.1:{a.port}  (live calls capped at {a.max_calls} per run, {a.max_chars} spoken chars per server session)")
    ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()


if __name__ == "__main__":
    main()
