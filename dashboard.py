#!/usr/bin/env python3
"""
Web dashboard for Politics AI Swarm.

Usage:
    python dashboard.py              # http://127.0.0.1:7860
    python dashboard.py --port 8080
    python dashboard.py --host 0.0.0.0 --port 7860
"""

import sys
import json
import time
import copy
import threading
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from flask import Flask, Response, request, jsonify, send_file
except ImportError:
    print("Flask not found. Run menu.py option [8] which sets up the venv automatically.")
    print(f"Or manually: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt")
    sys.exit(1)

from src.orchestrator import Orchestrator, load_config, load_personas
from src.tabby_client import TabbyClient
from src import live_status

# ── App & shared state ────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

_debates: dict = {}
_podcast_jobs: dict = {}
_config = None
_personas = None
_stdout_lock = threading.Lock()


def _get_config():
    global _config, _personas
    if _config is None:
        _config = load_config("config/settings.yaml")
        _personas = load_personas("config/personas.yaml")
    return _config, _personas


# ── Log capture ───────────────────────────────────────────────────────────────

class _LogCapture:
    def __init__(self, log_list, real_stdout):
        self._logs = log_list
        self._real = real_stdout
        self._buf = ""

    def write(self, text):
        with _stdout_lock:
            self._real.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            s = line.rstrip()
            if s:
                self._logs.append({"ts": time.time(), "text": s})

    def flush(self):
        self._real.flush()


# ── Background runners ────────────────────────────────────────────────────────

def _run_debate(debate_id, params):
    info = _debates[debate_id]
    info["status"] = "running"
    info["started_at"] = time.time()
    config, personas = _get_config()
    cfg = copy.deepcopy(config)
    if not params.get("research_enabled", True):
        cfg["research"]["enabled"] = False
    orig = sys.stdout
    sys.stdout = _LogCapture(info["logs"], orig)
    try:
        orch = Orchestrator(config=cfg, personas=personas,
                            us_persona=params.get("us_persona") or None,
                            china_persona=params.get("china_persona") or None,
                            eu_persona=params.get("eu_persona") or None)
        state = orch.run_debate(
            topic=params["topic"],
            max_rounds=params.get("rounds") or None,
            time_limit_minutes=params.get("time_limit") or None,
            produce_podcast=params.get("podcast", False),
        )
        info["state"] = state.to_dict()
        info["round_num"] = state.round_num
        info["finish_reason"] = state.finish_reason
        info["status"] = "done"
    except Exception as e:
        info["status"] = "error"
        info["error"] = str(e)
        info["logs"].append({"ts": time.time(), "text": f"[ERROR] {e}"})
    finally:
        sys.stdout = orig
        info["finished_at"] = time.time()


def _run_podcast(job_id, tx_filename, voice_refs):
    info = _podcast_jobs[job_id]
    info["status"] = "running"
    config, personas = _get_config()
    orig = sys.stdout
    sys.stdout = _LogCapture(info["logs"], orig)
    try:
        from src.context.debate_state import DebateState
        from src.tts.podcast import PodcastProducer
        tx_path = Path(config["output"]["transcripts_dir"]) / tx_filename
        state = DebateState.from_dict(json.loads(tx_path.read_text()))
        client = TabbyClient(base_url=config["tabbyapi"]["url"],
                             api_key=config["tabbyapi"].get("api_key", ""))
        result = PodcastProducer(config, personas).produce(
            state, client, voice_refs or None, source_transcript=tx_filename)
        info["result"] = result.name if result else None
        info["status"] = "done"
    except Exception as e:
        info["status"] = "error"
        info["error"] = str(e)
        info["logs"].append({"ts": time.time(), "text": f"[ERROR] {e}"})
    finally:
        sys.stdout = orig
        info["finished_at"] = time.time()


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse_stream(store, job_id):
    def generate():
        info = store.get(job_id)
        if not info:
            yield f"data: {json.dumps({'error': 'not found'})}\n\n"
            return
        cursor = 0
        while True:
            while cursor < len(info["logs"]):
                yield f"data: {json.dumps(info['logs'][cursor])}\n\n"
                cursor += 1
            if info["status"] in ("done", "error"):
                yield f"data: {json.dumps({'done': True, 'status': info['status'], 'result': info.get('result')})}\n\n"
                return
            time.sleep(0.25)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── API: system ───────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    config, _ = _get_config()
    url = config["tabbyapi"]["url"]
    alive = TabbyClient(base_url=url, api_key=config["tabbyapi"].get("api_key", "")).is_alive()
    txdir = Path(config["output"]["transcripts_dir"])
    txcount = len(list(txdir.glob("*.json"))) if txdir.exists() else 0
    active = sum(1 for d in _debates.values() if d["status"] == "running")
    cli_status = live_status.read()
    if cli_status and cli_status.get("active"):
        active += 1
    return jsonify({"tabbyapi": {"url": url, "alive": alive},
                    "active_debates": active, "total_debates": len(_debates),
                    "transcript_count": txcount,
                    "cli_debate": cli_status})


@app.route("/api/live")
def api_live():
    """Live status of CLI-launched debates (file-based cross-process tracking)."""
    status = live_status.read()
    if not status:
        return jsonify({"active": False})
    return jsonify(status)


@app.route("/api/live/debate")
def api_live_debate():
    """Full live debate state — turns, research, scores, verdict."""
    debate = live_status.read_debate()
    if not debate:
        return jsonify({"active": False})
    return jsonify(debate)


# ── API: debates ──────────────────────────────────────────────────────────────

@app.route("/api/debates")
def api_debates():
    out = [{"id": d["id"], "topic": d["topic"], "status": d["status"],
            "round_num": d.get("round_num"), "finish_reason": d.get("finish_reason"),
            "created_at": d["created_at"], "error": d.get("error")}
           for d in sorted(_debates.values(), key=lambda x: x["created_at"], reverse=True)]
    return jsonify({"debates": out})


@app.route("/api/debate/start", methods=["POST"])
def api_start_debate():
    params = request.get_json(force=True) or {}
    topic = (params.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic required"}), 400
    did = f"debate_{int(time.time()*1000)}"
    _debates[did] = {"id": did, "topic": topic, "status": "starting", "logs": [],
                     "state": None, "error": None, "round_num": 0,
                     "finish_reason": None, "created_at": time.time(),
                     "started_at": None, "finished_at": None}
    threading.Thread(target=_run_debate, args=(did, params), daemon=True).start()
    return jsonify({"debate_id": did, "topic": topic})


@app.route("/api/debate/<did>/logs")
def api_debate_logs(did):
    info = _debates.get(did)
    if not info:
        return jsonify({"error": "not found"}), 404
    since = float(request.args.get("since", 0))
    return jsonify({"logs": [l for l in info["logs"] if l["ts"] > since],
                    "status": info["status"]})


@app.route("/api/debate/<did>/stream")
def api_debate_stream(did):
    return _sse_stream(_debates, did)


# ── API: transcripts ──────────────────────────────────────────────────────────

@app.route("/api/transcripts")
def api_transcripts():
    config, _ = _get_config()
    txdir = Path(config["output"]["transcripts_dir"])
    if not txdir.exists():
        return jsonify({"transcripts": []})
    files = sorted(txdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
    out = []
    for f in files:
        try:
            d = json.loads(f.read_text())
            out.append({"filename": f.name, "topic": d.get("topic", "?"),
                        "rounds": d.get("round_num", 0),
                        "finish_reason": d.get("finish_reason", ""),
                        "mtime": f.stat().st_mtime})
        except Exception:
            pass
    return jsonify({"transcripts": out})


@app.route("/api/transcripts/<fname>")
def api_transcript_get(fname):
    if not fname.endswith(".json") or "/" in fname or ".." in fname:
        return jsonify({"error": "invalid"}), 400
    config, _ = _get_config()
    p = Path(config["output"]["transcripts_dir"]) / fname
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(p.read_text()))


# ── API: podcast ──────────────────────────────────────────────────────────────

@app.route("/api/podcast/generate", methods=["POST"])
def api_podcast_generate():
    params = request.get_json(force=True) or {}
    fname = (params.get("filename") or "").strip()
    if not fname or "/" in fname or ".." in fname or not fname.endswith(".json"):
        return jsonify({"error": "invalid filename"}), 400
    config, _ = _get_config()
    if not (Path(config["output"]["transcripts_dir"]) / fname).exists():
        return jsonify({"error": "transcript not found"}), 404
    voice_refs = {k: v for k in ("us", "china", "judge")
                  if (v := (params.get(f"voice_{k}") or "").strip())}
    job_id = f"podcast_{int(time.time()*1000)}"
    _podcast_jobs[job_id] = {"id": job_id, "filename": fname, "status": "starting",
                              "logs": [], "result": None, "error": None,
                              "created_at": time.time(), "finished_at": None}
    threading.Thread(target=_run_podcast, args=(job_id, fname, voice_refs), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/podcast/<job_id>/stream")
def api_podcast_stream(job_id):
    return _sse_stream(_podcast_jobs, job_id)


@app.route("/api/podcast/for-transcript/<tx_fname>")
def api_podcast_for_transcript(tx_fname):
    if "/" in tx_fname or ".." in tx_fname:
        return jsonify({"error": "invalid"}), 400
    config, _ = _get_config()
    audio_dir = Path(config["output"]["audio_dir"])
    if not audio_dir.exists():
        return jsonify({"podcast": None})
    for mf in sorted(audio_dir.glob("*.manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(mf.read_text())
            if data.get("source_transcript") == tx_fname:
                wav = audio_dir / data["audio_file"]
                return jsonify({"podcast": {
                    "manifest_file": mf.name,
                    "audio_file": data["audio_file"],
                    "audio_exists": wav.exists(),
                    "total_duration": data.get("total_duration", 0),
                    "segments": data.get("segments", []),
                }})
        except Exception:
            pass
    return jsonify({"podcast": None})


# ── API: audio ────────────────────────────────────────────────────────────────

@app.route("/api/audio/<fname>")
def api_audio(fname):
    if "/" in fname or ".." in fname:
        return jsonify({"error": "invalid"}), 400
    config, _ = _get_config()
    p = Path(config["output"]["audio_dir"]) / fname
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    mime = "audio/mpeg" if fname.endswith(".mp3") else "audio/wav"
    return send_file(str(p), mimetype=mime, conditional=True)


# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Politics AI Swarm</title>
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bd:#30363d;--tx:#e6edf3;
      --mu:#8b949e;--bl:#58a6ff;--gn:#3fb950;--rd:#f85149;--yl:#d29922;--or:#ffa657;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh;}
a{color:var(--bl);}

/* layout */
header{background:var(--bg2);border-bottom:1px solid var(--bd);padding:.7rem 1.5rem;
       display:flex;align-items:center;gap:1rem;}
header h1{font-size:1.1rem;font-weight:700;}
header h1 em{color:var(--bl);font-style:normal;}
.sbar{background:var(--bg2);border-bottom:1px solid var(--bd);padding:.35rem 1.5rem;
      display:flex;gap:1.5rem;font-size:.78rem;flex-wrap:wrap;align-items:center;}
.si{display:flex;align-items:center;gap:.4rem;}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.dot-g{background:var(--gn);box-shadow:0 0 5px var(--gn);}
.dot-r{background:var(--rd);}
main{padding:1.1rem 1.5rem;max-width:1500px;margin:0 auto;
     display:grid;grid-template-columns:340px 1fr;gap:1.1rem;}
.span2{grid-column:1/-1;}

/* cards */
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;overflow:hidden;}
.ch{padding:.55rem 1rem;border-bottom:1px solid var(--bd);font-size:.72rem;font-weight:700;
    color:var(--mu);text-transform:uppercase;letter-spacing:.06em;
    display:flex;align-items:center;justify-content:space-between;}
.cb{padding:.9rem;}

/* form */
.fg{margin-bottom:.65rem;}
label{display:block;font-size:.775rem;color:var(--mu);margin-bottom:.2rem;}
input[type=text],input[type=number],select,textarea{
  width:100%;background:var(--bg3);border:1px solid var(--bd);border-radius:5px;
  color:var(--tx);padding:.42rem .65rem;font-size:.8125rem;font-family:inherit;}
input:focus,select:focus,textarea:focus{outline:none;border-color:var(--bl);}
textarea{resize:vertical;min-height:58px;}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;}
.cbrow{display:flex;align-items:center;gap:.45rem;}
.cbrow input{width:auto;}
.cbrow label{margin:0;color:var(--tx);font-size:.8rem;}
.checks{display:flex;gap:1.25rem;margin:.15rem 0 .65rem;}
button.pri{display:block;width:100%;padding:.52rem;border-radius:6px;
  background:var(--bl);color:#0d1117;font-weight:700;font-size:.84rem;
  border:none;cursor:pointer;transition:opacity .15s;}
button.pri:hover{opacity:.85;}
button.pri:disabled{opacity:.35;cursor:not-allowed;}
button.sm{background:none;border:1px solid var(--bd);color:var(--mu);
  padding:.22rem .55rem;border-radius:4px;font-size:.72rem;cursor:pointer;}
button.sm:hover{border-color:var(--tx);color:var(--tx);}
button.ghost{background:none;border:none;color:var(--bl);font-size:.8rem;cursor:pointer;padding:0;}
button.ghost:hover{text-decoration:underline;}
button.danger{background:none;border:1px solid var(--rd);color:var(--rd);
  padding:.22rem .55rem;border-radius:4px;font-size:.72rem;cursor:pointer;}

/* log */
.log{background:#010409;border-radius:5px;padding:.65rem .75rem;
     font-family:'Monaco','Menlo','Consolas',monospace;font-size:.7rem;
     height:560px;overflow-y:auto;line-height:1.6;white-space:pre-wrap;word-break:break-word;}
.ll{padding:1px 0;color:var(--mu);}
.ll.ph{color:var(--bl);font-weight:bold;}
.ll.us{color:#79c0ff;}
.ll.cn{color:var(--or);}
.ll.ju{color:#ffd700;}
.ll.er{color:var(--rd);}
.ll.wn{color:var(--yl);}
.ll.ok{color:var(--tx);}

/* session items */
.ditem{padding:.6rem .9rem;border-bottom:1px solid var(--bd);cursor:pointer;transition:background .1s;}
.ditem:hover{background:var(--bg3);}
.ditem:last-child{border-bottom:none;}
.dtopic{font-size:.82rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:.18rem;}
.dmeta{font-size:.7rem;color:var(--mu);display:flex;gap:.65rem;flex-wrap:wrap;}
.st-run{color:var(--yl);}
.st-done{color:var(--gn);}
.st-err{color:var(--rd);}

/* transcript panel */
.tsplit{display:grid;grid-template-columns:290px 1fr;min-height:400px;}
.tlist{border-right:1px solid var(--bd);overflow-y:auto;max-height:560px;}
.tview{padding:1.1rem 1.25rem;overflow-y:auto;max-height:560px;font-size:.8375rem;line-height:1.75;}

/* turn */
.turn{margin-bottom:1.35rem;}
.turn-hdr{display:flex;align-items:center;gap:.45rem;margin-bottom:.35rem;flex-wrap:wrap;}
.badge{display:inline-block;padding:.1em .45em;border-radius:4px;font-size:.7rem;font-weight:700;}
.b-us{background:rgba(88,166,255,.12);color:var(--bl);}
.b-cn{background:rgba(255,166,87,.12);color:var(--or);}
.b-ju{background:rgba(255,215,0,.09);color:#ffd700;}
.rnd{color:var(--mu);font-size:.7rem;}
.rep{color:var(--rd);font-size:.7rem;}
.tcontent{color:#c9d1d9;white-space:pre-wrap;word-break:break-word;}
.qs{color:var(--mu);font-size:.7rem;}

/* audio player */
.player-wrap{background:var(--bg3);border:1px solid var(--bd);border-radius:7px;
             padding:.8rem;margin-bottom:1rem;}
audio{width:100%;margin-bottom:.5rem;height:36px;}
audio::-webkit-media-controls-panel{background:var(--bg3);}

/* synced segments */
.seg{padding:.55rem .7rem;border-left:3px solid transparent;margin-bottom:.5rem;
     border-radius:0 5px 5px 0;cursor:pointer;transition:background .15s,border-color .15s;}
.seg:hover{background:var(--bg3);}
.seg.active{border-left-color:var(--bl);background:rgba(88,166,255,.07);}
.seg.active.ag-us{border-left-color:var(--bl);}
.seg.active.ag-cn{border-left-color:var(--or);}
.seg.active.ag-ju{border-left-color:#ffd700;}
.seg-hdr{display:flex;align-items:center;gap:.4rem;margin-bottom:.25rem;}
.seg-time{color:var(--mu);font-size:.7rem;font-family:monospace;}
.seg-text{font-size:.8125rem;color:#c9d1d9;line-height:1.6;}

/* podcast controls */
.pod-form{background:var(--bg3);border:1px solid var(--bd);border-radius:6px;padding:.8rem;margin-bottom:.85rem;}
.pod-form h4{font-size:.8rem;margin-bottom:.6rem;color:var(--mu);text-transform:uppercase;letter-spacing:.05em;}
.pod-log{background:#010409;border-radius:4px;padding:.5rem .6rem;
         font-family:monospace;font-size:.68rem;height:80px;overflow-y:auto;margin-top:.5rem;}

/* tabs */
.tabs{display:flex;border-bottom:1px solid var(--bd);margin-bottom:.85rem;}
.tab{padding:.45rem .8rem;font-size:.78rem;cursor:pointer;color:var(--mu);
     border-bottom:2px solid transparent;margin-bottom:-1px;}
.tab.on{color:var(--tx);border-bottom-color:var(--bl);}

.empty{text-align:center;padding:2rem;color:var(--mu);font-size:.8rem;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--bd);border-radius:3px;}
</style>
</head>
<body>

<header>
  <h1>Politics AI <em>Swarm</em></h1>
  <span id="hdr-active" style="margin-left:auto;font-size:.78rem;color:var(--mu)"></span>
</header>

<div class="sbar">
  <div class="si"><div class="dot" id="tdot"></div><span id="ttxt">TabbyAPI …</span></div>
  <div class="si"><span id="actxt">—</span></div>
  <div class="si"><span id="txctxt">—</span></div>
  <div class="si" style="margin-left:auto"><span id="reftxt" style="color:var(--mu)"></span></div>
</div>

<main>
  <!-- Left column -->
  <div style="display:flex;flex-direction:column;gap:1.1rem;">

    <div class="card">
      <div class="ch">New Debate</div>
      <div class="cb">
        <div class="fg"><label>Topic</label>
          <textarea id="f-topic" placeholder="e.g. Should EU partner with NASA or CNSA for a lunar base?"></textarea>
        </div>
        <div class="row2">
          <div class="fg"><label>Max Rounds</label><input type="number" id="f-rounds" placeholder="8" min="1" max="20"></div>
          <div class="fg"><label>Time Limit (min)</label><input type="number" id="f-tlimit" placeholder="20" min="1" max="120"></div>
        </div>
        <div class="row2">
          <div class="fg"><label>US Persona</label>
            <select id="f-us">
              <option value="">Default (Strategist)</option>
              <option value="strategist">Ambassador V. Marsh</option>
              <option value="lobbyist">Director J. Harrington</option>
              <option value="analyst">Dr. Sarah Chen</option>
            </select>
          </div>
          <div class="fg"><label>China Persona</label>
            <select id="f-cn">
              <option value="">Default (Director)</option>
              <option value="director">Dir-Gen Wei Changming</option>
              <option value="enterprise_rep">Chairman Liu Peng</option>
              <option value="economist">Prof. Zhang Yifei</option>
            </select>
          </div>
        </div>
        <div class="fg"><label>EU Persona</label>
          <select id="f-eu">
            <option value="">Default (Strategist)</option>
            <option value="strategist">Commissioner Elise Fontaine</option>
            <option value="hawk">Admiral Henrik Sørensen</option>
            <option value="economist">Dr. Marie Leclerc</option>
          </select>
        </div>
        <div class="checks">
          <div class="cbrow"><input type="checkbox" id="f-res" checked><label for="f-res">Web Research</label></div>
          <div class="cbrow"><input type="checkbox" id="f-pod"><label for="f-pod">Podcast</label></div>
        </div>
        <button class="pri" id="start-btn" onclick="startDebate()">&#9654;  Start Debate</button>
      </div>
    </div>

    <div class="card" style="flex:1">
      <div class="ch">Sessions <button class="sm" onclick="loadDebates()">&#8635;</button></div>
      <div id="debates-list"><div class="empty">No debates this session</div></div>
    </div>

  </div>

  <!-- Right: live log -->
  <div class="card">
    <div class="ch">
      <span>Live Output</span>
      <span id="log-topic" style="font-weight:400;color:var(--mu);max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
    </div>
    <div style="padding:.65rem;">
      <div class="log" id="log-viewer"><div class="ll" style="color:var(--mu)">Start a debate to see live output here.</div></div>
    </div>
  </div>

  <!-- Bottom: transcripts + podcast player -->
  <div class="card span2">
    <div class="ch">Transcripts &amp; Podcast <button class="sm" onclick="loadTranscripts()">&#8635; Refresh</button></div>
    <div class="tsplit">
      <div class="tlist" id="tx-list"><div class="empty">Loading…</div></div>
      <div class="tview" id="tx-view"><div class="empty">Select a transcript.</div></div>
    </div>
  </div>
</main>

<script>
'use strict';
const $=id=>document.getElementById(id);

// ── Status ────────────────────────────────────────────────────────────────────
async function checkStatus(){
  try{
    const d=await fetch('/api/status').then(r=>r.json());
    $('tdot').className='dot '+(d.tabbyapi.alive?'dot-g':'dot-r');
    $('ttxt').textContent='TabbyAPI  '+(d.tabbyapi.alive?d.tabbyapi.url:'OFFLINE ('+d.tabbyapi.url+')');
    $('actxt').textContent=d.active_debates+' running';
    $('txctxt').textContent=d.transcript_count+' transcripts';
    $('hdr-active').textContent=d.active_debates>0?d.active_debates+' debate'+(d.active_debates>1?'s':'')+' running':'';
    $('reftxt').textContent='Updated '+new Date().toLocaleTimeString();
    // Show CLI-launched debate in live log
    if(d.cli_debate && d.cli_debate.active && !_cliWatching){
      _cliWatching=true;
      pollCliDebate(d.cli_debate);
    }
    if(d.cli_debate && !d.cli_debate.active){_cliWatching=false;}
  }catch{$('tdot').className='dot dot-r';}
}
let _cliWatching=false, _lastCliPhase='';
function pollCliDebate(st){
  const v=$('log-viewer');
  const phase=st.phase||'';const rnd=st.round||0;const agent=st.agent||'';
  const line=`[CLI] ${st.topic||''} — phase: ${phase}, round: ${rnd}${agent?' ('+agent+')':''}`;
  if(line!==_lastCliPhase){
    $('log-topic').textContent='[CLI] '+(st.topic||'');
    addLog(v,line);_lastCliPhase=line;
  }
}

// ── Debates list ──────────────────────────────────────────────────────────────
async function loadDebates(){
  const d=await fetch('/api/debates').then(r=>r.json());
  const el=$('debates-list');
  if(!d.debates.length){el.innerHTML='<div class="empty">No debates this session</div>';return;}
  el.innerHTML=d.debates.map(x=>`
    <div class="ditem" onclick="watchDebate(${JSON.stringify(x.id)},${JSON.stringify(x.topic)})">
      <div class="dtopic">${esc(x.topic)}</div>
      <div class="dmeta">
        <span class="${stClass(x.status)}">${x.status}</span>
        ${x.round_num!=null?'<span>Rd '+x.round_num+'</span>':''}
        ${x.finish_reason?'<span>'+x.finish_reason+'</span>':''}
        <span>${ago(x.created_at)}</span>
      </div>
    </div>`).join('');
}
function stClass(s){return 'st-'+(s==='running'?'run':s==='done'?'done':'err');}

// ── Start debate ──────────────────────────────────────────────────────────────
async function startDebate(){
  const topic=$('f-topic').value.trim();
  if(!topic){alert('Topic is required');return;}
  const params={topic,
    rounds:parseInt($('f-rounds').value)||null,
    time_limit:parseInt($('f-tlimit').value)||null,
    us_persona:$('f-us').value||null,
    china_persona:$('f-cn').value||null,
    eu_persona:$('f-eu').value||null,
    podcast:$('f-pod').checked,
    research_enabled:$('f-res').checked};
  const btn=$('start-btn');btn.disabled=true;btn.textContent='Starting…';
  try{
    const d=await fetch('/api/debate/start',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(params)}).then(r=>r.json());
    if(d.error){alert(d.error);return;}
    openStream(d.debate_id,topic);loadDebates();
  }finally{btn.disabled=false;btn.textContent='▶  Start Debate';}
}

// ── SSE streaming ─────────────────────────────────────────────────────────────
const streams={};
function openStream(id,topic){
  if(streams[id])streams[id].close();
  const v=$('log-viewer');v.innerHTML='';
  $('log-topic').textContent=topic;
  const es=new EventSource('/api/debate/'+id+'/stream');
  streams[id]=es;
  es.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.done){es.close();delete streams[id];loadDebates();loadTranscripts();checkStatus();return;}
    addLog(v,d.text);
  };
  es.onerror=()=>{es.close();delete streams[id];};
}
async function watchDebate(id,topic){
  const r=await fetch('/api/debate/'+id+'/logs').then(r=>r.json());
  const v=$('log-viewer');v.innerHTML='';$('log-topic').textContent=topic;
  r.logs.forEach(l=>addLog(v,l.text));
  if(r.status==='running'||r.status==='starting')openStream(id,topic);
}
function addLog(v,text){
  const d=document.createElement('div');
  d.className='ll '+clf(text);d.textContent=text;
  v.appendChild(d);v.scrollTop=v.scrollHeight;
}
function clf(t){
  if(/Phase \d|={4,}|DEBATE:|Debate:/.test(t))return 'ph';
  if(t.includes('[us]'))return 'us';
  if(t.includes('[china]'))return 'cn';
  if(/\[judge\]|\[verdict\]|\[quality\]|\[podcast\]/.test(t))return 'ju';
  if(/ERROR|error/.test(t))return 'er';
  if(/REPETITIVE|WARNING/.test(t))return 'wn';
  if(t.trim())return 'ok';return '';
}

// ── Transcripts ───────────────────────────────────────────────────────────────
async function loadTranscripts(){
  const d=await fetch('/api/transcripts').then(r=>r.json());
  const el=$('tx-list');
  if(!d.transcripts.length){el.innerHTML='<div class="empty">No transcripts yet</div>';return;}
  el.innerHTML=d.transcripts.map(t=>`
    <div class="ditem" onclick="viewTranscript(${JSON.stringify(t.filename)})">
      <div class="dtopic">${esc(t.topic)}</div>
      <div class="dmeta"><span>${t.rounds} rounds</span><span>${t.finish_reason||'—'}</span><span>${ago(t.mtime)}</span></div>
    </div>`).join('');
}

// ── Transcript viewer ─────────────────────────────────────────────────────────
let _audioEl=null, _manifest=null, _txFilename=null;

async function viewTranscript(fname){
  _txFilename=fname;
  const [tx,pod]=await Promise.all([
    fetch('/api/transcripts/'+fname).then(r=>r.json()),
    fetch('/api/podcast/for-transcript/'+fname).then(r=>r.json()),
  ]);
  renderTranscriptView(tx,pod.podcast,fname);
}

function renderTranscriptView(tx,podcast,fname){
  const v=$('tx-view');
  const lbl={us:'🇺🇸 US',china:'🇨🇳 China',judge:'🇪🇺 Judge'};
  const bcls={us:'b-us',china:'b-cn',judge:'b-ju'};

  let html=`<h3 style="font-size:.95rem;margin-bottom:.3rem">${esc(tx.topic)}</h3>
    <div style="font-size:.72rem;color:var(--mu);margin-bottom:.9rem">
      ${tx.round_num} rounds &middot; ${tx.finish_reason||'in progress'}
    </div>`;

  // Podcast area
  if(podcast && podcast.audio_exists){
    html+=`<div class="player-wrap" id="player-wrap">
      <audio id="audio-el" controls preload="metadata">
        <source src="/api/audio/${esc(podcast.audio_file)}" type="audio/wav">
      </audio>
      <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem;">
        <span style="font-size:.75rem;color:var(--mu)">&#9654; ${fmt(podcast.total_duration)}</span>
        <a href="/api/audio/${esc(podcast.audio_file)}" download style="font-size:.72rem">&#8659; Download</a>
      </div>
    </div>
    <div class="tabs">
      <div class="tab on" id="tab-sync" onclick="switchTab('sync')">&#9654; Synced Transcript</div>
      <div class="tab" id="tab-turns" onclick="switchTab('turns')">Debate Turns</div>
    </div>
    <div id="pane-sync">${renderSegments(podcast.segments)}</div>
    <div id="pane-turns" style="display:none">${renderTurns(tx.turns,lbl,bcls,tx.verdict)}</div>`;
  } else {
    // No podcast: show generate form + debate turns
    html+=renderPodcastForm(fname,podcast);
    html+=renderTurns(tx.turns,lbl,bcls,tx.verdict);
  }

  v.innerHTML=html;

  // Wire up audio player sync
  if(podcast && podcast.audio_exists){
    _audioEl=document.getElementById('audio-el');
    _manifest=podcast.segments;
    _audioEl.addEventListener('timeupdate',syncHighlight);
  }
}

function renderSegments(segs){
  if(!segs||!segs.length)return '<div class="empty">No segments</div>';
  const agCls={us:'ag-us',china:'ag-cn',judge:'ag-ju'};
  const lbl={us:'🇺🇸 US',china:'🇨🇳 China',judge:'🇪🇺 Judge'};
  const bcls={us:'b-us',china:'b-cn',judge:'b-ju'};
  return segs.map((s,i)=>`
    <div class="seg ${agCls[s.agent]||''}" id="seg-${i}"
         data-start="${s.start}" data-end="${s.end}" onclick="seekAudio(${s.start})">
      <div class="seg-hdr">
        <span class="badge ${bcls[s.agent]||''}">${lbl[s.agent]||s.agent}</span>
        <span class="seg-time">${fmt(s.start)}</span>
      </div>
      <div class="seg-text">${esc(s.text)}</div>
    </div>`).join('');
}

function renderResearch(sources){
  if(!sources||!sources.length)return '';
  return `<div style="margin:.4rem 0 .6rem;padding:.45rem .7rem;background:var(--bg3);border-radius:5px;border-left:3px solid var(--bl);font-size:.72rem;">
    <div style="color:var(--mu);margin-bottom:.25rem;font-weight:600;">Sources (${sources.length})</div>
    ${sources.map(s=>`<div style="margin-bottom:.15rem;"><a href="${esc(s.url)}" target="_blank" rel="noopener" style="color:var(--bl);text-decoration:none;">${esc(s.title||s.url)}</a></div>`).join('')}
  </div>`;
}
function renderTurns(turns,lbl,bcls,verdict){
  let h=(turns||[]).map(t=>`
    <div class="turn">
      <div class="turn-hdr">
        <span class="badge ${bcls[t.agent]||''}">${lbl[t.agent]||t.agent}</span>
        <span class="rnd">Round ${t.round}</span>
        ${t.is_repetitive?'<span class="rep">&#9888; repetitive</span>':''}
        ${t.quality_score?'<span class="qs">'+fmtQ(t.quality_score)+'</span>':''}
      </div>
      ${renderResearch(t.research)}
      <div class="tcontent">${esc(t.content)}</div>
    </div>`).join('');
  if(verdict)h+=`<div class="turn">
    <div class="turn-hdr"><span class="badge b-ju">🇪🇺 Final Verdict</span></div>
    <div class="tcontent">${esc(verdict)}</div>
  </div>`;
  return h;
}

function renderPodcastForm(fname,podcast){
  const hasScript=podcast&&!podcast.audio_exists;
  return `<div class="pod-form" id="pod-form">
    <h4>${hasScript?'Podcast script saved — audio needs Fish Speech':'Generate Podcast'}</h4>
    <div style="font-size:.75rem;color:var(--mu);margin-bottom:.6rem">
      Optional: path to voice reference .wav files (leave blank to use defaults)
    </div>
    <div class="row2" style="margin-bottom:.5rem">
      <div><label>US voice (.wav)</label><input type="text" id="pv-us" placeholder="/path/to/us.wav"></div>
      <div><label>China voice (.wav)</label><input type="text" id="pv-cn" placeholder="/path/to/china.wav"></div>
    </div>
    <div style="max-width:200px;margin-bottom:.5rem"><label>Judge voice (.wav)</label><input type="text" id="pv-ju" placeholder="/path/to/judge.wav"></div>
    <button class="pri" id="gen-btn" onclick="generatePodcast(${JSON.stringify(fname)})"
            style="max-width:200px">&#9654; Generate Podcast</button>
    <div class="pod-log" id="pod-log" style="display:none"></div>
  </div>`;
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name){
  $('tab-sync').className='tab'+(name==='sync'?' on':'');
  $('tab-turns').className='tab'+(name==='turns'?' on':'');
  $('pane-sync').style.display=name==='sync'?'':'none';
  $('pane-turns').style.display=name==='turns'?'':'none';
}

// ── Audio sync ────────────────────────────────────────────────────────────────
function syncHighlight(){
  if(!_audioEl||!_manifest)return;
  const t=_audioEl.currentTime;
  let active=null;
  document.querySelectorAll('.seg').forEach(el=>{
    const on=t>=parseFloat(el.dataset.start)&&t<parseFloat(el.dataset.end);
    el.classList.toggle('active',on);
    if(on)active=el;
  });
  if(active)active.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function seekAudio(t){if(_audioEl)_audioEl.currentTime=t;}
function fmt(s){const m=Math.floor(s/60);return m+':'+String(Math.floor(s%60)).padStart(2,'0');}

// ── Podcast generation ────────────────────────────────────────────────────────
async function generatePodcast(fname){
  const btn=$('gen-btn');if(!btn)return;
  btn.disabled=true;btn.textContent='Generating…';
  const logEl=$('pod-log');logEl.style.display='';logEl.innerHTML='';

  const params={filename:fname,
    voice_us:($('pv-us')||{value:''}).value.trim(),
    voice_china:($('pv-cn')||{value:''}).value.trim(),
    voice_judge:($('pv-ju')||{value:''}).value.trim()};

  const d=await fetch('/api/podcast/generate',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(params)}).then(r=>r.json());
  if(d.error){logEl.textContent=d.error;btn.disabled=false;btn.textContent='▶ Generate Podcast';return;}

  const es=new EventSource('/api/podcast/'+d.job_id+'/stream');
  es.onmessage=e=>{
    const msg=JSON.parse(e.data);
    if(msg.done){
      es.close();
      if(msg.result){
        // Reload view with player
        viewTranscript(_txFilename);
      } else {
        logEl.textContent+='[done — Fish Speech not installed, script saved only]';
        btn.disabled=false;btn.textContent='▶ Generate Podcast';
      }
      return;
    }
    logEl.textContent+=msg.text+'\n';
    logEl.scrollTop=logEl.scrollHeight;
  };
  es.onerror=()=>{es.close();btn.disabled=false;btn.textContent='▶ Generate Podcast';};
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function ago(ts){const s=Date.now()/1000-ts;if(s<60)return 'just now';if(s<3600)return Math.floor(s/60)+'m ago';if(s<86400)return Math.floor(s/3600)+'h ago';return Math.floor(s/86400)+'d ago';}
function fmtQ(q){if(!q)return '';return Object.entries(q).map(([k,v])=>k[0].toUpperCase()+':'+v).join(' ');}

// ── Init ──────────────────────────────────────────────────────────────────────
checkStatus();loadTranscripts();loadDebates();
setInterval(checkStatus,8000);setInterval(loadDebates,4000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return _HTML


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Politics AI Swarm dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    _get_config()
    print(f"\n  Politics AI Swarm  →  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
