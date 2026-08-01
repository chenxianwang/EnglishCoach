#!/usr/bin/env python3
"""
English Speaking Coach — web version
====================================

Runs the whole thing in your browser: the upload form AND the dashboard live
on one page, so there's no separate desktop window.

    pip install flask
    python english_coach_web.py
    # then open http://localhost:8000

Click "➕ New analysis" in the sidebar to upload a recording + transcript and
run it. Results drop straight into the dashboard with all your recordings and
the progress summary.
"""

import json
import os
import re
import secrets
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from datetime import timedelta
from urllib.parse import quote

try:
    from flask import Flask, request, redirect, session
    from werkzeug.security import generate_password_hash, check_password_hash
    from werkzeug.utils import secure_filename
except ImportError:
    raise SystemExit(
        "This needs Flask. Install it once with:\n\n    pip install flask\n")

import english_coach as ec

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY = ec.library_dir()            # <project>/VideoAudioFiles
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".english_coach.json")
AUDIO_EXTS = (".m4a", ".mp3", ".wav", ".mp4", ".mov", ".aac", ".flac", ".ogg")


def rec_dir(stem, create=True):
    return ec.rec_dir_for(stem, library=LIBRARY, create=create)


word_ipa = ec.word_ipa   # single CMUdict-backed IPA lookup, shared with the engine


def safe_name(filename):
    """Keep the original filename (spaces and all) but strip any path parts /
    traversal, so folder names match what the desktop app makes."""
    n = os.path.basename((filename or "").replace("\\", "/").split("/")[-1])
    n = n.replace("..", "").strip()
    return n or "recording"

app = Flask(__name__)

# Real-time transcription (sherpa-onnx) mounted directly on this app, so live
# transcribe works in ONE process — no separate service needed. Requires
# `pip install flask-sock sherpa-onnx` and a downloaded streaming model
# (SHERPA_MODEL_DIR). If those aren't present the route still exists and the
# session just reports "model not loaded" instead of failing to connect.
LIVE_WS_AVAILABLE = False
try:
    from flask_sock import Sock as _Sock
    import transcribe_service as _tsvc
    _live_sock = _Sock(app)

    @_live_sock.route("/v1/stream")
    def _live_stream(ws):
        # belt-and-suspenders: @app.before_request already gates this route,
        # but the sock upgrade lifecycle is different enough from a normal
        # request/response that it's worth checking again right here.
        if not session.get("authed"):
            ws.close()
            return
        try:
            sr = int(request.args.get("sample_rate", 16000))
        except (TypeError, ValueError):
            sr = 16000
        _tsvc.stream_session(ws, sr)
    LIVE_WS_AVAILABLE = True
except Exception:
    LIVE_WS_AVAILABLE = False   # flask-sock not installed — live transcribe off

# in-memory progress for running analyses: job_id -> dict
JOBS = {}


def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    except Exception:
        pass


# Practice scores, vocabulary, grammar log, listening SRS, ear-training and
# Mandarin-contrast stats — everything the client used to keep only in
# localStorage (device-only, invisible from a second browser/device). Stored
# next to history.json so it rides along with the recordings it's about.
PROGRESS_PATH = os.path.join(LIBRARY, "progress.json")


def load_progress():
    try:
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_progress(data):
    try:
        os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


@app.route("/api/progress", methods=["GET", "POST"])
def api_progress():
    """The server-side replacement for localStorage: GET returns everything,
    POST merges in whichever keys the client is updating (per-key last-write-
    wins — simple, and fine for one person on at most a couple of devices)."""
    from flask import jsonify
    if request.method == "GET":
        return jsonify(load_progress())
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="expected a JSON object"), 400
    data = load_progress()
    data.update(payload)
    save_progress(data)
    return jsonify(ok=True)


def _get_or_create_secret_key():
    """Flask session-signing key. Generated once and persisted, so restarting
    the server (or the launchd service) doesn't invalidate every session."""
    cfg = load_config()
    key = cfg.get("secret_key")
    if not key:
        key = secrets.token_hex(32)
        save_config({**cfg, "secret_key": key})
    return key


def _ensure_web_password():
    """First boot only: generate a random login password rather than exposing
    a public "choose your password" form — with the app now reachable on the
    open internet, the first stranger to hit that form before the owner does
    would get to set (and lock in) the password instead. Printed once, here,
    never sent over the network."""
    cfg = load_config()
    if cfg.get("web_password_hash"):
        return
    pw = secrets.token_urlsafe(9)
    save_config({**cfg, "web_password_hash": generate_password_hash(pw)})
    print("=" * 60)
    print("No login password was set for the web app — generated one:")
    print("    %s" % pw)
    print("Log in with it at /login, then set your own from the Setting Panel.")
    print("=" * 60)


app.secret_key = _get_or_create_secret_key()
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
_ensure_web_password()

_PUBLIC_PATHS = ("/login",)
_PUBLIC_PREFIXES = ("/static/",)


@app.before_request
def _require_login():
    p = request.path
    if p in _PUBLIC_PATHS or any(p.startswith(pre) for pre in _PUBLIC_PREFIXES):
        return
    if not session.get("authed"):
        nxt = quote(request.full_path if request.query_string else request.path, safe="")
        return redirect("/login?next=" + nxt)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        cfg = load_config()
        if check_password_hash(cfg.get("web_password_hash", ""), request.form.get("password", "")):
            session.permanent = True
            session["authed"] = True
            return redirect(request.args.get("next") or "/")
        error = "Wrong password."
    return ("""
    <!doctype html><html><head><title>English Coach — Login</title>
    <meta name='viewport' content='width=device-width,initial-scale=1'>
    <style>
      body{background:#0b0f16;color:#e7ecf3;font-family:system-ui,sans-serif;
           display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
      form{background:#151b26;padding:32px 36px;border-radius:14px;width:280px;
           border:1px solid #22343f}
      h1{font-size:20px;margin:0 0 18px}
      input{width:100%%;box-sizing:border-box;padding:9px 11px;border-radius:8px;
            border:1px solid #22343f;background:#0b0f16;color:#e7ecf3;font-size:15px}
      button{margin-top:12px;width:100%%;padding:10px;border-radius:8px;border:0;
             background:#46b3c9;color:#08222b;font-weight:700;cursor:pointer;font-size:15px}
      p.err{color:#ff6b6b;margin:0 0 12px}
    </style></head><body>
    <form method='post'>
      <h1>English Coach</h1>
      %s
      <input type='password' name='password' placeholder='Password' autofocus required>
      <button type='submit'>Log in</button>
    </form></body></html>
    """ % (("<p class='err'>%s</p>" % error) if error else ""))


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/change_password", methods=["POST"])
def change_password():
    new_pw = request.form.get("new_password", "").strip()
    if not new_pw:
        return redirect("/?p=settings&msg=" + quote("Password cannot be empty."))
    save_config({**load_config(), "web_password_hash": generate_password_hash(new_pw)})
    return redirect("/?p=settings&msg=" + quote("✓ Password updated."))


def _persist_keys_from_form(form):
    """Read Azure/DeepSeek/Anthropic/Kimi keys off a submitted form (falling
    back to whatever's already set), push them into the process environment,
    and persist them to disk. Shared by /settings and /analyze, since a key
    can be saved from either the Setting Panel or the analysis form."""
    cfg = load_config()
    azure_key = form.get("azure_key", "").strip() or os.environ.get("AZURE_SPEECH_KEY") or cfg.get("azure_key", "")
    azure_region = form.get("azure_region", "").strip() or cfg.get("azure_region", "eastus")
    anthropic_key = form.get("anthropic_key", "").strip() or os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_key", "")
    deepseek_key = form.get("deepseek_key", "").strip() or os.environ.get("DEEPSEEK_API_KEY") or cfg.get("deepseek_key", "")
    kimi_key = form.get("kimi_key", "").strip() or os.environ.get("KIMI_API_KEY") or cfg.get("kimi_key", "")
    if azure_key:
        os.environ["AZURE_SPEECH_KEY"] = azure_key
    if azure_region:
        os.environ["AZURE_SPEECH_REGION"] = azure_region
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if deepseek_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    if kimi_key:
        os.environ["KIMI_API_KEY"] = kimi_key
    # a DeepSeek key present means "use DeepSeek"; llm_analyze auto-detects
    os.environ["LLM_PROVIDER"] = "deepseek" if deepseek_key else "anthropic"
    keys = {"azure_key": azure_key, "azure_region": azure_region,
            "anthropic_key": anthropic_key, "deepseek_key": deepseek_key,
            "kimi_key": kimi_key}
    save_config({**cfg, **keys})
    return keys


def _settings_panel(msg="", active=""):
    """The Setting Panel — API keys and related config, split out of the
    analysis form so a key can be saved without uploading a recording."""
    cfg = load_config()
    akey = os.environ.get("AZURE_SPEECH_KEY") or cfg.get("azure_key", "")
    aregion = os.environ.get("AZURE_SPEECH_REGION") or cfg.get("azure_region", "eastus")
    ankey = os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_key", "")
    dkey = os.environ.get("DEEPSEEK_API_KEY") or cfg.get("deepseek_key", "")
    kkey = os.environ.get("KIMI_API_KEY") or cfg.get("kimi_key", "")
    ph = lambda k: "•••••• (saved — leave blank to keep)" if k else ""
    note = (("<p id='flash-msg' class='summary' style='border-left:4px solid var(--good)'>%s</p>" % msg)
            if msg and active == "settings" else "")
    return ("""
    <section id='settings' class='tabpanel hidden'>
      <h1>Setting Panel</h1>
      <p class='sub'>API keys and configuration shared across the app. Saved to
      <code>~/.english_coach.json</code>, outside the project directory — nothing
      here is committed.</p>
      %s
      <form method='post' action='/settings'>
        <div class='card'>
          <label>Azure key <span class='hint'>· pronunciation scoring</span>
            <input type='password' name='azure_key' placeholder="%s" style='width:100%%'></label><br>
          <label style='display:block;margin-top:6px'>Azure region
            <input type='text' name='azure_region' value='%s'></label><br>
          <p class='hint' style='margin:10px 0 4px'>Grammar analysis — provide ONE key. If both are set, DeepSeek is used.</p>
          <label style='display:block;margin-top:6px'>DeepSeek key <span class='hint'>· recommended — cheaper &amp; reachable from China (used by default)</span>
            <input type='password' name='deepseek_key' placeholder="%s" style='width:100%%'></label>
          <label style='display:block;margin-top:6px'>Anthropic key <span class='hint'>· optional</span>
            <input type='password' name='anthropic_key' placeholder="%s" style='width:100%%'></label>
          <p class='hint' style='margin:10px 0 4px'>Vocabulary — photo capture (Vocabulary &amp; chunks &rarr; From photo)
            needs a vision-capable model; DeepSeek's API is text-only. Kimi is used if set, else Anthropic.</p>
          <label style='display:block;margin-top:6px'>Kimi key <span class='hint'>· for photo vocabulary capture</span>
            <input type='password' name='kimi_key' placeholder="%s" style='width:100%%'></label>
        </div>
        <button type='submit' style='font-size:16px;padding:10px 20px;border-radius:10px;
           border:0;background:var(--accent);color:#08222b;font-weight:700;cursor:pointer'>
           💾 Save settings</button>
      </form>
      <form method='post' action='/change_password'>
        <div class='card'>
          <h2 style='margin-top:0'>Login password</h2>
          <p class='hint' style='margin:0 0 10px'>Guards every page on this
            server, including over the public tunnel. Changing it here signs
            out any other logged-in browser next time it loads a page.</p>
          <label>New password<br>
            <input type='password' name='new_password' required style='width:100%%'></label>
        </div>
        <button type='submit' style='font-size:16px;padding:10px 20px;border-radius:10px;
           border:0;background:var(--accent);color:#08222b;font-weight:700;cursor:pointer'>
           🔒 Update password</button>
      </form>
      <p class='hint' style='margin-top:16px'><a href='/logout' style='color:var(--accent)'>Log out</a></p>
    </section>
    """ % (note, ph(akey), aregion, ph(dkey), ph(ankey), ph(kkey)))


def _load_items_and_history():
    items, history = [], None
    av = (".m4a", ".mp3", ".wav", ".mp4", ".mov", ".aac", ".flac", ".ogg", ".webm")
    if os.path.isdir(LIBRARY):
        for root, _d, files in os.walk(LIBRARY):
            for fn in sorted(files):
                if fn.endswith(".result.json"):
                    try:
                        with open(os.path.join(root, fn), encoding="utf-8") as f:
                            item = json.load(f)
                    except Exception:
                        continue
                    # attach this recording's audio path (served by /VideoAudioFiles/…)
                    for af in sorted(os.listdir(root)):
                        if af.lower().endswith(av):
                            item["audio_rel"] = os.path.relpath(
                                os.path.join(root, af), HERE)
                            item["audio_abs"] = os.path.join(root, af)
                            try:
                                # fallback ordering signal for recordings whose
                                # filename carries no timestamp
                                item["_mtime"] = os.path.getmtime(item["audio_abs"])
                            except OSError:
                                pass
                            break
                    ec._backfill_prosody(item, os.path.join(root, fn))
                    items.append(item)
        hp = os.path.join(LIBRARY, "history.json")
        if os.path.exists(hp):
            try:
                with open(hp, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = None
    return items, history


def _form_panel(msg="", active=""):
    note = (("<p id='flash-msg' class='summary' style='border-left:4px solid var(--good)'>%s</p>" % msg)
            if msg and active == "newrec" else "")
    # live-transcribe WebSocket URL. Default: SAME origin (mounted on this app —
    # one process). Set TRANSCRIBE_SERVICE_URL to point at a separate service.
    _svc = os.environ.get("TRANSCRIBE_SERVICE_URL", "")
    _ws_url = ((_svc.replace("https://", "wss://").replace("http://", "ws://")
                .rstrip("/") + "/v1/stream") if _svc else "")
    return ("""
    <section id='newrec' class='tabpanel hidden'>
      <h1>New Speaking Analysis</h1>
      <p class='sub'>Upload your audio and the script you read aloud, then run.</p>
      %s
      <form method='post' action='/analyze' enctype='multipart/form-data'>
        <div class='card'>
          <label>Audio / video file<br><input type='file' id='audio-input' name='audio'
             accept='audio/*,video/*' onchange='lookupFiles()' required></label>
          <div style='margin-top:8px'>
            <button type='button' id='mic-btn' onclick='recordMic()'
               style='padding:6px 12px;border-radius:8px;border:0;background:#2c4a58;
               color:#fff;cursor:pointer'>● Record in browser</button>
            <span id='mic-status' class='hint' style='margin-left:8px'></span>
          </div>
          <div id='lookup-status' class='hint' style='margin-top:6px'></div>
        </div>
        <div class='card'>
          <label>Transcript — what you actually said<br>
          <textarea name='transcript' id='transcript' rows='6' style='width:100%%'></textarea></label>
          <p class='hint' style='margin:6px 0 0;border-left:3px solid var(--warn);padding-left:8px'>
            ⚠️ The grammar analysis grades <b>this text</b>. Fix any mis-heard words so it matches
            what you really said — otherwise a transcription slip looks like your mistake.
            (Pronunciation is scored separately from your audio by Azure.)</p>
          <div id='tx-check' class='hint' style='margin-top:6px'></div>
          <div style='margin-top:8px'>📖 Load a polished reading
            <select id='reading-sel' onchange='loadReading(this.value)'>
              <option value=''>choose…</option></select></div>
          <button type='button' id='tx-btn' onclick='transcribeAudio()'
             style='margin-top:8px;padding:6px 12px;border-radius:8px;border:0;
             background:#2c4a58;color:#fff;cursor:pointer'>Transcribe audio ⮯</button>
          <label style='margin-left:10px'>model
            <select id='tx-model'>
              <option value='base'>base · fastest, least accurate</option>
              <option value='small'>small · good balance</option>
              <option value='medium' selected>medium · more accurate, slower</option>
              <option value='large-v3'>large-v3 · best accuracy, big &amp; slow on CPU</option>
            </select></label>
          <label style='margin-left:10px'>language
            <select id='tx-lang'>
              <option value='en' selected>English</option>
              <option value=''>Auto-detect</option>
              <option value='zh'>中文 (Chinese)</option>
            </select></label>
          <span id='tx-status' class='hint' style='margin-left:8px'></span>
          <div style='margin-top:10px'>
            <button type='button' id='live-btn' onclick='liveTranscribe()'
               style='padding:6px 12px;border-radius:8px;border:0;background:#2c4a58;
               color:#fff;cursor:pointer'>🔴 Live transcribe</button>
            <span class='hint' style='margin-left:8px'>real-time captions as you speak (offline, on-device)</span>
            <span id='live-status' class='hint' style='margin-left:6px'></span>
            <div id='live-caption' style='margin-top:6px;min-height:20px;padding:8px 10px;
               border-radius:8px;background:var(--panel);border:1px solid var(--line);
               color:var(--ink);font-size:14px'></div>
          </div>
        </div>
        <div class='card'>
          <label><input type='checkbox' name='do_azure' checked> Pronunciation scoring (Azure)</label><br>
          <label style='display:block;margin-top:8px'>Scoring strictness
            <select name='strictness'>
              <option value='standard'>Standard (Azure default)</option>
              <option value='strict' selected>Strict</option>
              <option value='very_strict'>Very strict</option>
            </select></label>
        </div>
        <button type='submit' style='font-size:16px;padding:10px 20px;border-radius:10px;
           border:0;background:var(--accent);color:#08222b;font-weight:700;cursor:pointer'>
           Analyze ▶</button>
      </form>
      <script>
        // when a file is picked, ask the server if we already have its
        // transcript / analysis JSON and auto-fill them.
        function lookupFiles(){
          var inp=document.getElementById('audio-input');
          var st=document.getElementById('lookup-status');
          if(!inp.files || !inp.files[0]){ st.textContent=''; return; }
          fetch('/lookup?name='+encodeURIComponent(inp.files[0].name))
            .then(function(r){return r.json();})
            .then(function(j){
              if(j.transcript){
                document.getElementById('transcript').value=j.transcript;
                st.textContent='Auto-loaded: saved transcript for this recording.';
              } else {
                st.textContent='No saved transcript for this recording yet.';
              }
            }).catch(function(){ st.textContent=''; });
        }
        function transcribeAudio(){
          var inp=document.getElementById('audio-input');
          var st=document.getElementById('tx-status');
          var btn=document.getElementById('tx-btn');
          if(!inp.files || !inp.files[0]){ st.textContent='Choose an audio/video file first.'; return; }
          var fd=new FormData(); fd.append('audio', inp.files[0]);
          fd.append('model', document.getElementById('tx-model').value);
          fd.append('lang', document.getElementById('tx-lang').value);
          btn.disabled=true;
          st.innerHTML='<span style=\"color:#43c59e\">●</span> Uploading…';
          fetch('/transcribe',{method:'POST',body:fd})
            .then(function(r){return r.json();})
            .then(function(j){
              if(j.error){ btn.disabled=false; st.textContent='Failed: '+j.error; return; }
              pollTx(j.job, Date.now());
            })
            .catch(function(e){ btn.disabled=false; st.textContent='Failed: '+e; });
        }
        function pollTx(job, t0){
          var st=document.getElementById('tx-status');
          var btn=document.getElementById('tx-btn');
          fetch('/tprogress/'+job).then(function(r){return r.json();}).then(function(j){
            var secs=Math.round((Date.now()-t0)/1000);
            if(j.error){ btn.disabled=false; st.textContent='Failed: '+j.error; return; }
            if(j.done){
              btn.disabled=false;
              document.getElementById('transcript').value=j.text||'';
              st.innerHTML='<span style=\"color:#43c59e\">✓</span> Done in '+secs+'s — review/fix it, then Analyze.';
              var chk=document.getElementById('tx-check');
              if(chk){
                if(j.uncertain && j.uncertain.length){
                  var words=j.uncertain.map(function(w){
                    return '<span style=\"background:rgba(255,180,84,.18);color:var(--warn);'+
                      'padding:1px 6px;border-radius:5px;margin:0 3px 3px 0;display:inline-block\">'+
                      w.replace(/</g,'&lt;')+'</span>';
                  }).join('');
                  chk.innerHTML='🔎 <b>Whisper was unsure about:</b> '+words+
                    '<br><span class=\"hint\">Check these against what you actually said and fix any wrong ones before Analyze.</span>';
                } else { chk.innerHTML='<span class=\"hint\">✓ No low-confidence words flagged.</span>'; }
              }
              return;
            }
            st.innerHTML='<span style=\"color:#43c59e\">●</span> '+(j.status||'Working…')+
              ' <span style=\"opacity:.7\">· '+secs+'s</span>';
            setTimeout(function(){pollTx(job,t0);}, 800);
          }).catch(function(){ setTimeout(function(){pollTx(job,t0);}, 1200); });
        }
        // ---- real-time captions via the sherpa-onnx transcribe service ----
        var _live={ws:null,ctx:null,node:null,stream:null,on:false,opened:false,finishing:false,committed:'',finTimer:null};
        function _dsp(buf,inRate){ if(16000>=inRate) return buf;
          var ratio=inRate/16000, n=Math.floor(buf.length/ratio), out=new Float32Array(n);
          for(var i=0;i<n;i++){ var a=Math.floor(i*ratio),b=Math.floor((i+1)*ratio),s=0,c=0;
            for(var j=a;j<b&&j<buf.length;j++){s+=buf[j];c++;} out[i]=c?s/c:0; } return out; }
        function liveTranscribe(){
          var btn=document.getElementById('live-btn'), st=document.getElementById('live-status');
          if(_live.on || _live.finishing){ stopLive(); return; }
          if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){ st.textContent='Mic not supported here.'; return; }
          var base=window.LIVE_WS; if(!base){ base=(location.protocol==='https:'?'wss://':'ws://')+location.host+'/v1/stream'; }
          var url=base+'?sample_rate=16000';
          var ws; try{ ws=new WebSocket(url); }catch(e){ st.textContent='Cannot open live transcribe.'; return; }
          ws.binaryType='arraybuffer'; _live.ws=ws; _live.committed=''; _live.opened=false;
          document.getElementById('live-caption').textContent='';
          st.textContent='Connecting…';
          ws.onopen=function(){ _live.opened=true; _live.on=true; btn.textContent='■ Stop live'; btn.style.background='#ff6b6b'; st.textContent='Listening…'; _startMic(); };
          ws.onerror=function(){
            // only blame the connection if it never opened; otherwise keep the
            // server's real message (e.g. "model not loaded")
            if(!_live.opened){ st.textContent='Could not open live transcribe. Is flask-sock installed and the app restarted?'; }
            _teardown(); btn.textContent='🔴 Live transcribe'; btn.style.background='#2c4a58'; };
          ws.onclose=function(){ _live.on=false; btn.textContent='🔴 Live transcribe'; btn.style.background='#2c4a58'; };
          ws.onmessage=function(ev){ var m; try{m=JSON.parse(ev.data);}catch(_){return;}
            var cap=document.getElementById('live-caption');
            if(m.type==='error'){ document.getElementById('live-status').textContent='Service error: '+m.error; return; }
            if(m.type==='final'){ _live.committed=(_live.committed+' '+(m.text||'')).replace(/\s+/g,' ').trim(); cap.textContent=_live.committed;
              if(_live.finishing) _finalize(); }
            else if(m.type==='partial'){ cap.textContent=(_live.committed+' '+(m.text||'')).replace(/\s+/g,' ').trim(); }
          };
        }
        function _startMic(){
          navigator.mediaDevices.getUserMedia({audio:true}).then(function(stream){
            _live.stream=stream; var Ctx=window.AudioContext||window.webkitAudioContext; var ctx=new Ctx(); _live.ctx=ctx;
            var src=ctx.createMediaStreamSource(stream), node=ctx.createScriptProcessor(4096,1,1); _live.node=node;
            var inRate=ctx.sampleRate;
            node.onaudioprocess=function(e){ if(!_live.ws||_live.ws.readyState!==1) return;
              var ds=_dsp(e.inputBuffer.getChannelData(0),inRate), i16=new Int16Array(ds.length);
              for(var i=0;i<ds.length;i++){ var v=Math.max(-1,Math.min(1,ds[i])); i16[i]=v<0?v*0x8000:v*0x7fff; }
              _live.ws.send(i16.buffer); };
            src.connect(node); node.connect(ctx.destination);
          }).catch(function(){ document.getElementById('live-status').textContent='Mic permission denied.'; stopLive(); });
        }
        function _teardown(){
          try{ if(_live.node) _live.node.disconnect(); }catch(_){}
          try{ if(_live.ctx) _live.ctx.close(); }catch(_){}
          try{ if(_live.stream) _live.stream.getTracks().forEach(function(t){t.stop();}); }catch(_){}
          _live.on=false;
        }
        function _finalize(){
          if(!_live.finishing) return; _live.finishing=false; if(_live.finTimer){clearTimeout(_live.finTimer);}
          if(_live.committed){ var ta=document.getElementById('transcript'); if(ta){ ta.value=(ta.value?ta.value.trim()+' ':'')+_live.committed; } }
          var btn=document.getElementById('live-btn'); if(btn){ btn.textContent='🔴 Live transcribe'; btn.style.background='#2c4a58'; }
          var st=document.getElementById('live-status'); if(st) st.textContent=_live.committed?'Stopped — captions added to the transcript.':'Stopped.';
          try{ if(_live.ws) _live.ws.close(); }catch(_){} _live.ws=null;
        }
        function stopLive(){
          _teardown(); _live.finishing=true;
          try{ if(_live.ws && _live.ws.readyState===1) _live.ws.send('{"type":"stop"}'); }catch(_){}
          _live.finTimer=setTimeout(_finalize, 900);
        }
        // ---- record audio directly in the browser, use it as the upload ----
        var _mic={};
        function recordMic(){
          var btn=document.getElementById('mic-btn');
          var st=document.getElementById('mic-status');
          if(btn.classList.contains('on')){ if(_mic.mr) _mic.mr.stop(); return; }
          if(!navigator.mediaDevices){ st.textContent='Recording not supported here.'; return; }
          navigator.mediaDevices.getUserMedia({audio:true}).then(function(stream){
            var mr=new MediaRecorder(stream); var chunks=[]; _mic.mr=mr;
            mr.ondataavailable=function(e){chunks.push(e.data);};
            mr.onstop=function(){
              stream.getTracks().forEach(function(t){t.stop();});
              var blob=new Blob(chunks,{type:'audio/webm'});
              var d=new Date(); var pad=function(n){return (n<10?'0':'')+n;};
              var name='Recording '+d.getFullYear()+pad(d.getMonth()+1)+pad(d.getDate())+
                       '-'+pad(d.getHours())+pad(d.getMinutes())+pad(d.getSeconds())+'.webm';
              var file=new File([blob],name,{type:'audio/webm'});
              try{ var dt=new DataTransfer(); dt.items.add(file);
                   document.getElementById('audio-input').files=dt.files; }catch(_){}
              btn.classList.remove('on'); btn.style.background='#2c4a58'; btn.textContent='● Record in browser';
              st.textContent='Captured '+name+' — fill the transcript, then Analyze.';
              lookupFiles();
            };
            mr.start(); btn.classList.add('on'); btn.style.background='#ff6b6b'; btn.textContent='■ Stop';
            st.textContent='Recording…';
          }).catch(function(){ st.textContent='Microphone permission denied.'; });
        }
        // ---- practice: record one word and re-score it via Azure ----
        var _pr={};
        function practiceRecord(btn){
          var card=btn.closest('.drill');
          var word=card.getAttribute('data-word');
          var out=card.querySelector('.presult');
          if(btn.classList.contains('on')){ if(_pr.mr) _pr.mr.stop(); return; }
          if(!navigator.mediaDevices){ out.textContent='Recording not supported.'; return; }
          navigator.mediaDevices.getUserMedia({audio:true}).then(function(stream){
            var mr=new MediaRecorder(stream); var chunks=[]; _pr.mr=mr;
            mr.ondataavailable=function(e){chunks.push(e.data);};
            mr.onstop=function(){
              stream.getTracks().forEach(function(t){t.stop();});
              btn.classList.remove('on'); btn.textContent='● Record';
              out.textContent='Scoring…';
              var blob=new Blob(chunks,{type:'audio/webm'});
              var fd=new FormData(); fd.append('word',word); fd.append('audio',blob,'w.webm');
              fetch('/practice',{method:'POST',body:fd}).then(function(r){return r.json();})
                .then(function(j){
                  if(j.error){ out.textContent='Failed: '+j.error; return; }
                  var col=j.score>=85?'#43c59e':(j.score>=70?'#ffb454':'#ff6b6b');
                  out.innerHTML='<b style=\"color:'+col+'\">'+j.score+'/100</b> · accuracy '+j.accuracy;
                }).catch(function(e){ out.textContent='Failed: '+e; });
            };
            mr.start(); btn.classList.add('on'); btn.textContent='■ Stop';
            out.textContent='Recording… say \"'+word+'\"';
          }).catch(function(){ out.textContent='Mic permission denied.'; });
        }
        window.recordMic=recordMic; window.practiceRecord=practiceRecord;
        // polished texts from past recordings can be reused as the script for a reread
        window.loadReading=function(i){ if(i===''||i==null)return; var s=(window.READING_ITEMS||[])[i];
          if(s){ document.getElementById('transcript').value=s.text; } };
        window.addEventListener('load',function(){ var sel=document.getElementById('reading-sel');
          if(!sel)return; (window.READING_ITEMS||[]).forEach(function(s,i){ var o=document.createElement('option');
            o.value=i; o.textContent=s.name; sel.appendChild(o); }); });
      </script>
    </section>
    """ % (note,)) + (
        "<script>window.LIVE_WS=%r;"
        # flash banner: strip msg from the URL (so a refresh won't repeat it)
        # and auto-dismiss after a few seconds
        "(function(){var m=document.getElementById('flash-msg');if(!m)return;"
        "try{var u=new URL(location.href);u.searchParams.delete('msg');"
        "history.replaceState(null,'',u.pathname+(u.search||''));}catch(e){}"
        "setTimeout(function(){m.style.transition='opacity .5s';m.style.opacity='0';"
        "setTimeout(function(){m.style.display='none';},500);},6000);})();</script>"
        % _ws_url)


_PRACTICE_JS = r"""
(function(){
 var S; var SORT={key:'avg',dir:'asc'}; var IPA={}; var _ipaBusy=false;
 function base(){ return window.PRACTICE_WORDS||[]; }
 function st(){ return window.SkillStore; }
 function custom(){ return st().get('pw_custom',[]); }
 function hidden(){ return st().get('pw_hidden',[]); }
 function words(){ var h=hidden(),seen={},out=[]; base().concat(custom()).forEach(function(w){ w=(''+w).trim(); var lw=w.toLowerCase(); if(w&&!seen[lw]&&h.indexOf(w)<0){seen[lw]=1;out.push(w);} }); return out; }
 function body(){ return document.getElementById('pw-body'); }
 function curve(a){ if(!a.length) return "<span class='hint'>—</span>";
   var pts=a.slice(-14),n=pts.length,W=140,H=30,p=3;
   function xs(i){return n===1?W/2:p+(W-2*p)*i/(n-1);}
   function ys(s){return H-p-(H-2*p)*(s/100);}
   var d=pts.map(function(o,i){return (i?'L':'M')+xs(i).toFixed(1)+' '+ys(o.s).toFixed(1);}).join(' ');
   function pc(s){ return s>=85?'#43c59e':s>=75?'#ffb454':'#ff6b6b'; }
   var last=pts[n-1].s,col=pc(last);
   var dots=pts.map(function(o,i){return "<circle cx='"+xs(i).toFixed(1)+"' cy='"+ys(o.s).toFixed(1)+"' r='2.1' fill='"+pc(o.s)+"'/>";}).join('');
   return "<svg width='"+W+"' height='"+H+"' style='vertical-align:middle'>"+
     "<line x1='0' y1='"+ys(85).toFixed(1)+"' x2='"+W+"' y2='"+ys(85).toFixed(1)+"' stroke='#43c59e' stroke-opacity='.35' stroke-dasharray='3 3'/>"+
     "<path d='"+d+"' fill='none' stroke='"+col+"' stroke-width='1.7'/>"+dots+"</svg>";
 }
 function num(v){ if(v==null)return "<span class='hint'>—</span>"; var col=v>=85?'#43c59e':v>=70?'#ffb454':'#ff6b6b'; return "<b style='color:"+col+"'>"+v+"</b>"; }
 function metrics(word){ var a=(st().get('ec_scores',{}))['word:'+word.toLowerCase()]||[];
   var last=a.length?a[a.length-1].s:null;
   var best=a.length?Math.max.apply(null,a.map(function(x){return x.s;})):null;
   var avg=a.length?Math.round(a.reduce(function(x,y){return x+y.s;},0)/a.length):null;
   return {word:word,arr:a,last:last,best:best,avg:avg,tries:a.length}; }
 function sortVal(m){ var k=SORT.key; if(k==='word')return m.word.toLowerCase();
   var v=(k==='avg'?m.avg:k==='last'?m.last:k==='best'?m.best:m.tries); return v==null?-1:v; }
 function render(){
   S=st(); var el=body(); if(!el)return;
   var ms=words().map(metrics);
   ms.sort(function(a,b){
     var at=a.tries>0, bt=b.tries>0; if(at!==bt) return at?-1:1;   // tried words first
     var va=sortVal(a),vb=sortVal(b);
     if(SORT.key==='word'){ return SORT.dir==='asc'?(va<vb?-1:va>vb?1:0):(va<vb?1:va>vb?-1:0); }
     return SORT.dir==='asc'? va-vb : vb-va; });
   function isMastered(m){ var a=m.arr||[]; if(a.length<3) return false; return a.slice(-3).every(function(x){return x.s>=85;}); }
   var mastered=ms.filter(isMastered).length;
   var left=ms.length-mastered;
   var pct=ms.length?Math.round(mastered/ms.length*100):0;
   // The headline answers "how far through the list am I?". Attempt count and
   // average-last-score were noise: both drift as words are added and neither
   // says how much work is left.
   var summary="<div class='card' style='display:flex;gap:22px;flex-wrap:wrap;align-items:center'>"+
     "<span><b style='font-size:22px'>"+ms.length+"</b> <span class='hint'>words</span></span>"+
     "<span><b style='font-size:22px;color:var(--good)'>"+mastered+"</b> <span class='hint'>mastered ("+pct+"%)</span></span>"+
     "<span><b style='font-size:22px'>"+left+"</b> <span class='hint'>still to go</span></span>"+
     "<span style='flex:1'></span>"+
     "<input id='pw-new' placeholder='add a word…' style='max-width:180px' onkeydown='if(event.key===\"Enter\")PW.add()'> "+
     "<button class='btn small' onclick='PW.add()'>➕ Add</button></div>"+
     "<div class='sb-t' style='height:8px;margin:0 2px 4px'><div class='sb-f' style='width:"+pct+"%'></div></div>";
   var status="<div id='pw-status' class='hint' style='margin:8px 2px;min-height:18px'></div>";
   var mask=st().get('pw_hidemastered',false);
   var shown=mask?ms.filter(function(m){return !isMastered(m);}):ms;
   var fbtn="<button class='btn small' onclick='PW.toggleMask()' style='"+(mask?'background:var(--accent);color:#08222b;border-color:var(--accent)':'')+"'>"+(mask?'☑':'☐')+" Hide mastered (last 3 ≥ 85)</button>";
   var fbar="<div style='display:flex;gap:12px;align-items:center;margin:10px 2px 4px;flex-wrap:wrap'>"+fbtn+
     "<span class='hint'>showing "+shown.length+" of "+ms.length+"</span>"+
     "<span style='flex:1'></span>"+
     "<button class='btn small' onclick='PW.reset()' style='background:#3a2030;border-color:#5a2a3a;color:#ff9db0'>🗑 Reset history</button></div>";
   if(!ms.length){ el.innerHTML=summary+fbar+status+"<div class='card'>No words yet — add one above, or analyze a recording to auto-fill your blind spots.</div>"; return; }
   function hd(k,label,center){ var ar=SORT.key===k?(SORT.dir==='asc'?' ▲':' ▼'):'';
     return "<th onclick='PW.sort(\""+k+"\")' style='cursor:pointer"+(center?';text-align:center':'')+"'>"+label+ar+"</th>"; }
   if(!shown.length){ el.innerHTML=summary+fbar+status+"<div class='card'>🎉 Every word is mastered (last 3 tries ≥ 85). Untick the filter to see them all.</div>"; return; }
   var rows=shown.map(function(m){
     return "<tr>"+
       "<td><b style='font-size:16px'>"+S.esc(m.word)+"</b></td>"+
       "<td class='hint' style='font-family:ui-monospace,monospace;white-space:nowrap'>"+S.esc(IPA[m.word.toLowerCase()]||'')+"</td>"+
       "<td>"+curve(m.arr)+"</td>"+
       "<td style='text-align:center'>"+num(m.last)+"</td>"+
       "<td style='text-align:center'>"+num(m.best)+"</td>"+
       "<td style='text-align:center' class='hint'>"+m.tries+"</td>"+
       "<td style='text-align:right;white-space:nowrap'>"+
         "<button class='btn small' data-say=\""+S.esc(m.word)+"\">🔊</button> "+
         "<button class='btn small rec pwrec' data-word=\""+S.esc(m.word)+"\">● Rec</button> "+
         "<button class='btn small' style='background:#1f3542' onclick='PW.del("+JSON.stringify(m.word)+")'>✕</button></td></tr>";
   }).join('');
   el.innerHTML=summary+fbar+status+"<table class='pwt'><tr>"+
     hd('word','Word')+"<th>IPA</th>"+hd('avg',"Trend (avg · goal 85)")+hd('last','Last',1)+
     hd('best','Best',1)+hd('tries','Tries',1)+"<th></th></tr>"+rows+"</table>";
   fetchIPA(ms.map(function(m){return m.word;}));
 }
 function fetchIPA(list){
   if(_ipaBusy) return;
   var miss=[]; list.forEach(function(w){ if(IPA[w.toLowerCase()]===undefined) miss.push(w); });
   if(!miss.length) return;
   _ipaBusy=true;
   fetch('/ipa?words='+encodeURIComponent(miss.join(','))).then(function(r){return r.json();}).then(function(j){
     miss.forEach(function(w){ var k=w.toLowerCase(); IPA[k]=(j[k]||''); });
     st().set('pw_ipa',IPA); _ipaBusy=false; render();
   }).catch(function(){ miss.forEach(function(w){ IPA[w.toLowerCase()]=''; }); _ipaBusy=false; });
 }
 document.addEventListener('click',function(e){
   var b=e.target.closest && e.target.closest('.pwrec'); if(!b)return;
   var word=b.getAttribute('data-word'); var key='word:'+word.toLowerCase(); var status=document.getElementById('pw-status');
   if(b.__mr){ b.__mr.stop(); return; }
   if(!navigator.mediaDevices){ if(status)status.textContent='Recording not supported.'; return; }
   navigator.mediaDevices.getUserMedia({audio:true}).then(function(s){
     var mr=new MediaRecorder(s),ch=[]; b.__mr=mr; b.__t0=Date.now(); b.textContent='■ Stop'; b.classList.add('on');
     if(status)status.textContent='Recording… say "'+word+'", then tap Stop.';
     mr.ondataavailable=function(ev){ if(ev.data&&ev.data.size)ch.push(ev.data); };
     mr.onstop=function(){ s.getTracks().forEach(function(t){t.stop();}); b.__mr=null; b.textContent='● Rec'; b.classList.remove('on');
       var blob=new Blob(ch,{type:'audio/webm'});
       if(blob.size<1200 || (Date.now()-b.__t0)<500){ if(status)status.innerHTML='<span style="color:var(--warn)">Too short — hold Rec, say "'+word+'", then Stop.</span>'; return; }
       if(status)status.textContent='Scoring "'+word+'"…';
       var fd=new FormData(); fd.append('word',word); fd.append('audio',blob,'w.webm');
       fetch('/practice',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(j){
         if(j.error){ if(status)status.textContent=j.error; return; }
         var sc=Math.max(0,Math.min(100,Math.round(j.score)));
         window.SkillStore.logScore(key,sc); if(status)status.innerHTML='<b>'+word+'</b> → '+sc+'/100 logged'; render();
       }).catch(function(){ if(status)status.textContent='No server / offline.'; });
     };
     mr.start(200);
   }).catch(function(){ if(status)status.textContent='Mic permission denied.'; });
 });
 window.PW={ add:function(){ var i=document.getElementById('pw-new'); var v=(i.value||'').trim(); if(!v)return;
     var lw=v.toLowerCase();
     if(words().some(function(x){return x.toLowerCase()===lw;})){
       var s=document.getElementById('pw-status'); if(s) s.innerHTML="⚠️ <b>"+st().esc(v)+"</b> is already in your list.";
       var inp=document.getElementById('pw-new'); if(inp){inp.focus(); inp.select();} return; }
     var h=hidden(); if(h.some(function(x){return x.toLowerCase()===lw;})) st().set('pw_hidden', h.filter(function(x){return x.toLowerCase()!==lw;}));
     var c=custom(); c.push(v); st().set('pw_custom',c); render();
     var s2=document.getElementById('pw-status'); if(s2) s2.textContent='Added “'+v+'”.';
     var ni=document.getElementById('pw-new'); if(ni)ni.focus(); },
   del:function(word){ st().set('pw_custom', custom().filter(function(x){return x!==word;})); var h=hidden(); if(h.indexOf(word)<0){h.push(word); st().set('pw_hidden',h);} render(); },
   sort:function(k){ if(SORT.key===k){ SORT.dir=SORT.dir==='asc'?'desc':'asc'; } else { SORT.key=k; SORT.dir=(k==='word')?'asc':'desc'; } render(); },
   toggleMask:function(){ st().set('pw_hidemastered', !st().get('pw_hidemastered',false)); render(); },
   reset:function(){
     var all=st().get('ec_scores',{}); var n=0;
     Object.keys(all).forEach(function(k){ if(k.indexOf('word:')===0) n++; });
     if(!n){ var s=document.getElementById('pw-status'); if(s)s.textContent='No single-word history to reset.'; return; }
     if(!confirm('Reset ALL single-word score history?\n\nThis permanently clears every recorded attempt for all '+n+' words (story scores are kept). This cannot be undone.')) return;
     var kept={}; Object.keys(all).forEach(function(k){ if(k.indexOf('word:')!==0) kept[k]=all[k]; });
     st().set('ec_scores',kept); render();
     var s2=document.getElementById('pw-status'); if(s2)s2.textContent='Single-word score history cleared.'; } };
 window.addEventListener('load', function(){
   IPA=Object.assign({}, st().get('pw_ipa',{}), window.PRACTICE_IPA||{});
   // heal any legacy out-of-range scores (e.g. a stray 150) already in storage
   var all=st().get('ec_scores',{}),ch=false;
   Object.keys(all).forEach(function(k){ (all[k]||[]).forEach(function(x){
     if(x&&typeof x.s==='number'){ var c=Math.max(0,Math.min(100,Math.round(x.s))); if(c!==x.s){x.s=c;ch=true;} } }); });
   if(ch) st().set('ec_scores',all);
   if(document.getElementById('pw-body')) render(); });
})();
"""


# Minimal-pair words collected from handwritten practice notes — always
# available to drill, on top of whatever an analysis surfaces.
_PRACTICE_SEED = [
    "dress", "quilt", "fold", "bed", "bad", "bird", "bedspread", "bedpad",
    "wrong", "long", "wring", "ring", "comb", "towel", "tower", "mouth", "mouse",
    "wash", "what's", "dinner", "dessert", "desert", "dream",
    "show", "shoe", "advert", "edward", "word", "world", "turn", "ten", "red",
    "led", "lad", "celery", "salary", "are", "ah", "lamb", "lamp", "them", "then",
    "dyed", "tied", "brown", "belong", "perm", "pan", "live", "leave", "all", "or",
]


def _practice_panel(items):
    from collections import Counter
    words = Counter()
    for d in items or []:
        az = d.get("azure") or {}
        for w in az.get("words", []):
            if w.get("error") == "Mispronunciation":
                tok = (w.get("word", "") or "").strip(".,!?;:")
                if tok:
                    words[tok.lower()] += 1
    top = [w for w, _ in words.most_common(20)]
    # append the notebook seed words, de-duplicated, preserving order
    top = list(dict.fromkeys(top + _PRACTICE_SEED))
    ipa_map = {w.lower(): word_ipa(w) for w in top}
    payload = json.dumps(top).replace("</", "<\\/")
    ipa_payload = json.dumps(ipa_map, ensure_ascii=False).replace("</", "<\\/")
    return ("<section id='practice' class='tabpanel hidden'>"
            "<h1>Practice your blind spots</h1>"
            "<p class='sub'>Say each word, record it, get an instant Azure score — and every "
            "attempt is logged so you can watch each word improve. Add your own words too.</p>"
            "<div id='pw-body'></div></section>"
            "<script>window.PRACTICE_WORDS=%s;window.PRACTICE_IPA=%s;%s</script>"
            % (payload, ipa_payload, _PRACTICE_JS))


def _h_attr(s):
    import html as _h
    return _h.escape(str(s), quote=True)


def _page(active="summary", msg=""):
    items, history = _load_items_and_history()
    _sum_cls = " class='active'" if active == "summary" else ""
    extra_nav = ("<a data-panel='settings'>⚙️ Setting Panel</a>"
                 "<a data-panel='summary'%s>📈 Summary &amp; progress</a>"
                 "<a data-panel='newrec'>➕ New Speaking Analysis</a>"
                 "<a data-panel='practice'>🎯 Practice single word</a>"
                 % _sum_cls)
    extra_panels = (_settings_panel(msg, active) + _form_panel(msg, active)
                    + _practice_panel(items))
    return ec.generate_dashboard_html(
        items, history, extra_nav=extra_nav,
        extra_panels=extra_panels, active=active)


@app.route("/")
def index():
    return _page(active=request.args.get("p", "summary"),
                 msg=request.args.get("msg", ""))


@app.route("/VideoAudioFiles/<path:p>")
def media(p):
    from flask import send_from_directory
    return send_from_directory(LIBRARY, p)


IJOBS = {}   # listening-import jobs: id -> {log, done, error}
IMPORT_SOURCES = ("tatoeba",)   # only sources that need no extra arguments


def _run_import(job, source, count):
    """Shell out to listening_import.py and stream its output into the job log.

    Fixed argv with a validated source and an integer count — nothing from the
    request is ever interpolated into a shell string.
    """
    import subprocess
    cmd = [sys.executable, os.path.join(HERE, "listening_import.py"),
           source, "--count", str(count)]
    try:
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            line = line.rstrip()
            if line:
                IJOBS[job]["log"].append(line)
                del IJOBS[job]["log"][:-40]
        p.wait()
        if p.returncode != 0:
            IJOBS[job]["error"] = "importer exited with code %d" % p.returncode
    except Exception as e:
        IJOBS[job]["error"] = str(e)[:200]
    finally:
        IJOBS[job]["done"] = True


@app.route("/import_listening", methods=["POST"])
def import_listening():
    from flask import jsonify
    source = (request.form.get("source") or "tatoeba").strip().lower()
    if source not in IMPORT_SOURCES:
        return jsonify(error="Only %s can be imported from the panel; the other "
                             "sources need a --dir on the command line."
                             % ", ".join(IMPORT_SOURCES))
    try:
        count = max(1, min(1000, int(request.form.get("count", "200"))))
    except (TypeError, ValueError):
        count = 200
    job = uuid.uuid4().hex[:8]
    IJOBS[job] = {"log": [], "done": False, "error": None}
    threading.Thread(target=_run_import, args=(job, source, count),
                     daemon=True).start()
    return jsonify(job=job)


@app.route("/import_progress/<job>")
def import_progress(job):
    from flask import jsonify
    j = IJOBS.get(job)
    if not j:
        return jsonify(error="unknown job", done=True, log=[])
    return jsonify(log=j["log"], done=j["done"], error=j["error"])


@app.route("/backup", methods=["GET", "POST"])
def backup():
    """Send a zip of the project — source, data and notes, no media.

    Built in memory and streamed as a download, so the backup lands somewhere
    other than the folder it is backing up. Scores/schedules/error logs live in
    progress.json now (server-side, next to history.json), so they're already
    picked up by _backup_members() like any other project file — nothing to
    collect from the browser.
    """
    from flask import Response
    data, name, count = ec.build_project_backup()
    return Response(data, mimetype="application/zip", headers={
        "Content-Disposition": 'attachment; filename="%s"' % name,
        "Content-Length": str(len(data)),
        "X-Backup-Name": name,
        "X-Backup-Files": str(count),
    })


@app.route("/listening/<path:p>")
def listening_media(p):
    """Serve the imported listening clips."""
    from flask import send_from_directory
    return send_from_directory(ec.listening_dir(), p)


@app.route("/dictation", methods=["POST"])
def dictation():
    """Grade a dictation attempt against the clip's reference transcript.

    The reference stays server-side and is looked up by clip id, so the answer
    can't be read out of the page before you've attempted it.
    """
    from flask import jsonify
    cid = (request.form.get("id") or "").strip()
    typed = (request.form.get("typed") or "").strip()
    if not cid or not typed:
        return jsonify(error="missing clip id or text")
    clip = next((c for c in ec.load_listening_library() if c["id"] == cid), None)
    if not clip:
        return jsonify(error="unknown clip — re-run the importer and reload the page")
    res = ec.dictation_check(clip["text"], typed)
    res["text"] = clip["text"]
    return jsonify(res)


@app.route("/tts")
def tts():
    """Server-side text-to-speech using the OS voice (macOS `say`), so playback
    doesn't depend on the browser's flaky Web Speech engine."""
    from flask import Response
    import shutil, subprocess, tempfile
    text = (request.args.get("text") or "").strip()[:2000]
    try:
        rate = max(80, min(300, int(request.args.get("r", "175"))))
    except Exception:
        rate = 175
    say = shutil.which("say")
    if not text or not say:
        return ("", 404)  # browser will fall back to Web Speech
    out = os.path.join(tempfile.gettempdir(), "tts_" + uuid.uuid4().hex[:8] + ".wav")
    try:
        subprocess.run([say, "-o", out, "--file-format=WAVE",
                        "--data-format=LEI16@22050", "-r", str(rate), text],
                       check=True, timeout=25,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out, "rb") as f:
            data = f.read()
        return Response(data, mimetype="audio/wav")
    except Exception:
        return ("", 500)
    finally:
        try:
            os.remove(out)
        except Exception:
            pass


@app.route("/ipa")
def ipa_lookup():
    from flask import jsonify
    words = (request.args.get("words") or "").split(",")
    out = {}
    for w in words:
        w = w.strip()
        if w:
            out[w.lower()] = word_ipa(w)
    return jsonify(out)


@app.route("/practice", methods=["POST"])
def practice():
    """Score a short re-recording of a single word against itself (Azure)."""
    from flask import jsonify
    import tempfile
    word = (request.form.get("word") or "").strip()
    f = request.files.get("audio")
    if not word or not f:
        return jsonify(error="missing word or audio")
    cfg = load_config()
    key = os.environ.get("AZURE_SPEECH_KEY") or cfg.get("azure_key", "")
    region = os.environ.get("AZURE_SPEECH_REGION") or cfg.get("azure_region", "eastus")
    if not key:
        return jsonify(error="No Azure key set — run an analysis once to save it.")
    os.environ["AZURE_SPEECH_KEY"] = key
    os.environ["AZURE_SPEECH_REGION"] = region
    tmp = os.path.join(tempfile.gettempdir(), "practice_" + uuid.uuid4().hex[:6] + ".webm")
    f.save(tmp)
    if os.path.getsize(tmp) < 1200:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return jsonify(error="The recording was empty or too short — hold the button, "
                             "say the word clearly, then Stop.")
    try:
        # Single-word drill: prosody/fluency are meaningless for one word and
        # only drag the number down, so skip prosody and grade on Accuracy —
        # the phoneme-level quality that actually reflects the pronunciation.
        # Miscue OFF too, or homophones (tied/tide, dyed/died) get docked for a
        # spelling mismatch the recogniser invents even when the sound is right.
        az = ec.azure_pronunciation(tmp, word, enable_prosody=False,
                                    enable_miscue=False, locale="en-US")
        acc = az.get("accuracy")
        score = acc if acc is not None else az.get("pron_score")
        return jsonify(score=score, accuracy=acc, pron=az.get("pron_score"),
                       words=[{"w": w["word"], "a": w["accuracy"], "e": w["error"]}
                              for w in az.get("words", [])])
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "end of file" in low or "invalid data" in low or "541478725" in msg:
            msg = ("Couldn't read the recording — it looks empty or cut off. "
                   "Hold the button, speak for about a second, then Stop.")
        return jsonify(error=msg[:200])
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


@app.route("/vocab_photo", methods=["POST"])
def vocab_photo():
    """Turn an uploaded photo into candidate vocabulary items (vision LLM).

    ec.vision_vocab_from_image() picks Kimi if a key is set, else Anthropic —
    both keys are passed through here so it can choose. The browser already
    downscales the image to a JPEG before this runs (see the vocab panel's
    downscaleImage), so the size guard here is just a backstop.
    """
    from flask import jsonify
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image received.")
    data = f.read()
    if not data:
        return jsonify(error="The image was empty.")
    if len(data) > 6 * 1024 * 1024:
        return jsonify(error="Image is too large (max 6MB).")
    cfg = load_config()
    kimi_key = os.environ.get("KIMI_API_KEY") or cfg.get("kimi_key", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_key", "")
    if not (kimi_key or anthropic_key):
        return jsonify(error="No Kimi or Anthropic key set — photo capture needs "
                             "a vision-capable model (DeepSeek's API is "
                             "text-only). Add one in Settings.")
    if kimi_key:
        os.environ["KIMI_API_KEY"] = kimi_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    try:
        items = ec.vision_vocab_from_image(data, mime_type=f.mimetype or "image/jpeg")
        return jsonify(items=items)
    except Exception as e:
        return jsonify(error=str(e)[:300])


def _stem_of(filename):
    return os.path.splitext(safe_name(filename))[0]


@app.route("/lookup")
def lookup():
    """Does this recording already have a saved transcript on disk?"""
    from flask import jsonify
    stem = _stem_of(request.args.get("name", ""))
    if not stem:
        return jsonify(transcript="")
    d = rec_dir(stem, create=False)
    txt = os.path.join(d, stem + ".txt")
    transcript = ""
    if os.path.exists(txt):
        try:
            with open(txt, encoding="utf-8") as f:
                transcript = f.read()
        except Exception:
            transcript = ""
    return jsonify(transcript=transcript)


TJOBS = {}  # transcription jobs: id -> {status, t, done, text, error}


def _run_transcribe(job, audio_path, d, stem, model_name="base", language=None):
    def prog(m):
        TJOBS[job]["status"] = m
        TJOBS[job]["t"] = time.time()
    try:
        text, _w, _dur = ec.transcribe(audio_path, language=language,
                                       model_name=model_name, progress=prog)
        with open(os.path.join(d, stem + ".txt"), "w", encoding="utf-8") as t:
            t.write(text)
        # words Whisper was unsure about — the user should double-check these
        uncertain, seen = [], set()
        for w in _w or []:
            tok = (w.get("w", "") or "").strip().strip(".,!?;:\"'“”‘’()").strip()
            if not tok:
                continue
            if w.get("prob", 1) < 0.55 and tok.lower() not in seen:
                seen.add(tok.lower())
                uncertain.append(tok)
        TJOBS[job]["uncertain"] = uncertain[:12]
        TJOBS[job]["text"] = text
        TJOBS[job]["done"] = True
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        TJOBS[job]["error"] = str(tb).splitlines()[-1][:300]
        TJOBS[job]["done"] = True


@app.route("/transcribe", methods=["POST"])
def transcribe():
    from flask import jsonify
    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify(error="no audio file")
    name = safe_name(f.filename)
    stem = os.path.splitext(name)[0]
    d = rec_dir(stem)
    audio_path = os.path.join(d, name)
    try:
        f.save(audio_path)
    except PermissionError:
        if not os.path.exists(audio_path):
            raise
    model_name = request.form.get("model", "base")
    if model_name not in ("base", "small", "medium", "large-v3", "large-v2"):
        model_name = "base"
    lang = (request.form.get("lang", "") or "").strip() or None   # None = auto-detect
    if lang not in (None, "en", "zh"):
        lang = None
    job = uuid.uuid4().hex[:8]
    TJOBS[job] = {"status": "Starting…", "t": time.time(), "done": False,
                  "text": None, "error": None}
    threading.Thread(target=_run_transcribe,
                     args=(job, audio_path, d, stem, model_name, lang),
                     daemon=True).start()
    return jsonify(job=job)


@app.route("/tprogress/<job>")
def tprogress(job):
    from flask import jsonify
    j = TJOBS.get(job)
    if not j:
        return jsonify(error="unknown job", done=True)
    return jsonify(status=j.get("status", ""), done=j.get("done", False),
                   text=j.get("text"), error=j.get("error"),
                   uncertain=j.get("uncertain"),
                   since=round(time.time() - j.get("t", time.time()), 1))


def _error_page(title, detail):
    """A readable error page instead of a bare 500, with the traceback to copy."""
    import html as _h
    return ("<!doctype html><meta charset='utf-8'>"
            "<body style='font:15px/1.6 -apple-system,Segoe UI,Arial;max-width:820px;"
            "margin:40px auto;padding:0 20px;color:#1a1f36'>"
            "<h2 style='color:#b91c1c'>%s</h2>"
            "<pre style='background:#f4f5f8;padding:14px;border-radius:8px;"
            "white-space:pre-wrap;overflow:auto'>%s</pre>"
            "<p><a href='/?p=newrec'>&larr; Back to the form</a></p></body>"
            % (_h.escape(title), _h.escape(detail)), 200)


def _run_job(job, audio_path, d, stem, transcript, do_azure, do_llm, strictness):
    """Background worker: does the slow analysis and updates JOBS[job]."""
    def prog(m):
        now = time.time()
        j = JOBS[job]
        j["status"] = m
        j["t"] = now
        log = j.setdefault("log", [])
        if not log or log[-1]["m"] != m:          # skip exact repeats
            log.append({"s": round(now - j.get("t0", now), 1), "m": m})
            if len(log) > 40:
                del log[0:len(log) - 40]
    try:
        data = ec.analyze_recording(
            audio_path, reference_text=transcript or None, do_llm=do_llm,
            do_azure=do_azure, strictness=strictness, progress=prog)
        data.setdefault("title", stem)
        prog("Saving results…")
        with open(os.path.join(d, stem + ".result.json"), "w", encoding="utf-8") as r:
            json.dump(data, r, ensure_ascii=False, indent=2)
        if data.get("polished"):
            with open(os.path.join(d, stem + ".polished.txt"), "w", encoding="utf-8") as p:
                p.write(data["polished"])
        ec.log_session(data, os.path.join(LIBRARY, "history.json"))
        prog("Building dashboard…")
        with open(os.path.join(HERE, "dashboard.html"), "w", encoding="utf-8") as h:
            h.write(ec.build_dashboard_for_dir(LIBRARY))
        prog("Done ✓")
        JOBS[job]["status"] = "Done"
        JOBS[job]["done"] = True
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        JOBS[job]["error"] = tb
        JOBS[job]["done"] = True


@app.route("/progress/<job>")
def progress(job):
    from flask import jsonify
    j = JOBS.get(job)
    if not j:
        return jsonify(error="unknown job", done=True)
    return jsonify(status=j.get("status", ""), done=j.get("done", False),
                   error=j.get("error"), log=j.get("log", []),
                   since=round(time.time() - j.get("t", time.time()), 1))


@app.route("/delete_session", methods=["POST"])
def delete_session():
    """Remove one entry (by index) from history.json and rebuild the dashboard.
    Keeps the audio file and analysis on disk — this only clears the progress row."""
    from flask import jsonify
    try:
        i = int((request.get_json(silent=True) or {}).get("i"))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="bad index")
    hp = os.path.join(LIBRARY, "history.json")
    try:
        with open(hp, encoding="utf-8") as f:
            hist = json.load(f)
    except Exception:
        return jsonify(ok=False, error="no history file")
    if not (0 <= i < len(hist)):
        return jsonify(ok=False, error="index out of range")
    hist.pop(i)
    try:
        with open(hp, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2, ensure_ascii=False)
        with open(os.path.join(HERE, "dashboard.html"), "w", encoding="utf-8") as h:
            h.write(ec.build_dashboard_for_dir(LIBRARY))
    except Exception as e:
        return jsonify(ok=False, error=str(e)[:150])
    return jsonify(ok=True)


def _waiting_page(job):
    return ("<!doctype html><meta charset='utf-8'>"
            "<title>Analyzing…</title>"
            "<style>"
            "body{font:16px/1.6 -apple-system,Segoe UI,Arial;background:#0d1620;color:#eaf3f2;"
            "display:flex;min-height:100vh;margin:0;align-items:center;justify-content:center}"
            ".box{width:640px;max-width:92vw;padding:32px;text-align:center}"
            ".dot{display:inline-block;width:12px;height:12px;border-radius:50%;background:#43c59e;"
            "margin-right:8px;animation:beat 1s infinite ease-in-out}"
            "@keyframes beat{0%,100%{transform:scale(.7);opacity:.5}50%{transform:scale(1.3);opacity:1}}"
            ".bar{height:6px;background:#24404c;border-radius:6px;overflow:hidden;margin:18px 0}"
            ".bar i{display:block;height:100%;width:35%;background:#46b3c9;border-radius:6px;"
            "animation:slide 1.4s infinite}"
            "@keyframes slide{0%{margin-left:-35%}100%{margin-left:100%}}"
            ".muted{color:#8fa6ad;font-size:14px}"
            ".steps{display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin:16px 0 8px}"
            ".step{font-size:12.5px;padding:4px 10px;border-radius:999px;background:#172530;"
            "color:#8fa6ad;border:1px solid #22343f}"
            ".step.on{background:rgba(70,179,201,.18);color:#46b3c9;border-color:#46b3c9}"
            ".step.done{background:rgba(67,197,158,.16);color:#43c59e;border-color:#43c59e}"
            "#log{text-align:left;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;"
            "background:#0a1119;border:1px solid #22343f;border-radius:10px;padding:10px 12px;"
            "max-height:190px;overflow:auto;margin:14px 0}"
            ".logrow{padding:1px 0;color:#c7d3d6}.logrow:last-child{color:#eaf3f2}"
            ".ts{color:#5f7b84;display:inline-block;min-width:44px}"
            "</style>"
            "<div class='box'>"
            "<h2><span class='dot'></span>Analyzing your recording</h2>"
            "<div class='bar'><i></i></div>"
            "<div class='steps' id='steps'>"
            "<span class='step' data-k='transcribe'>1 · Transcribe</span>"
            "<span class='step' data-k='grammar'>2 · Grammar</span>"
            "<span class='step' data-k='pron'>3 · Pronunciation</span>"
            "<span class='step' data-k='save'>4 · Save</span>"
            "<span class='step' data-k='dash'>5 · Dashboard</span></div>"
            "<p id='status'>Starting…</p>"
            "<div id='log'></div>"
            "<p class='muted'><span id='elapsed'>0</span>s elapsed · "
            "<span id='beat'>live</span></p>"
            "<p class='muted'>Whisper transcription and Azure scoring can take a "
            "minute or two for longer clips — this log updates live as it works.</p>"
            "</div>"
            "<script>"
            "var job='" + job + "', t0=Date.now();"
            "function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}"
            "setInterval(function(){document.getElementById('elapsed').textContent="
            "Math.round((Date.now()-t0)/1000);},1000);"
            "function stage(msg){msg=(msg||'').toLowerCase();"
            "  if(msg.indexOf('transcrib')>=0||msg.indexOf('whisper')>=0||msg.indexOf('model')>=0)return 'transcribe';"
            "  if(msg.indexOf('grammar')>=0||msg.indexOf('claude')>=0||msg.indexOf('deepseek')>=0)return 'grammar';"
            "  if(msg.indexOf('pronunciation')>=0||msg.indexOf('azure')>=0||msg.indexOf('scoring')>=0||msg.indexOf('grading')>=0)return 'pron';"
            "  if(msg.indexOf('saving')>=0)return 'save';"
            "  if(msg.indexOf('dashboard')>=0||msg.indexOf('done')>=0)return 'dash';"
            "  return '';}"
            "var ORDER=['transcribe','grammar','pron','save','dash'];"
            "function setSteps(cur){var ci=ORDER.indexOf(cur);"
            "  document.querySelectorAll('.step').forEach(function(el){"
            "    var i=ORDER.indexOf(el.getAttribute('data-k'));el.classList.remove('on','done');"
            "    if(ci<0)return; if(i<ci)el.classList.add('done'); else if(i===ci)el.classList.add('on');});}"
            "function poll(){fetch('/progress/'+job).then(function(r){return r.json();})"
            ".then(function(j){"
            "  if(j.status){document.getElementById('status').textContent=j.status; setSteps(stage(j.status));}"
            "  if(j.log){var el=document.getElementById('log');"
            "    el.innerHTML=j.log.map(function(e){return \"<div class='logrow'><span class='ts'>\"+e.s+\"s</span> \"+esc(e.m)+\"</div>\";}).join('');"
            "    el.scrollTop=el.scrollHeight;}"
            "  document.getElementById('beat').textContent="
            "    (j.since!=null? ('updated '+j.since+'s ago'):'live');"
            "  if(j.error){document.body.innerHTML="
            "    '<div style=\"max-width:820px;margin:40px auto;padding:0 20px\">"
            "<h2 style=\"color:#ff6b6b\">Analysis failed</h2><pre style=\"white-space:pre-wrap;"
            "background:#172530;padding:14px;border-radius:8px\">'+esc(j.error)+'</pre>"
            "<a style=\"color:#46b3c9\" href=\"/?p=newrec\">&larr; Back</a></div>'; return;}"
            "  if(j.done){window.location='/?p=rec0&msg='+encodeURIComponent('\\u2713 Analysis complete — here is your report.'); return;}"
            "  setTimeout(poll,1000);"
            "}).catch(function(){setTimeout(poll,1500);});}"
            "poll();"
            "</script>")


@app.route("/settings", methods=["POST"])
def settings():
    _persist_keys_from_form(request.form)
    return redirect("/?p=settings&msg=" + quote("✓ Settings saved."))


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        f = request.files.get("audio")
        if not f or not f.filename:
            return redirect("/?p=newrec&msg=Please+choose+an+audio+file.")
        name = safe_name(f.filename)
        stem = os.path.splitext(name)[0]
        d = rec_dir(stem)
        audio_path = os.path.join(d, name)
        try:
            f.save(audio_path)
        except PermissionError:
            if not os.path.exists(audio_path):
                raise

        transcript = (request.form.get("transcript") or "").strip()
        if not transcript and os.path.exists(os.path.join(d, stem + ".txt")):
            with open(os.path.join(d, stem + ".txt"), encoding="utf-8") as t:
                transcript = t.read().strip()
        if transcript:
            with open(os.path.join(d, stem + ".txt"), "w", encoding="utf-8") as t:
                t.write(transcript)

        # keys are configured on the Setting Panel now; this form only carries
        # per-analysis options, so fall back to whatever's already saved there
        keys = _persist_keys_from_form(request.form)

        do_azure = request.form.get("do_azure") == "on"
        do_llm = bool(keys["anthropic_key"] or keys["deepseek_key"])
        strictness = request.form.get("strictness", "strict")

        if do_azure and not transcript:
            return redirect("/?p=newrec&msg=Azure+scoring+needs+the+transcript+you+read.")
        if not do_azure and not do_llm:
            return redirect("/?p=newrec&msg=Nothing+to+run:+tick+Azure,+or+add+a+"
                            "DeepSeek+or+Anthropic+key.")

        # kick off the slow work in the background and show a live progress page
        job = uuid.uuid4().hex[:8]
        _now = time.time()
        JOBS[job] = {"status": "Starting…", "t": _now, "t0": _now, "done": False,
                     "error": None, "log": [{"s": 0.0, "m": "Starting…"}]}
        threading.Thread(
            target=_run_job,
            args=(job, audio_path, d, stem, transcript, do_azure, do_llm, strictness),
            daemon=True,
        ).start()
        return _waiting_page(job)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return _error_page("Analysis failed", tb)


def main():
    port = int(os.environ.get("PORT", "8000"))
    url = "http://localhost:%d" % port
    print("English Coach is running at %s  (Ctrl+C to stop)" % url)
    if not os.environ.get("EC_NO_BROWSER"):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    # threaded=True so live-transcribe WebSockets and normal requests coexist
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
