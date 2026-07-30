#!/usr/bin/env python3
"""
Sherpa-ONNX Transcription Service
=================================

A small, self-contained speech-to-text microservice built on sherpa-onnx
(offline, on-device, no cloud). It exposes ONE clean API that any project can
call — batch transcription over HTTP and real-time streaming over WebSocket.

Why it exists
-------------
- Fully local / offline: audio never leaves the machine, works with no internet
  (great where cloud STT is blocked).
- Real-time: streaming Zipformer models transcribe as you speak, even on CPU.
- Reusable: one service, many projects. Point any app at it; the "/api" endpoint
  describes itself so an AI can learn to use it from a single request.

Quick start
-----------
    pip install sherpa-onnx flask flask-sock av numpy
    # download a streaming model (see Transcribe_Service_API.md), then:
    export SHERPA_MODEL_DIR=/path/to/sherpa-onnx-streaming-zipformer-model
    python transcribe_service.py            # serves on http://localhost:8100

Endpoints (see GET /api for the machine-readable spec)
------------------------------------------------------
  GET  /health              -> service + model status
  GET  /api                 -> self-describing JSON API spec
  POST /v1/transcribe       -> batch: upload an audio file, get the transcript
  WS   /v1/stream           -> real-time: stream PCM, receive partial/final text
"""

import io
import json
import os
import threading

try:
    from flask import Flask, request, jsonify
except ImportError:
    raise SystemExit("This needs Flask:  pip install flask flask-sock av numpy")

PORT = int(os.environ.get("PORT", "8100"))
MODEL_DIR = os.environ.get("SHERPA_MODEL_DIR", "")   # optional explicit path
_MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
NUM_THREADS = int(os.environ.get("SHERPA_NUM_THREADS", "2"))
TARGET_SR = 16000  # sherpa streaming models expect 16 kHz mono

app = Flask(__name__)

# The recognizer is created once and shared; every connection/request gets its
# own decoding stream, which is the sherpa-onnx-recommended pattern.
_RECOGNIZER = None
_LOAD_ERROR = None
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Model loading (lazy, tolerant)                                              #
# --------------------------------------------------------------------------- #
def _find(dirpath, *substrings):
    """Return the first file in dirpath whose name contains all substrings."""
    try:
        for fn in sorted(os.listdir(dirpath)):
            low = fn.lower()
            if all(s in low for s in substrings):
                return os.path.join(dirpath, fn)
    except OSError:
        return None
    return None


def _is_model_dir(d):
    """True if d looks like a streaming transducer model (has tokens + encoder)."""
    return bool(d and os.path.isdir(d) and _find(d, "tokens") and _find(d, "encoder"))


def _model_candidate_roots():
    """Folders to scan for a model, in priority order. Covers both the app's own
    ./models and the user's Desktop/models (and ~/models), so wherever you drop
    the extracted model, it's found."""
    here = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser("~")
    roots = [
        _MODELS_ROOT,                                # <app>/models
        os.path.join(here, os.pardir, "models"),     # parent (e.g. ~/Desktop/models)
        os.path.join(home, "Desktop", "models"),     # ~/Desktop/models
        os.path.join(home, "models"),                # ~/models
    ]
    out, seen = [], set()
    for r in roots:
        r = os.path.abspath(r)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _resolve_model_dir():
    """Where the model lives: SHERPA_MODEL_DIR if set, else the first usable model
    found in any candidate root — either the root itself (files placed directly in
    it) or a model subfolder inside it. No environment variable required."""
    if _is_model_dir(MODEL_DIR):
        return MODEL_DIR
    for root in _model_candidate_roots():
        if not os.path.isdir(root):
            continue
        if _is_model_dir(root):          # model files sit directly in this folder
            return root
        for name in sorted(os.listdir(root)):  # or in a subfolder
            d = os.path.join(root, name)
            if _is_model_dir(d):
                return d
    return MODEL_DIR or _MODELS_ROOT


def _load_recognizer():
    """Build the streaming recognizer from files in MODEL_DIR. Auto-discovers
    encoder/decoder/joiner/tokens so it works across model naming schemes."""
    global _RECOGNIZER, _LOAD_ERROR
    if _RECOGNIZER is not None or _LOAD_ERROR is not None:
        return _RECOGNIZER
    with _LOCK:
        if _RECOGNIZER is not None or _LOAD_ERROR is not None:
            return _RECOGNIZER
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            _LOAD_ERROR = ("sherpa-onnx is not installed. Run:  "
                           "pip install sherpa-onnx")
            return None
        mdir = _resolve_model_dir()
        if not os.path.isdir(mdir):
            _LOAD_ERROR = ("No model found. Download a streaming Zipformer model "
                           "and extract it into %s (or set SHERPA_MODEL_DIR). "
                           "See Transcribe_Service_API.md." % _MODELS_ROOT)
            return None
        tokens = _find(mdir, "tokens")
        encoder = _find(mdir, "encoder") or _find(mdir, "encoder", ".onnx")
        decoder = _find(mdir, "decoder")
        joiner = _find(mdir, "joiner")     # present -> transducer; absent -> paraformer
        if not (tokens and encoder and decoder):
            _LOAD_ERROR = ("Could not find encoder/decoder/tokens in %s . "
                           "Is this a streaming model folder?" % mdir)
            return None
        endpoint = dict(enable_endpoint_detection=True,
                        rule1_min_trailing_silence=2.4,
                        rule2_min_trailing_silence=1.2,
                        rule3_min_utterance_length=300)
        try:
            if joiner:
                # transducer model (encoder + decoder + joiner)
                _RECOGNIZER = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=tokens, encoder=encoder, decoder=decoder, joiner=joiner,
                    num_threads=NUM_THREADS, sample_rate=TARGET_SR, feature_dim=80,
                    decoding_method="greedy_search", **endpoint)
            else:
                # paraformer model (encoder + decoder, no joiner)
                try:
                    _RECOGNIZER = sherpa_onnx.OnlineRecognizer.from_paraformer(
                        tokens=tokens, encoder=encoder, decoder=decoder,
                        num_threads=NUM_THREADS, sample_rate=TARGET_SR, feature_dim=80,
                        decoding_method="greedy_search", **endpoint)
                except TypeError:
                    # older sherpa-onnx: from_paraformer without endpoint kwargs
                    _RECOGNIZER = sherpa_onnx.OnlineRecognizer.from_paraformer(
                        tokens=tokens, encoder=encoder, decoder=decoder,
                        num_threads=NUM_THREADS, sample_rate=TARGET_SR, feature_dim=80,
                        decoding_method="greedy_search")
        except Exception as e:  # pragma: no cover - depends on model files
            _LOAD_ERROR = "Failed to load model: %s" % (str(e)[:300])
            return None
    return _RECOGNIZER


# --------------------------------------------------------------------------- #
# Audio helpers                                                               #
# --------------------------------------------------------------------------- #
def _resample_to_16k(samples, sr):
    """Linear-resample a float32 mono array to 16 kHz (defensive; clients should
    already send 16 kHz)."""
    import numpy as np
    if sr == TARGET_SR or samples.size == 0:
        return samples
    n_out = int(round(samples.size * TARGET_SR / float(sr)))
    if n_out <= 1:
        return samples
    xp = np.linspace(0.0, 1.0, samples.size, dtype=np.float64)
    x = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
    return np.interp(x, xp, samples).astype("float32")


def _decode_file_to_16k(raw_bytes):
    """Decode any audio/video bytes to a 16 kHz mono float32 numpy array (PyAV)."""
    import numpy as np
    import av
    with av.open(io.BytesIO(raw_bytes)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio stream in the uploaded file")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
        chunks = []
        for frame in container.decode(stream):
            frame.pts = None
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray().reshape(-1).astype("float32") / 32768.0
                chunks.append(arr)
    if not chunks:
        return np.zeros(0, dtype="float32")
    return np.concatenate(chunks)


def _transcribe_samples(samples):
    """Run the recognizer over a whole waveform and return the final text."""
    import numpy as np
    rec = _load_recognizer()
    if rec is None:
        raise RuntimeError(_LOAD_ERROR or "recognizer unavailable")
    stream = rec.create_stream()
    stream.accept_waveform(TARGET_SR, samples)
    # tail padding so the last words flush out
    stream.accept_waveform(TARGET_SR, np.zeros(int(0.5 * TARGET_SR), dtype="float32"))
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    return (rec.get_result(stream) or "").strip()


# --------------------------------------------------------------------------- #
# HTTP API                                                                    #
# --------------------------------------------------------------------------- #
API_SPEC = {
    "service": "sherpa-onnx-transcribe",
    "version": "1.0",
    "engine": "sherpa-onnx (offline, on-device)",
    "audio": "16 kHz mono; the service resamples uploads automatically",
    "endpoints": [
        {"method": "GET", "path": "/health",
         "desc": "Service and model status.",
         "returns": {"ok": "bool", "model_loaded": "bool", "model_dir": "str",
                     "error": "str|null", "ws_streaming": "bool"}},
        {"method": "GET", "path": "/api",
         "desc": "This self-describing spec."},
        {"method": "POST", "path": "/v1/transcribe",
         "desc": "Batch transcription of one audio/video file.",
         "request": "multipart/form-data with field 'audio' (a file), OR raw "
                    "audio bytes as the request body.",
         "returns": {"text": "str", "sample_rate": 16000},
         "example_curl": "curl -F audio=@clip.wav http://localhost:%d/v1/transcribe" % PORT},
        {"method": "WEBSOCKET", "path": "/v1/stream",
         "desc": "Real-time streaming transcription.",
         "query": {"sample_rate": "int, default 16000"},
         "send": "binary frames of 16-bit little-endian PCM mono at sample_rate; "
                 "send text {\"type\":\"stop\"} or close the socket to finish.",
         "receive": "JSON text messages: {\"type\":\"partial\"|\"final\"|\"error\","
                    "\"text\":\"...\"}"},
    ],
    "notes": [
        "partial = live hypothesis (may change); final = committed after a pause "
        "or at end of stream.",
        "Streaming Zipformer models usually output lower-case, no punctuation.",
    ],
}


@app.route("/health")
def health():
    rec = _load_recognizer()
    return jsonify(ok=True, model_loaded=rec is not None, model_dir=_resolve_model_dir(),
                   error=_LOAD_ERROR, ws_streaming=_HAS_SOCK, target_sample_rate=TARGET_SR)


@app.route("/")
@app.route("/api")
def api():
    return jsonify(API_SPEC)


@app.route("/v1/transcribe", methods=["POST"])
def transcribe_endpoint():
    f = request.files.get("audio")
    raw = f.read() if (f and f.filename) else request.get_data()
    if not raw:
        return jsonify(error="no audio provided (multipart field 'audio' or raw body)"), 400
    if _load_recognizer() is None:
        return jsonify(error=_LOAD_ERROR or "model not loaded"), 503
    try:
        samples = _decode_file_to_16k(raw)
        text = _transcribe_samples(samples)
        return jsonify(text=text, sample_rate=TARGET_SR)
    except Exception as e:
        return jsonify(error=str(e)[:300]), 500


# --------------------------------------------------------------------------- #
# WebSocket streaming (optional; needs flask-sock)                            #
# --------------------------------------------------------------------------- #
def stream_session(ws, sample_rate=TARGET_SR):
    """Run one real-time streaming session over a WebSocket. Reusable: the
    standalone service and the English Coach app both call this."""
    import numpy as np
    rec = _load_recognizer()
    if rec is None:
        ws.send(json.dumps({"type": "error", "error": _LOAD_ERROR or "model not loaded"}))
        return
    s = rec.create_stream()
    last = ""
    try:
        while True:
            data = ws.receive()
            if data is None:
                break
            if isinstance(data, str):
                try:
                    if json.loads(data).get("type") == "stop":
                        break
                except ValueError:
                    pass
                continue
            pcm = np.frombuffer(data, dtype=np.int16).astype("float32") / 32768.0
            if sample_rate != TARGET_SR:
                pcm = _resample_to_16k(pcm, sample_rate)
            s.accept_waveform(TARGET_SR, pcm)
            while rec.is_ready(s):
                rec.decode_stream(s)
            text = (rec.get_result(s) or "").strip()
            if rec.is_endpoint(s):
                rec.reset(s)
                if text:
                    ws.send(json.dumps({"type": "final", "text": text}))
                last = ""
            elif text != last:
                last = text
                ws.send(json.dumps({"type": "partial", "text": text}))
        # flush whatever is left when the client stops
        s.accept_waveform(TARGET_SR, np.zeros(int(0.5 * TARGET_SR), dtype="float32"))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        ws.send(json.dumps({"type": "final", "text": (rec.get_result(s) or "").strip()}))
    except Exception as e:
        try:
            ws.send(json.dumps({"type": "error", "error": str(e)[:200]}))
        except Exception:
            pass


_HAS_SOCK = False
try:
    from flask_sock import Sock
    _sock = Sock(app)
    _HAS_SOCK = True

    @_sock.route("/v1/stream")
    def stream(ws):
        try:
            sr = int(request.args.get("sample_rate", TARGET_SR))
        except (TypeError, ValueError):
            sr = TARGET_SR
        stream_session(ws, sr)
except ImportError:
    _HAS_SOCK = False


def main():
    print("Transcribe service on http://localhost:%d  (model dir: %s)" % (PORT, MODEL_DIR))
    print("  GET /health · GET /api · POST /v1/transcribe · WS /v1/stream")
    if _load_recognizer() is None:
        print("  ⚠ model not loaded yet:", _LOAD_ERROR)
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
