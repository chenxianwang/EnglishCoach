# Sherpa-ONNX Transcribe Service — API

A small local speech-to-text service (offline, on-device, no cloud) built on
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). One service, reusable by any
project: **batch** transcription over HTTP and **real-time** transcription over
WebSocket.

> Give this file to another project's AI assistant. The service also self-describes
> at `GET /api`, so an agent can learn the interface from a single request.

---

## 1. Setup (once)

```bash
pip install sherpa-onnx flask flask-sock av numpy
```

Download a **streaming Zipformer** model. For a Mandarin + English speaker, a
bilingual zh-en streaming model is ideal; an English-only streaming model also
works. Browse the current list here and pick a **streaming** (a.k.a. "online")
transducer model:

- Model list: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html

Example (bilingual zh-en streaming Zipformer — check the page for the latest URL):

```bash
cd "~/Desktop/English Coach"
mkdir -p models
# download + extract the .tar.bz2 from the models page into models/
# then point the service at the extracted folder:
export SHERPA_MODEL_DIR="$PWD/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
```

The folder must contain a `tokens.txt` and `encoder*.onnx`, `decoder*.onnx`,
`joiner*.onnx` (the service auto-discovers them).

## 2. Run

```bash
export SHERPA_MODEL_DIR=/path/to/streaming-model      # required
export PORT=8100                                      # optional (default 8100)
python transcribe_service.py
```

Check it: `curl http://localhost:8100/health`

Environment variables:

| Var                  | Default            | Meaning                              |
| -------------------- | ------------------ | ------------------------------------ |
| `SHERPA_MODEL_DIR`   | `./models/sherpa-stream` | Folder with the streaming model |
| `PORT`               | `8100`             | HTTP/WS port                         |
| `SHERPA_NUM_THREADS` | `2`                | Decoding threads                     |

---

## 3. Endpoints

### `GET /health`
Service and model status.
```json
{ "ok": true, "model_loaded": true, "model_dir": "...", "error": null,
  "ws_streaming": true, "target_sample_rate": 16000 }
```

### `GET /api`
Returns the machine-readable spec (the source of truth). Point an AI at this.

### `POST /v1/transcribe`  — batch
Transcribe one audio/video file. Send it either as multipart field `audio` or as
the raw request body. Any format PyAV can read (wav, mp3, m4a, webm, mp4, …); the
service resamples to 16 kHz mono automatically.

```bash
curl -F audio=@clip.wav http://localhost:8100/v1/transcribe
# -> { "text": "the transcript ...", "sample_rate": 16000 }
```

Python:
```python
import requests
r = requests.post("http://localhost:8100/v1/transcribe",
                  files={"audio": open("clip.wav", "rb")})
print(r.json()["text"])
```

### `WS /v1/stream`  — real-time
Bidirectional streaming. Connect, then send audio as it arrives; receive text as
it's recognized.

- **Connect:** `ws://localhost:8100/v1/stream?sample_rate=16000`
- **Send (client → server):** binary frames of **16-bit little-endian PCM, mono**,
  at `sample_rate`. To finish, send the text message `{"type":"stop"}` or close.
- **Receive (server → client):** JSON text messages:
  - `{"type":"partial","text":"..."}` — live hypothesis, may still change
  - `{"type":"final","text":"..."}` — committed after a pause or at end of stream
  - `{"type":"error","error":"..."}`

Browser (Web Audio → 16 kHz PCM → WebSocket):
```js
const ws = new WebSocket("ws://localhost:8100/v1/stream?sample_rate=16000");
ws.binaryType = "arraybuffer";
ws.onmessage = (e) => { const m = JSON.parse(e.data); console.log(m.type, m.text); };

const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const ctx = new AudioContext();
const src = ctx.createMediaStreamSource(stream);
const node = ctx.createScriptProcessor(4096, 1, 1);
src.connect(node); node.connect(ctx.destination);
node.onaudioprocess = (ev) => {
  const f32 = ev.inputBuffer.getChannelData(0);
  const ds = downsampleTo16k(f32, ctx.sampleRate);   // your resampler
  const i16 = new Int16Array(ds.length);
  for (let i = 0; i < ds.length; i++) i16[i] = Math.max(-1, Math.min(1, ds[i])) * 0x7fff;
  if (ws.readyState === 1) ws.send(i16.buffer);
};
// to stop: node.disconnect(); stream.getTracks().forEach(t=>t.stop()); ws.send('{"type":"stop"}');
```

Python (streaming a WAV in chunks):
```python
import json, wave, websocket   # pip install websocket-client
ws = websocket.create_connection("ws://localhost:8100/v1/stream?sample_rate=16000")
w = wave.open("clip16k_mono.wav", "rb")           # 16 kHz mono s16
while (frames := w.readframes(3200)):             # ~0.2s chunks
    ws.send_binary(frames)
    ws.settimeout(0.01)
    try:
        while True: print(json.loads(ws.recv()))
    except Exception: pass
ws.send('{"type":"stop"}')
print("FINAL:", json.loads(ws.recv()))
```

---

## 4. Notes & limits

- **Streaming models** typically output **lower-case with no punctuation**. Add a
  separate punctuation model if you need it.
- `partial` results are provisional; treat `final` as the committed transcript.
- The service shares one recognizer across requests and gives each connection its
  own decoding stream (sherpa-onnx's recommended pattern) — safe for concurrent use.
- It's **offline**: no data leaves the machine, and it needs no internet at runtime.
- Before the model is installed, `/health` reports `model_loaded: false` and the
  transcribe endpoints return `503` with a helpful message — the service still runs.
