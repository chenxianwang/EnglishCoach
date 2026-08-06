# English Coach — Project Notes (handoff / context)

A personal English-pronunciation & speaking coach app for a Mandarin-L1 speaker
(Jonah), also used as a portfolio project. It's publicly deployed now (see
**Deployment**), so treat every change as production-facing, not local-only.

If you're a new assistant session: **read the source files before editing** —
most logic lives in `english_coach.py` and `english_coach_web.py`, with lots
of JavaScript embedded inside Python raw strings. This file is a map, not a
substitute for reading the code.

## Run it

It's very likely **already running** as a background service (see
Deployment) — check before starting a second copy:

```bash
lsof -i :8000                        # if something's listening, it's already up
pgrep -fl english_coach_web
```

If it's not running:
```bash
cd ~/Desktop/"English Coach"
python english_coach_web.py          # serves http://localhost:8000
```

Python deps: `flask flask-sock faster-whisper azure-cognitiveservices-speech av
numpy sherpa-onnx zhconv cmudict werkzeug`. Config/keys are stored in
`~/.english_coach.json` (also holds the login password hash and session secret
key now — see Auth below).

**The whole app is behind a login** (added this session — it's reachable from
the open internet now, not just localhost). If you're testing and don't know
the current password, check the Setting Panel isn't your first stop — you
can't reach it unauthenticated. Ask the user, or if truly locked out, reset it
directly by editing `~/.english_coach.json`:
```python
import json
from werkzeug.security import generate_password_hash
p = "/Users/chenxianwang/.english_coach.json"
cfg = json.load(open(p)); cfg["web_password_hash"] = generate_password_hash("temp-password-here")
json.dump(cfg, open(p, "w"))
```

## Deployment

The app is publicly reachable at **`https://my-english-coach.com`**, proxied
through a Cloudflare Tunnel from this Mac — nothing runs in the cloud, your
own machine is still the server. Two independent pieces, both self-healing
launchd services:

1. **The Flask app itself** — `~/Library/LaunchAgents/com.chenxianwang.englishcoach.plist`
   (a *LaunchAgent*, no sudo needed). Runs
   `/Users/chenxianwang/miniforge3/bin/python3 english_coach_web.py` with
   `EC_NO_BROWSER=1` (so it doesn't pop a browser tab every restart). Logs to
   `debug/web.out.log` / `debug/web.err.log` in this folder.
2. **cloudflared** — `/Library/LaunchDaemons/com.cloudflare.cloudflared.plist`
   (a system *LaunchDaemon*, needs sudo to touch). Runs `cloudflared tunnel
   run` against the named tunnel `english-coach`, config at
   `/usr/local/etc/cloudflared/config.yml` (**not** `~/.cloudflared/config.yml`
   — that's a different file `cloudflared service install` doesn't read; if
   you ever reinstall the service, remember to recreate the ingress config
   there too, pointing `my-english-coach.com` → `http://localhost:8000`).

**Restarting either:** just `kill` the running PID — both are configured
`KeepAlive`, so launchd restarts them automatically within seconds. To check:
```bash
pgrep -fl english_coach_web; pgrep -fl cloudflared
curl -s -o /dev/null -w "%{http_code}\n" https://my-english-coach.com/login
```

**Gotcha already hit once:** `cloudflared service install` by default writes
a `ProgramArguments` with no subcommand at all (bare `cloudflared`, which just
prints a hint and exits) — if the tunnel ever goes fully dark after a
reinstall, check `/Library/LaunchDaemons/com.cloudflare.cloudflared.plist`
actually has `["...cloudflared", "tunnel", "run"]`, not just `["...cloudflared"]`.

## GitHub

Repo: **https://github.com/chenxianwang/EnglishCoach** — **public**, so treat
every push like it'll be read by a stranger.

This working directory (`~/Desktop/English Coach`) did **not** have a `.git`
until this session — the original push came from a since-deleted sibling copy
("English Coach github"). It's now connected properly: `origin` is the plain
HTTPS URL (no token in it, so it never lands in `.git/config`); auth for
fetch/push is passed inline per-command
(`git push https://<token>@github.com/chenxianwang/EnglishCoach.git main`) —
ask the user for a token each time you need to push, don't assume one is
lying around. Git identity for this repo is set locally (not global) to
`chenxianwang <chenxianwang@users.noreply.github.com>`, matching the existing
commit history — don't let it fall back to the real-name auto-detected one
(`git config user.name`/`user.email`, scoped to this repo only, already set).

**Before every push, actually review the diff** — this repo is public and
has real personal content nearby (recordings, transcripts, names). Known
exclusions via `.gitignore`: `VideoAudioFiles/` (covers `history.json`,
`*.result.json`, and now `progress.json` too), `debug/`, generated HTML
reports, API-key-shaped filenames. `PROJECT_NOTES.md` (this file) and
`Intro_Feedback.md` are **not gitignored but also never `git add`ed** —
deliberate, they're personally identifying; don't add them.

**Known trap:** the local `README.md` can silently drift from GitHub's if
edits ever happen from a different clone/copy again. Before editing it,
`curl -s https://raw.githubusercontent.com/chenxianwang/EnglishCoach/main/README.md`
and diff against local — don't just push local over remote blind, you can
delete already-published content that way (happened once).

## Files

- **english_coach.py** — the engine. Transcription (faster-whisper), Azure
  pronunciation scoring, LLM grammar analysis (DeepSeek default / Anthropic),
  vision-LLM vocabulary capture (Kimi default / Anthropic), a pure-NumPy
  prosody analyzer, IPA (CMUdict), and the whole self-contained **dashboard
  HTML generator** + every "skill panel" (their JS is embedded in Python
  `r"""..."""` strings) — including `SkillStore`, the shared client that
  hydrates from and persists to `/api/progress` (see Progress sync below).
- **english_coach_web.py** — Flask app. Routes: `/analyze`, `/transcribe`,
  `/practice`, `/tts`, `/ipa`, `/delete_session`, `/vocab_photo` (photo →
  vocabulary), `/api/progress` (GET/POST — server-side progress store),
  `/login` `/logout` `/change_password` (auth), `/settings` (API keys), and
  the `/v1/stream` **WebSocket** for live transcription. Builds the page via
  `ec.generate_dashboard_html(...)`. Also owns the login/session layer — every
  route is gated behind `@app.before_request` except `/login` and `/static/*`.
- **transcribe_service.py** — standalone sherpa-onnx STT microservice (HTTP
  `/v1/transcribe` + WebSocket `/v1/stream`), reusable by other projects. Its
  streaming code (`stream_session`) is also mounted on the web app so live
  transcribe works in one process.
- **Transcribe_Service_API.md** — API docs for that service.
- **PronScore reference.html**, **Pronunciation how-to.html** — printable refs.
- **Practice scripts/** — story `.txt` files (minimal-pair practice).
- **phonemes.json / mandarin_contrasts.json / sound_system.json /
  daily_phrases.json** — data for the training modules.
- **VideoAudioFiles/** — each recording + its `*.result.json`; `history.json`
  drives Summary & progress; **`progress.json`** (new) is the server-side
  store for everything that used to live only in browser localStorage
  (Vocabulary, Practice-word scores, Grammar log, Listening SRS, ear-training
  stats). All of `VideoAudioFiles/` is gitignored.
- **english_coach_gui.py** — older Tkinter desktop version (not the main path).

## Models (shared across projects, in `~/Desktop/models/`)
- `faster-whisper-medium/` — batch transcription (accurate English + punctuation).
- `sherpa-onnx-streaming-...-bilingual-zh-en/` — live streaming (paraformer;
  encoder+decoder, no joiner). Handles English AND Chinese.
- `sherpa-onnx-streaming-zipformer-en-2023-06-26/` — English streaming (transducer).
The app auto-detects models in `~/Desktop/models/` (and `<app>/models/`); override
with `SHERPA_MODEL_DIR` / `WHISPER_MODEL_DIR`. HuggingFace is blocked in China →
downloads use the mirror `HF_ENDPOINT=https://hf-mirror.com`.

## Architecture — what's new this session

**Auth.** Single password, no per-user accounts (personal tool). Hash stored
as `web_password_hash` in `~/.english_coach.json` (werkzeug scrypt hash);
session-signing `secret_key` also persisted there so restarts don't log
everyone out. First boot with no password set auto-generates a random one and
prints it once to the console/service log — deliberately *not* a public
"choose your password" form, since the first stranger to hit that on the open
internet would get to set it. Sessions last 30 days.

**Server-side progress sync** (`/api/progress`, `VideoAudioFiles/progress.json`).
Replaces what used to be pure browser localStorage — the whole reason this
existed was "my Mac and my phone showed different progress." Client side,
`SkillStore.get`/`.set` in `_SKILLS_UTIL_JS` still have to be **synchronous**
(every panel calls them mid-render like a plain object lookup), so hydration
happens once via a **blocking synchronous XHR** at page load into an
in-memory cache; `get()` reads that cache instantly, `set()` writes it
instantly and fires an async POST in the background. This is deliberate, not
an oversight — don't "fix" it into async without rewriting every panel's
render() to handle promises.

Two real bugs were found and fixed here, worth knowing before touching this
code again:
- `save_progress()` used to `open(path, 'w')` directly, which truncates
  before writing. A GET landing in that window saw a partial/empty file and
  reported "no data yet" — a client seeing that then treated itself as
  never-synced and pushed its own (smaller) local snapshot up, silently
  wiping richer data from another device. Fixed: atomic write (temp file +
  `os.replace`) plus a `threading.Lock()` around the read-modify-write in
  `/api/progress`'s POST handler. Don't remove either.
- Plain per-key last-write-wins is fine for most keys, but `ec_scores` (word
  → list of `{s, d}` score entries) needs a real union-merge, not a replace —
  two devices syncing minutes apart would otherwise have whichever posts
  second discard the other's newly-logged scores. See `_merge_progress` /
  `_is_score_history` in `english_coach_web.py`. If you ever add another
  append-only, dict-of-lists-shaped key, it gets this treatment automatically
  (the check is structural, not by key name).

**Photo → vocabulary capture** (`/vocab_photo`, Vocabulary panel's "From
photo" tab). Sends a client-downscaled JPEG (canvas resize, no server-side
image library needed) to a vision-capable LLM. DeepSeek's API is text-only,
so this always uses Kimi (if `KIMI_API_KEY` set) or falls back to
Anthropic/Claude — never DeepSeek. Model is `kimi-k3` via raw `urllib`
(stdlib, no package, same pattern as the DeepSeek grammar calls). Two Kimi
quirks that cost real debugging time:
- `kimi-k3` **cannot disable reasoning** — `reasoning_effort` is stuck at
  `max`, and the hidden `reasoning_content` shares the same `max_tokens`
  budget as the visible answer. Default is `16000`, not the ~2000 that'd be
  plenty for a non-reasoning model — don't lower it without understanding why.
- Don't send a `temperature` param to Kimi at all — reasoning models reject
  anything but their fixed value and error out (`_kimi_chat` omits it
  entirely rather than hardcoding 1, so it stays correct if the model changes).

Every photo also gets tagged with one **scenario** (`VOCAB_SCENARIOS` — a
fixed list shared between the prompt and the UI filter, so the model can't
invent one-off labels that fragment the filter buttons). Captured items are a
review queue, never auto-added — the user approves each one.

**Setting Panel** — API keys (Azure/DeepSeek/Anthropic/Kimi) and the login
password live here now, split out of the old "New Speaking Analysis" form so
a key can be saved without uploading a recording. `_persist_keys_from_form`
is the shared helper both `/settings` and `/analyze` call.

**Nav order** (top to bottom): Setting Panel · Summary & progress · New
Speaking Analysis · Practice single word · Speaking error log · Speaking
vocabulary · Listening — dictation · Listening error log · Listening
vocabulary · Reading · Vocabulary — then collapsible groups: Train (ear &
sound) · Train (words & usage) · Reference (read & understand) · Recordings.

**Word-frequency views** (Speaking vocabulary, Listening vocabulary) — a
shared component `window.VocabBars` (was a font-size "word cloud," rebuilt as
a sorted bar list with 30-row pagination, since size-as-signal was strictly
worse than just showing the number). Each also has a per-panel **skip-words**
list (🚫 a word to exclude it, e.g. a name that dominates the count) —
persisted via the same server-side progress store, key
`'skipwords:'+bodyId`.

## Key design decisions (pre-existing, still true)

- **Grammar analysis** → **DeepSeek** by default (cheaper, reachable from China).
  Model `deepseek-v4-flash`, thinking OFF, JSON mode ON. Anthropic optional.
  The prompt returns a strict-JSON schema: `level_estimate, overall_summary,
  top_fixes[], blind_spots[], grammar[], word_choice[], pronunciation_patterns[],
  polished`. DeepSeek often emits slightly invalid JSON, so `_extract_json` has a
  robust repair chain (comma fixes, inner-quote re-escaping that looks *past* a
  comma to disambiguate, `strict=False` for control chars) + a model-repairs-its-
  own-JSON fallback. Don't remove these — they were hard-won. Kimi's vision
  calls reuse the same `_extract_json` chain.
- **Pronunciation** → **Azure**. Single-word drills: `en-US`, prosody OFF, miscue
  OFF, graded on **AccuracyScore** (prosody/miscue distort one-word/homophone
  scores — e.g. tied/tide). Passages: `en-US` with prosody ON. Non-English
  transcripts are detected (`_looks_non_english`) and Azure is skipped to
  avoid all-zero scores.
- **Transcript feeds analysis** → if the user provides a transcript, grammar
  analysis uses it directly (no re-run of Whisper). Whisper only runs when the
  transcript is blank. There's a **language selector** (default English) because
  Whisper mis-detects accented English as Chinese and then translates it.
- **Prosody meter** → pure NumPy (FFT pitch tracking): pitch variation (monotone
  index), pitch range, speaking rate, pause ratio, nPVI rhythm + a pitch contour.
- **UI** → one page; panels toggled by `showPanel(id)`; sidebar sections are
  collapsible; client state goes through `window.SkillStore` — which is now
  server-backed (see Architecture above), not pure `localStorage`.

## Conventions when editing

- Embedded JS lives in Python strings — **watch quotes/apostrophes**. Prefer
  `data-*` attributes + `this` over inlining values in `onclick` (an apostrophe in
  a value once broke buttons). Use `data-x="..."` (double-quoted) with `S.esc`.
- After any change, validate: `python3 -m py_compile english_coach.py
  english_coach_web.py`; render + `HTMLParser().feed(html)`; and JS-parse each
  embedded `<script>` with `node -e "new Function(js)"` (or, safer, feed via
  stdin: `node -e "new Function(require('fs').readFileSync(0,'utf8'))"` — a
  broken version of this check once silently passed everything because
  `process.argv[1]` was undefined).
- **Never `import english_coach_web` (or run it) against the real
  `CONFIG_PATH`/`PROGRESS_PATH` while testing.** Importing the module executes
  its bootstrap immediately — it'll read/write the real
  `~/.english_coach.json` and `VideoAudioFiles/progress.json` just from being
  imported. This has actually happened: a test import once wrote a throwaway
  password into the real config, and a separate test run once seeded fake
  vocabulary into the real progress file. Always override `HOME` (subprocess
  env, before import) or reassign `web.CONFIG_PATH` / `web.PROGRESS_PATH`
  immediately after import, before calling anything — and remember
  `PROGRESS_PATH` is derived from `__file__`'s location, not `$HOME`, so a
  `HOME` override alone isn't enough to isolate it.
- Save deliverables into this folder (`~/Desktop/English Coach/`).

## Handy

- Reduce token cost: start fresh sessions periodically; avoid pasting large
  screenshots when a short description works.
- Login password, GitHub token, Cloudflare tunnel ID: none of these live in
  this file on purpose — ask the user fresh each session rather than
  expecting them here.
