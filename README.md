# English Coach

A self-hosted English pronunciation and speaking coach, built for a specific
problem: **generic language apps score you on whether you were understood.
They don't tell you which sound you are physically making wrong, or which
mistakes you keep repeating.**

This app records you speaking, then produces one diagnosis that combines four
independent signals:

| Signal | Source | What it catches |
|---|---|---|
| Phoneme accuracy | Azure Pronunciation Assessment | the specific words and sounds a listener would misjudge |
| Grammar & word choice | LLM (DeepSeek or Claude) | tense drift, dropped articles, translated-sounding phrasing |
| Prosody | a pure-NumPy pitch/rhythm analyzer | monotone delivery, syllable-timed rhythm, pace, phrasing |
| Recognition errors | Whisper's own mistakes | mis-transcriptions treated as evidence about *your* production |

That last row is the idea the project is built around. When a speech recognizer
hears "rope" as "lobes", that is not a bug to work around — it is a measurement.
The recognizer failed in the same way a human listener would, and it tells you
exactly which sound to drill.

Everything runs locally. Transcription and prosody analysis never leave the
machine; only the optional grammar check and pronunciation scoring call an API.

![Recording report](docs/screenshot-report.png)

## Try it without installing anything

The demo renders a complete report from a built-in synthetic fixture — no
models, no API keys, no audio:

```bash
python english_coach.py --demo --out demo_report.html
open demo_report.html
```

This is the **output** side only: a static file with the report, the progress
view, and every training panel. Recording and analysis need the server, below.

## Run the real thing

```bash
pip install -r requirements.txt
python english_coach_web.py          # http://localhost:8000
```

Everything happens on one page. **➕ New analysis** opens the capture workflow:

- **Record in the browser** via `MediaRecorder`, or upload any audio/video file.
  A recording made in-page is handed to the upload field directly, so there's no
  save-then-attach step.
- **Live captions while you speak** — mic audio is downsampled to 16 kHz PCM in
  an `AudioContext` and streamed over a WebSocket to an on-device sherpa-onnx
  model, which returns partial hypotheses and commits finals at each endpoint.
  Fully offline. Captions append straight into the transcript box.
- **Or transcribe after the fact** with Whisper, choosing the model
  (base → large-v3) and pinning the language. Transcription runs on a background
  thread; the page polls for progress and shows a live status line.
- **Low-confidence words are flagged.** Any word Whisper scored below 0.55 is
  highlighted for you to check, because the transcript is what the grammar
  analysis grades — a transcription slip would otherwise be reported as your
  mistake.
- **Pick a practice story** from `Practice scripts/` to load a known reference
  text, or let the app auto-fill a transcript it already has on disk for that
  file.

Results land in the dashboard alongside every previous session, the progress
curve, and a drill list seeded from the words you actually got wrong.

Keys are entered once in the UI and saved to `~/.english_coach.json`, outside
the project directory. Nothing is committed. Environment variables
(`AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `DEEPSEEK_API_KEY`,
`ANTHROPIC_API_KEY`) override the stored values.

Both API integrations are optional and degrade independently: with no Azure key
you still get grammar analysis and prosody, and with no LLM key you still get
pronunciation scoring, prosody and fluency. A failure in one never discards the
results of the other.

### Speech models

Place models under `~/Desktop/models/` (or `<project>/models/`, or set
`WHISPER_MODEL_DIR` / `SHERPA_MODEL_DIR`) — they are auto-detected, and the
location is shared so several projects can reuse one download.

| Model | Used for |
|---|---|
| `faster-whisper-medium/` | batch transcription (accurate English + punctuation) |
| `sherpa-onnx-streaming-paraformer-bilingual-zh-en/` | live captions, handles English *and* Chinese |
| `sherpa-onnx-streaming-zipformer-en-2023-06-26/` | English-only live captions |

Behind the Great Firewall, HuggingFace downloads need a mirror — the code
defaults to `HF_ENDPOINT=https://hf-mirror.com`.

## Architecture

```
audio ─┬─> faster-whisper ──────> transcript ─┬─> LLM ──────> grammar, word choice, polished rewrite
       │                                      │
       ├─> Azure Speech ────────────────────> phoneme scores, per-word errors, prosody sub-score
       │
       └─> NumPy FFT pitch tracker ────────> pitch variation, range, rate, pause ratio, nPVI rhythm
                                                          │
                                                          v
                                    one self-contained HTML dashboard (no build step,
                                    no external assets, no framework)
```

- **`english_coach.py`** — the engine. Transcription, scoring, LLM analysis, the
  prosody analyzer, IPA lookup via CMUdict, and the dashboard generator with its
  training panels.
- **`english_coach_web.py`** — Flask app. Upload, transcribe, score, and serve
  the dashboard; long jobs run on background threads and stream progress to the
  browser by polling.
- **`transcribe_service.py`** — a standalone, reusable sherpa-onnx STT
  microservice (HTTP + WebSocket, self-describing `/api` spec). Mounted into the
  web app so live captions work in a single process, but runnable on its own.
- **`listening_import.py`** — builds the listening library: pulls authentic
  recorded speech from VOA, the Santa Barbara Corpus, Tatoeba or your own
  folder, and normalizes every source into one clip format.
- **`sound_system.json`, `phonemes.json`, `mandarin_contrasts.json`** — drill
  content for the training modules.

## Listening by dictation

Passive replay can't verify comprehension — by the fifth listen you're
recognising the audio rather than parsing the language. So the listening module
grades you: play a sentence of real recorded speech, type what you heard, and
the answer is aligned against the reference word by word. Missed words are the
measurement.

The grader is the same alignment used for pronunciation, pointed the other way:
there the recognizer is the listener and its errors reveal your production;
here you are the listener and yours reveal your perception. Failed sentences
requeue through the existing SM-2 scheduler.

Text-to-speech is deliberately not used. The whole difficulty for a Mandarin-L1
listener is connected speech — linking, weak forms, the endings that vanish at
speed — and a synthesiser articulates too cleanly to train any of it.

Material comes from four sources with different trade-offs: VOA Learning English
(public domain, graded pace), the Santa Barbara Corpus (genuine unscripted
conversation, timestamped at intonation units), Tatoeba (single sentences with
native audio), or your own folder. VOA gives whole-article audio with no
sentence timings, so the importer recovers them by running Whisper for word
timings and aligning that hypothesis against the official transcript — the
clock comes from the audio, the words from the transcript, never from Whisper's
mishearing.

## Design notes

A few decisions that are less obvious than they look:

**Single-word drills need different Azure settings than passages.** A lone word
has no rhythm or intonation, so leaving prosody assessment on sinks the score
for a perfectly good vowel. Miscue detection has to go too: for a homophone
pair like *tied* / *tide* the recognizer picks its preferred spelling and then
penalizes the other one for a mismatch that was never audible. Single words are
therefore graded on raw `AccuracyScore` with both features off.

**Azure grades generously**, so scores are re-graded through an adjustable
strictness curve that amplifies the distance below 100. Useful when the default
"you got 92" stops being informative.

**Scores are aggregated per metric, not per segment.** Azure returns results
through a callback the SDK invokes on its own thread, and it swallows any
exception raised in there — so a failure is invisible. Azure also omits prosody
on segments too short to have rhythm, returning `None` rather than a number.
Multiplying that `None` by a weight threw inside the callback, which left the
numerators updated but the shared denominator not, and dropped that segment's
words. Every phrase-level score inflated past 100 and got quietly clamped to a
perfect 100 while the per-word list said otherwise. Each metric now carries its
own weight, the callback cannot throw, and a mean above 100 is reported as a
warning rather than clamped away — an impossible value should be loud.

**The transcript is the contract.** If you supply a transcript, the grammar
analysis reads it verbatim rather than re-running Whisper — you get graded on
what you actually said, not on what a model guessed. There is also an explicit
language selector, because Whisper reliably mis-detects accented English as
Chinese and then silently *translates* it.

**LLM JSON needs defence in depth.** Long structured outputs arrive with
trailing commas, missing commas, and unescaped inner quotes. The parser walks
the string tracking string state and decides whether a quote really closes a
value by looking at what follows it — a `,` only ends the string if a real token
comes after, which correctly keeps the comma inside `he said "hi", then`.

Two cases defeat any local rule, and both showed up in production.

A quote followed by a comma and then another quote is undecidable in isolation:

```
"why": "he said "hi", "bye" loudly"      <- inner quotes
"why": "he said hi", "next_key": "…"     <- a real close
```

Left to right they are identical. What settles it is whether the *whole
document* parses, so one repair enumerates the ambiguous sites and flips them —
one at a time, then two — taking the first reading that yields valid JSON.

Worse is a value that is never closed at all:

```json
"evidence": [
  "captured" -> likely heard as 'capture',
  "missed"   -> likely heard as 'miss'
]
```

Each element opens a string, emits a quote after the word, then continues in
bare prose. No closing quote exists, so quote-pairing swallows the rest of the
document and no amount of flipping recovers it. The fix anchors on structure
instead: a value runs until a delimiter genuinely followed by the next item —
inside an object that means a `"key":`, which is a strong signal — and
everything before it, stray quotes included, is the content.

The chain therefore escalates: parse → comma repairs → quote re-pairing →
structural re-lex → ambiguity search → ask the model to repair its own output.
`test_json_repair.py` pins all of it against real malformed replies, including a
check that 500 well-formed documents pass through untouched.

**Prosody without heavyweight dependencies.** Pitch tracking is autocorrelation
via FFT over 40 ms frames, with octave-jump outliers trimmed at the 5th/95th
percentile, plus a syllable-nucleus proxy from energy-envelope peaks. That
yields a monotone index, pitch range, speaking rate, pause ratio, and an nPVI
rhythm score — the metric that distinguishes stress-timed English from
syllable-timed Mandarin. No librosa, no PyTorch.

**Mandarin-L1 specificity.** The analysis prompt names the predictable
interference patterns directly — tense/aspect drift, dropped articles,
word-final consonant deletion, initial /r/, /θ ð/ → /s z/, dark /l/,
sentences losing weight at the end — and asks for root causes rather than a
flat list of errors.

## Progress tracking

Each session appends its metrics to `history.json`, which drives an
improvement curve across pronunciation, accuracy, fluency, and prosody, plus
recurring blind spots aggregated across recordings and a per-word score history
for the drill list.

Time on task is tracked too, measured from the actual length of every analyzed
recording and bucketed by day, week, or month. Empty periods are drawn as
zero-height stubs rather than skipped, so a gap reads as a day you didn't
practise instead of silently vanishing from the axis. Practice data stays local
and is not part of this repository.

![Prosody meter](docs/screenshot-prosody.png)

## Status

A working personal tool, used daily, not a packaged product. There are no tests
(validation is a manual compile / HTML-parse / JS-parse pass), the UI is one
generated HTML file by design, and `english_coach_gui.py` is an earlier Tkinter
version kept for reference.

## License

MIT — see [LICENSE](LICENSE).
