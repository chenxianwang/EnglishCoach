# English Listening & Speaking Coach — App Specification

A personal app that ingests your recordings, diagnoses listening and speaking
errors, tracks recurring blindspots over time, and prescribes targeted practice.

---

## 1. Product goal

Turn each recording into a concrete answer to four questions:

1. **What did I get wrong?** (pronunciation, grammar, word choice, fluency)
2. **What might I have misunderstood?** (listening)
3. **What's the better / more natural way to say or hear it?**
4. **What should I practice next?**

The differentiator over generic apps (Duolingo, Elsa) is the **personal
blindspot map**: it learns *your* recurring error patterns and adapts.

---

## 2. Core modules

### Module A — Capture & Transcribe
- Upload audio/video, or record in-app.
- Speech-to-text with **word-level timestamps** and **per-word confidence**.
- Store both the raw audio and the transcript, time-aligned, so feedback can
  link back to the exact moment ("0:42 — you said *X*").

### Module B — Speaking Diagnosis
For each recording, produce a scored report across five dimensions:

| Dimension | What it checks | Signal source |
|---|---|---|
| Pronunciation | Mispronounced phonemes, word/sentence stress, intonation | Pronunciation-assessment API (phoneme scores) |
| Grammar | Tense, articles, agreement, prepositions | LLM grammar pass |
| Word choice | Wrong/awkward vocabulary, collocations | LLM + collocation check |
| Fluency | Filler words, pauses, false starts, pace (words/min) | Timestamps + pause analysis |
| Naturalness | Correct-but-non-native phrasing | LLM rewrite to native-like form |

Each flagged item shows: *what you said → why it's off → the better version*.

### Module C — Listening Diagnosis
- App plays a target clip (with a known correct transcript).
- You type or speak what you understood.
- The app **diffs** your version against the truth and explains *why* you
  missed each word: linked sounds, reduced forms (*gonna / wanna / kinda*),
  weak forms, unknown vocab, or homophone confusion.

### Module D — Blindspot Tracker (the core differentiator)
- Every error is logged with a **category tag** (e.g. `grammar/article`,
  `pron/θ→s`, `listening/reduced-form`).
- Dashboards show recurring patterns and trends over time
  ("articles dropped: 18 → 9 → 4 over 3 weeks").
- Flags your **top 3 blindspots** at any moment.

### Module E — Personalized Practice
- Each session ends with 2–3 drills generated from *your* top blindspots,
  not a generic curriculum.
- Drill types: minimal-pair pronunciation, shadowing the corrected sentence,
  cloze listening on reduced forms, targeted grammar rewrites.

### Module F — Progress & Review
- Streaks, scores per dimension over time, a searchable archive of every
  recording + report, and spaced-repetition resurfacing of past errors.

---

## 3. The feedback report (what one analysis returns)

```
SESSION: Self-introduction · 3:20 · 24 Jun
Overall: B1+ → working toward B2

PRONUNCIATION  (3 issues)
  0:12  "th-ink"      /θ/ pronounced as /s/  →  drill: think/sink, three/free
  0:45  "COMfortable" stress on wrong syllable → COMF-ter-bl (3 syllables)

GRAMMAR  (2 issues)
  0:31  "I am working here since 2020"
        → "I have been working here since 2020"  (present perfect for duration)

WORD CHOICE / NATURALNESS  (2)
  1:10  "I very like my job"  →  "I really like my job" / "I love my job"

FLUENCY
  Pace 95 wpm (a bit slow) · 14 fillers ("uh","you know") · 3 long pauses

TOP BLINDSPOTS THIS WEEK
  1. Present perfect vs. present simple (since/for)
  2. /θ/ and /ð/ sounds
  3. Adverb placement ("very like" → "really like")

PRACTICE NEXT
  • Shadow 3 corrected sentences above
  • Minimal pairs: think/sink, this/dis
  • Rewrite 5 sentences using since/for + present perfect
```

---

## 4. Technical architecture

**Pipeline:** Upload → ASR → parallel analyzers → aggregator → report → tracker DB

| Layer | Recommended choice | Notes |
|---|---|---|
| Speech-to-text | OpenAI Whisper (`large-v3`) or a hosted Whisper API | Word timestamps + confidence; confidence dips often = pronunciation issues |
| Pronunciation scoring | **Azure Pronunciation Assessment** | Objective phoneme/word/stress scores — don't rely on an LLM to "guess" pronunciation from text |
| Grammar / word choice / naturalness | LLM (e.g. Claude) with a structured-output prompt | Returns JSON: errors, categories, rewrites |
| Fluency metrics | Compute from timestamps | wpm, pause count/length, filler detection |
| Listening diff | Sequence alignment + LLM explanation | Align your transcript vs. truth, label each gap |
| Storage | SQLite/Postgres for errors + object store for audio | Errors table powers the blindspot tracker |
| Frontend | Web (React) or mobile (React Native) | Audio recorder, report viewer, dashboards |

> **Key design note:** pronunciation feedback must come from an audio model
> (Azure/Whisper confidence), *not* from a text LLM. A text model can only
> judge grammar and word choice — it literally cannot hear you. This is exactly
> the limitation I hit trying to analyze your file in this sandbox.

---

## 5. Data model (blindspot tracking)

```
recordings(id, date, type, audio_url, transcript, duration, overall_level)
errors(id, recording_id, timestamp, category, dimension,
       said_text, correct_text, explanation, severity)
drills(id, error_category, prompt, type, completed_at, score)
```

The `errors.category` field is what makes the whole thing work — aggregate it
to surface patterns, trend it over time, and seed the practice generator.

---

## 6. Build plan (phased)

**Phase 1 — MVP (the loop).** Upload audio → Whisper transcript → LLM grammar/
word-choice/naturalness report. No pronunciation scoring yet. Proves the value.

**Phase 2 — Pronunciation & fluency.** Add Azure Pronunciation Assessment and
timestamp-based fluency metrics.

**Phase 3 — Blindspot tracker.** Persist errors with categories; build the
dashboard and top-3 blindspots view.

**Phase 4 — Listening module + personalized drills.** Add the listening diff
and auto-generated practice from your error history.

---

## 7. Honest scope notes

- **Pronunciation needs a real audio model.** Budget for Azure's assessment API
  (cheap per call) rather than trying to infer pronunciation from text.
- **Privacy:** voice is biometric data — store it encrypted, let yourself
  delete recordings, and keep it on your own infrastructure for a personal app.
- **Start narrow:** Phase 1 alone (upload → grammar/naturalness report) is
  genuinely useful and buildable in a weekend with an LLM API.
```
