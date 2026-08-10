#!/usr/bin/env python3
"""
English Listening & Speaking Coach — Phase 1 MVP
=================================================

Pipeline:  audio file  ->  Whisper transcript (+ word timestamps)
                        ->  fluency metrics (wpm, pauses, fillers)
                        ->  LLM analysis (grammar / word choice / naturalness)
                        ->  (optional) transcript-diff pronunciation check
                        ->  self-contained HTML report

Usage
-----
    # See an example report instantly (no models or API key needed):
    python english_coach.py --demo --out sample_report.html

    # Analyze a real recording:
    export ANTHROPIC_API_KEY=sk-ant-...
    python english_coach.py recording.m4a --out report.html

    # Add the pronunciation check: read a script aloud, pass the script as
    # reference. Any word the recognizer mishears is flagged as a pronunciation
    # issue.
    python english_coach.py recording.m4a --reference script.txt --out report.html

Dependencies for real analysis (not needed for --demo):
    pip install faster-whisper anthropic
    ffmpeg must be installed and on PATH.
"""

import argparse
import base64
import glob
import html
import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta


# ---------------------------------------------------------------------------
# 1. Transcription (Whisper)
# ---------------------------------------------------------------------------
def _to_simplified(text):
    """Convert Traditional Chinese to Simplified (no-op for non-Chinese / if no
    converter is installed). Works with any of these pure-Python options:
        pip install zhconv                      (recommended, tiny, no compiler)
        pip install opencc-python-reimplemented (pure-Python OpenCC)
    """
    if not text:
        return text
    # 1) zhconv — pure Python, smallest, always installs
    try:
        from zhconv import convert
        return convert(text, "zh-cn")
    except Exception:
        pass
    # 2) OpenCC (either package name / config form)
    try:
        from opencc import OpenCC
        try:
            return OpenCC("t2s").convert(text)
        except Exception:
            return OpenCC("t2s.json").convert(text)
    except Exception:
        return text


_ZH_PUNCT = "。！？，、；：…”’）】》"


def _punctuate_chinese(segs):
    """Turn Whisper's pause-separated Chinese segments into punctuated text.
    Long pause between clauses -> 。, short pause -> ，, ending -> 。
    Keeps any punctuation Whisper already produced."""
    parts = [s for s in segs if s[0]]
    out = []
    for i, (t, _start, end) in enumerate(parts):
        out.append(t)
        if t[-1] in _ZH_PUNCT:        # Whisper already punctuated this clause
            continue
        if i < len(parts) - 1:
            gap = parts[i + 1][1] - end       # silence before the next clause
            out.append("。" if gap > 0.6 else "，")
    text = "".join(out)
    if text and text[-1] not in "。！？":
        text += "。"
    return text


def _find_whisper_model(model_name):
    """Locate a locally-placed faster-whisper model folder in shared locations so
    it can be reused across projects. Returns an absolute path, or None (=> let
    faster-whisper download it). Set WHISPER_MODEL_DIR to override."""
    folder = "faster-whisper-" + model_name
    here = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser("~")
    cands = []
    override = os.environ.get("WHISPER_MODEL_DIR")
    if override:
        cands += [override, os.path.join(override, folder)]
    for root in (os.path.join(here, "models"),
                 os.path.join(here, os.pardir, "models"),   # e.g. ~/Desktop/models
                 os.path.join(home, "Desktop", "models"),
                 os.path.join(home, "models")):
        cands.append(os.path.join(root, folder))
    for c in cands:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "model.bin")):
            return os.path.abspath(c)
    return None


def transcribe(audio_path, language=None, model_name="base", simplified=True,
               progress=lambda m: None):
    """Return (full_text, words, duration).

    Uses the multilingual Whisper model and auto-detects the spoken language
    (so Chinese, etc. transcribe correctly). Pass language='en' to force English,
    or any Whisper code ('zh', 'es', …) to pin a language.

    progress(msg) is called with human-readable status strings (model detection,
    download, transcription progress) so a UI can show a live heartbeat.
    """
    # huggingface.co is often blocked in mainland China — default to the official
    # mirror so the one-time model download works. (Override by setting HF_ENDPOINT.)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from faster_whisper import WhisperModel  # lazy import
    # prefer a locally-placed model folder if present, searching common shared
    # locations so several projects can reuse ONE downloaded model:
    #   $WHISPER_MODEL_DIR, <project>/models, ~/Desktop/models, ~/models
    #   each holding faster-whisper-<name>/ (config.json, model.bin, tokenizer.json, …)
    local = _find_whisper_model(model_name)
    if local:
        progress("✓ Local model detected (%s) — loading…" % local)
        source = local
    else:
        progress("Loading Whisper '%s' model — first run downloads it (via %s)…"
                 % (model_name, os.environ.get("HF_ENDPOINT", "huggingface.co")))
        source = model_name
    model = WhisperModel(source, device="cpu", compute_type="int8")
    progress("✓ Model ready — starting transcription…")
    segments, info = model.transcribe(
        audio_path, language=language, word_timestamps=True, beam_size=5
    )
    total = round(getattr(info, "duration", 0) or 0, 1)
    progress("Detected language: %s · audio %.0fs — transcribing…"
             % (getattr(info, "language", "?"), total))
    segs, words = [], []
    for seg in segments:
        segs.append((seg.text.strip(), seg.start, seg.end))
        for w in (seg.words or []):
            words.append({
                "w": w.word.strip(),
                "start": round(w.start, 2),
                "end": round(w.end, 2),
                "prob": round(w.probability, 3),
            })
        if total:
            progress("Transcribing… %.0f%% (%.0fs / %.0fs)"
                     % (min(100, seg.end / total * 100), seg.end, total))
        else:
            progress("Transcribing… %d segments" % len(segs))

    lang = getattr(info, "language", "")
    if lang in ("zh", "yue"):
        text = _punctuate_chinese(segs)
        if simplified:
            progress("Converting to Simplified Chinese…")
            text = _to_simplified(text)
            for wd in words:
                wd["w"] = _to_simplified(wd["w"])
    else:
        text = " ".join(t for t, _, _ in segs if t)
    progress("✓ Transcription complete")
    return text, words, round(info.duration, 1)


# ---------------------------------------------------------------------------
# 2. Fluency metrics (from word timestamps)
# ---------------------------------------------------------------------------
FILLERS = {"uh", "um", "er", "ah", "like", "you know", "kind of", "sort of"}


def fluency_metrics(words, duration):
    if not words:
        return {"wpm": 0, "fillers": 0, "long_pauses": 0, "notes": "No words."}
    n = len(words)
    minutes = max(duration / 60.0, 1e-6)
    wpm = round(n / minutes)

    filler_count = sum(1 for w in words if w["w"].lower().strip(".,") in FILLERS)

    long_pauses = 0
    for prev, cur in zip(words, words[1:]):
        if cur["start"] - prev["end"] > 0.8:   # >0.8s gap
            long_pauses += 1

    notes = []
    if wpm < 110:
        notes.append("pace is on the slow side")
    elif wpm > 170:
        notes.append("pace is fast — risk of slurring")
    else:
        notes.append("pace is in a natural range")
    if filler_count >= 5:
        notes.append("noticeable filler words — replace with short pauses")
    return {
        "wpm": wpm,
        "fillers": filler_count,
        "long_pauses": long_pauses,
        "notes": "; ".join(notes),
    }


# ---------------------------------------------------------------------------
# 3. LLM analysis (grammar / word choice / naturalness / pronunciation / polish)
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = """You are an expert English speaking coach for a serious,
self-directed learner: a Mandarin (普通话) native speaker, ~10 years into a
fintech/quant career, who drills English the way a Go player drills
life-and-death. Coach him like a capable peer, not a beginner. Frame every
mistake as data — a "game record" to review — and prefer ROOT-CAUSE thinking:
if several errors share one underlying cause, say so and group them.

Analyze this spoken-English transcript from a recording. It may contain
speech-to-text errors; a mis-transcribed word is a SIGNAL about his production —
treat it as a likely PRONUNCIATION problem and name the sound (e.g. "math" heard
as "mass" = /θ/→/s/; "rope" heard as "lobes" = weak word-initial /r/).

Apply the Mandarin-L1 lens — his errors are largely predictable, so name the root
cause when relevant:
- Tense/aspect drift (Mandarin doesn't inflect verbs for tense) — the #1
  grammar-in-motion issue under real-time speaking pressure.
- Dropped articles (a/the have no direct Mandarin equivalent).
- Word-final consonants & clusters, word-initial /r/, /θ ð/→/s z/, /ɜː/ with
  r-coloring, /v/, and dark /l/.
- Stress-timing: sentences trail off or lose weight at the end.

Return ONLY valid JSON with this exact shape:
{
  "level_estimate": "e.g. B2 (reaching C1)",
  "overall_summary": "2-3 sentence summary",
  "top_fixes": [
    {"issue":"short label of the single most impactful thing to change",
     "example":"the exact thing they said",
     "better":"how to say it naturally instead",
     "why":"one line on why it matters / how it lands on a listener"}
  ],
  "blind_spots": [
    {"pattern":"a recurring weakness that shows up more than once",
     "kind":"pronunciation | grammar | word choice | fluency",
     "evidence":["example 1","example 2"],
     "fix":"one concrete thing to practise to break the habit"}
  ],
  "pronunciation_patterns": [
    {"name":"/v/ vs /w/","desc":"...","examples":["vendors -> windows"],"drill":"minimal pairs: vine/wine"}
  ],
  "grammar": [{"said":"...","correction":"...","rule":"..."}],
  "word_choice": [{"said":"...","suggestion":"...","note":"..."}],
  "polished": "a clean, natural rewrite of the whole thing the speaker can rehearse"
}

Rules for the new fields:
- "top_fixes": the 3 highest-impact changes, ordered most important first. Prioritise
  things that hurt clarity or sound unnatural to a listener over tiny slips.
- "blind_spots": only patterns that recur (appear 2+ times or are clearly systematic
  for a Chinese-L1 speaker). If nothing recurs, return an empty list.
- Be specific and quote the learner's own words in "example" / "evidence".

CRITICAL OUTPUT RULES — the response must be strictly valid JSON:
- Do NOT put a double-quote character (") inside any string value. If you must quote
  a word or sound, use single quotes ('like this') or write it plainly.
- Escape nothing else oddly; no trailing commas; no comments; no markdown fences.
- Return ONLY the JSON object, nothing before or after it.

Transcript:
\"\"\"{transcript}\"\"\"
"""


# A prompt edited in the Setting Panel, pushed in here by the web app at startup
# and on every save. Empty means "use the built-in ANALYSIS_PROMPT above".
ANALYSIS_PROMPT_OVERRIDE = ""


def analysis_prompt():
    """The prompt actually sent for grammar analysis."""
    return (ANALYSIS_PROMPT_OVERRIDE or "").strip() or ANALYSIS_PROMPT


# Default models for grammar analysis. Override with ANTHROPIC_MODEL /
# DEEPSEEK_MODEL env vars if your account uses different ones.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def _repair_json(s):
    """Best-effort fixes for the JSON slips LLMs make on long outputs:
    trailing commas, and missing commas between adjacent values/keys."""
    r = re.sub(r",(\s*[}\]])", r"\1", s)                      # trailing commas
    r = re.sub(r'([}\]"0-9eltn])\s*\n(\s*")', r"\1,\n\2", r)  # value \n "key/elem
    r = re.sub(r'([}\]])\s*\n(\s*[{\[])', r"\1,\n\2", r)      # } \n {  (arrays/objs)
    return r


def _real_close(s, idx):
    """Does the quote at `idx` actually END a JSON string value?

    It does only if what follows is a valid JSON continuation — `}`/`]`, or a
    `,`/`:` that is itself followed by a real token (`"{[`, a number, or
    true/false/null). Otherwise it's an inner quote the model forgot to escape
    (e.g. the comma in  "he said "hi", then"  is INSIDE the string).
    """
    n = len(s)
    j = idx + 1
    while j < n and s[j] in " \t\r\n":
        j += 1
    if j >= n:
        return True
    c = s[j]
    if c in "}]":
        return True
    if c in ",:":
        k = j + 1
        while k < n and s[k] in " \t\r\n":
            k += 1
        if k >= n:
            return True
        nc = s[k]
        if nc in '"{[' or nc in "-0123456789":
            return True
        if s[k:k + 4] in ("true", "null") or s[k:k + 5] == "false":
            return True
        return False           # e.g. ", then …"  -> comma is inside the string
    return False


def _is_ambiguous_close(s, idx):
    """True if the quote at `idx` looks like a real close but could equally be an
    unescaped inner quote.

    The undecidable case is a quote followed by a comma and then ANOTHER quote:

        "why": "he said "hi", "bye" loudly"      <- inner quotes
        "why": "he said hi", "next_key": "…"     <- a real close

    Both look identical to a left-to-right scanner, so `_real_close` has to guess
    and guesses "close". When that guess is wrong the document fails to parse a
    few lines later. `_parse_with_quote_search` flips these one at a time until
    the whole document parses.
    """
    n = len(s)
    j = idx + 1
    while j < n and s[j] in " \t\r\n":
        j += 1
    if j >= n or s[j] != ",":
        return False
    k = j + 1
    while k < n and s[k] in " \t\r\n":
        k += 1
    return k < n and s[k] == '"'


def _scan_quotes(s, force_inner=frozenset()):
    """Re-escape stray double-quotes inside JSON string values.

    Returns (repaired_text, ambiguous_indices). Any index in `force_inner` is
    treated as an inner quote even though it looks like a real close — that is
    the knob `_parse_with_quote_search` turns.
    """
    n = len(s)
    out, in_str, i = [], False, 0
    ambiguous = []
    while i < n:
        c = s[i]
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
        else:
            if c == '\\' and i + 1 < n:
                out.append(c)
                out.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                close = _real_close(s, i)
                if close and _is_ambiguous_close(s, i):
                    ambiguous.append(i)
                    if i in force_inner:
                        close = False
                if close:
                    out.append('"')
                    in_str = False
                else:
                    out.append('\\"')     # inner quote → escape it
            else:
                out.append(c)
        i += 1
    return "".join(out), ambiguous


def _escape_inner_quotes(s):
    """Backwards-compatible wrapper: the plain left-to-right repair."""
    return _scan_quotes(s)[0]


_KEY_AHEAD = re.compile(r'"(?:[^"\\]|\\.)*"\s*:')


def _plausible_next_item(s, k, container):
    """Could position `k` start the next item of `container`?

    Used to tell a structural comma from one sitting inside a string value.
    Inside an object the next item must be a `"key":`, which is a strong signal;
    inside an array any value start is allowed, which is weaker — the remaining
    ambiguity is handled by `_parse_with_quote_search`.
    """
    if k >= len(s):
        return True
    c = s[k]
    if c in "}]":
        return True                      # trailing comma
    if container == "{":
        return bool(_KEY_AHEAD.match(s, k))
    return (c in '"{[' or c in "-0123456789"
            or s[k:k + 4] in ("true", "null") or s[k:k + 5] == "false")


def _escape_string_body(t):
    """Escape a raw string body for JSON, leaving existing valid escapes alone."""
    out, i, n = [], 0, len(t)
    while i < n:
        c = t[i]
        if c == "\\" and i + 1 < n:
            out.append(c)
            out.append(t[i + 1])
            i += 2
            continue
        if c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        elif ord(c) < 0x20:
            out.append("\\u%04x" % ord(c))
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _relex_json(s):
    """Rebuild every string token by structure rather than by quote pairing.

    Handles the failure the quote-flipping search cannot: a string value that is
    never closed at all, so it swallows the rest of the document. Real example
    from DeepSeek, inside an "evidence" array —

        "captured" -> likely heard as 'capture',
        "missed"   -> likely heard as 'miss'

    Each element opens a string, emits a quote after the word, then continues in
    bare prose with no closing quote. Pairing quotes left-to-right cannot recover
    this. Anchoring on structure can: a value runs until a delimiter that is
    genuinely followed by the next item, and everything before it — stray quotes
    included — is the content.

    Returns the rebuilt text, or None if the input doesn't look salvageable.
    """
    n = len(s)
    out, i = [], 0
    stack, expect_key = [], False
    rewrote = False
    while i < n:
        c = s[i]
        if c == '"':
            container = stack[-1] if stack else None
            j, end, term = i + 1, None, None
            while j < n:
                cj = s[j]
                if cj == "\\":
                    j += 2
                    continue
                if expect_key and cj == ":":
                    end, term = j, ":"
                    break
                if not expect_key and cj in ",}]":
                    k = j + 1
                    while k < n and s[k] in " \t\r\n":
                        k += 1
                    ok = False
                    if cj == ",":
                        ok = _plausible_next_item(s, k, container)
                    elif (cj == "}" and container == "{") or (cj == "]" and container == "["):
                        ok = k >= n or s[k] in ",}]"
                    if ok:
                        end, term = j, cj
                        break
                j += 1
            if end is None:
                end = n
            body = s[i + 1:end].rstrip()
            if body.endswith('"') and not body.endswith('\\"'):
                body = body[:-1]         # the token's own closing quote
            rebuilt = '"' + _escape_string_body(body) + '"'
            if rebuilt != s[i:end].rstrip():
                rewrote = True
            out.append(rebuilt)
            i = end
            if term == ":":
                expect_key = False
            continue
        if c == "{":
            stack.append("{")
            expect_key = True
        elif c == "[":
            stack.append("[")
            expect_key = False
        elif c in "}]":
            if stack:
                stack.pop()
            expect_key = False
        elif c == ",":
            expect_key = bool(stack) and stack[-1] == "{"
        out.append(c)
        i += 1
    return "".join(out) if rewrote else None


def _parse_with_quote_search(s, max_sites=12, max_attempts=4000):
    """Last-resort parse: brute-force the genuinely ambiguous quotes.

    Every `", "` boundary inside a malformed reply can be read two ways, and no
    local rule settles it — but the WHOLE document parsing is decisive. So try
    the default reading, then flip one ambiguous site, then two, and so on,
    returning the first variant that parses. Fewest flips wins, which keeps the
    result closest to what the model actually wrote.
    """
    import itertools
    base, sites = _scan_quotes(s)
    try:
        return json.loads(base, strict=False)
    except ValueError:
        pass
    try:
        return json.loads(_repair_json(base), strict=False)
    except ValueError:
        pass
    sites = sites[:max_sites]
    attempts = 0
    for r in range(1, len(sites) + 1):
        for combo in itertools.combinations(sites, r):
            if attempts >= max_attempts:
                raise ValueError("quote search exhausted after %d attempts" % attempts)
            attempts += 1
            cand, _ = _scan_quotes(s, frozenset(combo))
            for fixer in (lambda t: t, _repair_json):
                try:
                    return json.loads(fixer(cand), strict=False)
                except ValueError:
                    continue
    raise ValueError("no combination of quote readings produced valid JSON")


def _extract_json(text, who):
    """Pull a JSON object out of an LLM reply, tolerating code fences, prose
    wrapping, and the odd missing/trailing comma."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates.append(m.group(0))
    # try progressively more aggressive repairs
    repairs = [lambda s: s,
               _repair_json,
               lambda s: _escape_inner_quotes(_repair_json(s)),
               lambda s: _repair_json(_escape_inner_quotes(s))]
    last_err = None
    for c in candidates:
        for rep in repairs:
            try:
                # strict=False tolerates raw control chars (e.g. a literal
                # newline the model left inside a multi-line string value)
                return json.loads(rep(c), strict=False)
            except ValueError as e:
                last_err = e
        # every local rule failed — rebuild the strings from the structure,
        # which recovers values the model never closed
        try:
            relexed = _relex_json(c)
        except Exception:
            relexed = None
        if relexed:
            for rep in (lambda s: s, _repair_json):
                try:
                    return json.loads(rep(relexed), strict=False)
                except ValueError as e:
                    last_err = e
        # still no — brute-force the genuinely ambiguous quote boundaries
        for cand in ([c, relexed] if relexed else [c]):
            try:
                return _parse_with_quote_search(cand)
            except ValueError as e:
                last_err = e
    dump = _dump_failed_reply(text, who)
    raise RuntimeError(
        "%s did not return valid JSON (%s).%s First 300 chars of the reply:\n%s"
        % (who, last_err, (" Full reply saved to %s." % dump) if dump else "",
           (text[:300] if text else "(empty)")))


def _dump_failed_reply(text, who):
    """Save an unparseable LLM reply so the failure can actually be diagnosed.

    Only the first 300 characters reach the error message, which is rarely where
    the syntax error is. Returns the path written, or None."""
    if not text:
        return None
    try:
        import datetime
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
        os.makedirs(d, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        p = os.path.join(d, "llm-reply-%s-%s.txt" % (who.lower(), stamp))
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p
    except Exception:
        return None


def _active_provider():
    return (os.environ.get("LLM_PROVIDER")
            or ("deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "anthropic")).lower()


def _provider_label():
    """Human name of the grammar-analysis provider, for progress messages."""
    return "DeepSeek" if _active_provider() == "deepseek" else "Claude"


def llm_analyze(transcript, model=None, provider=None):
    """Grammar/word-choice/fluency analysis via an LLM.

    Provider is chosen by (in order): the `provider` arg, the LLM_PROVIDER env
    var, else 'deepseek' if a DEEPSEEK_API_KEY is set, otherwise 'anthropic'.
    """
    provider = (provider or _active_provider()).lower()
    if provider == "deepseek":
        return _deepseek_analyze(transcript, model)
    return _anthropic_analyze(transcript, model)


def _anthropic_analyze(transcript, model=None):
    try:
        import anthropic  # lazy import
    except ImportError:
        raise RuntimeError(
            "The Anthropic library isn't installed. Install it once with:\n"
            "    pip install anthropic")

    model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": analysis_prompt().replace("{transcript}", transcript),
            }],
        )
    except Exception as e:
        raise RuntimeError(
            "Anthropic API call failed (%s: %s). Check that your API key is valid, "
            "has credits, and that the model '%s' is available — override it with "
            "the ANTHROPIC_MODEL environment variable."
            % (type(e).__name__, str(e)[:180], model))

    text = "".join(getattr(b, "text", "") for b in (msg.content or []))
    return _extract_json(text, "Claude")


def _deepseek_chat(messages, model, max_tokens=8000):
    """One call to DeepSeek's OpenAI-compatible chat endpoint (stdlib urllib)."""
    import urllib.request
    import urllib.error

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("No DeepSeek API key set (DEEPSEEK_API_KEY).")
    base = (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        # V4 defaults to "thinking" mode, which can burn the budget and leave
        # content empty; we only want the JSON answer.
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:220]
        raise RuntimeError(
            "DeepSeek API call failed (HTTP %s): %s — check your key/credits, or "
            "override the model with the DEEPSEEK_MODEL env var (current: '%s')."
            % (e.code, detail, model))
    except Exception as e:
        raise RuntimeError("DeepSeek API call failed (%s: %s)."
                           % (type(e).__name__, str(e)[:180]))
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("DeepSeek returned an unexpected response shape: "
                           + json.dumps(payload)[:200])
    if choice.get("finish_reason") == "length":
        raise RuntimeError("DeepSeek's reply was cut off (hit the output limit). "
                           "Try a shorter transcript.")
    return text


def _deepseek_analyze(transcript, model=None):
    model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
    text = _deepseek_chat(
        [{"role": "user", "content": analysis_prompt().replace("{transcript}", transcript)}],
        model)
    try:
        return _extract_json(text, "DeepSeek")
    except RuntimeError as first:
        # Local repairs couldn't fix it (e.g. an inner quote followed by a comma
        # is genuinely ambiguous). Let the model repair its own JSON — cheap and
        # reliable, since fixing near-valid JSON is easy for it.
        fix = ("The text below is meant to be a SINGLE JSON object but has syntax "
               "errors — unescaped double-quotes inside string values, missing "
               "commas, or raw newlines inside strings. Return ONLY the corrected, "
               "strictly valid JSON with the exact same content and fields: escape "
               "any inner double-quote as \\\", add missing commas, and turn raw "
               "newlines inside strings into spaces. Add/remove nothing else.\n\n"
               + text)
        try:
            # the repair has to re-emit the whole object, so give it more room
            # than the original call — otherwise it gets cut off and we lose
            # the real error behind a misleading "reply was cut off".
            fixed = _deepseek_chat([{"role": "user", "content": fix}], model,
                                   max_tokens=12000)
            return _extract_json(fixed, "DeepSeek")
        except RuntimeError as second:
            raise RuntimeError(
                "DeepSeek returned malformed JSON and the repair attempt also "
                "failed.\n  original: %s\n  repair:   %s" % (first, second))


# ---------------------------------------------------------------------------
# 3b. Vocabulary capture from a photo (vision LLM)
# ---------------------------------------------------------------------------
# Shared with the front end (embedded verbatim into the Vocabulary panel) so
# the scenario a photo gets tagged with always matches one of the filter
# buttons — a model free-texting its own label would fragment the filter into
# one-off buckets instead of a few useful ones.
VOCAB_SCENARIOS = [
    "General", "Food & Dining", "Sports & Fitness", "Travel & Places",
    "Home & Daily Life", "Work & Study", "Shopping",
    "Nature & Outdoors", "Technology", "People & Relationships",
]

VOCAB_PHOTO_PROMPT = """You are helping a Mandarin-L1 English learner build
vocabulary from their everyday surroundings. Look at the photo and pick out
8-15 English words and short natural phrases a fluent speaker would actually
use to talk about what's pictured — concrete objects, actions, and the
collocations that go with them (e.g. not just "light" but "turn on the
light" if a light fixture is visible). Skip anything trivial (a, the, is)
and skip anything that doesn't relate to what is actually visible.

Also pick the ONE scenario/topic that best describes the whole photo, from
this exact list: %s.

Describe the photo in 3-5 sentences in English: what is in it, what is
happening, and any detail worth noticing. Write natural prose the learner can
read aloud for shadowing practice, and use some of the words you picked out
above so the description and the word list reinforce each other.

Return ONLY a JSON object with this exact shape:

{
  "scenario": "one value from the list above, verbatim",
  "description": "the 3-5 sentence description, as one plain string",
  "items": [
    {"headword": "the word or phrase",
     "definition": "a short, plain English gloss or explanation",
     "example": "one natural example sentence using it",
     "type": "single_word | collocation | idiom | phrasal_verb"}
  ]
}

CRITICAL OUTPUT RULES — the response must be strictly valid JSON:
- Do NOT put a double-quote character (") inside any string value. If you must
  quote a word, use single quotes ('like this') or write it plainly.
- No trailing commas, no comments, no markdown fences.
- Return ONLY the JSON object, nothing before or after it.
""" % ", ".join('"%s"' % s for s in VOCAB_SCENARIOS)


# Same Setting Panel override mechanism as ANALYSIS_PROMPT above.
VOCAB_PHOTO_PROMPT_OVERRIDE = ""


def vocab_photo_prompt():
    """The prompt actually sent with an uploaded photo."""
    return (VOCAB_PHOTO_PROMPT_OVERRIDE or "").strip() or VOCAB_PHOTO_PROMPT


# kimi-k3 is Moonshot's current flagship with vision built in natively (no
# separate "-vision-preview" model needed, unlike the older moonshot-v1-*
# line). Override with KIMI_VISION_MODEL if your account is provisioned
# differently — a 404 "Not found the model ... or Permission denied" from
# Kimi means either the model name or your key/region doesn't match (see
# KIMI_BASE_URL: api.moonshot.cn for the China platform vs api.moonshot.ai
# for the international one — keys are not portable between the two).
DEFAULT_KIMI_VISION_MODEL = "kimi-k3"


def _vision_provider():
    """DeepSeek's hosted chat API (used for grammar analysis) is text-only —
    it has no image input — so photo vocabulary capture never uses it. Kimi
    is preferred when a key is set (cheap, reachable from China, matching
    this app's DeepSeek-first bias); Anthropic/Claude is the fallback."""
    return "kimi" if os.environ.get("KIMI_API_KEY") else "anthropic"


def vision_vocab_from_image(image_bytes, mime_type="image/jpeg"):
    """Turn a photo into a short vocabulary list via a vision-capable LLM.

    Returns {"items": [...], "description": "..."}. Every item is stamped with
    the photo's one detected scenario (falling back to "General" if the model
    drifted off the fixed list), so the Vocabulary panel's scenario filter has
    a consistent bucket to file it under.
    """
    provider = _vision_provider()
    data = (_kimi_vision_vocab(image_bytes, mime_type) if provider == "kimi"
            else _anthropic_vision_vocab(image_bytes, mime_type))
    if not isinstance(data, dict):
        return {"items": [], "description": ""}
    scenario = data.get("scenario") if data.get("scenario") in VOCAB_SCENARIOS else "General"
    items = data.get("items") or []
    for it in items:
        if isinstance(it, dict):
            it["scenario"] = scenario
    # an edited prompt may drop the description, so never assume it's there
    desc = data.get("description")
    return {"items": items,
            "description": desc.strip() if isinstance(desc, str) else ""}


def _anthropic_vision_vocab(image_bytes, mime_type):
    try:
        import anthropic  # lazy import, mirrors _anthropic_analyze
    except ImportError:
        raise RuntimeError(
            "The Anthropic library isn't installed. Install it once with:\n"
            "    pip install anthropic")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "No Anthropic API key set. Photo vocabulary capture needs a "
            "vision-capable model — DeepSeek's API is text-only. Add a Kimi "
            "or Anthropic key in Settings.")

    model = os.environ.get("ANTHROPIC_VISION_MODEL") or DEFAULT_ANTHROPIC_MODEL
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    b64 = base64.b64encode(image_bytes).decode("ascii")
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": mime_type, "data": b64}},
                    {"type": "text", "text": vocab_photo_prompt()},
                ],
            }],
        )
    except Exception as e:
        raise RuntimeError(
            "Anthropic vision call failed (%s: %s). Check that your API key "
            "is valid, has credits, and that model '%s' is available."
            % (type(e).__name__, str(e)[:180], model))

    text = "".join(getattr(b, "text", "") for b in (msg.content or []))
    return _extract_json(text, "Claude vision")


def _kimi_chat(messages, model, max_tokens=16000):
    """One call to Moonshot's (Kimi) OpenAI-compatible chat endpoint (stdlib
    urllib, same shape as `_deepseek_chat` above).

    max_tokens defaults high because kimi-k3 can't turn reasoning off — unlike
    DeepSeek's `thinking: disabled`, Kimi's `reasoning_effort` is currently
    stuck at "max" — and its (hidden but token-costing) reasoning_content
    comes out of the same budget as the visible answer. A short JSON reply
    still needs headroom for however long the model chose to think first.
    """
    import urllib.request
    import urllib.error

    key = os.environ.get("KIMI_API_KEY")
    if not key:
        raise RuntimeError("No Kimi API key set (KIMI_API_KEY).")
    base = (os.environ.get("KIMI_BASE_URL") or "https://api.moonshot.cn/v1").rstrip("/")
    # No "temperature" here on purpose: reasoning models like kimi-k3 reject
    # anything but their fixed value (1) and error on an explicit override,
    # so omitting the field lets each model use its own default/only value.
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:220]
        raise RuntimeError(
            "Kimi API call failed (HTTP %s): %s — check your key/credits, or "
            "override the model with the KIMI_VISION_MODEL env var (current: "
            "'%s')." % (e.code, detail, model))
    except Exception as e:
        raise RuntimeError("Kimi API call failed (%s: %s)."
                           % (type(e).__name__, str(e)[:180]))
    try:
        choice = payload["choices"][0]
        text = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Kimi returned an unexpected response shape: "
                           + json.dumps(payload)[:200])
    if choice.get("finish_reason") == "length":
        raise RuntimeError("Kimi's reply was cut off (hit the output limit).")
    return text


def _kimi_vision_vocab(image_bytes, mime_type):
    model = os.environ.get("KIMI_VISION_MODEL") or DEFAULT_KIMI_VISION_MODEL
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = "data:%s;base64,%s" % (mime_type, b64)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": vocab_photo_prompt()},
        ],
    }]
    text = _kimi_chat(messages, model)
    try:
        return _extract_json(text, "Kimi")
    except RuntimeError as first:
        # same self-repair fallback as _deepseek_analyze: ask the model to fix
        # its own near-valid JSON rather than trying to out-guess it locally.
        fix = ("The text below is meant to be a SINGLE JSON object but has syntax "
               "errors — unescaped double-quotes inside string values, missing "
               "commas, or raw newlines inside strings. Return ONLY the corrected, "
               "strictly valid JSON with the exact same content and fields: escape "
               "any inner double-quote as \\\", add missing commas, and turn raw "
               "newlines inside strings into spaces. Add/remove nothing else.\n\n"
               + text)
        try:
            fixed = _kimi_chat([{"role": "user", "content": fix}], model)
            return _extract_json(fixed, "Kimi")
        except RuntimeError as second:
            raise RuntimeError(
                "Kimi returned malformed JSON and the repair attempt also "
                "failed.\n  original: %s\n  repair:   %s" % (first, second))


# ---------------------------------------------------------------------------
# 4. Transcript-diff pronunciation check (the free pronunciation signal)
# ---------------------------------------------------------------------------
def _looks_non_english(text):
    """True if the text is predominantly CJK (Chinese/Japanese) rather than
    English — used to skip English pronunciation scoring on a wrong-language
    transcript."""
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    cjk = len(re.findall(r"[一-鿿぀-ヿ가-힯]", text or ""))
    return cjk > max(3, latin)


def _norm(s):
    return re.sub(r"[^a-z0-9']+", " ", s.lower()).split()


def dictation_check(reference_text, typed_text):
    """Grade a dictation attempt: what was actually said vs what you heard.

    The same alignment idea as `pronunciation_diff`, but pointed the other way —
    there the recognizer is the listener, here you are. Returns a render-ready
    sequence of ops so the UI can show exactly which words were missed, plus a
    score.

    Comparison ignores case and punctuation (mishearing "committee" as
    "community" matters; forgetting a comma does not) but is otherwise strict:
    "didn't" and "did not" are genuinely different things to hear.
    """
    import difflib
    ref, hyp = _norm(reference_text), _norm(typed_text)
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    ops, correct = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            correct += i2 - i1
        ops.append({"op": tag, "ref": ref[i1:i2], "hyp": hyp[j1:j2]})
    total = len(ref)
    return {
        "score": round(100 * correct / total) if total else 0,
        "total": total,
        "correct": correct,
        "missed": total - correct,
        "perfect": correct == total and len(hyp) == len(ref),
        "ops": ops,
    }


def pronunciation_diff(reference_text, hypothesis_text):
    """Align the script the user MEANT to say against what the recognizer heard.
    Each mismatch is a likely pronunciation issue."""
    import difflib
    ref, hyp = _norm(reference_text), _norm(hypothesis_text)
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    flags = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        flags.append({
            "expected": " ".join(ref[i1:i2]) or "(—)",
            "heard_as": " ".join(hyp[j1:j2]) or "(dropped)",
            "type": tag,
        })
    total = max(len(ref), 1)
    correct = sum((i2 - i1) for tag, i1, i2, j1, j2 in sm.get_opcodes()
                  if tag == "equal")
    score = round(100 * correct / total)
    return {"score": score, "flags": flags}


# ---------------------------------------------------------------------------
# 4b. Azure Pronunciation Assessment (real phoneme-level scores)
# ---------------------------------------------------------------------------
def _to_wav_16k_mono(audio_path):
    """Azure wants 16 kHz mono PCM WAV.

    Convert in-process with PyAV (installed alongside faster-whisper) so no
    system ffmpeg binary is required. Falls back to an ffmpeg CLI if PyAV
    isn't available for some reason.
    """
    import tempfile
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        import av  # bundled with faster-whisper
        in_c = av.open(audio_path)
        in_s = in_c.streams.audio[0]
        out_c = av.open(out, mode="w")
        out_s = out_c.add_stream("pcm_s16le", rate=16000)
        out_s.layout = "mono"
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        try:
            for frame in in_c.decode(in_s):
                frame.pts = None
                for rf in resampler.resample(frame):
                    for pkt in out_s.encode(rf):
                        out_c.mux(pkt)
        except (av.error.EOFError, EOFError):
            # Browser MediaRecorder clips can end without a finalized cue; keep
            # whatever frames we already decoded rather than failing outright.
            pass
        for pkt in out_s.encode(None):   # flush
            out_c.mux(pkt)
        out_c.close()
        in_c.close()
        return out
    except ImportError:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", out],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return out


# The full set of Azure pronunciation-assessment error categories, with the
# badge colours used in Speech Studio. (key, label, badge-bg, badge-fg)
AZURE_ERROR_CATS = [
    ("Mispronunciation", "Mispronunciations", "#e6b800", "#08222b"),
    ("Omission", "Omissions", "#6b7280", "#ffffff"),
    ("Insertion", "Insertions", "#b91c1c", "#ffffff"),
    ("UnexpectedBreak", "Unexpected break", "#e6a3a3", "#08222b"),
    ("MissingBreak", "Missing break", "#c7ccd9", "#08222b"),
    ("Monotone", "Monotone", "#8b3fc6", "#ffffff"),
]
AZURE_ERROR_KEYS = [c[0] for c in AZURE_ERROR_CATS]

# Azure's own scores run generous. "Strictness" re-grades them with a tougher
# curve that amplifies how far each score sits below 100, and (above standard)
# flags otherwise-clean words that fall under a threshold as "Low".
#   level -> (k amplifier, low-word threshold or None)
STRICTNESS = {
    "standard": (1.0, None),
    "strict": (1.8, None),
    "very_strict": (2.6, None),
}
STRICTNESS_LABEL = {
    "standard": "Standard (Azure default)",
    "strict": "Strict",
    "very_strict": "Very strict",
}


def apply_strictness(az, level="strict"):
    """Return a re-graded copy of an Azure result dict under the given level."""
    k, thr = STRICTNESS.get(level, (1.0, None))
    if k == 1.0 and thr is None:
        return az

    def regrade(s):
        try:
            return max(0, round(100 - k * (100 - float(s))))
        except (TypeError, ValueError):
            return s

    out = dict(az)
    out["strictness"] = level
    for key in ("pron_score", "accuracy", "fluency", "completeness", "prosody"):
        if az.get(key) is not None:
            out[key] = regrade(az[key])
    counts = {kk: vv for kk, vv in az.get("error_counts", {}).items()}
    new_words, low = [], 0
    for w in az.get("words", []):
        nw = dict(w)
        nw["accuracy"] = regrade(w.get("accuracy", 0))
        if not nw.get("error") and thr is not None and nw["accuracy"] < thr:
            nw["error"] = "Low"
            low += 1
        new_words.append(nw)
    out["words"] = new_words
    if low:
        counts["Low"] = counts.get("Low", 0) + low
    out["error_counts"] = counts
    return out


# --- Azure usage metering ---------------------------------------------------
# Azure has no "remaining quota" endpoint. The Speech key can spend quota but
# cannot ask what is left: usage is only visible as an Azure Monitor metric
# (`AudioSecondsTranscribed` on the Cognitive Services account), which needs
# Azure AD credentials and the subscription/resource IDs — none of which this
# app has or should have.
#
# But the app knows exactly what it sent, because it is the thing sending it.
# So it meters itself: every call appends the measured audio length here, and
# "remaining" is that subtracted from whatever monthly allowance you set. This
# is authoritative for this app and blind to anything else on the same key —
# which is the honest trade, and the reason the UI says "sent by this app".

USAGE_FILE = "azure_usage.json"


def _wav_seconds(path):
    """Length of a PCM wav, in seconds. 0 if it cannot be read."""
    import wave
    try:
        with wave.open(path, "rb") as w:
            rate = w.getframerate() or 0
            return round(w.getnframes() / rate, 2) if rate else 0.0
    except Exception:                                  # noqa: BLE001
        return 0.0


def usage_path(library=None):
    return os.path.join(library or library_dir(), USAGE_FILE)


def _seed_usage(library=None):
    """Reconstruct past usage from the analyses on disk, once.

    Only an approximation of history: a recording re-analysed twice is counted
    once, and single-word drills left no duration behind at all. Marked
    `seeded` so the summary can say so instead of implying it was measured.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(library or library_dir(), "**",
                                           "*.result.json"), recursive=True)):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:                              # noqa: BLE001
            continue
        if not ((d.get("azure") or {}).get("words")):
            continue                                   # Azure never ran on it
        sec = d.get("duration_sec") or 0
        if sec:
            rows.append({"t": d.get("recorded_at") or d.get("date") or "",
                         "sec": round(float(sec), 2), "kind": "analysis",
                         "seeded": True})
    return rows


def read_azure_usage(library=None):
    """The whole ledger, seeding it from disk the first time it is asked for."""
    p = usage_path(library)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except FileNotFoundError:
        rows = _seed_usage(library)
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)
        except OSError:
            pass
        return rows
    except Exception:                                  # noqa: BLE001
        pass
    return []


def log_azure_usage(seconds, kind="analysis", library=None):
    """Append one Azure call. Never raises — metering must not break scoring."""
    try:
        seconds = float(seconds or 0)
        if seconds <= 0:
            return
        rows = read_azure_usage(library)
        rows.append({"t": datetime.now().isoformat(timespec="seconds"),
                     "sec": round(seconds, 2), "kind": kind})
        p = usage_path(library)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except Exception:                                  # noqa: BLE001
        pass


def azure_usage_summary(library=None, allowance_hours=0, month=None):
    """This month's Azure audio, against whatever allowance you have.

    `allowance_hours` of 0 means "no cap set" — pay-as-you-go, or simply not
    configured. The summary then reports usage and stays quiet about what is
    left, rather than inventing a limit.
    """
    rows = read_azure_usage(library)
    month = month or date.today().strftime("%Y-%m")
    sec, by_kind, seeded = 0.0, {}, False
    for r in rows:
        if not str(r.get("t", "")).startswith(month):
            continue
        s = float(r.get("sec") or 0)
        sec += s
        k = r.get("kind") or "analysis"
        by_kind[k] = round(by_kind.get(k, 0) + s, 1)
        seeded = seeded or bool(r.get("seeded"))
    hours = sec / 3600.0
    out = {"month": month, "seconds": round(sec, 1), "hours": round(hours, 2),
           "by_kind": by_kind, "seeded": seeded, "calls":
           sum(1 for r in rows if str(r.get("t", "")).startswith(month)),
           "allowance_hours": allowance_hours or 0}
    if allowance_hours:
        out["remaining_hours"] = round(max(0.0, allowance_hours - hours), 2)
        out["pct"] = round(min(100.0, 100.0 * hours / allowance_hours), 1)
    return out


def azure_pronunciation(audio_path, reference_text, progress=lambda m: None,
                        enable_prosody=True, enable_miscue=True, locale="en-US",
                        usage_kind="analysis"):
    """Score a recording against the script the user read aloud.

    Returns a dict with overall + breakdown scores, per-word error list, and
    error counts — the same data shown in Azure's Speech Studio demo.

    `enable_prosody` should be left on for connected speech, but turned OFF for
    single-word drills: a lone word (especially an interjection like "ah") has
    no rhythm or intonation to assess, so the prosody sub-score sinks and drags
    the overall PronScore down even when the vowel itself is spot on.

    `enable_miscue` compares the recognized words to the reference and penalises
    mismatches — essential for catching skipped/inserted words in a passage, but
    wrong for a single-word drill: for a homophone (tied vs tide, dyed vs died)
    the recogniser picks its favourite spelling, so the other spelling gets
    docked for a "miscue" even though the sound is identical. Turn it OFF for
    single words so grading is purely phoneme accuracy.

    `locale` is the accent you're graded against. Single-word drills use "en-GB"
    (British target, matching this app's IPA). Passages default to "en-US"
    because prosody / syllable / spoken-phoneme / IPA-name features are en-US
    only — so keeping stories on en-US preserves rhythm & monotone feedback.

    Requires env vars AZURE_SPEECH_KEY and AZURE_SPEECH_REGION, plus
    `pip install azure-cognitiveservices-speech`.
    """
    import azure.cognitiveservices.speech as speechsdk

    key = os.environ["AZURE_SPEECH_KEY"]
    region = os.environ["AZURE_SPEECH_REGION"]
    wav = _to_wav_16k_mono(audio_path)
    # Metered here rather than at each call site: Azure bills the audio it is
    # sent, and this is the one place every caller — analysis, single-word
    # practice, reading, the phoneme backfill — passes through.
    log_azure_usage(_wav_seconds(wav), kind=usage_kind)

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.audio.AudioConfig(filename=wav)

    pa_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=enable_miscue,
    )
    if enable_prosody:
        try:
            pa_config.enable_prosody_assessment()
        except Exception:
            pass  # older SDKs lack prosody

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config, language=locale
    )
    pa_config.apply_to(recognizer)

    words, done = [], []
    timings = []        # (offset_ticks, duration_ticks) per word, for fluency
    prosody_errs = {}  # UnexpectedBreak / MissingBreak / Monotone, from the result JSON
    skipped = []       # segments whose scores couldn't be read, for reporting
    # Weighted accumulators for phrase-level scores (weight = #words in phrase).
    # Each metric carries its OWN weight: Azure omits prosody on some segments
    # (a short trailing phrase has no rhythm to assess), and a metric that is
    # missing for one segment must not distort the average of the others.
    acc = {k: [0.0, 0.0] for k in
           ("accuracy", "fluency", "completeness", "pron", "prosody")}

    def _num(v):
        """Azure returns None for a metric it didn't assess — not 0."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f else None          # reject NaN

    def on_recognized(evt):
        # The Speech SDK dispatches this from its own thread and swallows any
        # exception, so a failure in here is INVISIBLE and leaves the
        # accumulators half-updated. That silently inflated every phrase-level
        # score (numerator counted a segment the denominator didn't) and
        # dropped that segment's words entirely. Never let anything escape.
        try:
            res = evt.result
            if res.reason != speechsdk.ResultReason.RecognizedSpeech:
                return
            pr = speechsdk.PronunciationAssessmentResult(res)
            n = max(len(pr.words), 1)
        except Exception as e:
            skipped.append(str(e)[:120])
            return

        for key, val in (("accuracy", getattr(pr, "accuracy_score", None)),
                         ("fluency", getattr(pr, "fluency_score", None)),
                         ("completeness", getattr(pr, "completeness_score", None)),
                         ("pron", getattr(pr, "pronunciation_score", None)),
                         ("prosody", getattr(pr, "prosody_score", None))):
            v = _num(val)
            if v is None:
                continue                      # not assessed — skip this metric only
            acc[key][0] += v * n
            acc[key][1] += n

        # the raw JSON carries per-word Offset/Duration (for playback + fluency)
        json_words = []
        try:
            raw = res.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
            if raw:
                nb = json.loads(raw).get("NBest", [])
                if nb:
                    json_words = nb[0].get("Words", []) or []
        except Exception:
            json_words = []
        try:
            for i, w in enumerate(pr.words):
                start = None
                if i < len(json_words):
                    off, dur = json_words[i].get("Offset"), json_words[i].get("Duration")
                    if off is not None:
                        start = round(off / 1e7, 2)
                        if dur is not None:
                            timings.append((off, dur))
                wa = _num(getattr(w, "accuracy_score", None))
                # The phoneme sequence Azure expected for this word — i.e. what
                # it actually graded you against. Worth keeping: Azure's lexicon
                # is occasionally just wrong (it expects "tied" as /t iy d/,
                # "teed"), and without this there is no way to tell a genuine
                # mispronunciation from being marked against the wrong target.
                #
                # Each phoneme also carries its OWN accuracy score, kept here in
                # `pacc` (same length and order as `phones`). Without it the
                # error log can only say "this word scored 62" and guess which
                # sound sank it; with it, "you produced /r/ 1227 times and 123
                # of them were weak" is a count rather than an inference.
                phones, pacc = [], []
                if i < len(json_words):
                    for p in (json_words[i].get("Phonemes") or []):
                        sym = p.get("Phoneme")
                        if not sym:
                            continue
                        phones.append(sym)
                        ps = _num((p.get("PronunciationAssessment") or {})
                                  .get("AccuracyScore"))
                        # -1, not 0: "Azure did not score this" and "you scored
                        # zero" mean opposite things to the stats that read it.
                        pacc.append(-1 if ps is None else max(0, min(100, round(ps))))
                words.append({
                    "word": w.word,
                    "accuracy": 0 if wa is None else max(0, min(100, round(wa))),
                    "error": w.error_type if w.error_type != "None" else "",
                    "t": start,   # start time in seconds, for click-to-play
                    "phones": phones,
                    "pacc": pacc,
                })
        except Exception as e:
            skipped.append("words: %s" % str(e)[:100])
        # prosody break / intonation errors (also JSON-only)
        for wd in json_words:
            fb = (wd.get("PronunciationAssessment", {}) or {}).get("Feedback", {}) or {}
            pros = fb.get("Prosody", {}) or {}
            for sub in ("Break", "Intonation"):
                for et in (pros.get(sub, {}) or {}).get("ErrorTypes", []) or []:
                    if et and et != "None":
                        prosody_errs[et] = prosody_errs.get(et, 0) + 1
        progress("Scoring pronunciation (Azure)… %d words" % len(words))

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(lambda e: done.append(True))
    recognizer.canceled.connect(lambda e: done.append(True))

    import time
    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.3)
    recognizer.stop_continuous_recognition()

    # always report the full six Azure categories (0 when none)
    error_counts = {k: 0 for k in AZURE_ERROR_KEYS}
    for x in words:
        if x["error"]:
            error_counts[x["error"]] = error_counts.get(x["error"], 0) + 1
    for k, v in prosody_errs.items():
        error_counts[k] = error_counts.get(k, 0) + v

    warnings = []

    def mean(key):
        """Weighted mean of one metric over the segments that reported it."""
        total, weight = acc[key]
        if weight <= 0:
            return None                       # never assessed — say so, don't invent 0
        v = total / weight
        # A weighted mean of 0-100 values cannot exceed 100. If it does, the
        # numerator and denominator disagree about which segments were counted —
        # exactly the corruption that silently produced a straight 100/100 here
        # before. Surface it instead of hiding it behind the clamp.
        if v > 100.5:
            warnings.append("%s aggregated to %.1f (>100) — segment weights "
                            "disagree; score is unreliable" % (key, v))
        return max(0, min(100, round(v)))

    if skipped:
        warnings.append("%d recognition segment(s) could not be scored: %s"
                        % (len(skipped), "; ".join(skipped[:3])))
        progress("⚠️ %d segment(s) skipped during scoring" % len(skipped))

    out = {
        "pron_score": mean("pron"),
        "accuracy": mean("accuracy"),
        "fluency": mean("fluency"),
        "completeness": mean("completeness"),
        "prosody": mean("prosody"),
        "words": words,
        "error_counts": error_counts,
        "fluency_derived": _fluency_from_timings(timings, words),
    }
    if warnings:
        out["warnings"] = warnings
    return out


_FILLER_WORDS = {"uh", "um", "er", "erm", "ah", "hmm", "mmm", "uhh", "umm"}


def _fluency_from_timings(timings, words):
    """Derive words/min, filler count, and long-pause count from Azure word
    timings (offset/duration in 100-ns ticks)."""
    out = {"wpm": None, "fillers": None, "long_pauses": None}
    # fillers: count recognised filler tokens
    out["fillers"] = sum(
        1 for x in words
        if (x.get("word", "").lower().strip(".,!?;:")) in _FILLER_WORDS)
    if timings:
        timings = sorted(timings)
        starts = [o / 1e7 for o, _ in timings]
        ends = [(o + du) / 1e7 for o, du in timings]
        span = max(ends) - min(starts)
        if span > 0:
            out["wpm"] = round(len(timings) / span * 60)
        gaps = [starts[i] - ends[i - 1] for i in range(1, len(timings))]
        out["long_pauses"] = sum(1 for g in gaps if g > 0.7)
    return out


# ---------------------------------------------------------------------------
# 5. HTML report generator (self-contained, no external assets)
# ---------------------------------------------------------------------------
def _esc(x):
    return html.escape(str(x))


def _attr(x):
    return html.escape(str(x), quote=True)


def _rows(items, cols):
    out = []
    for it in items:
        cells = "".join("<td>%s</td>" % _esc(it.get(c, "")) for c in cols)
        out.append("<tr>%s</tr>" % cells)
    return "\n".join(out)


def _drill_words(p):
    """Pull clean, real practice words out of a pronunciation pattern's examples
    (the target word — left of '->') and its drill minimal pairs."""
    words = []
    for ex in p.get("examples", []) or []:
        left = ex.split("->")[0] if "->" in ex else ex
        words += re.findall(r"[A-Za-z']{2,}", left)
    drill = p.get("drill", "") or ""
    drill = drill.split(":", 1)[1] if ":" in drill else drill
    words += re.findall(r"[A-Za-z']{2,}", drill)
    stop = {"minimal", "pairs", "practice", "and", "the", "or", "is", "with",
            "like", "words", "word", "sound", "vs"}
    out, seen = [], set()
    for w in words:
        lw = w.strip("'").lower()
        if 2 <= len(lw) <= 16 and lw not in stop and lw not in seen:
            seen.add(lw)
            out.append(lw)
    return out[:10]


def _prosody_section(d):
    """Render the statistical Prosody meter (pitch contour + metrics + targets)."""
    pm = d.get("prosody_metrics")
    if not pm:
        return ""
    GOOD, WARN, BAD = "var(--good)", "#ffb454", "var(--bad)"

    def band(v, good, ok, higher=True):
        # good/ok are thresholds; higher=True means bigger is better
        if higher:
            return GOOD if v >= good else (WARN if v >= ok else BAD)
        return GOOD if v <= good else (WARN if v <= ok else BAD)

    pv = pm.get("pitch_var_st")
    pr = pm.get("pitch_range_st")
    rate = pm.get("speech_rate_syl_s")
    pause = pm.get("pause_ratio_pct")
    npvi = pm.get("npvi")

    cards = []

    def card(label, valstr, color, target, note):
        cards.append(
            "<div class='card' style='min-width:150px;flex:1;text-align:left;margin:0'>"
            "<div style='font-size:24px;font-weight:800;color:%s'>%s</div>"
            "<div style='font-weight:600;font-size:14px'>%s</div>"
            "<div class='hint'>%s</div><div class='hint'>%s</div></div>"
            % (color, valstr, label, target, note))

    if pv is not None:
        c = band(pv, 3.0, 2.0, higher=True)
        msg = ("Good melody" if c == GOOD else
               "A bit flat — add pitch movement" if c == WARN else
               "Monotone — widen your pitch")
        card("Pitch variation", "%.1f st" % pv, c, "aim 3–6 st", msg)
    if pr is not None:
        c = band(pr, 8.0, 5.0, higher=True)
        msg = ("Expressive range" if c == GOOD else
               "Somewhat narrow" if c == WARN else "Very narrow pitch")
        card("Pitch range", "%.1f st" % pr, c, "aim 8–16 st", msg)
    if rate is not None:
        c = GOOD if 3.5 <= rate <= 5.5 else (WARN if 2.8 <= rate <= 6.5 else BAD)
        msg = ("Natural pace" if c == GOOD else
               "Slow/hesitant" if rate < 3.5 else "Fast")
        card("Speaking rate", "%.1f/s" % rate, c, "aim 3.5–5.5 syl/s", msg)
    if pause is not None:
        c = GOOD if 10 <= pause <= 35 else (WARN if pause <= 50 else BAD)
        msg = ("Well-phrased" if c == GOOD else
               "Few pauses — phrase more" if pause < 10 else "Choppy — long pauses")
        card("Pause ratio", "%d%%" % pause, c, "aim 10–35%", msg)
    if npvi is not None:
        c = band(npvi, 50, 40, higher=True)
        msg = ("English-like rhythm" if c == GOOD else
               "Developing rhythm" if c == WARN else
               "Syllable-timed — stress content words")
        card("Rhythm (nPVI)", "%d" % npvi, c, "higher = more English; aim 50+", msg)

    # pitch contour graph (gaps where unvoiced)
    pts = pm.get("contour") or []
    vals = [v for v in pts if v is not None]
    contour_html = ""
    if len(vals) >= 2:
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        W, H, nn = 560, 90, len(pts)
        dpath, pen = "", False
        for i, v in enumerate(pts):
            if v is None:
                pen = False
                continue
            xx = 8 + (W - 16) * i / max(1, nn - 1)
            yy = H - 8 - (H - 16) * (v - lo) / rng
            dpath += "%s%.1f %.1f " % ("M" if not pen else "L", xx, yy)
            pen = True
        contour_html = (
            "<div class='card'><div class='hint' style='margin-bottom:4px'>"
            "Your pitch over time — a flat line is monotone; more up-and-down is more melodic.</div>"
            "<svg width='100%%' viewBox='0 0 %d %d' preserveAspectRatio='none' style='height:96px;display:block'>"
            "<path d='%s' fill='none' stroke='var(--accent)' stroke-width='1.6' "
            "stroke-linejoin='round'/></svg>"
            "<div class='hint'>pitch range %.0f–%.0f Hz</div></div>"
            % (W, H, dpath, lo, hi))

    az = d.get("azure") or {}
    azline = ""
    if az.get("prosody") is not None:
        azline = ("<p class='hint'>Azure prosody score for this clip: "
                  "<b>%s/100</b> — the metrics below explain what drives it.</p>"
                  % az["prosody"])

    return ("<h2>Prosody meter</h2>"
            "<p class='sub'>Measured from your audio (pitch, rhythm, pace). For a "
            "Mandarin speaker the usual gap is a flat pitch and even rhythm — "
            "watch pitch variation and nPVI climb as you shadow and exaggerate stress.</p>"
            + azline + contour_html
            + "<div class='metrics' style='gap:10px'>" + "".join(cards) + "</div>")


def _report_body(d):
    """Inner HTML (header + sections) for ONE recording — embedded in the dashboard."""
    pron = ""
    for p in d.get("pronunciation_patterns", []):
        ex = ", ".join(p.get("examples", []))
        dw = _drill_words(p)
        add_btn = ""
        if dw:
            add_btn = ("<button class='btn small' data-words=\"%s\" "
                       "onclick='addDrillWords(this)' style='margin-top:6px'>"
                       "➕ Add drill words to Practice</button>"
                       % _attr(" ".join(dw)))
        pron += (
            "<div class='card'><h4>%s</h4><p>%s</p>"
            "<p class='ex'><b>Examples:</b> %s</p>"
            "<p class='drill'><b>Drill:</b> %s</p>%s</div>"
            % (_esc(p.get("name", "")), _esc(p.get("desc", "")),
               _esc(ex), _esc(p.get("drill", "")), add_btn)
        )
    # one-click: add every drill word from all patterns to Practice single word
    all_dw, _seen = [], set()
    for p in d.get("pronunciation_patterns", []):
        for w in _drill_words(p):
            if w not in _seen:
                _seen.add(w)
                all_dw.append(w)
    if all_dw:
        pron = ("<button class='btn' data-words=\"%s\" onclick='addDrillWords(this)' "
                "style='margin-bottom:12px'>➕ Add all %d mispronounced words to "
                "Practice</button>" % (_attr(" ".join(all_dw)), len(all_dw))) + pron

    pron_diff = ""
    pd = d.get("pron_diff")
    if pd:
        rows = "".join(
            "<tr><td>%s</td><td class='bad'>%s</td><td>%s</td></tr>"
            % (_esc(f["expected"]), _esc(f["heard_as"]), _esc(f["type"]))
            for f in pd["flags"]
        )
        pron_diff = (
            "<h2>Pronunciation check (script vs. what was heard)</h2>"
            "<p class='score'>Intelligibility score: <b>%s%%</b> "
            "— each mismatch is a word a listener likely misheard.</p>"
            "<table><tr><th>You meant</th><th>Heard as</th><th>Type</th></tr>"
            "%s</table>" % (pd["score"], rows)
        )

    # --- Azure real pronunciation scores (if available) ---
    azure_html = ""
    az = d.get("azure")
    if az:
        def bar(label, val):
            return ("<div class='sb'><div class='sb-h'><span>%s</span>"
                    "<b>%s / 100</b></div><div class='sb-t'>"
                    "<div class='sb-f' style='width:%s%%'></div></div></div>"
                    % (label, val, val))
        bars = (bar("Accuracy", az["accuracy"]) + bar("Fluency", az["fluency"])
                + bar("Completeness", az["completeness"])
                + bar("Prosody", az["prosody"]))
        strict_tag = ""
        if az.get("strictness") and az["strictness"] != "standard":
            strict_tag = (" <span class='score'>· %s grading</span>"
                          % _esc(STRICTNESS_LABEL.get(az["strictness"], az["strictness"])))
        warn_html = ""
        if az.get("warnings"):
            warn_html = (
                "<p class='summary' style='border-left:4px solid var(--warn)'>"
                "⚠️ <b>These scores may not be trustworthy.</b><br>%s"
                "<br><span class='hint'>Re-run the analysis; if it recurs, the "
                "per-word list below is still reliable.</span></p>"
                % "<br>".join(_esc(x) for x in az["warnings"]))
        ring_style = ("background:conic-gradient(var(--good) 0 %s%%,"
                      "var(--line) %s%% 100%%)" % (az["pron_score"], az["pron_score"]))
        azure_html = (
            "<h2>Pronunciation score (Azure)%s</h2>%s"
            "<div class='azwrap'><div class='ring' style='%s'><b>%s</b><small>overall</small></div>"
            "<div class='bars'>%s</div></div>"
            % (strict_tag, warn_html, ring_style, az["pron_score"], bars)
        )

    if not az and d.get("azure_note"):
        azure_html = ("<h2>Pronunciation score (Azure)</h2>"
                      "<p class='summary' style='border-left:4px solid #ffb454'>⚠️ %s</p>"
                      % _esc(d["azure_note"]))

    # Grammar analysis failed but the rest of the run succeeded — say so plainly
    # rather than showing empty Grammar / Word choice tables that look like a
    # clean bill of health.
    analysis_note = ""
    if d.get("analysis_error"):
        analysis_note = (
            "<p class='summary' style='border-left:4px solid var(--bad)'>"
            "⚠️ <b>Grammar analysis didn't run for this recording.</b> "
            "Pronunciation, prosody and fluency below are unaffected. "
            "Re-run the analysis to try again — the transcript is already saved."
            "<br><span class='hint' style='font-family:ui-monospace,monospace'>%s</span></p>"
            % _esc(str(d["analysis_error"])[:400]))

    fl = d.get("fluency", {}) or {}

    def _dash(v):
        return "—" if v in (None, "", "—") else v
    fluency = (
        "<div class='metrics'>"
        "<div class='m'><span>%s</span>words/min</div>"
        "<div class='m'><span>%s</span>filler words</div>"
        "<div class='m'><span>%s</span>long pauses</div>"
        "</div><p>%s</p>"
        % (_dash(fl.get("wpm")), _dash(fl.get("fillers")),
           _dash(fl.get("long_pauses")), _esc(fl.get("notes", "")))
    )

    # --- Top fixes: the prioritised, most-impactful changes ---
    top_fixes = ""
    tf = d.get("top_fixes", [])
    if tf:
        cards = ""
        for i, f in enumerate(tf, 1):
            cards += (
                "<div class='card' style='border-left:3px solid var(--warn)'>"
                "<h4>%d. %s</h4>"
                "<p class='ex'><b>You said:</b> “%s”</p>"
                "<p class='drill'><b>Better:</b> “%s”</p>"
                "<p class='hint'>%s</p></div>"
                % (i, _esc(f.get("issue", "")), _esc(f.get("example", "")),
                   _esc(f.get("better", "")), _esc(f.get("why", "")))
            )
        top_fixes = "<h2>Top fixes — start here</h2>" + cards

    # --- Blind spots: recurring / systematic weaknesses ---
    blind_spots = ""
    bs = d.get("blind_spots", [])
    if bs:
        cards = ""
        for b in bs:
            ev = "; ".join(b.get("evidence", []) or [])
            cards += (
                "<div class='card'><h4>%s <span class='hint'>· %s</span></h4>"
                "<p class='ex'><b>Shows up as:</b> %s</p>"
                "<p class='drill'><b>Fix:</b> %s</p></div>"
                % (_esc(b.get("pattern", "")), _esc(b.get("kind", "")),
                   _esc(ev), _esc(b.get("fix", "")))
            )
        blind_spots = ("<h2>Your blind spots — recurring habits</h2>"
                       "<p class='sub'>Patterns that showed up more than once — "
                       "the highest-leverage things to drill.</p>" + cards)

    grammar = _rows(d.get("grammar", []), ["said", "correction", "rule"])
    wc = _rows(d.get("word_choice", []), ["said", "suggestion", "note"])

    polished = _esc(d.get("polished", "")).replace("\n\n", "</p><p>")

    return ("<header>"
            "<h1>Speaking Diagnosis — %(title)s</h1>"
            "<div class='meta'>%(date)s &middot; %(duration)s</div>"
            "<span class='level'>Level: %(level)s</span></header>"
            "%(analysis_note)s"
            "<h2 class='plain'>Summary</h2><p class='summary'>%(summary)s</p>"
            "%(top_fixes)s%(blind_spots)s"
            "<h2>Pronunciation patterns</h2>%(pron)s%(azure)s%(pron_diff)s"
            "<h2>Grammar</h2><table><tr><th>You said</th><th>Correction</th>"
            "<th>Why</th></tr>%(grammar)s</table>"
            "<h2>Word choice &amp; naturalness</h2><table><tr><th>You said</th>"
            "<th>More natural</th><th>Note</th></tr>%(wc)s</table>"
            "<h2>Fluency</h2>%(fluency)s"
            "%(prosody)s"
            "<h2>Polished version (rehearse this)</h2>"
            "<div class='polish' data-text=\"%(polished_plain)s\"><p>%(polished)s</p></div>"
            "<div class='shadow'>"
            "<button class='btn' onclick='shadowSpeak(this)'>🔊 Listen &amp; shadow</button>"
            "<button class='btn rec' onclick='shadowRecord(this)'>● Record yourself</button>"
            "<span class='hint shadowmsg'></span>"
            "<audio class='shadowaudio' controls style='display:none;margin-top:8px'></audio>"
            "</div>") % {
        "title": _esc(d.get("title", "Recording")),
        "date": _esc(d.get("date", str(date.today()))),
        "duration": _esc(d.get("duration", "")),
        "level": _esc(d.get("level_estimate", "—")),
        "summary": _esc(d.get("overall_summary", "")
                        or ("(not available — see the note above)"
                            if d.get("analysis_error") else "")),
        "analysis_note": analysis_note,
        "top_fixes": top_fixes,
        "blind_spots": blind_spots,
        "pron": pron or "<p>No major pronunciation flags.</p>",
        "azure": azure_html,
        "pron_diff": pron_diff,
        "grammar": grammar,
        "wc": wc,
        "fluency": fluency,
        "prosody": _prosody_section(d),
        "polished": polished,
        "polished_plain": _attr(d.get("polished", "").replace("\n\n", "  ").replace("\n", " ")),
    }


# --- the page shell (CSS + sidebar + tab-switching JS), shared by all views ---
_DASHBOARD_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
  :root{--bg:#0d1620;--panel:#0a1119;--card:#172530;--ink:#eaf3f2;--mut:#8fa6ad;
        --accent:#46b3c9;--accent-ink:#08222b;--seafoam:#a8dadc;
        --good:#43c59e;--warn:#ffb454;--bad:#ff6b6b;--line:#22343f;--surface2:#1f3542;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);display:flex;
       font:16px/1.62 'IBM Plex Sans',-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  h1,h2,h3,h4,.brand,.level{font-family:'Space Grotesk','IBM Plex Sans',sans-serif;letter-spacing:-.01em}
  ::selection{background:var(--accent);color:var(--accent-ink)}
  .sidenav{width:240px;flex-shrink:0;position:sticky;top:0;align-self:flex-start;
           height:100vh;overflow:auto;padding:24px 14px;border-right:1px solid var(--line);
           background:var(--panel)}
  .brand{font-weight:800;font-size:16px;margin:0 10px 16px}
  .navsec{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
          color:var(--mut);margin:18px 10px 6px;cursor:pointer;user-select:none;
          display:flex;align-items:center;gap:6px;border-radius:6px;padding:4px 6px}
  .navsec:hover{color:var(--ink);background:var(--card)}
  .navcaret{display:inline-block;width:10px;font-size:9px;opacity:.7;flex-shrink:0}
  .sidenav a{display:block;color:var(--mut);text-decoration:none;padding:8px 10px;
             border-radius:8px;font-size:14px;border-left:3px solid transparent;cursor:pointer}
  .sidenav a small{display:block;color:var(--mut);font-size:11px;opacity:.8}
  .sidenav a:hover{background:var(--card);color:var(--ink)}
  .sidenav a.active{background:var(--card);color:var(--ink);border-left-color:var(--accent)}
  .content{flex:1;min-width:0}
  .wrap{max-width:860px;margin:0 auto;padding:32px 24px 80px}
  .tabpanel.hidden{display:none}
  header{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:8px}
  h1{margin:0 0 6px;font-size:26px}
  .meta{color:var(--mut);font-size:14px}
  .level{display:inline-block;margin-top:10px;background:var(--accent);
         color:#08222b;font-weight:700;padding:4px 12px;border-radius:999px;font-size:14px}
  h2{margin:34px 0 12px;font-size:20px;border-left:4px solid var(--accent);padding-left:10px}
  h2.plain{border:0;padding:0;margin:22px 0 8px}
  h4{margin:0 0 6px;color:var(--accent)}
  p{margin:8px 0}
  .sub{color:var(--mut);margin:0 0 18px}
  .summary{background:var(--card);padding:16px 18px;border-radius:12px;color:var(--ink);border:1px solid var(--line)}
  .card{background:var(--card);padding:14px 16px;border-radius:12px;margin:10px 0;border:1px solid var(--line)}
  .ex,.drill{font-size:14px;color:var(--mut)}
  .drill{color:var(--good)}
  table{width:100%;border-collapse:collapse;background:var(--card);
        border-radius:12px;overflow:hidden;font-size:15px;margin-top:6px}
  th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}
  th{background:#1f3542;color:var(--mut);font-weight:600}
  tr:last-child td{border-bottom:none}
  td.bad{color:var(--bad)}
  .score{color:var(--mut)}
  .metrics{display:flex;gap:12px;flex-wrap:wrap;margin:6px 0}
  .m{background:var(--card);border-radius:12px;padding:14px 18px;text-align:center;min-width:120px}
  .m span{display:block;font-size:28px;font-weight:700;color:var(--accent)}
  .polish{background:var(--card);padding:18px 20px;border-radius:12px;border-left:4px solid var(--good)}
  .azwrap{display:flex;gap:24px;align-items:center;flex-wrap:wrap;background:var(--card);padding:18px;border-radius:12px}
  .ring{position:relative;width:120px;height:120px;border-radius:50%;display:flex;
        flex-direction:column;align-items:center;justify-content:center;color:var(--ink)}
  .ring::before{content:'';position:absolute;inset:16px;border-radius:50%;
        background:var(--card);z-index:0}
  .ring b{position:relative;z-index:1;font-size:34px;font-weight:800;line-height:1}
  .ring small{position:relative;z-index:1;font-size:12px;font-weight:600;color:var(--mut);margin-top:2px}
  .bars{flex:1;min-width:260px}
  .sb{margin:8px 0}
  .sb-h{display:flex;justify-content:space-between;font-size:14px;color:var(--mut)}
  .sb-t{height:8px;background:var(--line);border-radius:6px;overflow:hidden;margin-top:3px}
  .sb-f{height:100%;background:var(--good)}
  /* .words and .wpill outlived the per-recording word list they were built
     for -- the error-stats examples and the dictation skip list use them. */
  .words{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .wpill{font-size:13px;padding:3px 8px;border-radius:6px;background:#1f3542;color:var(--mut)}
  .wpill.bad{background:#3a2029;color:var(--bad)}
  .wpill.seek{cursor:pointer}
  .wpill.seek:hover{filter:brightness(1.25);outline:1px solid var(--accent)}
  /* Error-stats rows. The bar is deliberately relative to YOUR average, not to
     a fixed 0-100 -- an 8% failure rate means nothing until you know whether
     your own baseline is 3% or 20%. The tick marks where average sits. */
  .es{display:grid;grid-template-columns:minmax(190px,1.4fr) auto minmax(110px,1fr);
      gap:10px 14px;align-items:center;padding:9px 0;border-bottom:1px solid var(--line)}
  .es:last-child{border-bottom:0}
  .es.thin{opacity:.55}
  .es-n{font-size:13px;color:var(--mut);white-space:nowrap;text-align:right}
  .es-n b{color:var(--ink);font-weight:700}
  .lb{position:relative;height:9px;background:var(--line);border-radius:6px}
  .lb i{display:block;height:100%;border-radius:6px;background:var(--good)}
  .lb.warn i{background:var(--warn)} .lb.bad i{background:var(--bad)}
  .lb u{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--mut);opacity:.65}
  .es-x{font-size:12px;color:var(--mut);margin-top:3px;display:block}
  .esnote{grid-column:1/-1;margin:-2px 0 2px;font-size:12.5px;color:var(--mut)}
  .esbadge{display:inline-block;font-size:12px;font-weight:700;padding:3px 9px;
           border-radius:999px;background:#1f3542;color:var(--mut);margin-left:8px}
  .esbadge.exact{background:rgba(67,197,158,.18);color:var(--good)}
  .hint{color:var(--mut);font-weight:400;font-size:13px}
  .btn{font-size:14px;padding:8px 14px;border-radius:9px;border:0;cursor:pointer;
       background:var(--accent);color:#08222b;font-weight:700;margin-right:8px}
  .btn.rec{background:#2c4a58;color:#fff}
  .btn.rec.on{background:var(--bad)}
  .shadow{margin:10px 0 0;display:flex;align-items:center;flex-wrap:wrap;gap:8px}
  .legend{margin-top:8px;color:var(--mut);font-size:13px}
  .lg{margin-right:14px}.lg i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:middle}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
  .chip{font-size:13px;padding:4px 10px;border-radius:999px;background:#1f3542;color:var(--mut)}
  .chip.up{color:var(--good)}.chip.down{color:var(--bad)}
  .bschip{cursor:pointer;display:inline-flex;align-items:center;gap:6px}
  .bschip:hover{outline:1px solid var(--accent)}
  /* nothing to jump to, so don't look clickable -- the ✓ inside still is */
  .bschip.nojump{cursor:default}
  .bschip.nojump:hover{outline:0}
  .bschip.mastered{opacity:.45;text-decoration:line-through}
  .masterbtn{border:0;border-radius:50%;width:18px;height:18px;line-height:1;
       cursor:pointer;background:#24404c;color:#9aa3bf;font-size:11px;padding:0}
  .bschip.mastered .masterbtn{background:var(--good);color:#08222b}
  .bsfilter{display:flex;gap:8px;margin:8px 0 14px}
  .btn.small{font-size:13px;padding:5px 12px;background:#1f3542;color:var(--mut);font-weight:600}
  .btn.small.active{background:var(--accent);color:#08222b}
  .flash{animation:flash 1.6s ease}
  @keyframes flash{0%,100%{background:transparent}25%{background:rgba(70,179,201,.25)}}
  .pwt td,.pwt th{padding:7px 12px;vertical-align:middle}
  .hwcard{background:var(--card);border-radius:16px;padding:34px 24px;min-height:250px;
    display:flex;flex-direction:column;justify-content:center;cursor:pointer;
    border-top:3px solid var(--accent);transition:transform .06s}
  .hwcard:active{transform:scale(.995)}
  .hwctl{display:flex;gap:8px;align-items:center;margin-top:14px;flex-wrap:wrap}
  .lsipa{display:none}
  #listening.ipa-on .lsipa{display:inline}
  .pwt tr:hover td{background:rgba(70,179,201,.05)}
  .drillnav,.ssbtypes{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 16px}
  .ssopts{display:flex;flex-direction:column;gap:8px;max-width:420px}
  .ssopt{text-align:left;font-size:16px;padding:12px 14px}
  .ssgrades{margin:12px 0;color:var(--mut);font-size:14px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
  .chartcard{background:var(--card);border-radius:14px;padding:18px}
  footer{margin-top:50px;color:var(--mut);font-size:13px;text-align:center}
  .navtoggle,.navbackdrop{display:none}
  @media(max-width:720px){
    .navtoggle{display:inline-flex;align-items:center;position:fixed;top:10px;left:10px;z-index:80;
      background:var(--accent);color:#08222b;border:0;border-radius:9px;padding:9px 13px;font-weight:800;
      cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.45)}
    .sidenav{display:block;position:fixed;left:0;top:0;height:100vh;width:84%;max-width:300px;
      transform:translateX(-105%);transition:transform .25s ease;z-index:90}
    .sidenav.open{transform:translateX(0)}
    .navbackdrop.show{display:block;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:85}
    .wrap{padding-top:64px}
  }
"""

_DASHBOARD_JS = """
  var links=document.querySelectorAll('.sidenav a[data-panel]');
  var panels=document.querySelectorAll('.tabpanel');
  function showPanel(id){
    for(var i=0;i<panels.length;i++){
      panels[i].classList.toggle('hidden', panels[i].id!==id);
    }
    for(var j=0;j<links.length;j++){
      links[j].classList.toggle('active', links[j].getAttribute('data-panel')===id);
    }
    window.scrollTo(0,0);
  }
  for(var k=0;k<links.length;k++){
    links[k].addEventListener('click', function(e){
      e.preventDefault(); showPanel(this.getAttribute('data-panel'));
    });
  }
  // mobile: slide-in sidebar
  function toggleNav(){ var s=document.querySelector('.sidenav'); var b=document.querySelector('.navbackdrop');
    var open=s.classList.toggle('open'); if(b) b.classList.toggle('show', open); }
  window.toggleNav=toggleNav;
  var _navlinks=document.querySelectorAll('.sidenav a');
  for(var n=0;n<_navlinks.length;n++){ _navlinks[n].addEventListener('click', function(){
    var s=document.querySelector('.sidenav'); if(s.classList.contains('open')) toggleNav(); }); }
  // shadowing: read the polished text aloud with the browser's TTS
  function shadowSpeak(btn){
    var panel = btn.closest('.tabpanel') || document;
    var box = panel.querySelector('.polish');
    var msg = panel.querySelector('.shadowmsg');
    var text = box ? box.getAttribute('data-text') : '';
    if(window.SkillStore){ window.SkillStore.speak(text, 0.95);
      if(msg) msg.textContent='Speaking — shadow the rhythm and intonation…'; return; }
    if(!('speechSynthesis' in window)){ if(msg) msg.textContent='No speech synthesis in this browser.'; return; }
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text); u.lang='en-US'; u.rate=0.95;
    if(msg) msg.textContent='Speaking — shadow the rhythm and intonation…';
    window.speechSynthesis.speak(u);
  }
  window.shadowSpeak = shadowSpeak;
  // record yourself, then play back to compare with the shadow
  var _rec = {};
  function shadowRecord(btn){
    var panel = btn.closest('.tabpanel') || document;
    var msg = panel.querySelector('.shadowmsg');
    var au = panel.querySelector('.shadowaudio');
    if(btn.classList.contains('on')){ if(_rec.mr){ _rec.mr.stop(); } return; }
    if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
      if(msg) msg.textContent='Recording not supported here.'; return; }
    navigator.mediaDevices.getUserMedia({audio:true}).then(function(stream){
      var mr = new MediaRecorder(stream); var chunks=[]; _rec.mr=mr;
      mr.ondataavailable=function(e){ chunks.push(e.data); };
      mr.onstop=function(){
        stream.getTracks().forEach(function(t){t.stop();});
        var blob=new Blob(chunks,{type:'audio/webm'});
        if(au){ au.src=URL.createObjectURL(blob); au.style.display='block'; }
        btn.classList.remove('on'); btn.textContent='● Record yourself';
        if(msg) msg.textContent='Recorded — play it back and compare.';
      };
      mr.start(); btn.classList.add('on'); btn.textContent='■ Stop';
      if(msg) msg.textContent='Recording…';
    }).catch(function(){ if(msg) msg.textContent='Microphone permission denied.'; });
  }
  window.shadowRecord = shadowRecord;
  // ---- blind spots: click to locate, mark mastered (persisted) ----
  function _mget(){ try{return JSON.parse(localStorage.getItem('ec_mastered')||'{}');}catch(_){return {};} }
  function _mset(o){ try{localStorage.setItem('ec_mastered',JSON.stringify(o));}catch(_){} }
  function applyMastered(){
    var st=_mget();
    var chips=document.querySelectorAll('.bschip');
    for(var i=0;i<chips.length;i++){
      chips[i].classList.toggle('mastered', !!st[chips[i].getAttribute('data-id')]);
    }
  }
  function toggleMaster(ev, btn){
    ev.stopPropagation();
    var chip=btn.closest('.bschip'); var id=chip.getAttribute('data-id');
    var st=_mget(); if(st[id]) delete st[id]; else st[id]=1; _mset(st);
    chip.classList.toggle('mastered');
    var act=document.querySelector('.bsfilter .active');
    if(act) bsFilter(act);
  }
  window.toggleMaster=toggleMaster;
  function bsFilter(btn){
    var f=btn.getAttribute('data-f');
    var btns=document.querySelectorAll('.bsfilter .btn');
    for(var i=0;i<btns.length;i++) btns[i].classList.remove('active');
    btn.classList.add('active');
    var chips=document.querySelectorAll('.bschip');
    for(var j=0;j<chips.length;j++){
      var m=chips[j].classList.contains('mastered');
      var show=(f==='all')||(f==='mastered'&&m)||(f==='active'&&!m);
      chips[j].style.display=show?'inline-flex':'none';
    }
  }
  window.bsFilter=bsFilter;
  function _flash(el){ el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); }
  document.addEventListener('click', function(e){
    var chip=e.target.closest ? e.target.closest('.bschip') : null;
    if(!chip || (e.target && e.target.classList.contains('masterbtn'))) return;
    if(chip.classList.contains('nojump')) return;   // no target to scroll to
    var key=(chip.getAttribute('data-key')||'').trim().toLowerCase();
    var recs=document.querySelectorAll(\"section.tabpanel[id^='rec']\");
    for(var i=0;i<recs.length;i++){
      var panel=recs[i], hit=null;
      var cells=panel.querySelectorAll('td');
      for(var c=0;c<cells.length;c++){
        if((cells[c].textContent||'').trim().toLowerCase()===key){ hit=cells[c]; break; }
      }
      if(hit){
        showPanel(panel.id);
        setTimeout(function(el){return function(){
          el.scrollIntoView({behavior:'smooth',block:'center'}); _flash(el);
        };}(hit), 60);
        return;
      }
    }
  });
  applyMastered();
  // ---- add pronunciation-drill words into Practice single word ----
  window.addDrillWords=function(btn){
    var raw=(btn.getAttribute('data-words')||'').trim();
    var S=window.SkillStore;
    if(!raw || !S){ return; }
    var words=raw.split(/\\s+/).filter(Boolean);
    var added=0;
    // update() re-reads the server first, so this can't clobber words added
    // from another tab or device since this page loaded
    S.update('pw_custom',[],function(cur){ cur=(cur||[]).slice();
      var have={}; cur.forEach(function(w){ have[(''+w).toLowerCase()]=1; });
      words.forEach(function(w){ var lw=(''+w).toLowerCase();
        if(!have[lw]){ have[lw]=1; cur.push(lw); added++; } });
      return cur; });
    // if any were previously removed/hidden, bring them back
    S.update('pw_hidden',[],function(hid){ return (hid||[]).filter(function(h){
      return words.indexOf((''+h).toLowerCase())<0; }); });
    btn.textContent = added>0 ? ('\\u2713 Added '+added+' to Practice single word')
                              : '\\u2713 Already in Practice';
    btn.disabled=true; btn.style.opacity='.7';
  };
  // ---- practice-time chart: switch between daily / weekly / monthly ----
  document.addEventListener('click', function(e){
    var b = e.target.closest ? e.target.closest('.ptimebtn') : null;
    if(!b) return;
    var want = b.getAttribute('data-range');
    var modes = ['day','week','month'];
    for(var i=0;i<modes.length;i++){
      var el = document.getElementById('ptime-'+modes[i]);
      if(el) el.style.display = (modes[i]===want) ? '' : 'none';
    }
    var btns = document.querySelectorAll('.ptimebtn');
    for(var j=0;j<btns.length;j++) btns[j].classList.toggle('active', btns[j]===b);
  });
  // ---- download a zip of the project (code and data, no media) ----
  // Scores, review schedules and error logs live server-side now (progress.json,
  // next to history.json) so they're swept into the zip automatically — no need
  // to ask the browser for its old localStorage copy.
  function S_esc(s){ return (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;'); }
  document.addEventListener('click', function(e){
    var b = e.target.closest ? e.target.closest('#backup-btn') : null;
    if(!b) return;
    var msg = document.getElementById('backup-msg');
    var label = b.textContent;
    b.disabled = true; b.textContent = 'Zipping…';
    if(msg) msg.textContent = '';
    fetch('/backup').then(function(r){
      if(!r.ok) throw new Error('server returned ' + r.status);
      var n = r.headers.get('X-Backup-Files') || '?';
      var fn = r.headers.get('X-Backup-Name') || 'english-coach-backup.zip';
      return r.blob().then(function(blob){ return {blob:blob, n:n, fn:fn}; });
    }).then(function(o){
      var url = URL.createObjectURL(o.blob);
      var a = document.createElement('a');
      a.href = url; a.download = o.fn;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      setTimeout(function(){ URL.revokeObjectURL(url); }, 4000);
      if(msg) msg.innerHTML = '✓ ' + S_esc(o.fn) + ' — ' + S_esc(o.n) + ' files, ' +
        (Math.round(o.blob.size/1024/102.4)/10) + ' MB, including your practice ' +
        'history. Check your Downloads folder.';
      b.disabled = false; b.textContent = label;
    }).catch(function(err){
      if(msg) msg.textContent = 'Backup failed: ' + err.message +
        ' (this needs the app running, not a saved copy of the page).';
      b.disabled = false; b.textContent = label;
    });
  });
  // ---- restore practice history from a backup's progress.json ----
  // Pushes straight to the server (POST /api/progress) rather than localStorage,
  // since that's the shared source of truth now. Still accepts an old-style
  // practice_data.json from a pre-migration backup — same {key: value} shape.
  document.addEventListener('click', function(e){
    var b = e.target.closest ? e.target.closest('#restore-btn') : null;
    if(!b) return;
    var f = document.getElementById('restore-file');
    if(f) f.click();
  });
  document.addEventListener('change', function(e){
    if(!e.target || e.target.id !== 'restore-file') return;
    var file = e.target.files && e.target.files[0];
    var msg = document.getElementById('restore-msg');
    if(!file) return;
    var reader = new FileReader();
    reader.onload = function(){
      var data;
      try{ data = JSON.parse(reader.result); }
      catch(err){ if(msg) msg.textContent = 'That file is not valid JSON.'; return; }
      var keys = Object.keys(data || {});
      if(!keys.length){ if(msg) msg.textContent = 'No practice data in that file.'; return; }
      // merging would interleave two histories and corrupt both, so be explicit
      if(!confirm('Replace practice data with the ' + keys.length + ' entries in this file?\\n\\n' +
                  'Current scores, review schedules and error logs on the server ' +
                  'will be overwritten. This cannot be undone.')) return;
      if(msg) msg.textContent = 'Restoring…';
      // __replace: the server normally MERGES score history (so a stale tab can
      // never delete attempts) — but a restore promises replacement above, so
      // it has to opt out of that, key by key, or the confirm text would lie.
      var body = {}; keys.forEach(function(k){ body[k] = data[k]; });
      body.__replace = keys;
      fetch('/api/progress', {method:'POST', headers:{'Content-Type':'application/json'},
                              body: JSON.stringify(body)}).then(function(r){
        if(!r.ok) throw new Error('server returned ' + r.status);
        if(msg) msg.textContent = '✓ Restored ' + keys.length + ' entries — reloading…';
        setTimeout(function(){ location.reload(); }, 900);
      }).catch(function(err){
        if(msg) msg.textContent = 'Restore failed: ' + err.message;
      });
    };
    reader.readAsText(file);
  });
  // ---- delete a session row from the Summary history ----
  window.deleteSession=function(i){
    if(!confirm('Remove this session from your progress history?\\n\\nThis deletes the row and its scores (the audio file is kept). Cannot be undone.')) return;
    fetch('/delete_session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({i:i})}).then(function(r){return r.json();}).then(function(j){
        if(j&&j.ok){ location.reload(); }
        else { alert('Delete failed: '+((j&&j.error)||'unknown')); }
      }).catch(function(){ alert('Delete only works while the app is running (not on a saved copy).'); });
  };
  // ---- collapsible sidebar sections (state saved per section) ----
  (function(){
    var secs=document.querySelectorAll('.sidenav .navsec');
    for(var i=0;i<secs.length;i++){ (function(sec){
      var label=(sec.textContent||'').trim();
      var key='navsec_'+label.replace(/[^a-z0-9]+/gi,'_').toLowerCase();
      var group=[], el=sec.nextElementSibling;
      while(el && !(el.classList && el.classList.contains('navsec'))){ group.push(el); el=el.nextElementSibling; }
      var car=document.createElement('span'); car.className='navcaret';
      sec.insertBefore(car, sec.firstChild);
      var collapsed=false; try{ collapsed=localStorage.getItem(key)==='1'; }catch(_){ }
      function apply(){ for(var g=0;g<group.length;g++){ group[g].style.display=collapsed?'none':''; }
        car.textContent=collapsed?'\\u25B8':'\\u25BE'; }
      apply();
      sec.addEventListener('click', function(){ collapsed=!collapsed;
        try{ localStorage.setItem(key, collapsed?'1':'0'); }catch(_){ } apply(); });
    })(secs[i]); }
  })();
"""


# ---------------------------------------------------------------------------
# Listening library — authentic recorded speech, normalized across sources.
#
# Every source (VOA, Santa Barbara Corpus, Tatoeba, a local folder) is imported
# into ONE shape by listening_import.py, so the panel can shuffle across all of
# them without knowing where a clip came from:
#
#   {"id": "voa-20260701-07",        # stable — SRS scheduling keys off it
#    "source": "VOA",                 # shown as attribution
#    "license": "Public domain",      # shown too; CC BY and VOA both require it
#    "source_url": "https://…",
#    "audio": "voa/2026-07-01.mp3",   # relative to listening/
#    "start": 12.4, "end": 17.9,      # optional window inside a longer file
#    "text": "The committee said it would review the decision.",
#    "accent": "US", "speaker": "…"}
# ---------------------------------------------------------------------------
LISTENING_DIR = "listening"


def listening_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), LISTENING_DIR)


def _clean_clips(clips):
    """Keep only clips that can actually be played and graded."""
    out, seen = [], set()
    for i, c in enumerate(clips or []):
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or "").strip()
        audio = (c.get("audio") or "").strip()
        if not text or not audio:
            continue
        cid = str(c.get("id") or "").strip() or ("clip%d" % i)
        if cid in seen:
            continue
        seen.add(cid)
        item = {
            "id": cid,
            "source": (c.get("source") or "Unknown").strip(),
            "license": (c.get("license") or "").strip(),
            "source_url": (c.get("source_url") or "").strip(),
            "audio": audio,
            "text": text,
            "accent": (c.get("accent") or "").strip(),
        }
        for k in ("start", "end"):
            try:
                if c.get(k) is not None:
                    item[k] = round(float(c[k]), 2)
            except (TypeError, ValueError):
                pass
        out.append(item)
    return out


def _read_listening_file():
    p = os.path.join(listening_dir(), "library.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_listening_library():
    """Read listening/library.json, or return [] when nothing is imported yet."""
    data = _read_listening_file()
    clips = data.get("clips") if isinstance(data, dict) else data
    return _clean_clips(clips)


def load_listening_meta():
    """Per-source counts written by the importer: how much material exists
    upstream, so the panel can say how much is left to fetch."""
    data = _read_listening_file()
    return data.get("sources", {}) if isinstance(data, dict) else {}


def _load_sound_data():
    """Module 1 (Sound System) seed data — connected-speech + segmental drills."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sound_system.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


_SOUND_JS = r"""
<script>
(function(){
  function DATA(){ return (window.SOUND_DATA||[]).concat(get('ss_custom',[])); }
  var TYPE_LABEL = {linking:'Linking 连读', weak_form:'Weak forms 弱读',
    elision:'Elision 省音', assimilation:'Assimilation 同化', segmental:'Segmental 音素'};
  function $(s,r){ return (r||document).querySelector(s); }
  function get(k,d){ try{return JSON.parse(localStorage.getItem(k))||d;}catch(_){return d;} }
  function set(k,v){ try{localStorage.setItem(k,JSON.stringify(v));}catch(_){} }
  function today(){ return new Date().toISOString().slice(0,10); }
  function addDays(n){ var x=new Date(); x.setDate(x.getDate()+n); return x.toISOString().slice(0,10); }
  function esc(s){ return (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
  function label(it){ return it.type==='segmental'? it.targetForm : it.written; }
  function sound(it){ return it.type==='segmental'? it.ipa : it.soundsLike; }
  function refText(it){ if(it.type==='segmental') return it.referenceText;
    var ex=(it.example||'').split('→')[0].trim(); return ex||it.written; }
  function speak(t){ if(!window.speechSynthesis) return; speechSynthesis.cancel();
    var u=new SpeechSynthesisUtterance(t); u.lang='en-US'; u.rate=0.92; speechSynthesis.speak(u); }
  // SRS (SM-2)
  function schedule(id,grade){ var srs=get('ss_srs',{});
    var it=srs[id]||{ef:2.5,interval:1,reps:0,due:today()};
    if(grade===0){ it.reps=0; it.interval=1; }
    else { it.reps+=1; it.ef=Math.max(1.3, it.ef+(0.1-(3-grade)*(0.08+(3-grade)*0.02)));
      it.interval = it.reps===1?1 : it.reps===2?3 : Math.round(it.interval*it.ef); }
    it.due=addDays(it.interval); it.last=today(); srs[id]=it; set('ss_srs',srs); }
  function record(cat,ok){ var s=get('ss_stats',{}); var c=s[cat]||{attempts:0,correct:0};
    c.attempts++; if(ok)c.correct++; s[cat]=c; set('ss_stats',s); }
  function acc(cat){ var s=get('ss_stats',{})[cat]; return (s&&s.attempts)? s.correct/s.attempts : null; }
  function dueItems(){ var srs=get('ss_srs',{}), t=today();
    return DATA().filter(function(it){ var s=srs[it.id]; return !s||s.due<=t; }); }
  function weakFirst(){ var a=dueItems().slice();
    a.sort(function(x,y){ var ax=acc(x.category), ay=acc(y.category);
      ax=(ax==null?0.5:ax); ay=(ay==null?0.5:ay); return ax-ay; }); return a; }
  function distractors(it){ var pool=DATA().filter(function(d){return d.type===it.type&&d.category===it.category&&d.id!==it.id;});
    if(pool.length<3) pool=DATA().filter(function(d){return d.type===it.type&&d.id!==it.id;});
    pool=pool.slice(); var out=[]; while(out.length<3&&pool.length){ out.push(pool.splice(Math.floor(Math.random()*pool.length),1)[0]); } return out; }
  function shuffle(a){ for(var i=a.length-1;i>0;i--){ var j=Math.floor(Math.random()*(i+1)); var t=a[i];a[i]=a[j];a[j]=t; } return a; }

  var STATE={mode:'queue', cur:null};
  function body(){ return $('#ss-body'); }

  function gradeRow(it,onGrade){
    return "<div class='ssgrades'>Grade your recall: "+
      [['Again',0,'#ff6b6b'],['Hard',1,'#ffb454'],['Good',2,'#46b3c9'],['Easy',3,'#43c59e']]
      .map(function(g){return "<button class='btn small' style='background:"+g[2]+";color:#08222b' data-g='"+g[1]+"'>"+g[0]+"</button>";}).join('')+"</div>";
  }
  function bindGrades(it, ok, after){
    var btns=body().querySelectorAll('.ssgrades .btn');
    btns.forEach(function(b){ b.addEventListener('click',function(){
      var g=parseInt(b.getAttribute('data-g'),10);
      schedule(it.id,g); record(it.category, g>=2); after&&after(); }); });
  }

  function renderRecognition(queue){
    var pool = queue? weakFirst() : shuffle(dueItems().slice());
    if(!pool.length){ body().innerHTML="<div class='card'>🎉 Nothing due right now. Come back later, or switch to Browse.</div>"; return; }
    var it=pool[0]; STATE.cur=it;
    var opts=shuffle(distractors(it).map(sound).concat([sound(it)]));
    var h="<div class='card'><div class='hint'>"+esc(TYPE_LABEL[it.type])+" · "+esc(it.categoryLabel)+"</div>"+
      "<div style='font-size:30px;font-weight:800;margin:6px 0'>"+esc(label(it))+"</div>"+
      "<button class='btn ssplay' data-say=\""+esc(label(it))+"\">🔊 Play</button>"+
      "<p class='sub' style='margin-top:14px'>How does it actually sound?</p><div class='ssopts'>"+
      opts.map(function(o){return "<button class='btn small ssopt' data-o=\""+esc(o)+"\">"+esc(o)+"</button>";}).join('')+
      "</div><div class='ssreveal'></div></div>";
    body().innerHTML=h;
    body().querySelectorAll('.ssopt').forEach(function(b){ b.addEventListener('click',function(){
      var ok = b.getAttribute('data-o')===sound(it);
      body().querySelectorAll('.ssopt').forEach(function(x){
        var corr=x.getAttribute('data-o')===sound(it);
        x.style.background=corr?'#43c59e':(x===b?'#ff6b6b':'#1f3542');
        x.style.color=(corr||x===b)?'#08222b':'#9aa3bf'; x.disabled=true; });
      record(it.category, ok); schedule(it.id, ok?2:0);
      $('.ssreveal',body()).innerHTML="<div class='card' style='margin-top:10px'>"+
        "<b>"+esc(label(it))+"</b> → <b style='color:var(--accent)'>"+esc(sound(it))+"</b>"+
        "<p>"+esc(it.note||it.practiceNote||'')+"</p><p class='hint'>"+esc(it.example||it.referenceText||'')+"</p>"+
        gradeRow(it)+"<button class='btn' style='margin-top:8px' onclick='SS.next()'>Next →</button></div>";
      bindGrades(it, ok, null);
    }); });
  }

  function renderProduce(){
    var pool=weakFirst(); if(!pool.length) pool=DATA().slice();
    var it=pool[0]; STATE.cur=it;
    var ref=refText(it);
    var h="<div class='card'><div class='hint'>"+esc(TYPE_LABEL[it.type])+" · "+esc(it.categoryLabel)+"</div>"+
      "<div style='font-size:28px;font-weight:800;margin:6px 0'>"+esc(label(it))+"</div>"+
      "<div class='hint'>"+esc(sound(it))+"</div>"+
      "<p style='margin-top:8px'>Say: <b>"+esc(ref)+"</b></p>"+
      "<button class='btn ssplay' data-say=\""+esc(ref)+"\">🔊 Hear it</button>"+
      "<button class='btn rec ssrec'>● Record &amp; score</button>"+
      "<span class='ssmsg hint' style='margin-left:8px'></span>"+
      "<div class='ssreveal'></div></div>";
    body().innerHTML=h;
    var btn=$('.ssrec',body()), msg=$('.ssmsg',body()); var mr=null,chunks=[];
    btn.addEventListener('click',function(){
      if(btn.classList.contains('on')){ if(mr)mr.stop(); return; }
      if(!navigator.mediaDevices){ msg.textContent='Recording not supported.'; return; }
      navigator.mediaDevices.getUserMedia({audio:true}).then(function(stream){
        mr=new MediaRecorder(stream); chunks=[];
        mr.ondataavailable=function(e){chunks.push(e.data);};
        mr.onstop=function(){ stream.getTracks().forEach(function(t){t.stop();});
          btn.classList.remove('on'); btn.textContent='● Record & score'; msg.textContent='Scoring…';
          var blob=new Blob(chunks,{type:'audio/webm'});
          var fd=new FormData(); fd.append('word',ref); fd.append('audio',blob,'d.webm');
          fetch('/practice',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(j){
            if(j.error){ msg.textContent='Azure unavailable — grade yourself.'; selfGrade(it); return; }
            var g = j.score<50?0 : j.score<70?1 : j.score<90?2 : 3;
            schedule(it.id,g); record(it.category, g>=2);
            var col=j.score>=85?'#43c59e':(j.score>=70?'#ffb454':'#ff6b6b');
            msg.innerHTML="<b style='color:"+col+"'>"+j.score+"/100</b> (auto-graded)";
            $('.ssreveal',body()).innerHTML="<p>"+esc(it.note||it.practiceNote||'')+"</p>"+
              "<button class='btn' onclick='SS.next()'>Next →</button>";
          }).catch(function(){ msg.textContent='Offline — grade yourself.'; selfGrade(it); });
        };
        mr.start(); btn.classList.add('on'); btn.textContent='■ Stop'; msg.textContent='Recording…';
      }).catch(function(){ msg.textContent='Mic permission denied.'; });
    });
  }
  function selfGrade(it){ $('.ssreveal',body()).innerHTML="<div style='margin-top:8px'>"+
    "<p>"+esc(it.note||it.practiceNote||'')+"</p>"+gradeRow(it)+
    "<button class='btn' style='margin-top:8px' onclick='SS.next()'>Next →</button></div>";
    bindGrades(it,true,null); }

  function renderBrowse(){
    var types=Object.keys(TYPE_LABEL);
    var sel = STATE.bt || 'linking';
    var q = (STATE.bq||'').toLowerCase();
    var all = DATA();
    var list = all.filter(function(d){return d.type===sel;});
    if(q) list = list.filter(function(it){ return (label(it)+' '+sound(it)+' '+(it.note||it.practiceNote||'')+' '+(it.example||'')).toLowerCase().indexOf(q)>=0; });
    var rows = list.map(function(it){
      var del = it.custom? "<button class='btn small' style='background:#3a2029;color:#ff6b6b' onclick='SS.del(\""+it.id+"\")'>✕</button>" : "";
      var mine = it.custom? " <span class='hint'>(yours)</span>" : "";
      return "<tr><td><b>"+esc(label(it))+"</b>"+mine+"</td><td style='color:var(--accent)'>"+esc(sound(it))+"</td>"+
        "<td>"+esc(it.note||it.practiceNote||'')+"</td>"+
        "<td><button class='btn small ssplay' data-say=\""+esc(refText(it))+"\">🔊</button> "+del+"</td></tr>";
    }).join('');
    body().innerHTML="<div class='ssbtypes'>"+types.map(function(t){
      var n=all.filter(function(d){return d.type===t;}).length;
      return "<button class='btn small"+(t===sel?' active':'')+"' onclick='SS.browse(\""+t+"\")'>"+esc(TYPE_LABEL[t])+" ("+n+")</button>";}).join('')+"</div>"+
      "<div style='display:flex;gap:8px;align-items:center;margin:6px 0 12px;flex-wrap:wrap'>"+
        "<input id='ss-q' placeholder='Search this set…' value=\""+esc(STATE.bq||'')+"\" style='flex:1;min-width:200px;max-width:300px' oninput='SS.search(this.value)'>"+
        "<span class='hint'>"+list.length+" shown</span>"+
        "<button class='btn small' onclick='SS.addForm()'>➕ Add example</button></div>"+
      "<div id='ss-add'></div>"+
      "<table><tr><th>Written</th><th>Sounds like</th><th>Note</th><th></th></tr>"+rows+"</table>";
    var qi=document.getElementById('ss-q'); if(qi&&STATE.bq){ qi.focus(); qi.setSelectionRange(qi.value.length,qi.value.length); }
  }

  function renderStats(){
    var cats={};
    DATA().forEach(function(it){ if(!cats[it.category]) cats[it.category]={label:it.categoryLabel,type:it.type}; });
    var chips=Object.keys(cats).map(function(c){
      var a=acc(c); var col = a==null?'#2c4a58' : a>=0.8?'#43c59e' : a>=0.6?'#ffb454' : '#ff6b6b';
      var pc = a==null?'—':Math.round(a*100)+'%';
      return "<span class='chip' style='background:"+col+"22;color:"+col+";border:1px solid "+col+"55'>"+
        esc(cats[c].label)+" · "+pc+"</span>";
    }).join('');
    var due=dueItems().length, total=DATA().length, srs=get('ss_srs',{});
    var learned=Object.keys(srs).length;
    body().innerHTML="<div class='card'><b>Today</b><p>"+due+" item(s) due · "+learned+"/"+total+" seen</p></div>"+
      "<h2>Category heatmap</h2><p class='sub'>green = strong, red = weak. Drill the red ones in the Weakness queue.</p>"+
      "<div class='chips'>"+chips+"</div>"+
      "<button class='btn small' style='margin-top:16px' onclick='SS.reset()'>Reset all drill progress</button>";
  }

  function render(){
    if(STATE.mode==='queue') renderRecognition(true);
    else if(STATE.mode==='recognize') renderRecognition(false);
    else if(STATE.mode==='produce') renderProduce();
    else if(STATE.mode==='browse') renderBrowse();
    else renderStats();
  }
  window.SS={
    speak:speak,
    next:function(){ render(); },
    browse:function(t){ STATE.bt=t; STATE.bq=''; renderBrowse(); },
    search:function(v){ STATE.bq=v; renderBrowse(); },
    addForm:function(){ var t=STATE.bt||'linking';
      document.getElementById('ss-add').innerHTML="<div class='card'>"+
        "<div class='hint'>New example in <b>"+esc(TYPE_LABEL[t])+"</b></div>"+
        "<label>Written form<br><input id='sa-w' style='width:100%'></label>"+
        "<label style='display:block;margin-top:6px'>Sounds like<br><input id='sa-s' style='width:100%'></label>"+
        "<label style='display:block;margin-top:6px'>Category label (e.g. 打招呼)<br><input id='sa-cl' style='width:100%'></label>"+
        "<label style='display:block;margin-top:6px'>Example sentence<br><input id='sa-e' style='width:100%'></label>"+
        "<label style='display:block;margin-top:6px'>Note / rule<br><input id='sa-n' style='width:100%'></label>"+
        "<button class='btn' style='margin-top:10px' onclick='SS.addSave()'>Save example</button> <span id='sa-msg' class='hint'></span></div>"; },
    addSave:function(){ var w=(document.getElementById('sa-w').value||'').trim();
      if(!w){ document.getElementById('sa-msg').textContent='Written form required'; return; }
      var t=STATE.bt||'linking'; var s=(document.getElementById('sa-s').value||'').trim();
      var it={ id:'custom_'+Math.random().toString(36).slice(2,9), type:t, category:'custom',
        categoryLabel:(document.getElementById('sa-cl').value||'').trim()||'custom',
        written:w, soundsLike:s, example:(document.getElementById('sa-e').value||'').trim(),
        note:(document.getElementById('sa-n').value||'').trim(), custom:true };
      if(t==='segmental'){ it.targetForm=w; it.ipa=s; it.referenceText=it.example||w; }
      var a=get('ss_custom',[]); a.push(it); set('ss_custom',a); renderBrowse(); },
    del:function(id){ set('ss_custom', get('ss_custom',[]).filter(function(x){return x.id!==id;})); renderBrowse(); },
    reset:function(){ if(confirm('Clear all sound-drill progress?')){ localStorage.removeItem('ss_srs'); localStorage.removeItem('ss_stats'); render(); } }
  };
  window.ssMode=function(btn){ STATE.mode=btn.getAttribute('data-m');
    document.querySelectorAll('#drills .drillnav .btn').forEach(function(b){b.classList.remove('active');});
    btn.classList.add('active'); render(); };
  // (TTS play for [data-say] buttons is handled globally by SkillStore)
  // first render when the panel is shown (or immediately if visible)
  document.addEventListener('click',function(e){
    var a=e.target.closest && e.target.closest('a[data-panel=drills]'); if(a) setTimeout(render,30);
  });
  if(document.getElementById('drills')) render();
})();
</script>
"""


def _sound_panel():
    data = _load_sound_data()
    if not data:
        return ""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return ("<section id='drills' class='tabpanel hidden'>"
            "<h1>Sound drills — connected speech</h1>"
            "<p class='sub'>Diagnose &rarr; drill &rarr; space-repeat the linking, weak-form, "
            "elision, assimilation and segmental patterns that make fast English hard to follow. "
            "%d items, spaced-repetition scheduled.</p>"
            "<div class='drillnav'>"
            "<button class='btn small active' data-m='queue' onclick='ssMode(this)'>🎯 Weakness queue</button>"
            "<button class='btn small' data-m='recognize' onclick='ssMode(this)'>👂 Recognition</button>"
            "<button class='btn small' data-m='produce' onclick='ssMode(this)'>🎤 Production</button>"
            "<button class='btn small' data-m='browse' onclick='ssMode(this)'>📚 Browse</button>"
            "<button class='btn small' data-m='stats' onclick='ssMode(this)'>📊 Stats</button>"
            "</div><div id='ss-body'></div></section>"
            "<script>window.SOUND_DATA=%s;</script>%s"
            % (len(data), payload, _SOUND_JS))


_SKILLS_UTIL_JS = r"""
<script>
window.SkillStore=(function(){
 // ---- server-backed store (was localStorage-only — invisible across devices) ----
 // get()/set() must stay synchronous: every panel calls them mid-render like a
 // plain object lookup. So: hydrate a cache once at load (blocking, via a
 // synchronous XHR — the one place that's deliberate) and have get() read that
 // cache instantly; set() updates the cache instantly too, then persists to the
 // server in the background. First run only: if the server has nothing yet but
 // this browser's localStorage does (upgrading from the old client-only
 // storage), push the local data up once to seed the server.
 var CACHE=null;
 function _collectLocal(){
   var out={};
   try{
     for(var i=0;i<localStorage.length;i++){
       var k=localStorage.key(i);
       if(!k||k.indexOf('navsec_')===0) continue;  // per-device UI state, not progress
       try{ out[k]=JSON.parse(localStorage.getItem(k)); }catch(_){ out[k]=localStorage.getItem(k); }
     }
   }catch(_){}
   return out;
 }
 function _xhr(method,url,body){
   try{
     var x=new XMLHttpRequest(); x.open(method,url,false);
     if(body!=null) x.setRequestHeader('Content-Type','application/json');
     x.send(body!=null?JSON.stringify(body):null);
     if(x.status>=200&&x.status<300) return JSON.parse(x.responseText);
   }catch(_){}
   return null;
 }
 // One flaky request here must never look like "your progress is gone" — a
 // transient network hiccup during the ONE hydration call at page load would
 // otherwise silently fall back to an empty cache, indistinguishable from
 // actual data loss. Retry before giving up.
 function _xhrRetry(method,url,body,tries){
   tries=tries||3;
   for(var i=0;i<tries;i++){ var r=_xhr(method,url,body); if(r!==null) return r; }
   return null;
 }
 var HYDRATE_FAILED=false;
 (function init(){
   var server=_xhrRetry('GET','/api/progress');
   if(server===null){
     HYDRATE_FAILED=true;   // couldn't reach the server even after retries
     CACHE=_collectLocal();
     return;
   }
   if(!Object.keys(server).length){
     var local=_collectLocal();
     if(Object.keys(local).length){ CACHE=local; _xhrRetry('POST','/api/progress',local); return; }
   }
   CACHE=server;
 })();
 if(HYDRATE_FAILED){
   document.addEventListener('DOMContentLoaded', function(){
     try{
       var b=document.createElement('div');
       b.textContent='⚠️ Could not reach the server just now — showing whatever is cached on '+
         'this device. Nothing is deleted; reload the page once you are back online to see '+
         'your real progress.';
       b.style.cssText='position:fixed;top:0;left:0;right:0;z-index:9999;background:#3a2030;'+
         'color:#ff9db0;padding:10px 16px;font-size:13px;text-align:center';
       document.body.insertBefore(b, document.body.firstChild);
     }catch(_){}
   });
 }
 function get(k,d){ var v=CACHE&&CACHE[k]; return (v===undefined||v===null)?d:v; }
 // Keys written but not yet acknowledged by the server. update() re-reads the
 // whole store before mutating, and that read can be answered before an earlier
 // POST has been applied — which would silently undo it. Two clicks in quick
 // succession is enough. Anything in flight is re-applied over the server's
 // answer and dropped once the write is confirmed.
 var PENDING={};
 function set(k,v,replace){ CACHE=CACHE||{}; CACHE[k]=v;
   var body={}; body[k]=v;
   if(replace) body.__replace=[k];   // "delete this", not "here's my snapshot"
   PENDING[k]=v;
   try{ fetch('/api/progress',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
     .then(function(r){ if(r&&r.ok&&PENDING[k]===v) delete PENDING[k]; })
     .catch(function(){}); }catch(_){}
 }
 // Read-modify-write on a key that the server can only replace wholesale
 // (pw_custom, pw_hidden, …). CACHE is hydrated once at page load, so a tab
 // left open while you add words on another device would otherwise POST its
 // stale list and drop them. Re-read the server first, mutate that, write back.
 // `replace` is for lists whose entries get EDITED or REMOVED, not just
 // appended (ec_photos): the server's append-merge would otherwise keep the
 // old copy of an edited entry alongside the new one, and resurrect deletions.
 function update(k,d,fn,replace){
   var fresh=_xhrRetry('GET','/api/progress');
   if(fresh){
     CACHE=fresh;
     // our own in-flight writes outrank a server read that may predate them
     Object.keys(PENDING).forEach(function(pk){ CACHE[pk]=PENDING[pk]; });
   }
   var cur=get(k,d);
   set(k, fn(cur), replace);
 }
 function today(){return new Date().toISOString().slice(0,10);}
 function addDays(n){var x=new Date();x.setDate(x.getDate()+n);return x.toISOString().slice(0,10);}
 function esc(s){return (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
 function uid(){return 'x'+Math.random().toString(36).slice(2,9);}
 function schedule(srs,grade){ srs=srs||{ef:2.5,interval:1,reps:0,due:today()};
   if(grade===0){srs.reps=0;srs.interval=1;} else {srs.reps++; srs.ef=Math.max(1.3,srs.ef+(0.1-(3-grade)*(0.08+(3-grade)*0.02))); srs.interval=srs.reps===1?1:srs.reps===2?3:Math.round(srs.interval*srs.ef);} srs.due=addDays(srs.interval); srs.last=today(); return srs; }
 function pickVoice(){ var vs=(window.speechSynthesis.getVoices&&window.speechSynthesis.getVoices())||[];
   function f(re,local){ return vs.filter(function(v){return re.test(v.lang)&&(!local||v.localService);})[0]; }
   // prefer a LOCAL en-US voice (e.g. Samantha) — network voices often fail silently
   return f(/en[-_]US/i,true)||f(/^en/i,true)||f(/en[-_]US/i,false)||f(/^en/i,false)||vs[0]||null; }
 function webSpeak(t,rate){ if(!window.speechSynthesis||!t)return;
   var synth=window.speechSynthesis;
   try{ synth.cancel(); }catch(_){}
   var u=new SpeechSynthesisUtterance(String(t)); u.lang='en-US'; u.rate=rate||0.95;
   var v=pickVoice(); if(v)u.voice=v;
   try{ synth.resume(); }catch(_){}
   synth.speak(u); }
 var _au=null;
 function speak(t,rate){ if(!t)return;
   // when served locally, use the server's system TTS (reliable); else browser TTS
   if(/^https?:/.test(location.protocol)){
     try{ if(_au){_au.pause();} }catch(_){}
     var rt=Math.round((rate||0.95)*175);
     _au=new Audio('/tts?r='+rt+'&text='+encodeURIComponent(String(t)));
     _au.play().catch(function(){ webSpeak(t,rate); });
   } else { webSpeak(t,rate); } }
 function norm(s){ return (s||'').toLowerCase().replace(/[^a-z0-9 ]/g,'').replace(/\s+/g,' ').trim(); }
 // ---- score history (shared by Practice words + Stories) ----
 // Each attempt carries `i`, a unique id, so the server can tell two real
 // attempts apart even when they're the same score on the same day — without
 // it they're byte-identical and the merge used to silently drop one.
 function logScore(key,score){ var h=get('ec_scores',{});
   (h[key]=h[key]||[]).push({s:score,d:today(),i:uid()+Date.now().toString(36)});
   set('ec_scores',h); }
 function scores(key){ return (get('ec_scores',{})[key])||[]; }
 function spark(key){ var a=scores(key); if(!a.length) return "<span class='hint'>no attempts yet</span>";
   var last=a[a.length-1].s, best=Math.max.apply(null,a.map(function(x){return x.s;}));
   var bars=a.slice(-12).map(function(x){var col=x.s>=85?'#43c59e':x.s>=70?'#ffb454':'#ff6b6b';
     return "<i style='display:inline-block;width:6px;height:"+Math.max(3,Math.round(x.s/100*22))+"px;background:"+col+";margin-right:2px;vertical-align:bottom;border-radius:1px'></i>";}).join('');
   return "<span style='display:inline-flex;align-items:flex-end'>"+bars+"</span> <span class='hint' style='margin-left:6px'>best "+best+" · last "+last+" · "+a.length+" tries</span>"; }
 // warm up the voice list so the first Play isn't silent
 try{ window.speechSynthesis && window.speechSynthesis.getVoices();
   if(window.speechSynthesis) window.speechSynthesis.onvoiceschanged=function(){window.speechSynthesis.getVoices();}; }catch(_){}
 return {get:get,set:set,update:update,today:today,addDays:addDays,esc:esc,uid:uid,schedule:schedule,speak:speak,norm:norm,logScore:logScore,scores:scores,spark:spark};
})();
// generic TTS play via data-say (shared across skill panels)
document.addEventListener('click',function(e){
 var b=e.target.closest && e.target.closest('[data-say]'); if(!b) return;
 window.SkillStore.speak(b.getAttribute('data-say'), parseFloat(b.getAttribute('data-rate'))||0.95);
 var old=b.innerHTML; b.innerHTML='🔊 …'; setTimeout(function(){b.innerHTML=old;},600);
});
</script>
"""

_GRADE_BTNS = ("<div class='ssgrades'>Recall: "
               "<button class='btn small' style='background:#ff6b6b;color:#08222b' data-g='0'>Again</button>"
               "<button class='btn small' style='background:#ffb454;color:#08222b' data-g='1'>Hard</button>"
               "<button class='btn small' style='background:#46b3c9;color:#08222b' data-g='2'>Good</button>"
               "<button class='btn small' style='background:#43c59e;color:#08222b' data-g='3'>Easy</button></div>")


_GRADE_HTML = ("<div class='ssgrades'>Recall: "
  "<button class='btn small' style='background:#ff6b6b;color:#08222b' data-g='0'>Again</button>"
  "<button class='btn small' style='background:#ffb454;color:#08222b' data-g='1'>Hard</button>"
  "<button class='btn small' style='background:#46b3c9;color:#08222b' data-g='2'>Good</button>"
  "<button class='btn small' style='background:#43c59e;color:#08222b' data-g='3'>Easy</button></div>")

_VOCAB_JS = (r"""
(function(){var S=window.SkillStore; var MODE='review'; var SCFILTER='all';
 function scenarios(){ return window.VOCAB_SCENARIOS||['General']; }
 function items(){return S.get('lex_items',[]);} function save(a){S.set('lex_items',a);}
 function body(){return document.getElementById('vx-body');}
 function due(){var t=S.today(); return items().filter(function(x){return !x.srs||x.srs.due<=t;});}
 function render(){ MODE==='add'?addForm():MODE==='all'?all():MODE==='cover'?coverage():review(); }

 // ---- coverage: what my surroundings contain vs what I actually use --------
 // Tokenised, because a headword can be a phrase: "hit a forehand" is worth
 // knowing as hit + forehand, and matching it whole against a spoken-word list
 // would never hit. Function words are dropped — "a" proves nothing.
 function tokens(text){
   var out=[];
   (text||'').toLowerCase().replace(/[^a-z' ]+/g,' ').split(/\s+/).forEach(function(w){
     w=w.replace(/^'+|'+$/g,''); if(w.length>1||w==='a'||w==='i') out.push(w); });
   return out;
 }
 function stopwords(){
   var fw={}; (window.VOCAB_FUNCTION_WORDS||[]).forEach(function(w){ fw[w]=1; }); return fw;
 }
 // Coverage reads the permanent ledger (ec_seen), not the live photo list, so
 // deleting a photo to reclaim space never rewrites the history of what has
 // been around you. Photos are a working set; the ledger is the record.
 function seenLedger(){ return S.get('ec_seen',{}); }
 // A word enters your surroundings two ways: a photo turns it up, or you capture
 // it by hand. Both belong in the ledger, and both go in TOKENISED — a
 // collocation like "brick wall" is brick and wall. Storing only whole headwords
 // would hide every word that only ever arrives inside a phrase, which is most
 // of them once your deck fills up with chunks.
 function foldSource(seen, kind, id, headwords, date){
   var key = (kind==='photo') ? 'pids' : 'cids', changed=false;
   headwords.forEach(function(hw){
     tokens(hw).forEach(function(w){
       var e=seen[w] || (seen[w]={pids:[], cids:[], forms:{}, first:date||'', last:date||''});
       if(!e[key]) e[key]=[];
       if(e[key].indexOf(id)<0){ e[key].push(id); changed=true; }
       if(!e.forms[hw]){ e.forms[hw]=1; changed=true; }
       if(date && (!e.first || date<e.first)){ e.first=date; changed=true; }
       if(date && (!e.last  || date>e.last )){ e.last=date;  changed=true; }
     });
   });
   return changed;
 }
 // Written the moment a word is captured, so deleting it from the deck later
 // can't erase the fact that you met it.
 function recordCaptured(list){
   var rows=(list||[]).filter(function(x){ return x && x.id && tokens(x.headword).length; });
   if(!rows.length) return;
   S.update('ec_seen',{},function(seen){
     seen=seen||{};
     rows.forEach(function(x){ foldSource(seen,'capture',x.id,[x.headword],x.added||S.today()); });
     return seen; }, true);
 }
 // Self-healing migration: fold in any photo or captured word the ledger doesn't
 // know about yet. Idempotent on source id, so it also repairs a ledger that fell
 // behind for any other reason. Only writes when something was actually missing.
 function syncLedger(){
   var cur=seenLedger(), gotP={}, gotC={}, missP=[], missC=[];
   Object.keys(cur).forEach(function(w){
     (cur[w].pids||[]).forEach(function(p){ gotP[p]=1; });
     (cur[w].cids||[]).forEach(function(c){ gotC[c]=1; });
   });
   // a source with nothing tokenisable in it is skipped rather than retried
   // forever — it would never leave a trace to recognise it by
   photos().forEach(function(p){
     if(gotP[p.id]) return;
     if((p.items||[]).some(function(it){ return tokens(it.headword).length; })) missP.push(p);
   });
   items().forEach(function(x){
     if(!x.id || gotC[x.id]) return;
     if(tokens(x.headword).length) missC.push(x);
   });
   if(!missP.length && !missC.length) return false;
   S.update('ec_seen',{},function(seen){
     seen=seen||{};
     missP.forEach(function(p){
       foldSource(seen,'photo',p.id,(p.items||[]).map(function(it){return it.headword;}),p.d||'');
     });
     missC.forEach(function(x){ foldSource(seen,'capture',x.id,[x.headword],x.added||''); });
     return seen; }, true);
   return true;
 }
 // document frequency: how many separate places a word turned up in — each photo
 // it appeared in, plus each entry you captured by hand. Appearing in five photos
 // is what "keeps being around me" actually means; five mentions inside one photo
 // is just one scene.
 function surroundingFreq(){
   var fw=stopwords(), led=seenLedger(), live={}, freq={};
   photos().forEach(function(p){ live[p.id]=p; });
   Object.keys(led).forEach(function(w){
     if(fw[w]) return;
     var e=led[w], pids=e.pids||[], cids=e.cids||[];
     if(!pids.length && !cids.length) return;
     freq[w]={ n:pids.length+cids.length, cap:cids.length, words:e.forms||{},
               first:e.first||'', last:e.last||'',
               photos:pids.map(function(pid){
                 var p=live[pid];
                 return p && !p.imgGone
                   ? {img:p.img, pid:pid, d:p.d, scenario:p.scenario, live:true}
                   : {pid:pid, live:false}; }) };
   });
   return freq;
 }
 function spokenSet(){
   var s={}; (window.SPOKEN_WORDS||[]).forEach(function(w){ s[w]=1; }); return s;
 }
 // same derivation the Listening vocabulary panel uses: only clips you have
 // actually practised count as heard.
 function heardSet(){
   var clips=window.LISTEN_TEXTS||{}, srs=S.get('dict_srs',{}), sc=S.get('ec_scores',{}), ids={}, out={};
   Object.keys(srs).forEach(function(k){ ids[k]=1; });
   Object.keys(sc).forEach(function(k){ if(k.indexOf('dict:')===0) ids[k.slice(5)]=1; });
   Object.keys(ids).forEach(function(id){
     if(clips[id]===undefined) return;
     tokens(clips[id]).forEach(function(w){ out[w]=1; }); });
   return out;
 }
 function coverage(){
   syncLedger();
   var freq=surroundingFreq(), all=Object.keys(freq);
   if(!all.length){
     body().innerHTML="<div class='card'>Nothing here yet. Save a photo in <b>Describe a photo</b>, "+
       "or capture a word by hand, and this report shows which of the words around you you can "+
       "already say, which you've only ever heard, and which you've never met anywhere else.</div>";
     return;
   }
   var spoken=spokenSet(), heard=heardSet();
   var nSp=0, nHe=0, nBoth=0, gap=[];
   all.forEach(function(w){
     var s=!!spoken[w], h=!!heard[w];
     if(s) nSp++; if(h) nHe++; if(s&&h) nBoth++;
     if(!s&&!h) gap.push(w);
   });
   var pct=function(n){ return all.length?Math.round(100*n/all.length):0; };
   var stats="<div class='metrics'>"+
     "<div class='m'><span>"+all.length+"</span>distinct words around you</div>"+
     "<div class='m'><span style='color:var(--good)'>"+nSp+"</span>you've also spoken <div class='hint'>"+pct(nSp)+"%</div></div>"+
     "<div class='m'><span style='color:var(--accent)'>"+nHe+"</span>you've also heard <div class='hint'>"+pct(nHe)+"%</div></div>"+
     "<div class='m'><span style='color:var(--warn)'>"+gap.length+"</span>neither <div class='hint'>"+pct(gap.length)+"%</div></div></div>";

   gap.sort(function(a,b){ return freq[b].n-freq[a].n || (a<b?-1:1); });
   var recurring=gap.filter(function(w){ return freq[w].n>1; });
   var lead="<p class='sub'>Your surroundings against what you actually produce and take in. "+
     "Every word your photos turn up and every word you've captured counts, phrases included — "+
     "<i>brick wall</i> is counted as <i>brick</i> and <i>wall</i>, because those are the words "+
     "you'd have to reach for. <b>Spoken</b> comes from your recording transcripts, <b>heard</b> "+
     "from the dictation clips you've worked through — the same sources those two panels use. "+
     "This report is append-only: deleting a photo or a word frees the space but never removes "+
     "a word you've met.</p>";
   var gapNote="<h2>Never spoken, never heard <span class='hint' style='font-weight:400'>"+gap.length+"</span></h2>"+
     "<p class='sub'>Sorted by how many separate places each one turned up in — every photo it "+
     "appeared in, plus a hand capture, counts once. The ones at the top are all around you and "+
     "still absent from your English"+
     (recurring.length?(" — "+recurring.length+(recurring.length===1?" of them has":" of them have")+
       " turned up more than once"):"")+".</p>";
   var rows=gap.map(function(w){
     var e=freq[w];
     var thumbs=e.photos.slice(0,4).map(function(p){
       // the photo may be gone (deleted, or its image freed) — the word stays
       return p.live
         ? "<img src='"+S.esc(p.img)+"' title='"+S.esc((p.scenario||'')+' · '+(p.d||''))+
           "' style='width:38px;height:28px;object-fit:cover;border-radius:4px;cursor:pointer;margin-right:3px' "+
           "onclick=\"VX.openPhoto('"+p.pid+"')\">"
         : "<span title='Photo no longer stored — the word is kept' style='display:inline-block;"+
           "width:38px;height:28px;border-radius:4px;background:#1f3542;color:var(--mut);"+
           "font-size:14px;text-align:center;line-height:28px;margin-right:3px'>🗄</span>"; }).join('');
     // a word can reach you without a photo — through something you wrote down
     if(e.cap) thumbs+="<span title='Captured by hand"+(e.cap>1?" ("+e.cap+" entries)":"")+
       "' style='display:inline-block;width:38px;height:28px;border-radius:4px;"+
       "background:#1f3542;color:var(--mut);font-size:14px;text-align:center;line-height:28px'>✍️</span>";
     var forms=Object.keys(e.words).filter(function(x){return x.toLowerCase()!==w;});
     var when=e.first? ("<div class='hint'>first met "+S.esc(e.first)+"</div>") : "";
     return "<tr><td><b>"+S.esc(w)+"</b>"+(e.n>1?" <span class='chip up'>×"+e.n+"</span>":"")+when+"</td>"+
       "<td class='hint'>"+S.esc(forms.join(', '))+"</td><td>"+thumbs+"</td></tr>";
   }).join('');
   var table=gap.length
     ? "<table><tr><th>Word</th><th>Seen in</th><th>Where from</th></tr>"+rows+"</table>"
     : "<div class='card'>🎉 Every word your surroundings have produced has also shown up in your "+
       "speaking or your listening. Take a photo somewhere less familiar.</div>";
   body().innerHTML=lead+stats+gapNote+table; }
 function review(){ if(!items().length){body().innerHTML="<div class='card'>No words yet. Hit Capture to add your first.</div>";return;}
   var d=due(); if(!d.length){body().innerHTML="<div class='card'>🎉 All caught up — nothing due to review.</div>";return;}
   var it=d[0];
   body().innerHTML="<div class='card'><div class='hint'>"+S.esc(it.type||'')+(it.register?' · '+S.esc(it.register):'')+"</div>"+
     "<div style='font-size:28px;font-weight:800;margin:8px 0'>"+S.esc(it.headword)+"</div>"+
     "<button class='btn' onclick='VX.flip()'>Show meaning</button>"+
     "<div class='vxback' style='display:none;margin-top:10px'><p>"+S.esc(it.definition)+"</p>"+
       "<p class='hint'>"+S.esc(it.example||'')+"</p>"+
       "<button class='btn small' data-say=\""+S.esc(it.example||it.headword)+"\">🔊</button>"+__GRADE__+"</div></div>";
   body().querySelectorAll('.ssgrades .btn').forEach(function(b){b.addEventListener('click',function(){
     var g=parseInt(b.getAttribute('data-g'),10); it.srs=S.schedule(it.srs,g);
     var a=items(); for(var i=0;i<a.length;i++) if(a[i].id===it.id)a[i]=it; save(a); render(); });});
 }
 function addForm(){ body().innerHTML="<div class='card'>"+
   "<label>Headword<br><input id='vx-h' style='width:100%'></label>"+
   "<label style='display:block;margin-top:8px'>Meaning<br><input id='vx-d' style='width:100%'></label>"+
   "<label style='display:block;margin-top:8px'>Example<br><input id='vx-e' style='width:100%'></label>"+
   "<label style='display:block;margin-top:8px'>Type <select id='vx-t'><option>single_word</option><option>collocation</option><option>idiom</option><option>phrasal_verb</option></select></label>"+
   "<label style='display:block;margin-top:8px'>Register <select id='vx-r'><option>neutral</option><option>informal</option><option>formal</option></select></label>"+
   "<label style='display:block;margin-top:8px'>Scenario <select id='vx-sc'>"+scenarios().map(function(s){return "<option"+(s==='General'?' selected':'')+">"+S.esc(s)+"</option>";}).join('')+"</select></label>"+
   "<button class='btn' style='margin-top:12px' onclick='VX.add()'>Save</button> <span id='vx-msg' class='hint'></span>"+
   "<div style='margin-top:14px;border-top:1px solid var(--line);padding-top:12px'>"+
   "<button class='btn small' onclick='VX.loadPack()'>📦 Load Chinglish starter pack</button> <span id='vx-pack' class='hint'></span></div></div>"; }
 // Everything you've captured by hand, plus every word your photos produced.
 // Photo words that you've already added to the deck appear once, as captured —
 // the deck copy wins, because that's the one with the review schedule on it.
 function photos(){ return S.get('ec_photos',[]); }
 // A photo word can be dismissed from this list — a vision model picks out
 // things you already know, or splits one object into three near-duplicates.
 // It's hidden rather than cut out of the photo's own record, because that
 // record is what "Describe a photo" compares your recall against. Coverage is
 // unaffected either way: it reads the ledger, which never forgets.
 function hiddenSet(){
   var h={}; (S.get('lex_hidden',[])||[]).forEach(function(w){ h[w]=1; }); return h;
 }
 function setHidden(fn){ S.update('lex_hidden',[],fn,true); }
 function merged(){
   var out=items().map(function(x){
     return {id:x.id, headword:x.headword, definition:x.definition, example:x.example||'',
             type:x.type||'other', scenario:x.scenario||'General', src:'deck', srs:x.srs};
   });
   var have={}, hid=hiddenSet();
   out.forEach(function(x){ have[(x.headword||'').toLowerCase()]=1; });
   photos().forEach(function(p){
     (p.items||[]).forEach(function(it){
       var k=(it.headword||'').toLowerCase();
       if(!k || have[k] || hid[k]) return;
       have[k]=1;
       out.push({id:null, headword:it.headword, definition:it.definition||'',
                 example:it.example||'', type:it.type||'single_word',
                 scenario:it.scenario||p.scenario||'General', src:'photo',
                 img:p.img, pid:p.id, d:p.d});
     });
   });
   return out;
 }
 function chips(list, cur, fn){
   return "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px'>"+
     list.map(function(o){
       return "<button class='btn small"+(cur===o.k?' active':'')+"' onclick=\"VX."+fn+
         "('"+String(o.k).replace(/'/g,"\\'")+"')\">"+S.esc(o.label)+
         " <span class='hint'>"+o.n+"</span></button>"; }).join('')+"</div>";
 }
 function hideBar(){
   var hid=S.get('lex_hidden',[])||[];
   if(!hid.length) return "";
   return "<div class='bsfilter'><span class='hint'>"+hid.length+" photo word"+
     (hid.length===1?"":"s")+" removed from this list — still counted in Coverage</span> "+
     "<button class='btn small' onclick='VX.showHidden(this)'>👁 Show</button> "+
     "<button class='btn small' onclick='VX.unhideAll()'>↺ Restore all</button>"+
     "<div id='vx-hidden' style='display:none;margin-top:8px'>"+hid.map(function(w){
       return "<button class='btn small' title='Put it back' onclick=\"VX.unhide('"+
         encodeURIComponent(w)+"')\">"+S.esc(w)+" ↺</button> "; }).join('')+"</div></div>";
 }
 function all(){
   var full=merged();
   if(!full.length){ body().innerHTML=hideBar()+"<div class='card'>No words yet. Hit <b>Capture</b> to add "+
     "one by hand, or save a photo in <b>Describe a photo</b> and its words land here.</div>"; return; }
   // No source breakdown up here: where a word came from is already on its own
   // row, in the From column, where it's actually useful.
   var scount={}; full.forEach(function(x){ scount[x.scenario]=(scount[x.scenario]||0)+1; });
   if(SCFILTER!=='all' && !scount[SCFILTER]) SCFILTER='all';

   var present=Object.keys(scount).sort();
   var scBar=present.length>1 ? chips([{k:'all',label:'All scenarios',n:full.length}].concat(
       present.map(function(s){return {k:s,label:s,n:scount[s]};})), SCFILTER,'filterScenario') : "";

   var a=SCFILTER==='all'?full:full.filter(function(x){return x.scenario===SCFILTER;});
   if(!a.length){ body().innerHTML=hideBar()+scBar+"<div class='card'>Nothing matches that filter.</div>"; return; }
   a.sort(function(x,y){ return x.headword.toLowerCase()<y.headword.toLowerCase()?-1:1; });
   var by={}; a.forEach(function(x){(by[x.type||'other']=by[x.type||'other']||[]).push(x);});
   var h='';
   Object.keys(by).sort().forEach(function(t){
     h+="<h2>"+S.esc(t)+" <span class='hint' style='font-weight:400'>"+by[t].length+"</span></h2>"+
       "<table><tr><th>Word</th><th>Meaning</th><th>Example</th><th>From</th><th></th></tr>"+
       by[t].map(function(x){
         var from = x.src==='photo'
           ? "<img src='"+S.esc(x.img)+"' title='"+S.esc(x.scenario+' · '+(x.d||''))+
             "' style='width:54px;height:38px;object-fit:cover;border-radius:5px;cursor:pointer' "+
             "onclick=\"VX.openPhoto('"+x.pid+"')\">"
           : "<span class='hint'>"+(x.srs?'reviewing':'captured')+"</span>";
         var act = x.src==='photo'
           ? "<button class='btn small' title='Add to your review deck' onclick=\"VX.toDeck('"+
             encodeURIComponent(x.headword)+"')\">➕</button> "+
             "<button class='btn small' style='background:#3a2029;color:#ff9db0' "+
             "title='Remove from this list — the photo and the Coverage report keep it' "+
             "onclick=\"VX.hide('"+encodeURIComponent(x.headword)+"')\">&#10005;</button>"
           : "<button class='btn small' onclick='VX.del(\""+x.id+"\")'>✕</button>";
         return "<tr><td><b>"+S.esc(x.headword)+"</b></td><td>"+S.esc(x.definition)+
           "</td><td class='hint'>"+S.esc(x.example||'')+"</td><td>"+from+
           "</td><td style='text-align:right;white-space:nowrap'>"+act+"</td></tr>"; }).join('')+"</table>";
   });
   body().innerHTML=hideBar()+scBar+"<p class='hint' id='vx-allmsg'></p>"+h; }
 window.VX={ flip:function(){var b=body().querySelector('.vxback'); if(b)b.style.display='block';},
   add:function(){var h=document.getElementById('vx-h').value.trim(); if(!h){document.getElementById('vx-msg').textContent='Headword required';return;}
     var it={id:S.uid(),headword:h,definition:document.getElementById('vx-d').value.trim(),example:document.getElementById('vx-e').value.trim(),type:document.getElementById('vx-t').value,register:document.getElementById('vx-r').value,scenario:document.getElementById('vx-sc').value,added:S.today(),srs:null};
     var a=items(); a.push(it); save(a); recordCaptured([it]);
     document.getElementById('vx-msg').textContent='Saved ✓'; ['vx-h','vx-d','vx-e'].forEach(function(i){document.getElementById(i).value='';}); },
   // Deleting a captured word that a photo ALSO produced would otherwise just
   // unmask the photo copy and the row would appear not to go away, so the
   // dismissal has to cover both sources.
   del:function(id){
     var it=items().filter(function(x){return x.id===id;})[0];
     var hw=it?(it.headword||'').toLowerCase():'';
     var inPhoto=hw && photos().some(function(p){
       return (p.items||[]).some(function(i){ return (i.headword||'').toLowerCase()===hw; }); });
     // hide BEFORE the deck write: setHidden re-reads the whole store, so doing
     // it second would restore the entry we just removed here
     if(inPhoto) setHidden(function(cur){ cur=cur||[];
       return cur.indexOf(hw)<0 ? cur.concat([hw]) : cur; });
     save(items().filter(function(x){return x.id!==id;}));
     all();
   },
   hide:function(hw){
     hw=decodeURIComponent(hw).toLowerCase();
     setHidden(function(cur){ cur=cur||[];
       return cur.indexOf(hw)<0 ? cur.concat([hw]) : cur; });
     all();
   },
   unhide:function(hw){
     hw=decodeURIComponent(hw).toLowerCase();
     setHidden(function(cur){ return (cur||[]).filter(function(w){ return w!==hw; }); });
     all();
   },
   unhideAll:function(){ setHidden(function(){ return []; }); all(); },
   showHidden:function(btn){
     var d=document.getElementById('vx-hidden'); if(!d) return;
     var on=d.style.display==='none';
     d.style.display=on?'block':'none'; btn.textContent=on?'👁 Hide':'👁 Show';
   },
   filterScenario:function(s){ SCFILTER=s; all(); },
   openPhoto:function(pid){
     var a=document.querySelector("a[data-panel=photodesc]"); if(a) a.click();
     if(window.PD) window.PD.open(pid);
     window.scrollTo(0,0);
   },
   // promote a photo word into the spaced-repetition deck; until then it is
   // only a thing you've seen, not a thing you're being asked to recall
   toDeck:function(hw){
     hw=decodeURIComponent(hw);
     var row=merged().filter(function(x){return x.headword===hw && x.src==='photo';})[0];
     if(!row) return;
     var made=null;
     S.update('lex_items',[],function(cur){
       cur=cur||[];
       if(cur.some(function(x){return (x.headword||'').toLowerCase()===hw.toLowerCase();})) return cur;
       made={id:S.uid(),headword:row.headword,definition:row.definition,
         example:row.example,type:row.type,register:'neutral',scenario:row.scenario,
         added:S.today(),srs:null};
       return cur.concat([made]);
     });
     if(made) recordCaptured([made]);
     all();
     var m=document.getElementById('vx-allmsg');
     if(m) m.textContent='Added “'+hw+'” to your review deck.';
   },
   loadPack:function(){ var seed=window.CHINGLISH_SEED||[]; var a=items(); var have={}; a.forEach(function(x){have[x.headword]=1;});
     var made=[]; seed.forEach(function(s){ if(!have[s.headword]){ var it={id:S.uid(),headword:s.headword,definition:s.definition,example:s.example||'',type:s.type||'collocation',register:'neutral',scenario:'General',added:S.today(),srs:null}; a.push(it); made.push(it); } });
     save(a); recordCaptured(made);
     var m=document.getElementById('vx-pack'); if(m)m.textContent='Added '+made.length+' chunk(s). See the All tab.'; }
 };
 window.vxMode=function(btn){MODE=btn.getAttribute('data-m'); document.querySelectorAll('#vocab .drillnav .btn').forEach(function(b){b.classList.remove('active');}); btn.classList.add('active'); render();};
 if(document.getElementById('vx-body')) render();
})();
""").replace("__GRADE__", json.dumps(_GRADE_HTML))

_GRAMMAR_SEED = [
    {"said": "I am student", "correction": "I am a student", "rule": "articles (a/an/the)"},
    {"said": "I have two cat", "correction": "I have two cats", "rule": "plural -s"},
    {"said": "He go to work every day", "correction": "He goes to work every day", "rule": "3rd-person -s"},
    {"said": "Yesterday I go to the park", "correction": "Yesterday I went to the park", "rule": "past tense"},
    {"said": "I got many informations", "correction": "I got a lot of information", "rule": "uncountable nouns"},
    {"said": "I am agree with you", "correction": "I agree with you", "rule": "'be' + verb (no extra 'be')"},
    {"said": "This very good", "correction": "This is very good", "rule": "missing 'be' (copula)"},
    {"said": "My sister, he is a doctor", "correction": "My sister, she is a doctor",
     "rule": "he/she (他/她 are both 'tā' — same sound in Mandarin)"},
    {"said": "I very like it", "correction": "I really like it", "rule": "adverb ('very' can't modify a verb)"},
    {"said": "There have many people", "correction": "There are many people", "rule": "there is/are"},
]

_GRAMMAR_JS = r"""
(function(){var S=window.SkillStore; var MODE='log';
 function recs(){return window.GRAMMAR_DATA||[];} function manual(){return S.get('gram_log',[]);}
 function seed(){return window.GRAMMAR_SEED||[];}
 function body(){return document.getElementById('gx-body');}
 function render(){ if(MODE==='gstats')return gstats(); if(MODE==='pstats')return pstats(); log(); }

 // ---- error stats -------------------------------------------------------
 // Everything here is a rate, never a bare count: "37 weak /r/" is unreadable
 // without "out of 1286 times you said one".
 function bar(lift){
   var MAX=2.5, w=Math.max(2,Math.min(100, lift/MAX*100));
   var cls = lift>=1.5?' bad' : (lift>=1.05?' warn':'');
   return "<div class='lb"+cls+"'><i style='width:"+w.toFixed(0)+"%'></i>"+
          "<u style='left:"+(100/MAX).toFixed(0)+"%' title='your average'></u></div>";
 }
 function soundRows(p){
   return (p.rows||[]).map(function(r){
     return "<div class='es"+(r.thin?" thin":"")+"'><div><b>"+S.esc(r.label)+"</b>"+
       (r.exact?"<span class='esbadge exact'>exact</span>":"")+
       (r.thin?"<span class='esbadge'>too few to rank</span>":"")+"</div>"+
       "<div class='es-n'>said <b>"+r.n+"</b>× · weak <b>"+r.bad+"</b> · <b>"+r.rate+"%</b></div>"+
       "<div>"+bar(r.lift)+"<span class='es-x'>"+r.lift.toFixed(2)+
         "× your average"+(!r.thin&&r.lift_lo>=1.15?" · clearly worse":"")+"</span></div>"+
       "<div class='esnote'>"+S.esc(r.note)+
         (r.examples&&r.examples.length? " <span class='words' style='display:inline-flex'>"+
           r.examples.map(function(w){return "<span class='wpill bad'>"+S.esc(w)+"</span>";}).join('')+
           "</span>":"")+"</div></div>";
   }).join('');
 }
 // ---- the trend charts --------------------------------------------------
 // One chart engine, two datasets. The only real differences are the unit and
 // what counts as a thin week, so those are the only things configured.
 var CH={
   pron:{ host:'gx-trend', overall:'All tracked sounds',
          data:function(){ return window.ERRSTATS.trend||{}; },
          ylab:function(v){ return v+'%'; },
          tip:function(p){ return p.rate+'% weak ('+p.bad+' of '+p.n+' attempts)'; },
          cell:function(p){ return p.rate+'% <span class="hint">('+p.n+')</span>'; },
          thin:function(p){ return p.n<40; },
          gap:function(n){ return n? ' — only '+n+' attempts' : ' — never came up'; },
          unit:'attempts',
          foot:'A <b>hollow</b> dot is a window with fewer than 40 attempts of that '+
               'sound — the rate is real but jumpy, so read the shape, not the step. '+
               'A <b>dotted</b> segment jumps a window too thin to score at all: it is '+
               'joined so the line stays followable, not because anything was measured '+
               'there. Hover any dot — or any gap — for the counts.' },
   gram:{ host:'gx-gtrend', overall:'All mistake types',
          data:function(){ return window.ERRSTATS.gtrend||{}; },
          ylab:function(v){ return v; },
          tip:function(p){ return p.rate+' per 1k words ('+p.n+' in '+p.words+' words)'; },
          cell:function(p){ return p.rate+' <span class="hint">('+p.n+')</span>'; },
          thin:function(p){ return !!p.thin; },
          gap:function(n){ return n? ' — only '+n+' words spoken' : ' — nothing recorded'; },
          unit:'words',
          foot:'Counts are per <b>1000 words spoken</b>, so a heavy week does not look '+
               'like a bad one. A <b>hollow</b> dot is a window with under 1000 words '+
               'behind it — real, but one long recording can swing it. A <b>dotted</b> '+
               'segment jumps a stretch you barely recorded in; the line is joined so '+
               'it stays followable, not because anything was measured there. Hover any '+
               'dot — or any gap — for the raw numbers.' }
 };
 var CHS={};                            // chart id -> {sel, slot, nums}
 function chs(id){ return CHS[id]||(CHS[id]={sel:{},slot:{},nums:false}); }
 function tcolors(id){ return CH[id].data().colors||[]; }
 function tpick(id,k){
   var st=chs(id), TSLOT=st.slot;
   // A colour belongs to a sound for as long as it is on screen. Handing them
   // out by rank instead would recolour every surviving line whenever one is
   // switched off, and a line that changes colour reads as a different line.
   if(TSLOT[k]!=null) return;
   var used={}; Object.keys(TSLOT).forEach(function(x){ used[TSLOT[x]]=1; });
   for(var i=0;i<tcolors(id).length;i++){ if(!used[i]){ TSLOT[k]=i; return; } }
 }
 window.GXT={
   toggle:function(id,k){
     var st=chs(id);
     if(st.sel[k]){ delete st.sel[k]; delete st.slot[k]; }
     else{
       if(Object.keys(st.sel).length>=tcolors(id).length) return;   // out of hues
       st.sel[k]=1; tpick(id,k);
     }
     drawChart(id);
   },
   nums:function(id){ chs(id).nums=!chs(id).nums; drawChart(id); }
 };
 function drawChart(id){
   var cfg=CH[id], st=chs(id), TSEL=st.sel, TSLOT=st.slot, TNUM=st.nums;
   var host=document.getElementById(cfg.host); if(!host) return;
   var T=cfg.data(), C=tcolors(id);
   var weeks=T.weeks||[], lines=[];
   if(T.overall && T.overall.length>1)
     lines.push({key:'', label:cfg.overall, short:'All', color:'#9aa3bf',
                 dash:'5 4', points:T.overall});
   (T.series||[]).forEach(function(s){
     if(TSEL[s.key]) lines.push({key:s.key, label:s.label, color:C[TSLOT[s.key]],
                                 short:s.label.split(' —')[0].split(' vs')[0]
                                        .split(' (')[0].split(' /')[0],
                                 dash:'', gaps:s.gaps||{}, points:s.points});
   });

   var W=920,H=300,L=46,R=118,Tp=18,B=46, iw=W-L-R, ih=H-Tp-B;
   var hi=0; lines.forEach(function(s){ s.points.forEach(function(p){ if(p.rate>hi) hi=p.rate; }); });
   var top=Math.max(10, Math.ceil(hi/5)*5+5);      // a rate starts at 0, always
   function y(v){ return Tp+ih*(1-v/top); }
   function x(w){ var i=weeks.indexOf(w);
     return L+(weeks.length<2?iw/2:iw*i/(weeks.length-1)); }
   function mon(ds){ var p=ds.split('-');
     return ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][+p[1]-1]+' '+(+p[2]); }
   // A bucket may cover more than one week. Name the span, so a pooled point
   // never passes for a single week that happened to be unusually busy.
   var WEND=T.wend||{};
   function bmon(w){
     var e=WEND[w]; if(!e||e===w) return mon(w);
     // "Aug 3 – 9" inside one month, "Jul 27 – Aug 2" across one. Repeating the
     // month on every tick costs width the axis does not have.
     return mon(w)+' – '+(w.slice(0,7)===e.slice(0,7) ? String(+e.slice(8)) : mon(e));
   }

   // Round tick values, whatever the unit. Dividing the range by five gives
   // steps like 8 and 0.9; readers want 10 and 1.
   function nice(v){
     var p=Math.pow(10, Math.floor(Math.log(v)/Math.LN10)), n=v/p;
     return (n<=1?1 : n<=2?2 : n<=2.5?2.5 : n<=5?5 : 10)*p;
   }
   var grid='', step=nice(top/5);
   for(var v=0; v<=top+0.001; v+=step)
     grid+="<line x1='"+L+"' y1='"+y(v).toFixed(1)+"' x2='"+(W-R)+"' y2='"+y(v).toFixed(1)+
           "' stroke='#24404c'/><text x='"+(L-8)+"' y='"+(y(v)+4).toFixed(1)+
           "' fill='#9aa3bf' font-size='11' text-anchor='end'>"+
           cfg.ylab(Math.round(v*100)/100)+"</text>";
   weeks.forEach(function(w){
     grid+="<text x='"+x(w).toFixed(1)+"' y='"+(H-16)+"' fill='#9aa3bf' font-size='11' "+
           "text-anchor='middle'>"+bmon(w)+"</text>";
   });

   var body='', legend='', ends=[];
   lines.forEach(function(s){
     var pts=s.points.map(function(p){ return [x(p.w), y(p.rate), p]; });
     // Segment by segment, not one polyline, so a jumped week can look
     // different from a measured one. Joining a gap with the same solid stroke
     // claims a reading that was never taken.
     for(var i=1;i<pts.length;i++){
       var a=pts[i-1], b=pts[i],
           skip=(weeks.indexOf(b[2].w)-weeks.indexOf(a[2].w))>1;
       body+="<line x1='"+a[0].toFixed(1)+"' y1='"+a[1].toFixed(1)+"' x2='"+
             b[0].toFixed(1)+"' y2='"+b[1].toFixed(1)+"' stroke='"+s.color+
             "' stroke-width='2'"+(skip?" stroke-dasharray='1 5' stroke-opacity='.55'":
             (s.dash?" stroke-dasharray='"+s.dash+"'":""))+"/>";
       if(skip){
         // name the missing week where the eye already is, mid-gap
         var g=[], j;
         for(j=weeks.indexOf(a[2].w)+1;j<weeks.indexOf(b[2].w);j++)
           g.push(bmon(weeks[j])+cfg.gap(s.gaps&&s.gaps[weeks[j]]));
         body+="<circle cx='"+((a[0]+b[0])/2).toFixed(1)+"' cy='"+((a[1]+b[1])/2).toFixed(1)+
               "' r='14' fill='transparent'><title>"+S.esc(s.label)+
               " — not measured\n"+S.esc(g.join('\n'))+"</title></circle>";
       }
     }
     pts.forEach(function(q){
       // Hollow when the week is thin. A rate off 24 attempts and one off 1000
       // are the same dot otherwise, and the thin one swings far more — which
       // is exactly the swing somebody reads as "I got worse".
       var thin=cfg.thin(q[2]);
       body+="<circle cx='"+q[0].toFixed(1)+"' cy='"+q[1].toFixed(1)+"' r='4.5' fill='"+
             (thin?'#172530':s.color)+"' stroke='"+(thin?s.color:'#172530')+"' stroke-width='2'/>"+
             // the visible dot stays small; the hit target does not
             "<circle cx='"+q[0].toFixed(1)+"' cy='"+q[1].toFixed(1)+"' r='12' fill='transparent'>"+
             "<title>"+S.esc(s.label)+" — "+(q[2].span>1?"":"week of ")+bmon(q[2].w)+
             (q[2].span>1?"  ("+q[2].span+" weeks pooled)":"")+"\n"+cfg.tip(q[2])+
             (thin?"\nthin week — read this point loosely":"")+"</title></circle>";
     });
     // direct label at the right edge, so identity is never colour alone.
     // Collected first, drawn after: two series ending on the same value would
     // otherwise print on top of each other.
     var last=pts[pts.length-1];
     if(last) ends.push({x:last[0]+10, y:last[1]+4, c:s.color, t:s.short});
     legend+="<span class='lg'><i style='background:"+s.color+"'></i>"+S.esc(s.label)+"</span>";
   });
   // Spread the end labels apart, then keep the whole stack inside the plot.
   // Pushing down alone walks the last label off the bottom and straight over
   // the date ticks, which is what four ticked series used to do.
   ends.sort(function(a,b){ return a.y-b.y; });
   var GAPY=14, lo=Tp+10, hi=Tp+ih;
   for(var e=1;e<ends.length;e++)
     if(ends[e].y-ends[e-1].y < GAPY) ends[e].y = ends[e-1].y + GAPY;
   var over = ends.length ? ends[ends.length-1].y - hi : 0;
   if(over > 0){
     for(var f=0;f<ends.length;f++) ends[f].y -= over;
     // shifting up can re-collide at the top; walk back down from the ceiling
     for(var g=0;g<ends.length;g++)
       if(ends[g].y < (g?ends[g-1].y+GAPY:lo)) ends[g].y = g?ends[g-1].y+GAPY:lo;
   }
   ends.forEach(function(e){
     body+="<text x='"+e.x.toFixed(1)+"' y='"+e.y.toFixed(1)+"' fill='"+e.c+
           "' font-size='11.5' font-weight='700'>"+S.esc(e.t)+"</text>";
   });

   var pills=(T.series||[]).map(function(s){
     var on=!!TSEL[s.key];
     return "<button class='btn small' onclick=\"GXT.toggle('"+id+"','"+s.key+"')\" style='"+
       (on?"background:"+C[TSLOT[s.key]]+";color:#0d1620":"background:#1f3542;color:var(--mut)")+
       "'>"+(on?"✓ ":"")+S.esc(s.label.split(' —')[0])+"</button>";
   }).join(' ');

   var table='';
   if(TNUM){
     table="<div style='overflow-x:auto;margin-top:12px'>"+
       "<table style='min-width:520px'><tr><th>"+(id==='pron'?'Sound':'Type')+"</th>"+
       weeks.map(function(w){return "<th>"+bmon(w)+"</th>";}).join('')+"</tr>"+
       lines.map(function(s){
         var by={}; s.points.forEach(function(p){ by[p.w]=p; });
         return "<tr><td>"+S.esc(s.label)+"</td>"+weeks.map(function(w){
           if(by[w]) return "<td>"+cfg.cell(by[w])+"</td>";
           // A bare dash reads as a bug. Say which kind of nothing it is.
           var n=s.gaps&&s.gaps[w];
           // Terse here on purpose: this cell repeats across every row and
           // every window, and the hover text carries the full sentence.
           return "<td style='color:var(--mut)'>— <span class='hint'>"+
                  (n? "("+n+" "+cfg.unit+")" : "(none)")+"</span></td>";
         }).join('')+"</tr>";
       }).join('')+"</table></div>";
   }

   host.innerHTML =
     "<div class='chips' style='margin:2px 0 10px'>"+pills+
       "<button class='btn small' onclick=\"GXT.nums('"+id+"')\" style='background:#1f3542;color:var(--mut)'>"+
       (TNUM?"hide numbers":"show numbers")+"</button></div>"+
     "<svg viewBox='0 0 "+W+" "+H+"' width='100%' style='max-width:"+W+"px'>"+grid+body+"</svg>"+
     "<div class='legend'>"+legend+"</div>"+
     "<p class='hint' style='margin:6px 0 0'>"+cfg.foot+"</p>"+table;
 }

 // The chart says nothing until something is on it, so open on the worst few.
 function seed3(id){
   var st=chs(id);
   if(Object.keys(st.sel).length) return;
   (CH[id].data().series||[]).slice(0,3).forEach(function(s){
     st.sel[s.key]=1; tpick(id,s.key);
   });
 }
 // A trimmed window is stated, never silently dropped: the chart is starting
 // later than your history does, and that is the reader's business.
 function trimNote(T){
   var n=T.trimmed||0; if(!n) return '';
   return " The chart starts at your first window with enough speech to carry a "+
     "rate; <b>"+n+"</b> earlier "+(n>1?"windows are":"window is")+" left out.";
 }
 function trendCard(id, title, intro, note){
   var T=CH[id].data(), n=(T.weeks||[]).length;
   if(n<2) return "<div class='card'><b>"+title+"</b><p class='hint' style='margin:6px 0 0'>"+
     "Needs at least two weeks of recordings — there "+(n?"is 1 so far":"are none yet")+
     ". Keep recording and this fills in.</p></div>";
   return "<div class='card'><b>"+title+"</b>"+
     "<p class='hint' style='margin:6px 0 2px'>"+intro+"</p>"+
     (note?"<p class='hint' style='margin:2px 0 8px'>"+note+"</p>":"")+
     "<div id='"+CH[id].host+"'></div></div>";
 }

 function pstats(){
   var D=window.ERRSTATS||{}, p=D.pron||{};
   if(!p.rows||!p.rows.length){ body().innerHTML="<div class='card'>No scored recordings yet — "+
     "run an analysis with Azure pronunciation scoring switched on.</div>"; return; }

   // Say plainly how the failures were counted. An attributed number that
   // looks exact is worse than no number.
   var honest = (p.mode==='exact')
     ? "<b>Exact.</b> Azure scored every sound individually, so a weak /r/ is counted where the /r/ actually is."
     : ((p.mode==='mixed')
        ? "<b>Partly exact.</b> "+p.exact_words+" of "+p.words+" words have per-sound scores; the rest are counted at word level (below)."
        : "<b>Counted at word level.</b> These recordings kept only Azure's verdict per <i>word</i>, so every tracked sound inside a word it flagged is charged for it. "+
          "The <i>said N×</i> counts are exact; the failure counts are an upper bound, and a sound sharing words with a genuinely bad one inherits blame. "+
          "Compare the ratios, not the raw rates.");

   var T=D.trend||{};
   body().innerHTML =
     trendCard('pron', 'Failure rate over time',
       "Lower is better. The dashed grey line is every tracked sound together; tick a "+
       "sound to follow it. Each point is a rolling "+(T.days||7)+" days ending on the "+
       "date shown, so the newest point always has a full week behind it. A window with "+
       "fewer than "+T.min_week+" attempts of a sound is skipped — one slip would move it "+
       "by more than "+Math.round(100/(T.min_week||10))+" points — and the line jumps it "+
       "as a faint dotted segment.",
       (T.skipped
         ? "Charting the <b>"+T.recordings+"</b> recordings with per-sound scores. <b>"+
           T.skipped+"</b> are left out because they are still measured at word level, and "+
           "the two methods give different numbers for identical speech — mixing them would "+
           "draw a cliff where the measurement changed, not where your speaking did. Run "+
           "<code>backfill_phonemes.py</code> to bring them in."
         : "All "+T.recordings+" scored recordings, measured the same way.")+trimNote(T)) +
     "<div class='card'><b>Pronunciation — how often each sound goes wrong</b>"+
       "<p class='hint' style='margin:6px 0 2px'>"+honest+"</p>"+
       "<p class='hint' style='margin:2px 0 10px'>From <b>"+p.recordings+"</b> scored recordings · <b>"+
         p.words+"</b> words · <b>"+p.slots+"</b> tracked sounds. Your average failure rate is <b>"+
         p.baseline+"%</b> — that is the tick on each bar. Sorted by how confidently a sound beats it.</p>"+
       soundRows(p)+"</div>";
   seed3('pron'); drawChart('pron');
 }

 function gstats(){
   var D=window.ERRSTATS||{}, g=D.gram||{}, T=D.gtrend||{};
   if(!g.rows||!g.rows.length){ body().innerHTML="<div class='card'>Nothing logged yet — "+
     "analyze a recording with grammar feedback switched on.</div>"; return; }
   body().innerHTML =
     trendCard('gram', 'Mistakes over time',
       "Lower is better. The dashed grey line is every type together; tick a type to follow "+
       "it. Each point is a rolling "+(T.days||7)+" days ending on the date shown, so the "+
       "newest point always has a full week behind it and today's recording is already in "+
       "it. Every point is the same length, so distance along the axis is time — a stretch "+
       "you did not record in shows up as space, not as a missing tick.",
       "From <b>"+T.recordings+"</b> recordings across <b>"+(T.weeks||[]).length+
       "</b> windows."+trimNote(T)+" These come from the language model, so "+
       "changing the analyser or its strictness nudges the counts a little on its own — "+
       "read a steady slope, not a single step.") +
     "<div class='card'><b>What you say wrong — by type</b>"+
       "<p class='hint' style='margin:6px 0 10px'>"+g.total+" distinct findings across "+g.spoken+
         " assessed words. <i>Per 1k</i> is the comparable number — raw counts just track how much you recorded.</p>"+
       "<table><tr><th>Type</th><th>Times</th><th>Share</th><th>Per 1k words</th><th></th></tr>"+
       (g.rows||[]).map(function(r,i){
         return "<tr><td><b>"+S.esc(r.label)+"</b></td><td>"+r.n+"</td><td>"+r.share+"%</td>"+
           "<td><b>"+r.per_1k+"</b></td>"+
           "<td><button class='btn small' onclick=\"GX.eg("+i+")\">examples</button></td></tr>"+
           "<tr id='eg-"+i+"' style='display:none'><td colspan='5'>"+
             (r.examples||[]).map(function(x){
               return "<div style='margin:4px 0'><span class='bad'>"+S.esc(x.said)+"</span> → "+
                 S.esc(x.correction)+"<br><span class='hint'>"+S.esc(x.rule)+"</span></div>";
             }).join('')+"</td></tr>";
       }).join('')+"</table></div>";
   seed3('gram'); drawChart('gram');
 }
 // Logging a mistake by hand is not a third of this panel — it is one button on
 // the log it writes into, revealed only when asked for.
 var ADD=false;
 function addBar(){
   return "<div class='bsfilter'><button class='btn small' onclick='GX.toggleAdd()'>"+
     (ADD?"✕ Cancel":"➕ Log a mistake by hand")+"</button></div>"+
     (ADD? "<div class='card'>"+
       "<label>You said (wrong)<br><input id='gx-s' style='width:100%'></label>"+
       "<label style='display:block;margin-top:8px'>Correction<br><input id='gx-c' style='width:100%'></label>"+
       "<label style='display:block;margin-top:8px'>Rule / tag<br><input id='gx-r' style='width:100%' placeholder='e.g. article, preposition'></label>"+
       "<button class='btn' style='margin-top:12px' onclick='GX.add()'>Save</button> "+
       "<span id='gx-msg' class='hint'></span></div>" : "");
 }
 function log(){ var all=recs().map(function(x){return {said:x.said,correction:x.correction,rule:x.rule,src:x.from||'recording'};})
     .concat(seed().map(function(x){return {said:x.said,correction:x.correction,rule:x.rule,src:'Mandarin L1'};}))
     .concat(manual().map(function(x){return {said:x.said,correction:x.correction,rule:x.rule,src:'manual'};}));
   if(!all.length){body().innerHTML=addBar()+"<div class='card'>No grammar errors yet — analyze a recording or log one by hand.</div>";return;}
   var by={}; all.forEach(function(x){(by[x.rule||'misc']=by[x.rule||'misc']||[]).push(x);});
   var mastered=S.get('gram_mastered',[])||[];
   var keys=Object.keys(by).filter(function(k){return mastered.indexOf(k)<0;})
     .sort(function(a,b){return by[b].length-by[a].length;});
   var restoreBar=mastered.length?("<div class='bsfilter'><span class='hint'>"+mastered.length+
     " pattern(s) mastered</span> <button class='btn small' onclick='GX.restore()'>↺ Restore all</button></div>"):"";
   if(!keys.length){ body().innerHTML=addBar()+restoreBar+"<div class='card'>🎉 All grammar patterns mastered — restore any time.</div>"; return; }
   body().innerHTML=addBar()+restoreBar+keys.map(function(r){return "<div class='card'>"+
     "<div style='display:flex;justify-content:space-between;align-items:center;gap:8px'>"+
       "<b>"+S.esc(r)+" · "+by[r].length+"×</b>"+
       "<button class='btn small' data-rule=\""+S.esc(r)+"\" onclick='GX.master(this)' "+
       "style='background:rgba(67,197,158,.18);border-color:var(--good);color:var(--good)'>✓ Mastered</button></div>"+
     "<table style='margin-top:8px'><tr><th>You said</th><th>Correction</th><th></th></tr>"+
     by[r].map(function(x){return "<tr><td class='bad'>"+S.esc(x.said)+"</td><td>"+S.esc(x.correction)+"</td><td class='hint'>"+S.esc(x.src)+"</td></tr>";}).join('')+"</table></div>";}).join(''); }

 window.GX={ add:function(){var s=document.getElementById('gx-s').value.trim(); if(!s){document.getElementById('gx-msg').textContent='Required';return;}
   var a=manual(); a.push({said:s,correction:document.getElementById('gx-c').value.trim(),rule:document.getElementById('gx-r').value.trim()||'misc'}); S.set('gram_log',a);
   document.getElementById('gx-msg').textContent='Saved ✓'; ['gx-s','gx-c','gx-r'].forEach(function(i){document.getElementById(i).value='';}); },
   toggleAdd:function(){ ADD=!ADD; render(); },
   eg:function(i){ var t=document.getElementById('eg-'+i); if(t) t.style.display = t.style.display==='none'?'':'none'; },
   master:function(btn){ var r=(btn&&btn.getAttribute)?btn.getAttribute('data-rule'):btn;
     var m=S.get('gram_mastered',[])||[]; if(r&&m.indexOf(r)<0)m.push(r); S.set('gram_mastered',m); render(); },
   restore:function(){ S.set('gram_mastered',[]); render(); } };
 window.gxMode=function(btn){MODE=btn.getAttribute('data-m'); document.querySelectorAll('#grammar .drillnav .btn').forEach(function(b){b.classList.remove('active');}); btn.classList.add('active'); render();};
 if(document.getElementById('gx-body')) render();
})();
"""

_LISTEN_JS = r"""
(function(){var S=window.SkillStore; var STATE={mode:'vowels',rate:0.8,cur:null,showIPA:!!S.get('ls_ipa',false),sub:(S.get('ls_sub','words')||'words')};
 function ph(kind){ return (window.PHONEMES||[]).filter(function(p){return p.kind===kind;}); }
 function phrases(){ var d=(window.DAILY_PHRASES||[]).map(function(t){return {text:t,note:''};});
   var extra=(window.SOUND_DATA||[]).filter(function(x){return x.type!=='segmental'&&x.example;}).map(function(x){var e=(x.example||'').split('→')[0].trim(); return {text:e||x.written,note:x.soundsLike};});
   return d.concat(extra); }
 function body(){ return document.getElementById('ls-body'); }
 function shuffle(a){for(var i=a.length-1;i>0;i--){var j=Math.floor(Math.random()*(i+1));var t=a[i];a[i]=a[j];a[j]=t;}return a;}
 function rec(kind,ok){ var key='listen:'+kind; var st=S.get('ls_stats',{}); var c=st[key]||{a:0,c:0}; c.a++; if(ok)c.c++; st[key]=c; S.set('ls_stats',st); }
 // pick a phoneme weighted by need: fewer attempts + lower correct rate => higher chance
 function weightedPick(list){ var st=S.get('ls_stats',{});
   var w=list.map(function(p){ var c=st['listen:'+p.ipa]||{a:0,c:0}; var a=c.a; var rate=a?c.c/a:0;
     return 1 + 3*(1/(1+a)) + 3*(1-rate); });
   var total=w.reduce(function(x,y){return x+y;},0), r=Math.random()*total, acc=0;
   for(var i=0;i<list.length;i++){ acc+=w[i]; if(r<=acc) return list[i]; }
   return list[list.length-1]; }
 // ---- vowels / consonants: choose what you heard ----
 function renderPh(kind){
   var list=ph(kind==='vowel'?'vowel':'consonant'); if(!list.length){ body().innerHTML="<div class='card'>No items.</div>"; return; }
   var p=weightedPick(list);
   var sets=p.sets||[{word:p.word,options:p.options}];
   var set=sets[Math.floor(Math.random()*sets.length)];
   var it={ipa:p.ipa, solo:p.solo, word:set.word, options:set.options}; STATE.cur=it;
   var opts=shuffle(it.options.slice());
   var say=(STATE.sub==='alone'?(it.solo||it.word):it.word);
   body().innerHTML="<div class='card'><div class='hint'>"+(kind==='vowel'?'Vowel':'Consonant')+"<span class='lsipa'> "+S.esc(it.ipa)+"</span></div>"+
     "<div class='drillnav' style='margin:6px 0'>"+
       "<button class='btn small lssub"+(STATE.sub==='words'?' active':'')+"' onclick='LS.sub(\"words\")'>🔤 In a word</button>"+
       "<button class='btn small lssub"+(STATE.sub==='alone'?' active':'')+"' onclick='LS.sub(\"alone\")'>🎵 Sound alone</button></div>"+
     "<p class='sub'>Listen, then choose the word you heard.</p>"+
     "<button class='btn lsplay-main' data-say=\""+S.esc(say)+"\">🔊 Play</button> <button class='btn small lsplay-main' data-say=\""+S.esc(say)+"\">↻ Again</button>"+
     "<div class='ssopts' style='margin-top:14px'>"+opts.map(function(o){var ip=(window.PHONEME_IPA||{})[o.toLowerCase()]||''; return "<button class='btn lsopt' data-w=\""+S.esc(o)+"\">"+S.esc(o)+(ip?" <span class='lsipa' style='opacity:.7;font-weight:400'>"+S.esc(ip)+"</span>":"")+"</button>";}).join('')+"</div>"+
     "<div class='lsrev'></div></div>";
   setTimeout(function(){S.speak(say,0.95);},200);
 }
 function answerPh(w){ var it=STATE.cur; if(!it)return; var ok=w===it.word;
   body().querySelectorAll('.lsopt').forEach(function(b){var bw=b.getAttribute('data-w'); b.style.background=bw===it.word?'#43c59e':(bw===w?'#ff6b6b':'#1f3542'); b.style.color=(bw===it.word||bw===w)?'#08222b':'#9aa3bf'; b.disabled=true;});
   rec(it.ipa, ok);
   body().querySelector('.lsrev').innerHTML="<div class='card' style='margin-top:10px'>"+(ok?"<b style='color:#43c59e'>✓ "+S.esc(it.word)+"</b>":"<b style='color:#ff6b6b'>It was “"+S.esc(it.word)+"”</b>")+
     " — "+S.esc(it.ipa)+" as in "+S.esc(it.word)+"."+
     "<div style='margin-top:8px'><button class='btn' onclick='LS.next()'>Next →</button></div></div>";
 }
 // ---- phrases: type what you hear ----
 function renderPhrase(keep){
   var list=phrases(); if(!list.length){ body().innerHTML="<div class='card'>No phrases.</div>"; return; }
   if(!keep) STATE.cur=list[Math.floor(Math.random()*list.length)];
   var it=STATE.cur;
   body().innerHTML="<div class='card'><div class='hint'>Speed</div><div class='drillnav'>"+
     [['0.6×',0.6],['0.8×',0.8],['1×',1.0]].map(function(s){return "<button class='btn small"+(STATE.rate===s[1]?' active':'')+"' onclick='LS.rate("+s[1]+")'>"+s[0]+"</button>";}).join('')+"</div>"+
     "<button class='btn' onclick='LS.play()'>🔊 Play sentence</button>"+
     "<label style='display:block;margin-top:12px'>Type what you hear<br><input id='ls-in' style='width:100%' autocomplete='off'></label>"+
     "<button class='btn' style='margin-top:10px' onclick='LS.check()'>Check</button> "+
     "<button class='btn small' onclick='LS.reveal()'>Reveal</button><div id='ls-res'></div></div>";
 }
 function score(a,b){a=S.norm(a).split(' ');b=S.norm(b).split(' ');var set={};b.forEach(function(w){set[w]=(set[w]||0)+1;});var hit=0;a.forEach(function(w){if(set[w]){hit++;set[w]--;}});return Math.round(hit/Math.max(b.length,1)*100);}
 function renderStats(){
   var st=S.get('ls_stats',{});
   function tbl(kind){
     var items=(window.PHONEMES||[]).filter(function(p){return p.kind===kind;}).map(function(p){
       var c=st['listen:'+p.ipa]||{a:0,c:0}; return {p:p,a:c.a,c:c.c,rate:c.a?c.c/c.a:null}; });
     items.sort(function(x,y){ var kx=(x.rate==null?2:x.rate), ky=(y.rate==null?2:y.rate); return kx-ky; });
     var rows=items.map(function(o){
       var r=o.rate==null?null:Math.round(o.rate*100);
       var col=r==null?'#5a6080':r>=80?'#43c59e':r>=60?'#ffb454':'#ff6b6b';
       return "<tr><td><b style='font-family:ui-monospace,monospace'>"+S.esc(o.p.ipa)+"</b> "+
         "<button class='btn small' data-say=\""+S.esc(o.p.word)+"\" style='padding:2px 8px'>🔊 "+S.esc(o.p.word)+"</button></td>"+
         "<td style='text-align:center'>"+o.a+"</td>"+
         "<td style='text-align:center'>"+(o.a?o.c:'—')+"</td>"+
         "<td style='text-align:center'><b style='color:"+col+"'>"+(r==null?'—':r+'%')+"</b></td></tr>";
     }).join('');
     return "<table class='pwt'><tr><th>Sound</th><th style='text-align:center'>Heard</th>"+
       "<th style='text-align:center'>Right</th><th style='text-align:center'>Correct rate</th></tr>"+rows+"</table>";
   }
   var tot={a:0,c:0}; Object.keys(st).forEach(function(k){ if(k.indexOf('listen:')===0){ tot.a+=st[k].a; tot.c+=st[k].c; } });
   var avg=tot.a?Math.round(tot.c/tot.a*100):null;
   body().innerHTML="<div class='card' style='display:flex;gap:22px;flex-wrap:wrap;align-items:center'>"+
       "<span><b style='font-size:22px'>"+tot.a+"</b> <span class='hint'>total heard</span></span>"+
       "<span><b style='font-size:22px'>"+(avg==null?'—':avg+'%')+"</b> <span class='hint'>overall correct</span></span>"+
       "<span style='flex:1'></span><button class='btn small' onclick='LS.resetStats()'>Reset stats</button></div>"+
     "<p class='sub'>Weakest first. Green ≥80%, red &lt;60% — drill the red ones.</p>"+
     "<h2>Vowels</h2>"+tbl('vowel')+
     "<h2 style='margin-top:22px'>Consonants</h2>"+tbl('consonant');
 }
 function render(){
   document.getElementById('ls-head').innerHTML="<div class='drillnav'>"+
     "<button class='btn small"+(STATE.mode==='vowels'?' active':'')+"' onclick='LS.mode(\"vowels\")'>👂 Vowels (20)</button>"+
     "<button class='btn small"+(STATE.mode==='consonants'?' active':'')+"' onclick='LS.mode(\"consonants\")'>👂 Consonants (24)</button>"+
     "<button class='btn small"+(STATE.mode==='phrases'?' active':'')+"' onclick='LS.mode(\"phrases\")'>🗣️ Daily phrases</button>"+
     "<button class='btn small"+(STATE.mode==='stats'?' active':'')+"' onclick='LS.mode(\"stats\")'>📊 Stats</button>"+
     ((STATE.mode==='vowels'||STATE.mode==='consonants')?"<span style='flex:1'></span><button id='ls-ipa-btn' class='btn small' onclick='LS.toggleIPA()'>"+(STATE.showIPA?'Hide IPA':'Show IPA')+"</button>":"")+"</div>";
   if(STATE.mode==='vowels') renderPh('vowel'); else if(STATE.mode==='consonants') renderPh('consonant'); else if(STATE.mode==='stats') renderStats(); else renderPhrase(false);
   var sec=document.getElementById('listening'); if(sec) sec.classList.toggle('ipa-on',STATE.showIPA);
 }
 window.LS={ mode:function(m){STATE.mode=m; render();},
   resetStats:function(){ if(confirm('Clear your listening stats?')){ localStorage.removeItem('ls_stats'); render(); } },
   sub:function(m){ STATE.sub=m; S.set('ls_sub',m); var it=STATE.cur; if(!it)return;
     var say=(m==='alone'?(it.solo||it.word):it.word);
     document.querySelectorAll('#ls-body .lsplay-main').forEach(function(b){ b.setAttribute('data-say',say); });
     document.querySelectorAll('#ls-body .lssub').forEach(function(b){ b.classList.toggle('active', b.getAttribute('onclick').indexOf('\"'+m+'\"')>=0); });
     S.speak(say,0.95); },
   toggleIPA:function(){ STATE.showIPA=!STATE.showIPA; S.set('ls_ipa',STATE.showIPA);
     var sec=document.getElementById('listening'); if(sec) sec.classList.toggle('ipa-on',STATE.showIPA);
     var b=document.getElementById('ls-ipa-btn'); if(b) b.textContent=STATE.showIPA?'Hide IPA':'Show IPA'; },
   rate:function(r){STATE.rate=r; renderPhrase(true);},
   play:function(){ S.speak(STATE.cur.text, STATE.rate); },
   check:function(){var v=document.getElementById('ls-in').value; var sc=score(v,STATE.cur.text); var col=sc>=80?'#43c59e':sc>=50?'#ffb454':'#ff6b6b';
     document.getElementById('ls-res').innerHTML="<p><b style='color:"+col+"'>"+sc+"% of words caught</b></p>";},
   reveal:function(){var it=STATE.cur; document.getElementById('ls-res').innerHTML="<div class='card' style='margin-top:8px'><b>"+S.esc(it.text)+"</b>"+(it.note?"<p class='hint'>sounds like: "+S.esc(it.note)+"</p>":"")+"<button class='btn' style='margin-top:6px' onclick='LS.next()'>Next →</button></div>";},
   next:function(){render();} };
 document.addEventListener('click',function(e){var b=e.target.closest&&e.target.closest('.lsopt'); if(b&&!b.disabled)answerPh(b.getAttribute('data-w'));});
 if(document.getElementById('ls-body')) render();
})();
"""

_REGISTER_JS = r"""
(function(){var S=window.SkillStore;
 var SC=[
  {scenario:"Asking your boss for a deadline extension",responses:{casual:"Hey, can I get a few more days on this?",neutral:"Would it be possible to extend the deadline by a couple of days?",formal:"I'd like to request a short extension to ensure the work meets our standards."},note:"With a manager, neutral is safest; casual can read as flippant."},
  {scenario:"Declining an invitation",responses:{casual:"Ah, can't make it this time!",neutral:"Thanks for the invite — I won't be able to make it.",formal:"Thank you for the kind invitation; regrettably I'm unable to attend."},note:"Formal can sound cold among friends; casual can seem dismissive at work."},
  {scenario:"Asking someone to repeat themselves",responses:{casual:"Sorry, what?",neutral:"Sorry, could you say that again?",formal:"I beg your pardon — could you repeat that?"},note:"'Sorry, what?' is fine with friends, abrupt with strangers."},
  {scenario:"Disagreeing in a meeting",responses:{casual:"Nah, I don't think so.",neutral:"I see it a bit differently.",formal:"I'd respectfully offer a different perspective."},note:"Soften disagreement upward; 'Nah' reads as dismissive."},
  {scenario:"Thanking someone for help",responses:{casual:"Thanks a ton!",neutral:"Thanks so much for your help.",formal:"I really appreciate you taking the time to help."},note:""},
  {scenario:"Introducing yourself at the start of an interview",responses:{casual:"Hi, I'm Mei — good to meet you!",neutral:"Hi, I'm Mei. Thanks for taking the time to meet with me today.",formal:"Good morning. My name is Mei Lin; thank you for the opportunity to speak with you."},note:"Aim for neutral-warm — sound human and confident, not scripted or stiff."},
  {scenario:"Describing a past project you're proud of",responses:{casual:"So basically I made this app that scores your spoken English.",neutral:"One project I'm proud of is an app that analyzes and scores spoken English.",formal:"A representative project of mine is a speaking-assessment application that evaluates pronunciation and fluency."},note:"Neutral is ideal in interviews: clear and confident, not over-rehearsed."},
  {scenario:"Explaining why you're looking for a new role",responses:{casual:"Yeah, I just wanted something new.",neutral:"I'm looking for a role with more room to grow in quant research.",formal:"I'm seeking a position that better aligns with my long-term focus on quantitative research."},note:"Stay positive and forward-looking; never criticize a past employer."},
  {scenario:"Answering a question you don't know",responses:{casual:"Uh, no idea honestly.",neutral:"I'm not sure, but here's how I'd approach finding out.",formal:"I don't have that at hand, though I'd be glad to reason through it."},note:"Never bluff — showing your approach ('here's how I'd figure it out') scores well."},
  {scenario:"Asking about next steps at the end of an interview",responses:{casual:"So what happens now?",neutral:"Could you tell me what the next steps in the process are?",formal:"May I ask what the timeline and next steps look like?"},note:"Always ask — it signals genuine interest. Neutral fits perfectly."},
  {scenario:"Following up after no reply",responses:{casual:"Hey, just checking in on this!",neutral:"Just following up to see if there's any update.",formal:"I wanted to follow up politely regarding my previous message."},note:"One gentle nudge is fine; keep it short and low-pressure."},
  {scenario:"Admitting a mistake at work",responses:{casual:"My bad, I messed that up.",neutral:"Sorry, that was my mistake — I'll fix it right away.",formal:"I apologize for the error; I'll correct it immediately."},note:"Own it briefly, then pivot to the fix. Over-apologizing weakens you."},
  {scenario:"Interrupting politely in a discussion",responses:{casual:"Oh wait, quick thing —",neutral:"Sorry to jump in — can I add something?",formal:"If I may briefly interject —"},note:"'If I may' is quite formal; 'Sorry to jump in' fits most meetings."},
  {scenario:"Giving your opinion",responses:{casual:"I reckon we should just ship it.",neutral:"I think we should go ahead and ship it.",formal:"My view is that we're ready to proceed with the release."},note:"'Reckon' is casual; neutral 'I think' is the safe default at work."}
 ];
 var IDX=0,LEVEL=1,L=['casual','neutral','formal'];
 function body(){return document.getElementById('rg-body');}
 function render(){var sc=SC[IDX],lv=L[LEVEL],txt=sc.responses[lv];
   body().innerHTML="<div class='card'><div class='hint'>Scenario "+(IDX+1)+"/"+SC.length+"</div>"+
     "<div style='font-size:20px;font-weight:700;margin:6px 0'>"+S.esc(sc.scenario)+"</div>"+
     "<input type='range' min='0' max='2' value='"+LEVEL+"' oninput='RG.slide(this.value)' style='width:280px'>"+
     "<div class='hint'>casual · neutral · formal</div>"+
     "<div class='card' style='margin-top:10px;border-left:4px solid var(--accent)'><b style='text-transform:capitalize'>"+lv+"</b>"+
       "<p style='font-size:18px'>"+S.esc(txt)+"</p><button class='btn small' data-say=\""+S.esc(txt)+"\">🔊</button></div>"+
     (sc.note?"<p class='hint'>"+S.esc(sc.note)+"</p>":"")+
     "<button class='btn' style='margin-top:8px' onclick='RG.next()'>Next scenario →</button></div>"; }
 window.RG={ slide:function(v){LEVEL=parseInt(v,10); render();}, next:function(){IDX=(IDX+1)%SC.length; LEVEL=1; render();} };
 if(document.getElementById('rg-body')) render();
})();
"""


_CHINGLISH_SEED = [
    {"headword": "turn on / turn off (the light)", "type": "collocation",
     "definition": "NOT 'open/close the light' (开灯/关灯). 'open' is for doors and boxes, not lights or devices.",
     "example": "Can you turn on the light?"},
    {"headword": "use my phone / be on my phone", "type": "collocation",
     "definition": "NOT 'play phone' (玩手机).", "example": "Stop being on your phone at dinner."},
    {"headword": "delicious", "type": "collocation",
     "definition": "Already means very tasty — don't say 'very delicious'. Use 'really good' / 'so good'.",
     "example": "This soup is delicious."},
    {"headword": "What's the word for…?", "type": "collocation",
     "definition": "NOT 'how to say' (怎么说). Or: 'How do you say X in English?'",
     "example": "What's the word for 筷子? — Chopsticks."},
    {"headword": "I really like it", "type": "collocation",
     "definition": "NOT 'I very like it' (我很喜欢). 'very' can't modify a verb — use 'really'.",
     "example": "I really like this song."},
    {"headword": "packed / crowded", "type": "idiom",
     "definition": "Natural way to say 人山人海.", "example": "The mall was packed on Saturday."},
    {"headword": "have a meeting", "type": "collocation",
     "definition": "NOT 'open a meeting' (开会). You 'have' or 'hold' a meeting.",
     "example": "We have a meeting at 3."},
    {"headword": "put on weight / gain weight", "type": "collocation",
     "definition": "For 长胖. 'grow fat' sounds blunt.", "example": "I put on weight over the holidays."},
]


def _vocab_panel():
    return ("<section id='vocab' class='tabpanel hidden'>"
            "<h1>Surrounding vocabulary</h1>"
            "<p class='sub'>The English for what's actually around you: every word your photos "
            "turn up, plus anything you capture by hand. Review it as flashcards, and check it "
            "against what you can already say and understand.</p>"
            "<div class='drillnav'>"
            "<button class='btn small active' data-m='review' onclick='vxMode(this)'>🃏 Review</button>"
            "<button class='btn small' data-m='add' onclick='vxMode(this)'>➕ Capture</button>"
            "<button class='btn small' data-m='all' onclick='vxMode(this)'>📚 All</button>"
            "<button class='btn small' data-m='cover' onclick='vxMode(this)'>📊 Coverage</button>"
            "</div><div id='vx-body'></div></section>"
            "<script>window.CHINGLISH_SEED=%s;window.VOCAB_SCENARIOS=%s;%s</script>"
            % (json.dumps(_CHINGLISH_SEED, ensure_ascii=False).replace("</", "<\\/"),
               json.dumps(VOCAB_SCENARIOS, ensure_ascii=False), _VOCAB_JS))


_PHOTO_DESC_JS = r"""
(function(){var S=window.SkillStore; var MODE='capture'; var OPEN=null;
 var P={blob:null, url:null, desc:'', items:null, busy:false, err:'', saved:''};
 var R={said:'', res:null, busy:false, err:'', revealed:false};
 function photos(){ return S.get('ec_photos',[]); }
 // ec_photos entries get EDITED (a new recall attempt) and DELETED, which the
 // server's append-merge can't express — so every write replaces the key.
 function writePhotos(fn){ S.update('ec_photos',[],fn,true); }

 // ---- the word ledger -----------------------------------------------------
 // ec_photos is a working set: you delete photos to reclaim space. ec_seen is
 // the permanent record of what has actually been around you, so the Coverage
 // report only ever grows. Keyed by word; `pids` is the set of photos it was
 // met in, which is both its frequency and its link back to the thumbnails.
 // (Words captured by hand land in the same ledger under `cids`, written by the
 // Surrounding vocabulary panel.)
 function tokenise(text){
   var out=[];
   (text||'').toLowerCase().replace(/[^a-z' ]+/g,' ').split(/\s+/).forEach(function(w){
     w=w.replace(/^'+|'+$/g,''); if(w.length>1||w==='a'||w==='i') out.push(w); });
   return out;
 }
 // Fold one photo into the ledger. Idempotent on the photo id, so re-running it
 // over photos already recorded changes nothing — that's what makes it safe to
 // call as a migration for photos saved before the ledger existed.
 function foldPhoto(seen, p){
   var changed=false;
   (p.items||[]).forEach(function(it){
     tokenise(it.headword).forEach(function(w){
       var e=seen[w] || (seen[w]={pids:[], cids:[], forms:{}, first:p.d||'', last:p.d||''});
       if(!e.pids) e.pids=[];
       if(e.pids.indexOf(p.id)<0){ e.pids.push(p.id); changed=true; }
       if(!e.forms[it.headword]){ e.forms[it.headword]=1; changed=true; }
       if(p.d && (!e.first || p.d<e.first)){ e.first=p.d; changed=true; }
       if(p.d && (!e.last  || p.d>e.last )){ e.last=p.d;  changed=true; }
     });
   });
   return changed;
 }
 function recordSeen(entry){
   S.update('ec_seen',{},function(cur){
     cur=cur||{}; foldPhoto(cur, entry); return cur; }, true);
 }
 function body(){ return document.getElementById('pd-body'); }
 function render(){ MODE==='review'?review():MODE==='all'?all():capture(); }
 function fmtDate(d){ return d||''; }

 function downscale(file,maxDim,cb){
   var img=new Image(), url=URL.createObjectURL(file);
   img.onload=function(){
     URL.revokeObjectURL(url);
     var w=img.width,h=img.height,scale=Math.min(1,maxDim/Math.max(w,h));
     var cw=Math.max(1,Math.round(w*scale)),ch=Math.max(1,Math.round(h*scale));
     var cv=document.createElement('canvas'); cv.width=cw; cv.height=ch;
     cv.getContext('2d').drawImage(img,0,0,cw,ch);
     cv.toBlob(function(b){cb(b);},'image/jpeg',0.85);
   };
   img.onerror=function(){ URL.revokeObjectURL(url); cb(null); };
   img.src=url;
 }

 function wordTable(list){
   if(!list||!list.length) return '';
   return "<table style='margin-top:10px'><tr><th>Word</th><th>Meaning</th><th>Example</th></tr>"+
     list.map(function(it){
       return "<tr><td><b>"+S.esc(it.headword)+"</b><div class='hint'>"+S.esc(it.type||'')+"</div></td>"+
              "<td>"+S.esc(it.definition||'')+"</td><td class='hint'>"+S.esc(it.example||'')+"</td></tr>";
     }).join('')+"</table>";
 }

 // ---- capture -------------------------------------------------------------
 function capture(){
   var prev=P.url?("<img src='"+P.url+"' style='max-width:320px;border-radius:10px;display:block;margin:10px 0'>"):"";
   var go=P.blob?("<button class='btn' onclick='PD.analyze()' "+(P.busy?'disabled':'')+">"+
     (P.busy?'Looking…':'🔎 Describe this photo')+"</button>"):"";
   var err=P.err?("<p class='hint' style='color:var(--bad)'>"+S.esc(P.err)+"</p>"):"";
   var out='';
   if(P.desc||(P.items&&P.items.length)){
     out="<div class='card' style='border-left:3px solid var(--accent)'>"+
       "<b>What's in this photo</b> <span class='hint'>· the description you'll be tested against</span>"+
       "<p style='margin:6px 0 0'>"+S.esc(P.desc)+"</p></div>"+
       "<div class='card'><b>"+((P.items||[]).length)+" words &amp; phrases</b>"+
       " <span class='hint'>· "+S.esc((P.items&&P.items[0]&&P.items[0].scenario)||'General')+
       "</span>"+wordTable(P.items)+"</div>"+
       "<button class='btn' onclick='PD.save()' "+(P.busy?'disabled':'')+">💾 Save this photo</button>"+
       "<span class='hint' style='margin-left:10px'>"+S.esc(P.saved)+"</span>";
   }
   body().innerHTML="<div class='card'><p class='sub' style='margin-top:0'>Take a photo of what's "+
     "around you. The vision model writes a short description and pulls out the words worth "+
     "knowing. Save it, and later you can describe the same photo from memory and see how close "+
     "you got.</p>"+
     "<input type='file' id='pd-file' accept='image/*' capture='environment' style='display:none' onchange='PD.chosen(this)'>"+
     "<button class='btn' onclick=\"document.getElementById('pd-file').click()\">📷 Take / choose photo</button> "+
     go+prev+err+"</div>"+out;
 }

 // ---- review --------------------------------------------------------------
 function thumbs(){
   var ps=photos().filter(function(p){ return !p.imgGone; });
   var retired=photos().length-ps.length;
   var note=retired?("<p class='hint'>"+retired+" photo"+(retired===1?" has":"s have")+
     " had the image freed — "+(retired===1?"its":"their")+
     " words and scores are kept, see All photos.</p>"):"";
   if(!ps.length) return note+"<div class='card'>No photos to review"+
     (retired?" — every saved image has been freed.":" yet — capture one first.")+"</div>";
   return "<div style='display:flex;flex-wrap:wrap;gap:10px'>"+ps.map(function(p){
     var best=(p.recalls||[]).reduce(function(m,r){return Math.max(m,r.score||0);},0);
     var badge=(p.recalls&&p.recalls.length)?("<div class='hint'>best "+best+"/100</div>"):
               "<div class='hint'>not tried</div>";
     return "<div style='width:150px;cursor:pointer' onclick=\"PD.open('"+p.id+"')\">"+
       "<img src='"+S.esc(p.img)+"' style='width:150px;height:110px;object-fit:cover;border-radius:8px'>"+
       "<div class='hint' style='margin-top:4px'>"+S.esc(fmtDate(p.d))+"</div>"+badge+"</div>";
   }).join('')+"</div>"+note;
 }
 function review(){
   var p=OPEN&&photos().filter(function(x){return x.id===OPEN;})[0];
   if(p&&p.imgGone){
     body().innerHTML="<button class='btn small' onclick='PD.back()'>← All photos</button>"+
       "<div class='card' style='margin-top:10px'><b>Image freed</b>"+
       "<p class='hint' style='margin:6px 0 0'>The picture for this one was deleted to save "+
       "space, so there's nothing left to describe. Its description, words and scores are "+
       "still here.</p></div>"+
       "<div class='card'><b>The description</b><p style='margin:6px 0 0'>"+S.esc(p.desc)+"</p></div>"+
       "<div class='card'><b>Words from this photo</b>"+wordTable(p.items)+"</div>";
     return;
   }
   if(!p){ OPEN=null; body().innerHTML="<p class='sub'>Pick a photo, then describe it from memory.</p>"+thumbs(); return; }
   var diff='';
   if(R.res){
     diff=renderDiff(R.res);
   }
   var reveal=(R.revealed||R.res)?
     ("<div class='card' style='border-left:3px solid var(--good)'><b>The AI's description</b>"+
      " <button class='btn small' style='margin-left:8px' onclick='PD.say()'>🔊 Hear it</button>"+
      "<p style='margin:6px 0 0'>"+S.esc(p.desc)+"</p></div>"+
      "<div class='card'><b>Words from this photo</b>"+wordTable(p.items)+"</div>"):"";
   var err=R.err?("<p class='hint' style='color:var(--bad)'>"+S.esc(R.err)+"</p>"):"";
   body().innerHTML=
     "<button class='btn small' onclick='PD.back()'>← All photos</button>"+
     "<div class='card' style='margin-top:10px'>"+
       "<img src='"+S.esc(p.img)+"' style='max-width:100%;border-radius:10px;display:block'>"+
       "<p class='hint' style='margin:8px 0 0'>"+S.esc(fmtDate(p.d))+" · "+
       S.esc(p.scenario||'General')+" · "+((p.items||[]).length)+" words</p></div>"+
     "<div class='card'><b>Describe it from memory</b>"+
       "<p class='hint' style='margin:4px 0 8px'>Say it out loud — that's the point — then compare. "+
       "Typing works too if you'd rather not record.</p>"+
       "<button class='btn' id='pd-mic' onclick='PD.mic()'>🎤 Record your description</button>"+
       "<span class='hint' id='pd-mic-status' style='margin-left:8px'></span>"+
       "<textarea id='pd-said' rows='4' style='width:100%;margin-top:10px'>"+S.esc(R.said)+"</textarea>"+
       "<button class='btn' onclick='PD.compare()' "+(R.busy?'disabled':'')+" style='margin-top:8px'>"+
       (R.busy?'Comparing…':'⇄ Compare with AI')+"</button>"+
       "<button class='btn small' style='margin-left:8px' onclick='PD.reveal()'>👁 Just show me</button>"+
       err+diff+"</div>"+reveal;
 }
 function renderDiff(res){
   var html='', missed=[];
   (res.ops||[]).forEach(function(o){
     if(o.op==='equal'){ html+="<span>"+S.esc(o.ref.join(' '))+" </span>"; return; }
     if(o.op==='replace'){ missed=missed.concat(o.ref);
       html+="<span style='color:var(--bad);text-decoration:line-through;opacity:.65'>"+S.esc(o.hyp.join(' '))+"</span> "+
             "<b style='color:var(--good)'>"+S.esc(o.ref.join(' '))+"</b> ";
     } else if(o.op==='delete'){ missed=missed.concat(o.ref);
       html+="<b style='color:var(--warn)'>"+S.esc(o.ref.join(' '))+"</b> ";
     } else if(o.op==='insert'){
       html+="<span style='color:var(--bad);text-decoration:line-through;opacity:.65'>"+S.esc(o.hyp.join(' '))+"</span> ";
     }
   });
   var col=res.score>=90?'var(--good)':(res.score>=70?'var(--warn)':'var(--bad)');
   var out="<p class='score' style='margin-top:12px'>You matched <b style='color:"+col+"'>"+
     res.correct+" of "+res.total+"</b> words · "+res.score+"/100</p>"+
     "<p class='summary' style='line-height:1.9'>"+html+"</p>";
   if(missed.length) out+="<p class='hint'>Words you didn't reach for: <b>"+S.esc(missed.join(', '))+
     "</b>. Matching the AI word-for-word isn't the goal — but these are the ones it found useful.</p>";
   return out;
 }

 // ---- all -----------------------------------------------------------------
 function all(){
   var ps=photos();
   if(!ps.length){ body().innerHTML="<div class='card'>No saved photos yet.</div>"; return; }
   body().innerHTML="<p class='hint'>Whatever you delete here, every word these photos "+
     "produced stays in <b>Surrounding vocabulary → Coverage</b> — that record only ever "+
     "grows.</p>"+
     "<table><tr><th>Photo</th><th>Date</th><th>Scenario</th><th>Words</th>"+
     "<th>Attempts</th><th>Best</th><th></th></tr>"+ps.map(function(p){
       var rc=(p.recalls||[]).length;
       var best=(p.recalls||[]).reduce(function(m,r){return Math.max(m,r.score||0);},0);
       var cell=p.imgGone
         ? "<span class='hint' title='Image freed to save space'>🗄 freed</span>"
         : "<img src='"+S.esc(p.img)+"' style='width:64px;height:44px;object-fit:cover;"+
           "border-radius:5px;cursor:pointer' onclick=\"PD.open('"+p.id+"')\">";
       var free=p.imgGone ? "" :
         "<button class='btn small' title='Delete just the image, keep the words and scores' "+
         "onclick=\"PD.freeImage('"+p.id+"')\">🗄 Free image</button> ";
       return "<tr><td>"+cell+"</td>"+
         "<td>"+S.esc(fmtDate(p.d))+"</td><td>"+S.esc(p.scenario||'General')+"</td>"+
         "<td>"+((p.items||[]).length)+"</td><td>"+rc+"</td><td>"+(rc?best+"/100":"—")+"</td>"+
         "<td style='text-align:right;white-space:nowrap'>"+free+
         "<button class='btn small' style='background:#3a2029;color:#ff9db0' "+
         "onclick=\"PD.del('"+p.id+"')\">&#10005;</button></td></tr>";
     }).join('')+"</table>";
 }

 window.PD={
   chosen:function(input){
     var f=input.files&&input.files[0]; if(!f) return;
     P.err=''; P.desc=''; P.items=null; P.saved='';
     downscale(f,1024,function(b){
       if(!b){ P.err='Could not read that image.'; render(); return; }
       P.blob=b; if(P.url) URL.revokeObjectURL(P.url); P.url=URL.createObjectURL(b); render();
     });
   },
   analyze:function(){
     if(!P.blob||P.busy) return;
     P.busy=true; P.err=''; P.saved=''; render();
     var fd=new FormData(); fd.append('image',P.blob,'photo.jpg');
     fetch('/vocab_photo',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(j){
       P.busy=false;
       if(j.error){ P.err=j.error; render(); return; }
       P.desc=j.description||'';
       P.items=(j.items||[]).map(function(it){return {headword:it.headword||'',definition:it.definition||'',
         example:it.example||'',type:it.type||'collocation',scenario:it.scenario||'General'};});
       if(!P.desc) P.err='The model returned no description — the prompt may have been edited to drop it (Setting Panel → Photo vocabulary prompt).';
       render();
     }).catch(function(){ P.busy=false; P.err='Could not reach the server.'; render(); });
   },
   save:function(){
     if(!P.blob||P.busy) return;
     P.busy=true; P.saved='Saving…'; render();
     var fd=new FormData(); fd.append('image',P.blob,'photo.jpg');
     fetch('/photo_save',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(j){
       P.busy=false;
       if(j.error){ P.saved=''; P.err=j.error; render(); return; }
       var entry={id:j.id, img:j.url, d:S.today(), desc:P.desc,
                  scenario:(P.items&&P.items[0]&&P.items[0].scenario)||'General',
                  items:P.items||[], recalls:[]};
       writePhotos(function(cur){ return [entry].concat(cur||[]); });
       recordSeen(entry);   // permanent, survives deleting the photo later
       P.saved='Saved ✓ — find it under Review.';
       render();
     }).catch(function(){ P.busy=false; P.saved=''; P.err='Could not reach the server.'; render(); });
   },
   open:function(id){ OPEN=id; MODE='review'; R={said:'',res:null,busy:false,err:'',revealed:false};
     setActive('review'); render(); },
   back:function(){ OPEN=null; render(); },
   reveal:function(){ R.revealed=true; render(); },
   say:function(){ var p=photos().filter(function(x){return x.id===OPEN;})[0];
     if(p&&S.speak) S.speak(p.desc,0.95); },
   compare:function(){
     var ta=document.getElementById('pd-said');
     R.said=ta?ta.value:'';
     var p=photos().filter(function(x){return x.id===OPEN;})[0];
     if(!p) return;
     if(!R.said.trim()){ R.err='Say or type your description first.'; render(); return; }
     R.busy=true; R.err=''; render();
     var fd=new FormData(); fd.append('ref',p.desc); fd.append('said',R.said);
     fetch('/photo_compare',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(j){
       R.busy=false;
       if(j.error){ R.err=j.error; render(); return; }
       R.res=j; R.revealed=true;
       writePhotos(function(cur){
         return (cur||[]).map(function(x){
           if(x.id!==p.id) return x;
           var rl=(x.recalls||[]).concat([{d:S.today(),score:j.score,text:R.said}]);
           return Object.assign({},x,{recalls:rl});
         });
       });
       render();
     }).catch(function(){ R.busy=false; R.err='Could not reach the server.'; render(); });
   },
   del:function(id){
     if(!confirm('Delete this photo entirely — image, description and your attempts?\n\n'+
                 'Its words stay in Surrounding vocabulary → Coverage either way. '+
                 'If you only want the disk space back, use "Free image" instead.')) return;
     var fd=new FormData(); fd.append('id',id);
     fetch('/photo_delete',{method:'POST',body:fd}).catch(function(){});
     writePhotos(function(cur){ return (cur||[]).filter(function(x){return x.id!==id;}); });
     if(OPEN===id) OPEN=null;
     render();
   },
   // reclaim the image bytes but keep the description, the word list and every
   // recall score — the photo just retires from the review rotation
   freeImage:function(id){
     if(!confirm('Delete just the image file?\n\nThe description, its words and your '+
                 'recall scores are kept — but without the picture there is nothing '+
                 'left to describe, so this photo drops out of Review.')) return;
     var fd=new FormData(); fd.append('id',id);
     fetch('/photo_delete',{method:'POST',body:fd}).catch(function(){});
     writePhotos(function(cur){ return (cur||[]).map(function(x){
       return x.id===id ? Object.assign({},x,{imgGone:true}) : x; }); });
     render();
   },
   mic:function(){
     var btn=document.getElementById('pd-mic'), st=document.getElementById('pd-mic-status');
     if(btn.classList.contains('on')){ if(window._pdmr) window._pdmr.stop(); return; }
     if(!navigator.mediaDevices){ st.textContent='Recording not supported here.'; return; }
     navigator.mediaDevices.getUserMedia({audio:true}).then(function(stream){
       var mr=new MediaRecorder(stream), chunks=[]; window._pdmr=mr;
       mr.ondataavailable=function(e){chunks.push(e.data);};
       mr.onstop=function(){
         stream.getTracks().forEach(function(t){t.stop();});
         btn.classList.remove('on'); btn.textContent='🎤 Record your description';
         st.textContent='Transcribing…';
         var fd=new FormData();
         fd.append('audio',new File(chunks,'recall.webm',{type:'audio/webm'}),'recall.webm');
         fd.append('model','base'); fd.append('lang','en');
         // saved beside the photos, not loose in the library root
         fd.append('save_folder','PhotoDescriptions/recall');
         fetch('/transcribe',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(j){
           if(j.error){ st.textContent='Transcription failed: '+j.error+' — type it instead.'; return; }
           poll(j.job, Date.now(), st);
         }).catch(function(){ st.textContent='Could not reach the server — type it instead.'; });
       };
       mr.start(); btn.classList.add('on'); btn.textContent='■ Stop';
       st.textContent='Recording — describe the photo…';
     }).catch(function(){ st.textContent='Microphone permission denied.'; });
   }
 };
 function poll(job,t0,st){
   fetch('/tprogress/'+job).then(function(r){return r.json();}).then(function(j){
     if(j.error){ st.textContent='Transcription failed: '+j.error+' — type it instead.'; return; }
     if(j.done){
       R.said=j.text||'';
       var ta=document.getElementById('pd-said'); if(ta) ta.value=R.said;
       st.textContent='Transcribed in '+Math.round((Date.now()-t0)/1000)+'s — fix any slips, then compare.';
       return;
     }
     st.textContent='Transcribing… '+Math.round((Date.now()-t0)/1000)+'s';
     setTimeout(function(){poll(job,t0,st);},1200);
   }).catch(function(){ st.textContent='Lost contact with the transcriber — type it instead.'; });
 }
 function setActive(m){
   MODE=m;
   document.querySelectorAll('#photodesc .drillnav .btn').forEach(function(b){
     b.classList.toggle('active', b.getAttribute('data-m')===m); });
 }
 window.pdMode=function(btn){ setActive(btn.getAttribute('data-m')); OPEN=null; render(); };
 if(document.getElementById('pd-body')) render();
})();
"""


def _photo_desc_panel():
    return ("<section id='photodesc' class='tabpanel hidden'>"
            "<h1>Describe a photo</h1>"
            "<p class='sub'>Photograph what's around you, let the model describe it and name the "
            "vocabulary, then come back later and describe the same photo from memory — out loud — "
            "and see word-for-word how close you got.</p>"
            "<div class='drillnav'>"
            "<button class='btn small active' data-m='capture' onclick='pdMode(this)'>📷 New photo</button>"
            "<button class='btn small' data-m='review' onclick='pdMode(this)'>🧠 Review</button>"
            "<button class='btn small' data-m='all' onclick='pdMode(this)'>🗂 All photos</button>"
            "</div><div id='pd-body'></div></section>"
            "<script>%s</script>" % _PHOTO_DESC_JS)


# ---------------------------------------------------------------------------
# 5b. Speaking error statistics — not "what went wrong" but "how often, out of
#     how many chances". A raw count of θ mistakes is meaningless on its own:
#     the question is how many times you said a θ at all. Everything below is
#     built to produce that denominator.
# ---------------------------------------------------------------------------

_SV = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
       "OW", "OY", "UH", "UW"}
#: Azure and CMUdict do not use the same ARPAbet. Azure emits `ax` for schwa
#: and `dx` for a flapped t; CMUdict writes those AH0 and T. Left unfolded, the
#: same sound is counted as two different ones and every rate is wrong.
_SFIX = {"AX": "AH", "AXR": "ER", "IX": "IH", "UX": "UW", "DX": "T",
         "NX": "N", "EL": "L", "EM": "M", "EN": "N", "WH": "W"}


def _stat_phone(p):
    """One ARPAbet symbol, stress stripped and alphabet folded."""
    p = re.sub(r"\d", "", (p or "").upper())
    return _SFIX.get(p, p)


def _stat_seq(word, stored=None):
    """The phonemes of one spoken word.

    Prefers what Azure says it graded you against; falls back to CMUdict for
    recordings analysed before that was recorded. Falling back matters — it is
    the difference between the report covering 12 recordings and all 43.
    """
    if stored:
        return [_stat_phone(p) for p in stored if p]
    w = re.sub(r"[^a-z']", "", (word or "").lower())
    prons = _cmu().get(w)
    return [_stat_phone(p) for p in prons[0]] if prons else []


def _stat_roles(seq):
    """(phone, role) per slot, where role is what makes a sound hard.

    Position is not cosmetic here. A /l/ before the vowel ("light") and a /l/
    after it ("feel") are different articulations — the second is the dark l
    that Mandarin has no equivalent for — and a stop at the end of a word is
    the one that gets dropped. Counting them together would average the
    problem away.
    """
    vi = [i for i, p in enumerate(seq) if p in _SV]
    out = []
    for i, p in enumerate(seq):
        if p in _SV:
            role = "nucleus"
        elif not vi:
            role = "onset"
        elif i < vi[0]:
            role = "onset"
        elif i > vi[-1]:
            role = "final" if i == len(seq) - 1 else "coda"
        else:
            role = "medial"
        out.append((p, role))
    return out


#: The sounds worth tracking, and why each one is on the list. Keyed so the
#: stats survive relabelling. `hit` decides whether a slot counts as an
#: encounter of this sound.
SOUND_GROUPS = [
    ("th_unvoiced", "θ — think, three, month", "Mandarin has no θ; it usually surfaces as /s/ or /t/.",
     lambda p, r: p == "TH"),
    ("th_voiced", "ð — this, other, breathe", "No ð in Mandarin either; commonly becomes /d/ or /z/.",
     lambda p, r: p == "DH"),
    ("final_stop", "final d t p k b g — need, hope, work", "Mandarin syllables cannot end in a stop, so the ending gets dropped.",
     lambda p, r: r == "final" and p in {"D", "T", "P", "K", "B", "G"}),
    ("dark_l", "dark l — feel, milk, well", "Coda /l/. Mandarin has no syllable-final l; it tends to vanish or turn into a vowel.",
     lambda p, r: p == "L" and r in ("coda", "final")),
    ("clear_l", "clear l — light, believe", "Onset /l/, the easy one — here as a control for the dark l row.",
     lambda p, r: p == "L" and r == "onset"),
    ("r_sound", "/r/ — right, very, world", "English /r/ is not the Mandarin r; the tongue must not touch.",
     lambda p, r: p == "R"),
    ("er_vowel", "/ɜr/ — work, learn, first", "R-coloured vowel. Often produced flat, without the r.",
     lambda p, r: p == "ER"),
    ("ng_sound", "/ŋ/ — thing, going, long", "Exists in Mandarin, but often loses its final closure in English.",
     lambda p, r: p == "NG"),
    ("v_sound", "/v/ — very, love, seven", "Mandarin has no /v/; it drifts toward /w/.",
     lambda p, r: p == "V"),
    ("final_sz", "final s / z — books, needs, is", "Grammatical endings. Dropping them costs plurals and tenses, not just sound.",
     lambda p, r: r == "final" and p in {"S", "Z"}),
    ("final_n", "final n — man, been, soon", "Often merged with /ŋ/ or nasalised into the vowel.",
     lambda p, r: p == "N" and r in ("coda", "final")),
    ("post_alveolar", "ʃ tʃ dʒ — she, church, judge", "Mandarin's nearest sounds are further forward and sound thinner.",
     lambda p, r: p in {"SH", "CH", "JH", "ZH"}),
    ("tense_lax", "iː vs ɪ — seat vs sit", "Mandarin has one high front vowel; English contrasts two.",
     lambda p, r: p in {"IY", "IH"}),
    ("ae_vowel", "/æ/ — cat, bad, hand", "No Mandarin equivalent; usually lands somewhere near /e/.",
     lambda p, r: p == "AE"),
]

#: Below this an Azure phoneme score counts as a weak production. Azure's own
#: banding treats the 60s as borderline, so this is the top of "clearly wrong"
#: rather than a cutoff invented here.
WEAK_BELOW = 60

#: Fewer attempts than this and the row is shown, but not ranked among the rest.
#: Not a statistical threshold — the interval below already handles uncertainty,
#: and it is quite happy to report that 2-of-3 means a real rate above 20%.
#: That is true and still the wrong thing to put at the top of a report: a
#: sound you have attempted three times is not a habit yet, and drilling it
#: instead of the one you get wrong 135 times a month is wasted practice.
MIN_ENCOUNTERS = 25


def _wilson_lo(k, n, z=1.96):
    """Lower bound of the 95% interval on k/n.

    Ranking on the raw rate puts 2-out-of-3 above 120-out-of-1000, which is how
    a report ends up telling you to drill a sound you have said three times.
    """
    if not n:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    return max(0.0, (p + z * z / (2 * n) -
                     z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / den)


def pronunciation_error_stats(items, weak_below=WEAK_BELOW):
    """How often each tracked sound was attempted, and how often it was weak.

    Two levels of evidence, and the report says which one it is using:

    * **exact** — Azure scored every phoneme individually (`pacc`), so a weak
      /r/ is counted where the /r/ actually is.
    * **attributed** — older recordings kept only a word-level verdict, so a
      word Azure flagged is charged to every tracked sound inside it. The
      encounter counts are still exact; the failure counts are an upper bound,
      and a sound that shares words with a genuinely bad one inherits blame it
      does not deserve. Read the ratio, not the absolute rate.
    """
    groups = {k: {"key": k, "label": lab, "note": note, "n": 0, "bad": 0,
                  "exact_n": 0, "words": {}}
              for k, lab, note, _ in SOUND_GROUPS}
    tot_n = tot_bad = 0          # baseline over every tracked slot
    words_seen = exact_words = recordings = 0

    for d in items or []:
        ws = ((d.get("azure") or {}).get("words")) or []
        if not ws:
            continue
        recordings += 1
        for w in ws:
            seq = _stat_seq(w.get("word"), w.get("phones"))
            if not seq:
                continue
            words_seen += 1
            pacc = w.get("pacc") or []
            exact = len(pacc) == len(seq) and any(s >= 0 for s in pacc)
            if exact:
                exact_words += 1
            flagged = w.get("error") == "Mispronunciation"
            for i, (p, role) in enumerate(_stat_roles(seq)):
                score = pacc[i] if exact and i < len(pacc) else -1
                if exact and score < 0:
                    continue                     # Azure declined to score it
                bad = (score < weak_below) if exact else flagged
                tot_n += 1
                tot_bad += bad
                for key, _lab, _note, hit in SOUND_GROUPS:
                    if not hit(p, role):
                        continue
                    g = groups[key]
                    g["n"] += 1
                    g["bad"] += bad
                    g["exact_n"] += exact
                    if bad:
                        ww = (w.get("word") or "").lower()
                        if ww:
                            g["words"][ww] = g["words"].get(ww, 0) + 1

    base = (tot_bad / tot_n) if tot_n else 0.0
    rows = []
    for key, lab, note, _ in SOUND_GROUPS:
        g = groups[key]
        if not g["n"]:
            continue
        rate = g["bad"] / g["n"]
        rows.append({
            "key": key, "label": lab, "note": note,
            "n": g["n"], "bad": g["bad"], "rate": round(100 * rate, 1),
            "lift": round(rate / base, 2) if base else 0,
            "lift_lo": round(_wilson_lo(g["bad"], g["n"]) / base, 2) if base else 0,
            "exact": g["exact_n"] == g["n"] and g["n"] > 0,
            "thin": g["n"] < MIN_ENCOUNTERS,
            # the words that actually went wrong, so a row is drillable
            "examples": [w for w, _c in sorted(g["words"].items(),
                                               key=lambda kv: -kv[1])[:8]],
        })
    rows.sort(key=lambda r: (r["thin"], -r["lift_lo"], -r["n"]))
    return {
        "mode": "exact" if exact_words and exact_words == words_seen else
                ("mixed" if exact_words else "attributed"),
        "exact_words": exact_words, "words": words_seen,
        "recordings": recordings, "slots": tot_n,
        "baseline": round(100 * base, 1), "weak_below": weak_below,
        "rows": rows,
    }


#: Colours for the trend lines. The app's chart hues, re-stepped into the
#: dark-mode lightness band so adjacent series actually separate: the original
#: cyan and green sit ΔE 9.4 apart to normal vision, which is below the
#: readable floor. Verified with the palette validator — all six checks pass on
#: the #172530 chart surface. Assign in this order; never cycle or recolour on
#: filtering, or a line changes meaning when you tick a checkbox.
TREND_COLORS = ["#ee5c5d", "#33a4b9", "#c9820e", "#1dac86", "#a279ea"]

#: A week needs at least this many attempts of a sound before its rate is
#: plotted. The number is not taste: at n attempts one slip moves the rate by
#: 100/n points, so below 10 a single word swings the week by more than ten
#: points and the dot stops being a measurement. Everything from here to 40 is
#: still jumpy rather than wrong, and the chart says so with a hollow dot —
#: that is the marker's whole job, so this floor only has to kill the
#: degenerate cells. Weeks it drops are still reported, as `gaps`: a rare sound
#: like θ can fall under the line in a perfectly ordinary week, and silently
#: erasing it makes the chart look broken.
TREND_MIN_WEEK = 10


#: Length of one point on both trend charts, and how many points are kept.
#: Rolling windows counted back from today, not ISO weeks. Two reasons. The
#: newest point always has a full seven days behind it — with calendar weeks,
#: every Monday starts a bucket holding one day of speech, and that lone point
#: swings wildly and reads as a sudden collapse. And because every window is
#: the same length, position on the axis is time: a month you did not record in
#: becomes visible space instead of being folded into the neighbouring tick.
TREND_WINDOW_DAYS = 7
TREND_MAX_POINTS = 20


def _day(d):
    """A YYYY-MM-DD string as a date, or None. Accepts a full timestamp."""
    try:
        return date.fromisoformat(str(d)[:10])
    except (TypeError, ValueError):
        return None


def _windows(items, end=None, days=TREND_WINDOW_DAYS, limit=TREND_MAX_POINTS):
    """Rolling windows covering the data, newest ending on `end` (today).

    Returns `(keys, index)`: `keys` are the window start dates oldest-first —
    every one of them, including windows you recorded nothing in, because a
    silent fortnight is information and an axis that omits it is lying about
    when you improved. `index` maps a date string to its window key, or drops
    it when it falls outside (older than `limit` windows, or in the future).
    """
    end = end or date.today()
    dates = [x for x in (_day(d.get("date")) for d in items or []) if x]
    dates = [x for x in dates if x <= end]
    if not dates:
        return [], {}
    oldest = (end - min(dates)).days // days          # window count, 0 = newest
    oldest = min(oldest, limit - 1)
    keys = [(end - timedelta(days=days * k + days - 1)).isoformat()
            for k in range(oldest, -1, -1)]
    index = {}
    for d in items or []:
        x = _day(d.get("date"))
        if not x or x > end:
            continue
        k = (end - x).days // days
        if k <= oldest:
            index[d.get("date")] = keys[oldest - k]
    return keys, index


def _trim_lead(keys, weight, floor):
    """Drop leading windows too small to open a chart on.

    A trend has to start somewhere the number means something, and a run of
    empty or near-empty windows at the front costs a column each while saying
    nothing. Only the *lead-in* is trimmed: once the first solid window is
    found everything after it is kept, gaps included, because a gap between two
    real measurements is information about you rather than dead space.
    """
    for i, k in enumerate(keys):
        if weight(k) >= floor:
            return keys[i:], i
    return keys, 0                    # nothing solid yet — show what there is


def _wend(keys, days=TREND_WINDOW_DAYS):
    """Window start -> its last day, so the axis can label the span it covers."""
    out = {}
    for k in keys:
        d = _day(k)
        if d:
            out[k] = (d + timedelta(days=days - 1)).isoformat()
    return out


def pronunciation_trend(items, weak_below=WEAK_BELOW, min_week=TREND_MIN_WEEK,
                        end=None):
    """Failure rate per sound over rolling weeks — the shape of getting better.

    Only recordings measured the *same way* are charted. This is the whole
    reason this function is not a two-line group-by: exact per-phoneme scoring
    and word-level attribution produce different numbers for identical speech
    (θ reads about 6% attributed and about 20% exact), so a library that was
    part-upgraded would show a cliff exactly where the backfill stopped. That
    cliff is a change in the instrument, not in the speaker, and drawing it as
    a trend would be a lie in chart form.

    Exact recordings win when there are any, because they are the ones worth
    trusting; the count of what was left out is returned so the UI can say so.
    """
    exact_items, att_items = [], []
    for d in items or []:
        ws = ((d.get("azure") or {}).get("words")) or []
        if not ws or not d.get("date"):
            continue
        n_exact = sum(1 for w in ws if w.get("pacc"))
        (exact_items if n_exact == len(ws) else att_items).append(d)

    use, mode = (exact_items, "exact") if exact_items else (att_items, "attributed")
    skipped = len(att_items) if exact_items else 0

    weeks, index = _windows(use, end)

    # window -> sound -> [encounters, weak];  plus "" for the all-sounds baseline
    grid = {}
    for d in use:
        wk = index.get(d.get("date"))
        if not wk:
            continue
        cell = grid.setdefault(wk, {})
        for w in ((d.get("azure") or {}).get("words")) or []:
            seq = _stat_seq(w.get("word"), w.get("phones"))
            if not seq:
                continue
            pacc = w.get("pacc") or []
            exact = len(pacc) == len(seq)
            flagged = w.get("error") == "Mispronunciation"
            for i, (p, role) in enumerate(_stat_roles(seq)):
                score = pacc[i] if exact and i < len(pacc) else -1
                if exact and score < 0:
                    continue
                bad = (score < weak_below) if exact else flagged
                base = cell.setdefault("", [0, 0])
                base[0] += 1
                base[1] += bad
                for key, _lab, _note, hit in SOUND_GROUPS:
                    if hit(p, role):
                        c = cell.setdefault(key, [0, 0])
                        c[0] += 1
                        c[1] += bad

    # 40 attempts is the same number the hollow dot uses: below it a window is
    # too jumpy to read, and one at the very front has nothing before it to be
    # read against either.
    weeks, trimmed = _trim_lead(weeks, lambda w: grid.get(w, {}).get("", [0])[0], 40)
    labels = {k: lab for k, lab, _n, _h in SOUND_GROUPS}

    def line(key):
        """Plottable points, plus the weeks that were too thin to plot.

        The thin weeks come back rather than vanishing: the chart needs to
        know a gap is "too few attempts", not "never came up", and the table
        needs to show the count so a dash is never unexplained.
        """
        pts, gaps = [], {}
        for wk in weeks:
            n, bad = grid.get(wk, {}).get(key, [0, 0])
            if n < min_week:
                if n:
                    gaps[wk] = n
                continue                      # too thin to mean anything
            pts.append({"w": wk, "n": n, "bad": bad,
                        "rate": round(100.0 * bad / n, 1)})
        return pts, gaps

    series = []
    for key, lab, _note, _hit in SOUND_GROUPS:
        pts, gaps = line(key)
        if len(pts) >= 2:                     # one point is not a trend
            n = sum(p["n"] for p in pts)
            bad = sum(p["bad"] for p in pts)
            series.append({"key": key, "label": lab, "points": pts,
                           "gaps": gaps,
                           "total": n, "rate": round(100.0 * bad / n, 1)})
    # worst first — the chart should open on what is costing you something,
    # not on whichever sound happens to be commonest in English
    series.sort(key=lambda s: (-s["rate"], -s["total"]))
    return {"weeks": weeks, "wend": _wend(weeks), "series": series,
            "overall": line("")[0],
            "mode": mode, "skipped": skipped, "min_week": min_week,
            "days": TREND_WINDOW_DAYS, "trimmed": trimmed,
            "colors": TREND_COLORS,
            "recordings": sum(1 for d in use if index.get(d.get("date")) in set(weeks))}


#: Error types for what you *say* wrong, as opposed to mis-pronounce. Ordered:
#: the first pattern that matches the analyser's own stated rule wins, so the
#: specific ones have to come before the general ones.
GRAMMAR_TYPES = [
    ("tense", "Verb tense", r"\btense\b|past (simple|continuous|perfect|form)|present (simple|continuous|perfect)|future|\b-?ed form\b|past participle"),
    ("agreement", "Subject–verb agreement", r"agreement|agrees? with|third[- ]person|subject.{0,4}verb"),
    ("article", "Articles (a / an / the)", r"\barticles?\b|definite|indefinite|\b(the|a|an)['’]?\s+(is|before|with|for|goes|needed|required|missing)"),
    ("number", "Plural / countability", r"\bplurals?\b|\bsingular\b|uncountable|countable|count noun|no plural"),
    ("preposition", "Prepositions", r"\bpreposition|\buse '(in|on|at|of|to|for|with|from|by|about|into|over|under)'|\badd '(of|to|in|on|for|with)'"),
    ("modal", "Modals & auxiliaries", r"\bmodal|auxiliar|after '(should|would|could|must|can|will|might)'|\bbase verb\b|\bbare infinitive\b"),
    ("wordform", "Word form", r"\b(adjective|adverb|noun form|verb form|gerund|infinitive|participle)\b|not a verb|as a verb"),
    ("pronoun", "Pronouns", r"\bpronoun|\bits?\b for|antecedent|reflexive|avoid repetition"),
    ("order", "Word order", r"word order|order of|placement|position of|restructure"),
    ("clause", "Clause & connectors", r"\bclause|conjunction|relative pronoun|run-on|comma splice|sentence fragment|unnecessary '(that|which)'"),
    ("comparative", "Comparatives", r"comparative|superlative|\bthan\b"),
    ("wordchoice", "Word choice / collocation", r"collocation|natural|idiomatic|more common|better word|word choice|phrasing|doesn't take|use '"),
]
_GT = [(k, lab, re.compile(pat, re.I)) for k, lab, pat in GRAMMAR_TYPES]


def classify_grammar_error(g):
    """Which type of mistake one finding is. '' when nothing fits."""
    rule = g.get("rule") or ""
    for key, _lab, rx in _GT:
        if rx.search(rule):
            return key
    # Nothing in the stated rule — fall back to what actually changed. An
    # article that appears in the correction and not in what you said is an
    # omitted article, whatever the analyser chose to call it.
    said = set(re.findall(r"[a-z']+", (g.get("said") or "").lower()))
    corr = set(re.findall(r"[a-z']+", (g.get("correction") or "").lower()))
    arts = {"a", "an", "the"}
    if len(corr & arts) > len(said & arts):
        return "article"
    blob = "%s %s" % (g.get("correction") or "", g.get("said") or "")
    for key, _lab, rx in _GT:
        if rx.search(blob):
            return key
    return ""


def grammar_error_stats(items):
    """Typed counts of what you say wrong, with a rate you can compare over time.

    Deduplicated on (said, correction) the same way the log above it is, so one
    mistake quoted in two analyses is one mistake.
    """
    labels = dict((k, lab) for k, lab, _ in GRAMMAR_TYPES)
    counts, examples, seen = {}, {}, set()
    spoken = 0
    for d in items or []:
        spoken += len(((d.get("azure") or {}).get("words")) or [])
        for g in d.get("grammar", []) or []:
            key = (g.get("said", ""), g.get("correction", ""))
            if not g.get("said") or key in seen:
                continue
            seen.add(key)
            t = classify_grammar_error(g) or "other"
            counts[t] = counts.get(t, 0) + 1
            examples.setdefault(t, []).append(
                {"said": g.get("said", "")[:120],
                 "correction": g.get("correction", "")[:120],
                 "rule": g.get("rule", "")[:140]})
    total = sum(counts.values())
    rows = []
    for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        rows.append({
            "key": t, "label": labels.get(t, "Other / unclassified"),
            "n": n, "share": round(100.0 * n / total, 1) if total else 0,
            # per 1000 words assessed — the only way a count means anything
            # across recordings of very different lengths
            "per_1k": round(1000.0 * n / spoken, 2) if spoken else 0,
            "examples": examples.get(t, [])[:6],
        })
    return {"rows": rows, "total": total, "spoken": spoken}


#: A grammar point wants about this many words behind it. Under the line the
#: rate is real but jumpy — 700 words is one long recording's opinion, and a
#: single analysis that ran strict swings the whole point — so it is drawn as a
#: hollow dot rather than merged away. The pronunciation chart has said exactly
#: this with exactly that marker since it was built; one mechanism for "thin"
#: across both charts beats two.
TREND_BUCKET_WORDS = 1000

#: Below this a window is not plotted at all. A rate off 80 words is arithmetic
#: on noise. Skipped windows come back as `gaps` so the line can dot across
#: them and the table can say how few words there were.
TREND_MIN_WORDS = 200


def grammar_trend(items, bucket_words=TREND_BUCKET_WORDS,
                  min_words=TREND_MIN_WORDS, end=None):
    """Rate of each mistake type over rolling weeks, per 1000 words spoken.

    Per 1000 rather than per window, because a raw count mostly measures how
    much you recorded: a heavy week looks like a bad week.

    Windows are the same rolling seven days the pronunciation chart uses, so
    the two read against each other and position on the axis means time. A
    window you barely spoke in is drawn hollow, and one with almost nothing in
    it is skipped and dotted across — the same two-tier treatment as a rare
    sound, for the same reason: silently merging or deleting a stretch of
    history hides when you were actually improving.

    Unlike the pronunciation trend there is no measurement-mode split to worry
    about, so every scored recording counts. The softer caveat is that these
    come from an LLM: change the analyser or its strictness and the counts move
    a little, independently of your English.
    """
    labels = dict((k, lab) for k, lab, _ in GRAMMAR_TYPES)
    weeks, index = _windows(items, end)
    weeks_words, grid = {}, {}
    for d in items or []:
        wk = index.get(d.get("date"))
        if not wk:
            continue
        spoken = len(((d.get("azure") or {}).get("words")) or [])
        if not spoken:
            spoken = len((d.get("polished") or "").split())
        weeks_words[wk] = weeks_words.get(wk, 0) + spoken
        cell = grid.setdefault(wk, {})
        # Deduplicated per recording, not globally: the same slip listed twice
        # in one analysis is one mistake, but making it again next week is
        # exactly the thing this chart exists to show.
        seen = set()
        for g in d.get("grammar", []) or []:
            key = (g.get("said", ""), g.get("correction", ""))
            if not g.get("said") or key in seen:
                continue
            seen.add(key)
            t = classify_grammar_error(g) or "other"
            cell[t] = cell.get(t, 0) + 1
            cell[""] = cell.get("", 0) + 1

    weeks, trimmed = _trim_lead(weeks, lambda w: weeks_words.get(w, 0), bucket_words)

    def line(key):
        """Points, plus the windows too quiet to carry a rate (`gaps`)."""
        pts, gaps = [], {}
        for wk in weeks:
            words = weeks_words.get(wk, 0)
            if words < min_words:
                if words:
                    gaps[wk] = words
                continue
            n = grid.get(wk, {}).get(key, 0)
            pts.append({"w": wk, "n": n, "words": words,
                        "thin": words < bucket_words,
                        "rate": round(1000.0 * n / words, 2)})
        return pts, gaps

    series = []
    for key in set(k for c in grid.values() for k in c if k):
        pts, gaps = line(key)
        if sum(p["n"] for p in pts) and len(pts) >= 2:
            series.append({"key": key, "label": labels.get(key, "Other / unclassified"),
                           "points": pts, "gaps": gaps,
                           "total": sum(p["n"] for p in pts),
                           "rate": round(sum(p["n"] for p in pts) * 1000.0 /
                                         sum(p["words"] for p in pts), 2)})
    series.sort(key=lambda s: (-s["rate"], -s["total"]))
    return {"weeks": weeks, "wend": _wend(weeks),
            "series": series, "overall": line("")[0],
            "colors": TREND_COLORS, "bucket_words": bucket_words,
            "min_words": min_words, "days": TREND_WINDOW_DAYS,
            "trimmed": trimmed,
            "recordings": sum(1 for d in items or []
                              if index.get(d.get("date")) in set(weeks)),
            "unit": "per 1k words"}


def _grammar_panel(items):
    from collections import OrderedDict
    seen = OrderedDict()
    for d in items or []:
        for g in d.get("grammar", []):
            key = (g.get("said", ""), g.get("correction", ""))
            if key not in seen and g.get("said"):
                seen[key] = {"said": g.get("said", ""), "correction": g.get("correction", ""),
                             "rule": g.get("rule", ""), "from": d.get("title", "")}
    payload = json.dumps(list(seen.values()), ensure_ascii=False).replace("</", "<\\/")
    seed = json.dumps(_GRAMMAR_SEED, ensure_ascii=False).replace("</", "<\\/")
    stats = json.dumps({"pron": pronunciation_error_stats(items),
                        "gram": grammar_error_stats(items),
                        "trend": pronunciation_trend(items),
                        "gtrend": grammar_trend(items)},
                       ensure_ascii=False).replace("</", "<\\/")
    return ("<section id='grammar' class='tabpanel hidden'>"
            "<h1>Speaking error log — your recurring mistakes</h1>"
            "<p class='sub'>Everything the analysis flags in what you <b>say</b>: grammar, "
            "word choice and pronunciation, pulled from your recordings — plus common "
            "Mandarin-speaker blind spots and anything you log by hand. "
            "<span class='hint'>(What you mis-<b>hear</b> is tracked separately, in the "
            "listening module.)</span></p>"
            "<div class='drillnav'>"
            "<button class='btn small active' data-m='log' onclick='gxMode(this)'>📋 Grammar error log</button>"
            "<button class='btn small' data-m='gstats' onclick='gxMode(this)'>📊 Grammar error by type stats</button>"
            "<button class='btn small' data-m='pstats' onclick='gxMode(this)'>🗣️ Pronunciation error stats</button>"
            "</div><div id='gx-body'></div>"
            "</section>"
            "<script>window.GRAMMAR_DATA=%s;window.GRAMMAR_SEED=%s;window.ERRSTATS=%s;%s</script>"
            % (payload, seed, stats, _GRAMMAR_JS))


def _load_json(name):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


_ARPA2IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b",
    "CH": "tʃ", "D": "d", "DH": "ð", "EH": "e", "ER": "ɜr", "EY": "eɪ", "F": "f",
    "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "iː", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "OW": "əʊ", "OY": "ɔɪ", "P": "p", "R": "r",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}
#: Overrides for stress 0 — see word_ipa(). CMUdict marks the weak forms with a
#: trailing 0 and reuses the strong vowel's symbol; IPA does not.
_ARPA2IPA_WEAK = {"AH": "ə", "ER": "ər"}
_CMU = None
_VOWELS_ARPA = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
                "OW", "OY", "UH", "UW"}
# valid English syllable onsets, so the stress mark lands before the whole onset
_ONSETS = {"pl", "pr", "bl", "br", "tr", "dr", "kl", "kr", "ɡl", "ɡr", "tw", "kw",
           "dw", "θr", "θw", "ʃr", "fl", "fr", "fj", "sl", "sm", "sn", "sp", "st",
           "sk", "sw", "pj", "bj", "kj", "ɡj", "mj", "hj", "vj", "lj", "nj", "sj",
           "tj", "dj", "spl", "spr", "str", "skr", "skw", "spj", "stj", "skj"}


def _cmu():
    global _CMU
    if _CMU is None:
        try:
            import cmudict
            _CMU = cmudict.dict()
        except Exception:
            _CMU = {}
    return _CMU


def cmu_arpabets(word, keep_stress=False):
    """Every pronunciation CMUdict lists for `word`, most common first.

    Heteronyms genuinely have more than one — "live" is /laɪv/ AND /lɪv/, and so
    are read, wind, lead, bow, close. Anything judging Azure against "the"
    pronunciation of a word has to consider all of them or it will call a
    perfectly good reading wrong.
    """
    variants = _cmu().get(re.sub(r"[^a-z']", "", (word or "").lower())) or []
    return [[p.lower() if keep_stress else re.sub(r"\d", "", p).lower() for p in v]
            for v in variants]


def cmu_arpabet(word, keep_stress=False):
    """CMUdict's primary pronunciation for `word`. [] if unknown."""
    variants = cmu_arpabets(word, keep_stress)
    return variants[0] if variants else []


# Azure and CMUdict describe the same mouth differently, and separating those
# notational gaps from real disagreements is the whole job of
# azure_lexicon_disagrees(). Everything folded here is a difference no listener
# would call a different word:
#   * reduced vowels — CMUdict AH0/IH0, Azure "ax", both schwa
#   * weak forms — "for" as /fɔːr/ (CMUdict) or /fər/ (Azure); "that", "them"
#   * syllabic consonants — "towel" as /taʊəl/ or /taʊl/, "I'll" either way
#   * pre-r vowels — /ʊr/~/ɜr/~/ɔr/ vary by dialect ("during", "are", "for")
#   * father/strut and cot/caught, which not every US accent separates
_ARPA_LOW = {"aa", "ao", "ah"}          # -> one low/central class
_ARPA_VOWELS = {"a", "ay", "iy", "ey", "ow", "uw", "aw", "oy", "ae", "eh",
                "ih", "uh", "er", "R", "@"}


def _norm_arpa(seq):
    """Fold one phoneme sequence into the comparison alphabet described above."""
    out = []
    for raw in seq or []:
        p = (raw or "").lower()
        stress = p[-1] if p and p[-1].isdigit() else ""
        p = re.sub(r"\d", "", p)
        if not p:
            continue
        if p in ("ax", "axr") or (p in ("ah", "ih") and stress == "0"):
            p = "@" if p != "axr" else "er"
        elif p in _ARPA_LOW:
            p = "a"
        out.append(p)
    # collapse any vowel + /r/ into one r-coloured slot
    folded = []
    for p in out:
        if p == "r" and folded and (folded[-1] in _ARPA_VOWELS or folded[-1] == "R"):
            folded[-1] = "R"
            continue
        folded.append("R" if p == "er" else p)
    return folded


def _phones_equivalent(a, b):
    """Can these two sequences describe the same utterance?

    A schwa matches any vowel, and may be present on one side only — that is
    what makes weak forms and syllabic consonants compare equal. Everything
    else has to line up exactly.
    """
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def reach(i, j):
        if i == len(a) and j == len(b):
            return True
        if i < len(a) and j < len(b):
            x, y = a[i], b[j]
            if x == y and reach(i + 1, j + 1):
                return True
            if ((x == "@" and y in _ARPA_VOWELS) or
                    (y == "@" and x in _ARPA_VOWELS)) and reach(i + 1, j + 1):
                return True
        if i < len(a) and a[i] == "@" and reach(i + 1, j):
            return True
        if j < len(b) and b[j] == "@" and reach(i, j + 1):
            return True
        return False

    return reach(0, 0)


def azure_lexicon_disagrees(word, azure_phones):
    """True when Azure graded `word` against a genuinely different pronunciation
    than CMUdict (and this app's own IPA display) gives.

    This is not hypothetical. Azure's lexicon has "tied" as /t iy d/ ("teed"),
    so saying it correctly as /taɪd/ scores the vowel 44 and the word 64, while
    the homophone "tide" scores 100 on the very same audio. Anything flagged
    here means the number describes Azure's dictionary, not your mouth.

    `azure_phones` must come from assessing the ACTUAL attempt: where Azure
    holds several pronunciations it returns whichever best matches the audio
    (say the noun "mouth" and it grades /maʊθ/, not the verb's /maʊð/). So a
    flag means Azure had no reading of this spelling that matches the word —
    not merely that it picked a different one from CMUdict's first entry.

    Returns False when either side is unknown — never guess.
    """
    theirs = _norm_arpa(azure_phones)
    variants = cmu_arpabets(word, keep_stress=True)
    if not theirs or not variants:
        return False
    # a heteronym is only wrong if it matches NONE of its readings
    for v in variants:
        ours = _norm_arpa(v)
        if ours and _phones_equivalent(tuple(ours), tuple(theirs)):
            return False
    return True


_HOMOPHONES = None


def cmu_homophone(word):
    """A different spelling with the identical CMUdict pronunciation, or ''.

    Used to re-score a word whose Azure lexicon entry is broken: "tied" grades
    wrongly, "tide" grades correctly, and they are the same sound — so the
    honest score for "tied" is the one Azure gives for "tide".
    """
    global _HOMOPHONES
    key = tuple(cmu_arpabet(word))
    if not key:
        return ""
    if _HOMOPHONES is None:
        _HOMOPHONES = {}
        for w, prons in _cmu().items():
            if not prons or "'" in w or not w.isalpha():
                continue
            k = tuple(re.sub(r"\d", "", p).lower() for p in prons[0])
            _HOMOPHONES.setdefault(k, []).append(w)
    lw = (word or "").lower()
    # shortest first, then alphabetical — favours the plain, common spelling
    alts = sorted((w for w in _HOMOPHONES.get(key, ()) if w != lw),
                  key=lambda w: (len(w), w))
    return alts[0] if alts else ""


def word_ipa(word):
    """Best-effort IPA for an English word via CMUdict (offline), with primary/
    secondary stress marks on words of 2+ syllables. '' if unknown."""
    _cmu()
    w = re.sub(r"[^a-z']", "", (word or "").lower())
    phones = _CMU.get(w)
    if not phones:
        return ""
    toks = []           # (ipa, is_vowel, stress)
    for p in phones[0]:
        base = re.sub(r"\d", "", p)
        s = p[-1] if p[-1].isdigit() else ""
        # Two ARPAbet vowels mean different things stressed and unstressed, and
        # the table can only hold one symbol each. AH is ʌ in "cup" but schwa in
        # "about"; ER is the NURSE vowel ɜr in "bird" but r-coloured schwa ər in
        # "letter" — printing ɜr there claims the second syllable of "letter"
        # sounds like "bird", which is a stress error, not a spelling nicety.
        ipa = _ARPA2IPA_WEAK.get(base) if s == "0" else None
        if ipa is None:
            ipa = _ARPA2IPA.get(base, "")
        toks.append([ipa, base in _VOWELS_ARPA, s])

    nsyl = sum(1 for t in toks if t[1])
    marks = {}
    if nsyl >= 2:                       # monosyllables carry no stress mark
        for i, t in enumerate(toks):
            if t[1] and t[2] in ("1", "2"):
                mark = "ˈ" if t[2] == "1" else "ˌ"
                onset = []
                k = i - 1
                while k >= 0 and not toks[k][1]:
                    onset.insert(0, toks[k][0])
                    k -= 1
                start = i
                for L in range(min(3, len(onset)), 0, -1):
                    if "".join(onset[len(onset) - L:]) in _ONSETS or L == 1:
                        start = i - L
                        break
                marks[start] = mark
    out = ""
    for i, t in enumerate(toks):
        if i in marks:
            out += marks[i]
        out += t[0]
    return ("/" + out + "/") if out else ""


_DICTATION_JS = r"""
(function(){
 var S=window.SkillStore;
 var ST={cur:null, rate:1, revealed:false, checked:false};
 function clips(){ return window.LISTEN_CLIPS||[]; }
 function body(){ return document.getElementById('dict-body'); }
 function srs(){ return S.get('dict_srs',{}); }
 function sources(){ var s={}; clips().forEach(function(c){ s[c.source]=1; }); return Object.keys(s).sort(); }
 function offSources(){ return S.get('dict_off',[]); }
 function pool(){
   var off=offSources();
   return clips().filter(function(c){ return off.indexOf(c.source)<0; });
 }
 // pick the next clip: anything due for review first (spaced repetition),
 // otherwise something never attempted, otherwise anything at random.
 function pick(){
   var p=pool(); if(!p.length) return null;
   var st=srs(), today=S.today();
   var due=p.filter(function(c){ var r=st[c.id]; return r && r.due && r.due<=today; });
   var fresh=p.filter(function(c){ return !st[c.id]; });
   var from = due.length?due : (fresh.length?fresh : p);
   var next = from[Math.floor(Math.random()*from.length)];
   if(from.length>1 && ST.cur && next.id===ST.cur.id) return pick();
   return next;
 }
 // a clip counts as practised once it has been graded OR checked at least once
 function practisedIds(){
   var st=srs(), sc=S.get('ec_scores',{}), ids={};
   Object.keys(st).forEach(function(k){ ids[k]=1; });
   Object.keys(sc).forEach(function(k){ if(k.indexOf('dict:')===0) ids[k.slice(5)]=1; });
   return ids;
 }
 function stats(){
   var st=srs(), p=pool(), done=0, due=0, today=S.today();
   var seen=practisedIds();
   p.forEach(function(c){
     if(seen[c.id]) done++;
     var r=st[c.id]; if(r && r.due && r.due<=today) due++;
   });
   // how much material is still sitting upstream, unimported
   var meta=window.LISTEN_META||{}, avail=0, known=false;
   Object.keys(meta).forEach(function(s){
     var e=(meta[s]||{}).eligible;
     if(typeof e==='number'){ avail+=e; known=true; }
   });
   return {total:p.length, done:done, fresh:p.length-done, due:due,
           library:clips().length, available:known?avail:null};
 }
 function esc(s){ return S.esc(s); }
 function audioURL(c){ return 'listening/'+c.audio; }

 function render(){
   var el=body(); if(!el) return;
   if(!clips().length){ el.innerHTML=emptyState()+importBar(stats()); return; }
   if(!ST.cur) ST.cur=pick();
   var c=ST.cur;
   if(!c){ el.innerHTML="<div class='card'>Every source is switched off — turn one back on below.</div>"+srcBar(); return; }
   var s=stats();
   var pct=s.total?Math.round(s.done/s.total*100):0;
   var head="<div class='card' style='display:flex;gap:22px;flex-wrap:wrap;align-items:center'>"+
     "<span><b style='font-size:22px'>"+s.total+"</b> <span class='hint'>clips</span></span>"+
     "<span><b style='font-size:22px;color:var(--good)'>"+s.done+"</b> <span class='hint'>practised ("+pct+"%)</span></span>"+
     "<span><b style='font-size:22px'>"+s.fresh+"</b> <span class='hint'>never heard</span></span>"+
     "<span><b style='font-size:22px;color:var(--warn)'>"+s.due+"</b> <span class='hint'>due for review</span></span>"+
     "</div>"+
     "<div class='sb-t' style='height:8px;margin:0 2px 4px'><div class='sb-f' style='width:"+pct+"%'></div></div>"+
     importBar(s);
   var rates=[0.6,0.75,1].map(function(r){
     return "<button class='btn small dictrate"+(ST.rate===r?' active':'')+"' data-rate='"+r+"'>"+
            (r===1?'1&#215; normal':(r+'&#215;'))+"</button>"; }).join('');
   var player="<div class='card'>"+
     "<audio id='dict-audio' src='"+esc(audioURL(c))+"' preload='auto'></audio>"+
     "<div class='hwctl'>"+
       "<button class='btn' id='dict-play'>&#9654; Play sentence</button>"+
       "<button class='btn small' id='dict-again'>&#8635; Again</button>"+
       "<span style='flex:1'></span>"+rates+
     "</div>"+
     "<div class='hint' style='margin-top:8px'>Listen as many times as you like. "+
     "Type what you hear — then it gets checked word by word.</div></div>";
   var box="<div class='card'>"+
     "<textarea id='dict-input' rows='3' style='width:100%' placeholder='type what you hear…' "+
     (ST.checked?'disabled':'')+"></textarea>"+
     "<div class='hwctl'><button class='btn' id='dict-check'>Check</button>"+
     "<button class='btn small' id='dict-reveal'>"+(ST.revealed?'Hide':'Show')+" transcript</button>"+
     "<span style='flex:1'></span>"+
     "<button class='btn small' id='dict-skip' title='Leave this sentence unanswered "+
       "and load a different one. Nothing is scored and it is not marked as practised, "+
       "so it can come round again.'>Skip this one &#8212; don&#39;t score it &#8594;</button>"+
     "<span class='hint' id='dict-msg'></span></div>"+
     (ST.revealed?("<p class='summary' style='margin-top:10px'>"+esc(c.text)+"</p>"):"")+
     "<div id='dict-result'></div></div>";
   var cite="<p class='hint'>Source: "+esc(c.source)+
     (c.license?(" · "+esc(c.license)):"")+
     (c.source_url?(" · <a href='"+esc(c.source_url)+"' target='_blank' rel='noopener' style='color:var(--accent)'>original</a>"):"")+
     (c.accent?(" · "+esc(c.accent)+" accent"):"")+"</p>";
   el.innerHTML=head+player+box+cite+srcBar();
   var ta=document.getElementById('dict-input');
   if(ta && !ST.checked){ ta.focus(); }
 }

 // Import more material without leaving the page. Runs the same
 // listening_import.py the command line does — the app just shells out to it.
 function importBar(s){
   var left = (s.available==null) ? null : Math.max(0, s.available - s.library);
   var note;
   if(left==null){
     note="<span class='hint'>Run the importer once to see how much material is available.</span>";
   } else if(left<=0){
     note="<span class='hint'>Every available clip is imported ("+s.library+").</span>";
   } else {
     note="<span class='hint'><b>"+left.toLocaleString()+"</b> more clips available to import "+
          "("+s.library.toLocaleString()+" of "+s.available.toLocaleString()+" fetched).</span>";
   }
   var sel="<select id='dict-n' style='margin:0 6px'>"+
     [50,100,200,500].map(function(n){return "<option value='"+n+"'"+(n===200?" selected":"")+">"+n+"</option>";}).join('')+
     "</select>";
   return "<div class='ssbtypes' style='margin:0 2px 10px;align-items:center'>"+note+
     "<span style='flex:1'></span>"+
     "<button class='btn small' id='dict-import'>&#11015; Import"+sel+"more</button>"+
     "</div><div id='dict-implog' class='hint' style='margin:0 2px 8px;white-space:pre-wrap'></div>";
 }

 function runImport(btn){
   var n=document.getElementById('dict-n');
   var count=n?parseInt(n.value,10):200;
   var log=document.getElementById('dict-implog');
   btn.disabled=true; btn.textContent='Importing…';
   if(log) log.textContent='Starting…';
   var fd=new FormData(); fd.append('source','tatoeba'); fd.append('count',count);
   fetch('/import_listening',{method:'POST',body:fd}).then(function(r){return r.json();})
    .then(function(j){
      if(j.error){ if(log) log.textContent=j.error; btn.disabled=false; btn.textContent='⬇ Import more'; return; }
      poll(j.job, btn, log);
    }).catch(function(){
      if(log) log.textContent='No server — importing needs the app running.';
      btn.disabled=false; btn.textContent='⬇ Import more';
    });
 }
 function poll(job, btn, log){
   fetch('/import_progress/'+job).then(function(r){return r.json();}).then(function(j){
     if(log) log.textContent=(j.log||[]).slice(-6).join('\n');
     if(j.done){
       if(j.error){ if(log) log.textContent+='\n'+j.error; }
       btn.disabled=false; btn.textContent='✓ Done — reloading';
       setTimeout(function(){ location.reload(); }, 900);
       return;
     }
     setTimeout(function(){ poll(job, btn, log); }, 1000);
   }).catch(function(){ setTimeout(function(){ poll(job, btn, log); }, 1500); });
 }

 function srcBar(){
   var off=offSources(), all=sources();
   if(all.length<2) return '';
   var btns=all.map(function(s){
     var on=off.indexOf(s)<0;
     return "<button class='btn small dictsrc"+(on?' active':'')+"' data-src='"+S.esc(s)+"'>"+
            (on?'☑':'☐')+' '+S.esc(s)+"</button>"; }).join('');
   return "<div class='ssbtypes' style='margin-top:14px'><span class='hint' style='align-self:center'>Sources:</span>"+btns+"</div>";
}

 function emptyState(){
   return "<div class='card'><h4>No clips imported yet</h4>"+
     "<p>This module practises listening against <b>real recorded speech</b>, not "+
     "text-to-speech — the whole point is connected speech: linking, weak forms, "+
     "and the endings that vanish in natural delivery.</p>"+
     "<p class='hint'>Populate it from the command line, then reload:</p>"+
     "<pre style='background:var(--panel);padding:10px 12px;border-radius:8px;overflow:auto;font-size:13px'>"+
     "python3 listening_import.py voa --count 20\n"+
     "python3 listening_import.py tatoeba --count 200\n"+
     "python3 listening_import.py sbc --dir ~/Downloads/sbc\n"+
     "python3 listening_import.py local --dir ~/my-audio</pre>"+
     "<p class='hint'>Run <code>python3 listening_import.py --help</code> for the "+
     "sources and their licences.</p></div>";
 }

 function renderDiff(res, c){
   var html='', missed=[];
   res.ops.forEach(function(o){
     if(o.op==='equal'){ html+="<span>"+S.esc(o.ref.join(' '))+" </span>"; return; }
     if(o.op==='replace'){
       missed=missed.concat(o.ref);
       html+="<span style='color:var(--bad);text-decoration:line-through;opacity:.65'>"+S.esc(o.hyp.join(' '))+"</span> "+
             "<b style='color:var(--good)'>"+S.esc(o.ref.join(' '))+"</b> ";
     } else if(o.op==='delete'){
       missed=missed.concat(o.ref);
       html+="<b style='color:var(--warn)'>"+S.esc(o.ref.join(' '))+"</b> ";
     } else if(o.op==='insert'){
       html+="<span style='color:var(--bad);text-decoration:line-through;opacity:.65'>"+S.esc(o.hyp.join(' '))+"</span> ";
     }
   });
   var col=res.score>=90?'var(--good)':(res.score>=70?'var(--warn)':'var(--bad)');
   var out="<p class='score' style='margin-top:12px'>You caught <b style='color:"+col+"'>"+
     res.correct+" of "+res.total+"</b> words · "+res.score+"/100</p>"+
     "<p class='summary' style='line-height:1.9'>"+html+"</p>";
   if(missed.length){
     out+="<p class='hint'>Missed: <b>"+S.esc(missed.join(', '))+"</b> — replay and listen for them "+
          "specifically; in natural speech these are usually swallowed by linking or reduced to a weak form.</p>";
   }
   out+="<div class='ssgrades'>How was it? "+
     "<button class='btn small dictgrade' data-g='0' style='background:#ff6b6b;color:#08222b'>Again</button>"+
     "<button class='btn small dictgrade' data-g='1' style='background:#ffb454;color:#08222b'>Hard</button>"+
     "<button class='btn small dictgrade' data-g='2' style='background:#46b3c9;color:#08222b'>Good</button>"+
     "<button class='btn small dictgrade' data-g='3' style='background:#43c59e;color:#08222b'>Easy</button></div>";
   return out;
 }

 function playClip(){
   var c=ST.cur; if(!c) return;
   var au=document.getElementById('dict-audio'); if(!au) return;
   try{ au.playbackRate=ST.rate; }catch(_){}
   var hasWindow = (c.start!=null && c.end!=null && c.end>c.start);
   try{ au.currentTime = hasWindow ? c.start : 0; }catch(_){}
   if(au.__stop){ clearTimeout(au.__stop); au.__stop=null; }
   au.play().then(function(){
     if(hasWindow){
       // stop at the sentence boundary, adjusted for playback speed
       var ms=((c.end-c.start)/ST.rate)*1000+120;
       au.__stop=setTimeout(function(){ try{au.pause();}catch(_){} }, ms);
     }
   }).catch(function(){
     var m=document.getElementById('dict-msg');
     if(m) m.textContent='Could not play the audio file — is it in the listening/ folder?';
   });
 }

 document.addEventListener('click', function(e){
   var t=e.target.closest?e.target:null; if(!t||!t.closest) return;
   var b;
   if((b=t.closest('#dict-play'))||(b=t.closest('#dict-again'))){ playClip(); return; }
   if((b=t.closest('.dictrate'))){ ST.rate=parseFloat(b.getAttribute('data-rate'))||1;
     var rs=document.querySelectorAll('.dictrate');
     for(var i=0;i<rs.length;i++) rs[i].classList.toggle('active', rs[i]===b);
     playClip(); return; }
   if((b=t.closest('#dict-skip'))){
     ST.cur=pick(); ST.revealed=false; ST.checked=false; render();
     // confirm what just happened — a tooltip is only discoverable on hover
     var m=document.getElementById('dict-msg');
     if(m) m.textContent='Skipped — not scored. That sentence stays in the pool.';
     return; }
   if((b=t.closest('#dict-reveal'))){ ST.revealed=!ST.revealed; render(); return; }
   if((b=t.closest('.dictsrc'))){
     var src=b.getAttribute('data-src'); var off=offSources(); var i2=off.indexOf(src);
     if(i2<0) off.push(src); else off.splice(i2,1);
     if(off.length>=sources().length){ off.splice(off.indexOf(src),1); }  // keep one on
     S.set('dict_off',off); ST.cur=pick(); ST.revealed=false; ST.checked=false; render(); return; }
   if((b=t.closest('#dict-import'))){ runImport(b); return; }
   if((b=t.closest('#dict-check'))){ check(); return; }
   if((b=t.closest('.dictgrade'))){ gradeIt(parseInt(b.getAttribute('data-g'),10)); return; }
 });

 function check(){
   var c=ST.cur; if(!c) return;
   var ta=document.getElementById('dict-input'); if(!ta) return;
   var typed=(ta.value||'').trim();
   var msg=document.getElementById('dict-msg');
   if(!typed){ if(msg) msg.textContent='Type what you heard first.'; return; }
   if(msg) msg.textContent='Checking…';
   var fd=new FormData(); fd.append('id',c.id); fd.append('typed',typed);
   fetch('/dictation',{method:'POST',body:fd}).then(function(r){return r.json();})
    .then(function(res){
      if(res.error){ if(msg) msg.textContent=res.error; return; }
      if(msg) msg.textContent='';
      ST.checked=true; ST.revealed=true;
      S.logScore('dict:'+c.id, res.score);
      recordErrors(res, c);
      var out=document.getElementById('dict-result');
      if(out) out.innerHTML=renderDiff(res, c);
      ta.disabled=true;
    }).catch(function(){
      if(msg) msg.textContent='No server — dictation checking needs the app running.';
    });
 }

 // Keep a record of every word you didn't catch. This is the listening
 // counterpart of the speaking error log: the same word failing again and again
 // is the signal worth acting on, and a single attempt never shows that.
 function recordErrors(res, c){
   var errs=S.get('dict_errors',{}), today=S.today();
   function bump(word, kind, heard){
     word=(word||'').toLowerCase(); if(!word) return;
     var e=errs[word]||{n:0, kinds:{}, heard:[], clips:[], last:''};
     e.n++; e.kinds[kind]=(e.kinds[kind]||0)+1; e.last=today;
     if(heard && e.heard.indexOf(heard)<0) e.heard.push(heard);
     if(e.heard.length>6) e.heard=e.heard.slice(-6);
     if(e.clips.indexOf(c.id)<0) e.clips.push(c.id);
     if(e.clips.length>8) e.clips=e.clips.slice(-8);
     errs[word]=e;
   }
   (res.ops||[]).forEach(function(o){
     if(o.op==='replace'){
       o.ref.forEach(function(w,i){ bump(w, 'misheard', o.hyp[i] || o.hyp.join(' ')); });
     } else if(o.op==='delete'){
       o.ref.forEach(function(w){ bump(w, 'missed', ''); });
     } else if(o.op==='insert'){
       // a word you typed that was never said — heard something that wasn't there
       o.hyp.forEach(function(w){ bump(w, 'imagined', ''); });
     }
   });
   S.set('dict_errors', errs);
 }

 function gradeIt(g){
   var c=ST.cur; if(!c||isNaN(g)) return;
   var st=srs(); st[c.id]=S.schedule(st[c.id], g); S.set('dict_srs',st);
   ST.cur=pick(); ST.revealed=false; ST.checked=false; render();
 }

 window.addEventListener('load', function(){ if(document.getElementById('dict-body')) render(); });
})();
"""


_LISTENLOG_JS = r"""
(function(){
 var S=window.SkillStore; var SORT='n'; var SHOWSKIP=false;
 function errs(){ return S.get('dict_errors',{}); }
 function body(){ return document.getElementById('llog-body'); }
 // Words you've decided aren't worth chasing. Missing "the" in dictation is a
 // fact about connected speech, not a gap you can drill — and left in, those
 // rows sit at the top of the table and bury the ones that matter.
 function skipped(){
   var m={}; (S.get('dict_skip',[])||[]).forEach(function(w){ m[(''+w).toLowerCase()]=1; });
   return m;
 }
 // removals have to stick, so skip-list writes replace the key wholesale
 function setSkip(fn){ S.update('dict_skip',[],fn,true); }
 function kindLabel(k){
   return k==='misheard' ? 'heard as something else'
        : k==='missed'   ? 'not heard at all'
        : 'heard but never said';
 }
 function rows(){
   var e=errs();
   return Object.keys(e).map(function(w){
     var x=e[w]; var top='misheard', best=0;
     Object.keys(x.kinds||{}).forEach(function(k){ if(x.kinds[k]>best){best=x.kinds[k];top=k;} });
     return {word:w, n:x.n||0, kind:top, heard:x.heard||[], last:x.last||'', clips:x.clips||[]};
   }).sort(function(a,b){
     if(SORT==='word') return a.word<b.word?-1:a.word>b.word?1:0;
     if(SORT==='last') return (b.last||'').localeCompare(a.last||'');
     return b.n-a.n || (a.word<b.word?-1:1);
   });
 }
 function render(){
   var el=body(); if(!el) return;
   var allRows=rows();
   if(!allRows.length){
     el.innerHTML="<div class='card'><h4>Nothing logged yet</h4>"+
       "<p>Every word you miss in a dictation lands here automatically. "+
       "Do a few clips on the practice tab and come back.</p></div>";
     return;
   }
   var sk=skipped();
   var r=allRows.filter(function(x){ return !sk[x.word.toLowerCase()]; });
   var hidden=allRows.filter(function(x){ return sk[x.word.toLowerCase()]; });
   var repeat=r.filter(function(x){return x.n>1;});
   var total=r.reduce(function(a,x){return a+x.n;},0);
   // one click for the whole function-word class — that's what's actually
   // clogging the top of this table
   var fw=(window.VOCAB_FUNCTION_WORDS||[]);
   var fwPending=fw.length ? allRows.filter(function(x){
     return !sk[x.word.toLowerCase()] && fw.indexOf(x.word.toLowerCase())>=0; }).length : 0;
   var head="<div class='card' style='display:flex;gap:22px;flex-wrap:wrap;align-items:center'>"+
     "<span><b style='font-size:22px'>"+r.length+"</b> <span class='hint'>distinct words missed</span></span>"+
     "<span><b style='font-size:22px'>"+total+"</b> <span class='hint'>total misses</span></span>"+
     "<span><b style='font-size:22px;color:var(--bad)'>"+repeat.length+"</b> <span class='hint'>missed more than once</span></span>"+
     (hidden.length?("<span><b style='font-size:22px;color:var(--mut)'>"+hidden.length+
       "</b> <span class='hint'>skipped</span></span>"):"")+
     "<span style='flex:1'></span>"+
     (fwPending?("<button class='btn small' id='llog-skipfw'>🚫 Skip "+fwPending+
       " function word"+(fwPending===1?"":"s")+"</button>"):"")+
     (hidden.length?("<button class='btn small' id='llog-showskip'>"+(SHOWSKIP?'▾':'▸')+
       " Skipped ("+hidden.length+")</button>"):"")+
     "<button class='btn small' id='llog-drill'>➕ Add repeats to Practice single word</button>"+
     "<button class='btn small' id='llog-reset' style='background:#3a2030;color:#ff9db0'>🗑 Reset</button></div>";
   if(SHOWSKIP && hidden.length){
     head+="<div class='card'><b>Skipped words</b> <span class='hint'>· still logged, just not "+
       "shown above. Click one to put it back.</span><div class='words' style='margin-top:8px'>"+
       hidden.map(function(x){
         return "<span class='wpill seek' data-unskip=\""+S.esc(x.word)+"\" title='Show this word again'>"+
           S.esc(x.word)+" <span class='hint'>×"+x.n+"</span> ↩</span>"; }).join('')+"</div></div>";
   }
   if(!r.length){
     el.innerHTML=head+"<div class='card'>Every logged word is skipped. Reveal some above to "+
       "see the table again.</div>";
     return;
   }
   function hd(k,label){ return "<th class='llsort' data-s='"+k+"' style='cursor:pointer'>"+label+(SORT===k?' ▼':'')+"</th>"; }
   var body_=r.map(function(x){
     var col=x.n>2?'var(--bad)':x.n>1?'var(--warn)':'var(--mut)';
     var heard=x.heard.length? ("<span class='hint'>you typed: "+S.esc(x.heard.join(', '))+"</span>") : "<span class='hint'>—</span>";
     return "<tr><td><b style='font-size:15px'>"+S.esc(x.word)+"</b></td>"+
       "<td style='text-align:center'><b style='color:"+col+"'>"+x.n+"</b></td>"+
       "<td class='hint'>"+kindLabel(x.kind)+"</td>"+
       "<td>"+heard+"</td>"+
       "<td class='hint'>"+S.esc(x.last)+"</td>"+
       "<td style='text-align:right;white-space:nowrap'>"+
       "<button class='btn small' data-say=\""+S.esc(x.word)+"\">🔊</button> "+
       "<button class='btn small llog-skip' data-skip=\""+S.esc(x.word)+
       "\" title='Skip this word — hide it from the table without losing the count'>🚫</button></td></tr>";
   }).join('');
   el.innerHTML=head+"<table class='pwt'><tr>"+hd('word','Word')+hd('n','Missed')+
     "<th>Mostly</th><th>What you wrote instead</th>"+hd('last','Last')+"<th></th></tr>"+body_+"</table>"+
     "<p class='hint'>“Heard but never said” means you typed a word that wasn't there — usually "+
     "a linking artefact, where two words ran together and your ear inserted a third.</p>";
 }
 document.addEventListener('click', function(ev){
   var t=ev.target; if(!t||!t.closest) return; var b;
   if((b=t.closest('.llsort'))){ SORT=b.getAttribute('data-s'); render(); return; }
   if((b=t.closest('#llog-reset'))){
     if(!confirm('Clear the whole listening error log?\n\nThis cannot be undone.')) return;
     S.set('dict_errors',{}); render(); return; }
   if((b=t.closest('.llog-skip'))){
     var w=(b.getAttribute('data-skip')||'').toLowerCase();
     setSkip(function(cur){ cur=(cur||[]).slice();
       if(cur.map(function(x){return (''+x).toLowerCase();}).indexOf(w)<0) cur.push(w);
       return cur; });
     render(); return; }
   if((b=t.closest('[data-unskip]'))){
     var u=(b.getAttribute('data-unskip')||'').toLowerCase();
     setSkip(function(cur){ return (cur||[]).filter(function(x){
       return (''+x).toLowerCase()!==u; }); });
     render(); return; }
   if((b=t.closest('#llog-showskip'))){ SHOWSKIP=!SHOWSKIP; render(); return; }
   if((b=t.closest('#llog-skipfw'))){
     var fw=window.VOCAB_FUNCTION_WORDS||[];
     var hits=rows().map(function(x){return x.word.toLowerCase();})
                    .filter(function(w){ return fw.indexOf(w)>=0; });
     if(!hits.length) return;
     setSkip(function(cur){ cur=(cur||[]).slice();
       var have={}; cur.forEach(function(x){ have[(''+x).toLowerCase()]=1; });
       hits.forEach(function(w){ if(!have[w]){ have[w]=1; cur.push(w); } });
       return cur; });
     render(); return; }
   if((b=t.closest('#llog-drill'))){
     // skipped words are ones you've said aren't worth chasing, so they don't
     // belong in the practice queue either
     var sk=skipped();
     var words=rows().filter(function(x){
       return x.n>1 && !sk[x.word.toLowerCase()]; }).map(function(x){return x.word;});
     if(!words.length){ alert('Nothing unskipped has been missed more than once yet.'); return; }
     var added=0;
     S.update('pw_custom',[],function(cur){ cur=(cur||[]).slice(); var have={};
       cur.forEach(function(w){ have[(''+w).toLowerCase()]=1; });
       words.forEach(function(w){ var lw=(''+w).toLowerCase();
         if(!have[lw]){ have[lw]=1; cur.push(w); added++; } });
       return cur; });
     S.update('pw_hidden',[],function(hid){ return (hid||[]).filter(function(h){
       return words.indexOf((''+h).toLowerCase())<0; }); });
     b.textContent = added ? ('✓ Added '+added+' to Practice single word') : '✓ Already there';
     b.disabled=true; return; }
 });
 window.addEventListener('load', function(){ if(document.getElementById('llog-body')) render(); });
})();
"""


def _listening_log_panel(hidden=True):
    """The listening counterpart of the speaking error log."""
    return ("<section id='listenlog' class='tabpanel%s'>"
            "<h1>Listening error log</h1>"
            "<p class='sub'>Every word you failed to catch in a dictation, counted. "
            "One miss is noise; the same word three times is a pattern — and for a "
            "Mandarin-L1 listener it's usually the same few causes: a weak form, a "
            "linked boundary, or a final consonant that never survived the sentence.</p>"
            "<p class='hint' style='margin-top:-8px'>🚫 skips a word: it stays counted, "
            "but drops out of the table and out of the practice queue. Use it on the "
            "function words — missing <i>the</i> is a fact about connected speech, not "
            "a gap you can drill.</p>"
            "<div id='llog-body'></div></section>"
            "<script>%s</script>" % ("" if not hidden else " hidden", _LISTENLOG_JS))


def _dictation_panel(hidden=True):
    """Listening practice against real recorded speech, graded by dictation."""
    clips = load_listening_library()
    payload = json.dumps(clips, ensure_ascii=False).replace("</", "<\\/")
    by_src = {}
    for c in clips:
        by_src[c["source"]] = by_src.get(c["source"], 0) + 1
    sub = ("Listen to a sentence of real speech, type what you hear, and get it "
           "checked word by word. Passive replay always feels like understanding — "
           "writing it down is what proves it.")
    if by_src:
        sub += ("<br><span class='hint'>%s</span>"
                % " · ".join("%s: %d clips" % (s, n) for s, n in sorted(by_src.items())))
    meta = json.dumps(load_listening_meta(), ensure_ascii=False).replace("</", "<\\/")
    return ("<section id='dictation' class='tabpanel%s'>"
            "<h1>Listening — dictation</h1>"
            "<p class='sub'>%s</p><div id='dict-body'></div></section>"
            "<script>window.LISTEN_CLIPS=%s;window.LISTEN_META=%s;\n%s</script>"
            % (" hidden" if hidden else "", sub, payload, meta, _DICTATION_JS))


def _listening_panel():
    phonemes = _load_json("phonemes.json")
    # IPA for every option word (shown when "Show IPA" is on)
    allwords = set()
    for p in phonemes:
        for s in p.get("sets", [{"word": p.get("word", ""), "options": p.get("options", [])}]):
            allwords.add(s.get("word", ""))
            for o in s.get("options", []):
                allwords.add(o)
    ipa_map = {w.lower(): word_ipa(w) for w in allwords if w}
    ph = json.dumps(phonemes, ensure_ascii=False).replace("</", "<\\/")
    dp = json.dumps(_load_json("daily_phrases.json"), ensure_ascii=False).replace("</", "<\\/")
    pi = json.dumps(ipa_map, ensure_ascii=False).replace("</", "<\\/")
    return ("<section id='listening' class='tabpanel hidden'>"
            "<h1>Listening — train your ear</h1>"
            "<p class='sub'>Vowels &amp; consonants: hear a word, choose which one you heard. "
            "Daily phrases: hear a real sentence at adjustable speed and type what you catch.</p>"
            "<div id='ls-head'></div><div id='ls-body'></div></section>"
            "<script>window.PHONEMES=%s;window.DAILY_PHRASES=%s;window.PHONEME_IPA=%s;%s</script>"
            % (ph, dp, pi, _LISTEN_JS))


def _register_panel():
    return ("<section id='register' class='tabpanel hidden'>"
            "<h1>Register — say it at the right formality</h1>"
            "<p class='sub'>Same intent, three formality levels. Slide to see what fits — and what "
            "sounds too casual or too stiff.</p>"
            "<div id='rg-body'></div></section>"
            "<script>" + _REGISTER_JS + "</script>")


_MANDARIN_JS = r"""
(function(){
 var S=window.SkillStore; var DATA=window.ZH_DATA||[];
 var CAT={vowel:'Vowels',consonant:'Consonants',final:'Finals & Clusters',stress:'Stress & Rhythm'};
 var ROOT={1:"① A contrast that does not exist in Mandarin — your ear has not learned it yet",
           2:"② Mandarin syllables cannot end the way English ones do",
           3:"③ Mandarin is syllable-timed; English is stress-timed"};
 var STATE={mode:'perception',cat:'all',cur:null,curP:null};
 function esc(s){return S.esc(s);}
 function body(){return document.getElementById('mz-body');}
 function inCat(d){return STATE.cat==='all'||d.cat===STATE.cat;}
 function record(id,ok){var s=S.get('mz_stats',{});var c=s[id]||{a:0,c:0};c.a++;if(ok)c.c++;s[id]=c;S.set('mz_stats',s);}
 function acc(id){var c=S.get('mz_stats',{})[id];return c&&c.a?c.c/c.a:null;}
 function srs(id,g){var s=S.get('mz_srs',{});s[id]=S.schedule(s[id],g);S.set('mz_srs',s);}
 function shuffle(a){for(var i=a.length-1;i>0;i--){var j=Math.floor(Math.random()*(i+1));var t=a[i];a[i]=a[j];a[j]=t;}return a;}

 function perceptionPool(){var out=[];DATA.forEach(function(d){if(!inCat(d))return;(d.sets||[]).forEach(function(set){if(set.length>=2)out.push({item:d,set:set});});});return out;}
 function renderPerception(){
   var pool=perceptionPool();
   if(!pool.length){body().innerHTML="<div class='card'>No minimal-pair sets in this category — try All, or use Reference / Say it.</div>";return;}
   var pick=pool[Math.floor(Math.random()*pool.length)];
   var target=pick.set[Math.floor(Math.random()*pick.set.length)];
   STATE.cur={item:pick.item,target:target};
   var opts=shuffle(pick.set.slice());
   body().innerHTML="<div class='card'><div class='hint'>"+esc(CAT[pick.item.cat])+" · "+esc(pick.item.title)+"</div>"+
     "<p class='sub'>Listen, then tap the word you heard.</p>"+
     "<button class='btn' data-say=\""+esc(target)+"\">🔊 Play</button> "+
     "<button class='btn small' data-say=\""+esc(target)+"\">↻ Again</button>"+
     "<div class='ssopts' style='margin-top:14px'>"+opts.map(function(o){return "<button class='btn mzopt' data-w=\""+esc(o)+"\">"+esc(o)+"</button>";}).join('')+"</div>"+
     "<div class='mzreveal'></div></div>";
   setTimeout(function(){S.speak(target,0.95);},200);
 }
 function answerPerception(w){
   var c=STATE.cur;if(!c)return;var ok=w===c.target;
   body().querySelectorAll('.mzopt').forEach(function(b){var bw=b.getAttribute('data-w');
     b.style.background=bw===c.target?'#43c59e':(bw===w?'#ff6b6b':'#1f3542');b.style.color=(bw===c.target||bw===w)?'#08222b':'#9aa3bf';b.disabled=true;});
   record(c.item.id,ok);srs(c.item.id,ok?2:0);
   body().querySelector('.mzreveal').innerHTML="<div class='card' style='margin-top:10px'>"+
     (ok?"<b style='color:#43c59e'>✓ Correct — it was "+esc(c.target)+"</b>":"<b style='color:#ff6b6b'>It was "+esc(c.target)+", you picked "+esc(w)+"</b>")+
     "<p>"+esc(c.item.explain)+"</p><p class='hint'>"+esc(c.item.note||'')+"</p>"+
     "<button class='btn small' data-say=\""+esc(c.target)+"\">🔊 "+esc(c.target)+"</button> "+
     "<button class='btn' onclick='MZ.next()'>Next →</button></div>";
 }
 function prodWords(){var w=[];DATA.forEach(function(d){if(!inCat(d))return;(d.sets||[]).forEach(function(set){set.forEach(function(x){w.push({w:x,item:d});});});});return w;}
 function renderProduction(){
   var words=prodWords();
   if(!words.length){body().innerHTML="<div class='card'>No drillable words here — try All or Reference.</div>";return;}
   var p=words[Math.floor(Math.random()*words.length)];STATE.curP=p;
   body().innerHTML="<div class='card'><div class='hint'>"+esc(CAT[p.item.cat])+" · "+esc(p.item.title)+"</div>"+
     "<div style='font-size:32px;font-weight:800;margin:8px 0'>"+esc(p.w)+"</div>"+
     "<button class='btn' data-say=\""+esc(p.w)+"\">🔊 Hear it</button>"+
     "<button class='btn rec mzrec'>● Record &amp; score</button> <span class='mzmsg hint'></span>"+
     "<div class='mzreveal'></div></div>";
   var btn=body().querySelector('.mzrec'),msg=body().querySelector('.mzmsg');var mr=null,ch=[];
   btn.addEventListener('click',function(){
     if(btn.classList.contains('on')){if(mr)mr.stop();return;}
     if(!navigator.mediaDevices){msg.textContent='Recording not supported.';return;}
     navigator.mediaDevices.getUserMedia({audio:true}).then(function(st){
       mr=new MediaRecorder(st);ch=[];mr.ondataavailable=function(e){ch.push(e.data);};
       mr.onstop=function(){st.getTracks().forEach(function(t){t.stop();});btn.classList.remove('on');btn.textContent='● Record & score';msg.textContent='Scoring…';
         var blob=new Blob(ch,{type:'audio/webm'});var fd=new FormData();fd.append('word',p.w);fd.append('audio',blob,'p.webm');
         fetch('/practice',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(j){
           if(j.error){msg.textContent='Azure unavailable: '+j.error;return;}
           var col=j.score>=85?'#43c59e':j.score>=70?'#ffb454':'#ff6b6b';record(p.item.id,j.score>=70);srs(p.item.id,j.score>=85?3:j.score>=70?2:0);
           msg.innerHTML="<b style='color:"+col+"'>"+j.score+"/100</b>";
           body().querySelector('.mzreveal').innerHTML="<p>"+esc(p.item.explain)+"</p><button class='btn' onclick='MZ.next()'>Next →</button>";
         }).catch(function(){msg.textContent='No server / offline.';});};
       mr.start();btn.classList.add('on');btn.textContent='■ Stop';msg.textContent='Recording…';
     }).catch(function(){msg.textContent='Mic permission denied.';});});
 }
 function renderReference(){
   var rc="<div class='card'><b>Three root causes</b><p class='hint' style='line-height:1.9'>"+esc(ROOT[1])+"<br>"+esc(ROOT[2])+"<br>"+esc(ROOT[3])+"</p></div>";
   var cards=DATA.filter(inCat).map(function(d){
     var chips=(d.words||[]).map(function(w){var lbl=w[1]?w[0]+'  '+w[1]:w[0];return "<button class='btn small' data-say=\""+esc(w[0])+"\">"+esc(lbl)+"</button>";}).join(' ');
     var a=acc(d.id);var tag=a==null?'':" · <b style='color:"+(a>=0.8?'#43c59e':a>=0.6?'#ffb454':'#ff6b6b')+"'>"+Math.round(a*100)+"%</b>";
     return "<div class='card'><div style='display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap'><b style='font-family:ui-monospace,monospace;font-size:17px'>"+esc(d.title)+"</b>"+
       "<span class='hint'>"+esc(CAT[d.cat])+" · root "+d.root+tag+"</span></div>"+
       "<div style='margin:8px 0;display:flex;gap:6px;flex-wrap:wrap'>"+chips+"</div><p>"+esc(d.explain)+"</p>"+
       "<p class='hint' style='border-left:3px solid var(--line);padding-left:8px'>"+esc(d.note||'')+"</p></div>";
   }).join('');
   body().innerHTML=rc+cards;
 }
 function cats(){return "<div class='ssbtypes'>"+['all','vowel','consonant','final','stress'].map(function(c){
   return "<button class='btn small"+(STATE.cat===c?' active':'')+"' onclick='MZ.cat(\""+c+"\")'>"+(c==='all'?'All':esc(CAT[c]))+"</button>";}).join('')+"</div>";}
 function render(){
   document.getElementById('mz-head').innerHTML="<div class='drillnav'>"+
     "<button class='btn small"+(STATE.mode==='perception'?' active':'')+"' onclick='MZ.mode(\"perception\")'>👂 Hear the difference</button>"+
     "<button class='btn small"+(STATE.mode==='production'?' active':'')+"' onclick='MZ.mode(\"production\")'>🎤 Say it</button>"+
     "<button class='btn small"+(STATE.mode==='reference'?' active':'')+"' onclick='MZ.mode(\"reference\")'>📚 Reference</button></div>"+cats();
   if(STATE.mode==='perception')renderPerception();else if(STATE.mode==='production')renderProduction();else renderReference();
 }
 window.MZ={mode:function(m){STATE.mode=m;render();},cat:function(c){STATE.cat=c;render();},next:function(){render();}};
 document.addEventListener('click',function(e){var b=e.target.closest&&e.target.closest('.mzopt');if(b&&!b.disabled)answerPerception(b.getAttribute('data-w'));});
 if(document.getElementById('mz-body'))render();
})();
"""


def _load_mandarin_data():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandarin_contrasts.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _mandarin_panel():
    data = _load_mandarin_data()
    if not data:
        return ""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return ("<section id='mandarin' class='tabpanel hidden'>"
            "<h1>Mandarin &rarr; English pronunciation</h1>"
            "<p class='sub'>The sounds Mandarin never trained your ear or mouth to make. "
            "Train perception first (hear the difference), then production — the order that "
            "actually works. %d contrast cards across vowels, consonants, finals/clusters and rhythm.</p>"
            "<div id='mz-head'></div><div id='mz-body'></div></section>"
            "<script>window.ZH_DATA=%s;%s</script>" % (len(data), payload, _MANDARIN_JS))


_READING_JS = r"""
(function(){
 var S=window.SkillStore;
 function all(){ return window.READING_ITEMS||[]; }
 function body(){ return document.getElementById('reading-body'); }
 function bestScore(key){ var a=S.scores(key); return a.length?Math.max.apply(null,a.map(function(x){return x.s;})):null; }
 function hideHigh(){ return S.get('reading_hide90',false); }
 function render(){
   var el=body(); if(!el)return;
   var full=all();
   var hide=hideHigh();
   var list=full.map(function(s,i){return {s:s,i:i};}).filter(function(o){
     if(!hide) return true;
     var b=bestScore('reading:'+o.s.id); return !(b!=null && b>=90); });
   var hiddenCount=full.length-list.length;
   var toggleBtn="<button class='btn small' onclick='READ.toggleHideHigh()' style='"+
     (hide?'background:var(--accent);color:#08222b;border-color:var(--accent)':'')+"'>"+
     (hide?'☑':'☐')+" Hide passages scoring 90+</button>";
   var controls=full.length>1 ? "<div style='display:flex;gap:8px;justify-content:flex-end;align-items:center;margin:0 0 10px;flex-wrap:wrap'>"+
     (hiddenCount?"<span class='hint' style='margin-right:auto'>"+hiddenCount+" hidden (score ≥ 90)</span>":"")+
     toggleBtn+
     "<button class='btn small' onclick='READ.expandAll(true)'>Expand all</button><button class='btn small' onclick='READ.expandAll(false)'>Collapse all</button></div>" : "";
   var cards = list.length? list.map(function(o,idx){
     var s=o.s, i=o.i;
     var key='reading:'+s.id;
     var open=idx===0;
     return "<div class='card reading-card' data-reading='"+i+"'><div style='display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:center'>"+
       "<div><b>"+S.esc(s.name)+"</b><div class='hint'>"+S.esc(s.when||'')+" · polished from your recording</div></div><span>"+
       "<button class='btn small' onclick='READ.toggle("+i+")' aria-expanded='"+(open?'true':'false')+"'>"+(open?'▾ Collapse':'▸ Expand')+"</button> "+
       "<button class='btn small' data-say=\""+S.esc(s.text)+"\">🔊 Listen</button> "+
       "<button class='btn small strec' data-key=\""+S.esc(key)+"\" data-ref=\""+S.esc(s.text)+"\">● Read &amp; score</button> "+
       "<button class='btn small' onclick='READ.use("+i+")'>↗ Read in speaking analysis</button>"+
       "</span></div>"+
       "<div class='reading-detail' style='"+(open?'':'display:none;')+"'>"+
       "<div class='sthist' style='margin-top:6px'>"+S.spark(key)+"</div>"+
       "<span class='stmsg hint'></span>"+
       "<p style='white-space:pre-wrap;margin-top:8px'>"+S.esc(s.text)+"</p></div></div>";
   }).join('') : (full.length? "<div class='card'><b>All passages scoring 90+ are hidden.</b><p class='hint'>Untick the filter above to see them.</p></div>" : "<div class='card'><b>No polished readings yet.</b><p class='hint'>Analyze an uploaded recording with grammar/word-choice feedback enabled. Its polished version will appear here automatically.</p></div>");
   el.innerHTML=controls+cards;
 }
 // record whole-story reading -> Azure score -> log to history
 document.addEventListener('click',function(e){
   var b=e.target.closest && e.target.closest('.strec'); if(!b) return;
   var card=b.closest('.card'); var msg=card.querySelector('.stmsg'); var hist=card.querySelector('.sthist');
   var key=b.getAttribute('data-key'), ref=b.getAttribute('data-ref');
   if(b.__mr){ b.__mr.stop(); return; }
   if(!navigator.mediaDevices){ msg.textContent='Recording not supported.'; return; }
   navigator.mediaDevices.getUserMedia({audio:true}).then(function(st){
     var mr=new MediaRecorder(st); var ch=[]; b.__mr=mr;
     mr.ondataavailable=function(ev){ch.push(ev.data);};
     mr.onstop=function(){ st.getTracks().forEach(function(t){t.stop();}); b.__mr=null; b.textContent='● Read & score'; b.classList.remove('on'); msg.textContent='Scoring…';
       var blob=new Blob(ch,{type:'audio/webm'}); var fd=new FormData(); fd.append('word',ref); fd.append('audio',blob,'s.webm');
       fetch('/practice',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(j){
         if(j.error){ msg.textContent='Azure unavailable: '+j.error; return; }
         S.logScore(key, j.score); var col=j.score>=85?'#43c59e':j.score>=70?'#ffb454':'#ff6b6b';
         msg.innerHTML=" <b style='color:"+col+"'>"+j.score+"/100</b>"; hist.innerHTML=S.spark(key);
       }).catch(function(){ msg.textContent='No server / offline.'; });
     };
     mr.start(); b.classList.add('on'); b.textContent='■ Stop'; msg.textContent='Reading… tap Stop when done.';
   }).catch(function(){ msg.textContent='Mic permission denied.'; });
 });
 window.READ={
   toggle:function(i){ var card=document.querySelector('.reading-card[data-reading="'+i+'"]'); if(!card)return;
     var detail=card.querySelector('.reading-detail'), btn=card.querySelector('[aria-expanded]');
     var open=detail.style.display==='none'; detail.style.display=open?'':'none';
     btn.textContent=open?'▾ Collapse':'▸ Expand'; btn.setAttribute('aria-expanded',open?'true':'false'); },
   expandAll:function(open){ document.querySelectorAll('.reading-card').forEach(function(card){
     var detail=card.querySelector('.reading-detail'), btn=card.querySelector('[aria-expanded]');
     detail.style.display=open?'':'none'; btn.textContent=open?'▾ Collapse':'▸ Expand'; btn.setAttribute('aria-expanded',open?'true':'false'); }); },
   use:function(i){ var s=all()[i]; if(!s)return; var t=document.getElementById('transcript'); if(t){ t.value=s.text; }
     var a=document.querySelector('a[data-panel=newrec]'); if(a){ a.click(); window.scrollTo(0,0); } },
   toggleHideHigh:function(){ S.set('reading_hide90', !hideHigh()); render(); }
 };
 if(document.getElementById('reading-body')) render();
})();
"""


def _reading_panel(items):
    """Personal reading material, one polished text for each analyzed upload."""
    readings = []
    for i, item in enumerate(reversed(sorted(items or [], key=_sort_ts))):
        text = (item.get("polished") or "").strip()
        if not text:
            continue
        title = item.get("title") or "Recording %d" % (i + 1)
        readings.append({
            "id": "%s:%s" % (item.get("date", ""), title),
            "name": title,
            "when": _when_label(item),
            "text": text,
        })
    payload = json.dumps(readings, ensure_ascii=False).replace("</", "<\\/")
    return ("<section id='stories' class='tabpanel hidden'>"
            "<h1>Reading</h1>"
            "<p class='sub'>Your personal reading library. Every passage is the polished version "
            "of an uploaded recording: listen, read it aloud, and score the reread against that "
            "improved version.</p>"
            "<div id='reading-body'></div></section>"
            "<script>window.READING_ITEMS=%s;%s</script>" % (payload, _READING_JS))


def _knowledge_panel():
    ph = _load_json("phonemes.json")
    by_ipa = {p["ipa"].strip("/"): p for p in ph}

    def chips(keys):
        out = []
        for k in keys:
            p = by_ipa.get(k)
            if p:
                out.append("<button class='btn small' data-say=\"%s\">/%s/ "
                           "<span class='hint'>%s</span></button>"
                           % (_attr(p["word"]), _esc(k), _esc(p["word"])))
        return " ".join(out)

    short_v = ["ɪ", "e", "æ", "ɒ", "ʊ", "ʌ", "ə"]
    long_v = ["iː", "ɑː", "ɔː", "uː", "ɜː"]
    dip_v = ["eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "ɪə", "eə", "ʊə"]
    stops = ["p", "b", "t", "d", "k", "g"]
    fric = ["f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h"]
    affr = ["tʃ", "dʒ"]
    nas = ["m", "n", "ŋ"]
    appr = ["l", "r", "w", "j"]

    return ("<section id='knowledge' class='tabpanel hidden'>"
            "<h1>Vowels &amp; consonants — the basics</h1>"
            "<p class='sub'>A plain-English guide to how the sounds work, so the drills make sense. "
            "Tap any chip to hear the sound.</p>"

            "<h2>The one big idea</h2>"
            "<div class='card'><p>Every English sound is a <b>vowel</b> or a <b>consonant</b>. "
            "A <b>vowel</b> is made with the mouth open and air flowing freely — your tongue and lips "
            "shape it, nothing blocks it, and your voice is always on. A <b>consonant</b> is made by "
            "squeezing or blocking the air somewhere in the mouth. Vowels are the loud musical "
            "centre of a syllable; consonants are the edges. In <b>cat</b>, /k/ and /t/ are the "
            "consonant edges and /æ/ is the vowel centre.</p></div>"

            "<h2>How vowels differ</h2>"
            "<div class='card'><p>You change a vowel with three “dials”:</p>"
            "<p>• <b>Tongue height</b> — high (see /iː/, boo /uː/), mid (bed /e/, the /ə/), or low (cat /æ/).<br>"
            "• <b>Front or back</b> — front (see /iː/, cat /æ/), central (about /ə/, bird /ɜː/), back (boo /uː/, hot /ɒ/).<br>"
            "• <b>Lip rounding</b> — rounded (boo /uː/, four /ɔː/) or spread/relaxed (see /iː/, cat /æ/).</p>"
            "<p>Two more things matter in English:</p>"
            "<p>• <b>Tense/long vs lax/short</b>: /iː/ (seat) is long and tight; /ɪ/ (sit) is short and "
            "relaxed. Same on the back vowels: /uː/ (pool) vs /ʊ/ (pull). Mandarin has <i>one</i> vowel "
            "where English has two — that's the classic <b>seat / sit</b> difficulty.<br>"
            "• <b>Monophthong vs diphthong</b>: a monophthong is one steady vowel (/æ/ cat); a "
            "<b>diphthong</b> glides from one vowel to another in a single syllable (/aɪ/ my, /eɪ/ day, "
            "/aʊ/ now).</p></div>"
            "<div class='card'><b>Short vowels</b><div style='margin-top:6px'>%s</div></div>"
            "<div class='card'><b>Long vowels</b><div style='margin-top:6px'>%s</div></div>"
            "<div class='card'><b>Diphthongs (gliding vowels)</b><div style='margin-top:6px'>%s</div></div>"

            "<h2>How consonants differ</h2>"
            "<div class='card'><p>Describe any consonant by answering three questions:</p>"
            "<p><b>1. Voiced or voiceless?</b> Is your voice (the throat buzz) on? Put your fingers on "
            "your throat and say “ffff” then “vvvv” — /v/ buzzes, /f/ doesn't. Voiceless/voiced pairs: "
            "p/b, t/d, k/g, f/v, s/z, θ/ð, ʃ/ʒ, tʃ/dʒ.</p>"
            "<p><b>2. Where is the air blocked?</b> (place) — the lips (/p b m/), lip + teeth (/f v/), "
            "tongue between the teeth (/θ ð/), the ridge behind the top teeth (/t d s z n l/), the hard "
            "palate (/ʃ ʒ tʃ dʒ j/), the soft palate at the back (/k g ŋ/), and the throat (/h/).</p>"
            "<p><b>3. How is the air blocked?</b> (manner):</p>"
            "<p>• <b>Stops</b> — air fully blocked, then released: /p b t d k g/.<br>"
            "• <b>Fricatives</b> — air squeezed through a gap, making a hiss: /f v θ ð s z ʃ ʒ h/.<br>"
            "• <b>Affricates</b> — a stop + a fricative glued together: /tʃ/ (chin), /dʒ/ (jump).<br>"
            "• <b>Nasals</b> — air goes out the nose: /m n ŋ/.<br>"
            "• <b>Approximants</b> — the mouth barely narrows, almost vowel-like: /l r w j/.</p></div>"
            "<div class='card'><b>Stops</b> <span class='hint'>(plosives)</span><div style='margin-top:6px'>%s</div></div>"
            "<div class='card'><b>Fricatives</b><div style='margin-top:6px'>%s</div></div>"
            "<div class='card'><b>Affricates</b><div style='margin-top:6px'>%s</div></div>"
            "<div class='card'><b>Nasals</b><div style='margin-top:6px'>%s</div></div>"
            "<div class='card'><b>Approximants</b><div style='margin-top:6px'>%s</div></div>"

            "<h2>A few things that trip up Mandarin speakers</h2>"
            "<div class='card'><p>• <b>Aspiration</b>: the little puff of air after English p/t/k at the "
            "start of a word (hold your hand near your mouth on “pin”). Mandarin has this puff too, but "
            "uses it to tell sounds apart <i>instead of</i> voicing — which is why English b/p, d/t, g/k "
            "get confused.<br>"
            "• <b>Endings &amp; clusters</b>: English ends syllables in consonants and stacks them "
            "(asked /ɑːskt/, texts /teksts/). Mandarin syllables end only in a vowel, /n/ or /ŋ/, so "
            "endings get dropped or an extra vowel sneaks in (bed → “bed-uh”).<br>"
            "• <b>Missing sounds</b>: /θ ð/ (think, this) and /v/ don't exist in Mandarin; /l/–/r/ and "
            "/n/–/l/ get mixed. Drill these in the <b>中→EN Pronunciation</b> module.</p></div>"

            "<h2>Reading IPA</h2>"
            "<div class='card'><p>IPA (the International Phonetic Alphabet) gives <b>one symbol per "
            "sound</b>, so spelling can't fool you — “sea”, “see” and “scene” all contain /iː/. Symbols "
            "in slashes like /iː/ mean “this sound.” Every chip on this page is tappable — listen and "
            "match the symbol to what you hear.</p></div>"
            "</section>"
            % (chips(short_v), chips(long_v), chips(dip_v),
               chips(stops), chips(fric), chips(affr), chips(nas), chips(appr)))


_HOWTO_CARDS = [
    {"word": "bad", "say": "bad", "ipa": "/bæd/", "map": "a → /æ/  (open · wide · long)",
     "how": "Drop the jaw, spread the lips, tongue low and front. Flatter and a touch longer than bed.",
     "alike": ["≠ bed — don't let it collapse to /e/"], "feel": "Wider mouth, longer vowel than bed."},
    {"word": "bed", "say": "bed", "ipa": "/bed/", "map": "e → /e/  (mid · short)",
     "how": "Jaw only slightly open, tongue mid and front, short and relaxed.",
     "alike": [], "feel": "Small, quick vowel."},
    {"word": "dress", "say": "dress", "ipa": "/dres/", "map": "d + r  ·  short /e/",
     "how": "/dr/ blend (say d, then slide the tongue back for r), then the ‘bed’ vowel /e/ + s.",
     "alike": ["off-target → ‘jazz’"], "feel": "Keep the vowel short /e/; don't over-‘j’ the start."},
    {"word": "dream", "say": "dream", "ipa": "/driːm/", "map": "d + r  ·  long /iː/",
     "how": "/dr/ blend + long /iː/ (smile, tongue high) + m (close the lips).",
     "alike": ["off-target → ‘drum’ (vowel too short)", "off-target → ‘deem’ (r dropped)"],
     "feel": "Hold the long ‘ee’."},
    {"word": "wrong", "say": "wrong", "ipa": "/rɒŋ/", "map": "r (float) + ŋ  ·  silent w",
     "how": "Silent w. Tongue tip pulls back and floats (no contact), lips slightly round; end with the nasal ‑ng.",
     "alike": ["≠ long — that one the tongue touches"], "feel": "Tongue floats = wrong."},
    {"word": "long", "say": "long", "ipa": "/lɒŋ/", "map": "l (touch) + ŋ",
     "how": "Tongue tip touches the ridge, air flows round the sides; end with the nasal ‑ng.",
     "alike": ["≠ wrong — that one the tongue floats"], "feel": "Tongue touches = long."},
    {"word": "wring = ring", "say": "wring", "ipa": "/rɪŋ/", "map": "silent w  ·  r + ɪ + ŋ",
     "how": "The ‘wr‑’ w is silent, so wring and ring are the SAME word in sound. Make r (float) + short ɪ + nasal ŋ.",
     "alike": ["wring and ring are homophones"], "feel": "No ‘w’ sound at all."},
    {"word": "mouth", "say": "mouth", "ipa": "/maʊθ/", "map": "aʊ + /θ/  (tongue → teeth)",
     "how": "m + glide ‘ah→oo’, then push the tongue tip to the teeth and blow. No voice.",
     "alike": ["off-target → ‘mouse’ (/s/ instead of /θ/)"], "feel": "Tongue peeks out to the teeth."},
    {"word": "mouse", "say": "mouse", "ipa": "/maʊs/", "map": "aʊ + /s/  (tongue behind teeth)",
     "how": "m + glide ‘ah→oo’, then hiss /s/ with the tongue behind the teeth.",
     "alike": ["≠ mouth — tongue stays behind"], "feel": "Tongue hides behind the teeth."},
    {"word": "girl", "say": "girl", "ipa": "/ɡɜːrl/", "map": "ɡ + /ɜː/ (bird vowel) + dark l",
     "how": "g + the ‘bird’ vowel /ɜː/ (r‑coloured, tongue central) + a dark l (tongue tip lands up).",
     "alike": ["hard — no /ɜː/ in Mandarin"], "feel": "One smooth ‘gurl’; land the tongue for the l."},
    {"word": "world", "say": "world", "ipa": "/wɜːrld/", "map": "w + ɜː + r + l + d",
     "how": "w (round lips) + /ɜː/ (r‑colour) + l + d — a tough final cluster ‑rld.",
     "alike": ["the ‑rld cluster often gets dropped"], "feel": "Slow it: wer‑l‑d, finish the d."},
    {"word": "three", "say": "three", "ipa": "/θriː/", "map": "θ + r + long /iː/",
     "how": "θ (tongue tip to teeth, blow) → glide into r (tongue back) → long /iː/.",
     "alike": ["off-target → ‘tree’ (/t/) or ‘free’ (/f/)"], "feel": "Start with the tongue on the teeth."},
    {"word": "very", "say": "very", "ipa": "/ˈveri/", "map": "v (teeth on lip) + r",
     "how": "v: top teeth touch the lower lip, voice buzzing. Then r (tongue floats) + /i/.",
     "alike": ["off-target → ‘wary’ (/w/) or ‘ferry’ (/f/)"], "feel": "Feel the buzz on your lip for /v/."},
    # --- Notebook page 1 (bedroom / dream) ---
    {"word": "quilt", "say": "quilt", "ipa": "/kwɪlt/", "map": "kw + short /ɪ/ + lt",
     "how": "Round the lips for /kw/, then a quick relaxed ‘i’ as in ‘sit’, finish ‑lt.",
     "alike": ["off-target → ‘kilt’ (w dropped)"], "feel": "Keep the w: k‑w‑ilt."},
    {"word": "fold", "say": "fold", "ipa": "/fəʊld/", "map": "f + /əʊ/ (oh) + dark l + d",
     "how": "Say ‘oh’, then curl the tongue tip up for a dark l before the d.",
     "alike": ["off-target → ‘food’ (vowel goes long/pure)"], "feel": "oh‑l‑d, not oo."},
    {"word": "bird", "say": "bird", "ipa": "/bɜːrd/", "map": "b + /ɜː/ (nurse) + d",
     "how": "Lips relaxed, tongue central, long r‑coloured vowel — no real ‘i’.",
     "alike": ["off-target → ‘bed’ (vowel too short)", "off-target → ‘bad’ (too open)"],
     "feel": "Same vowel as girl / word / world."},
    {"word": "bedspread", "say": "bedspread", "ipa": "/ˈbedspred/", "map": "two short /e/ (bed·spred)",
     "how": "Both vowels are the short ‘e’ of ‘bed’; keep them equal and flat.",
     "alike": ["off-target → ‘bedspraid’ (2nd vowel drifts to /eɪ/)"], "feel": "bed‑spred, short twice."},
    {"word": "bedpad", "say": "bedpad", "ipa": "/ˈbedpæd/", "map": "/e/ then /æ/",
     "how": "First vowel = bed /e/, second = bad /æ/ — open the jaw wider on the 2nd.",
     "alike": ["off-target if both vowels sound the same"], "feel": "e then a: bed→pad opens the jaw."},
    {"word": "comb", "say": "comb", "ipa": "/kəʊm/", "map": "k + /əʊ/ + m  ·  silent b",
     "how": "Say ‘koam’ and close the lips for m; the b is completely silent.",
     "alike": ["off-target → ‘komb’ (b sounded)"], "feel": "No b — like ‘foam’ with a k."},
    {"word": "towel", "say": "towel", "ipa": "/ˈtaʊəl/", "map": "t + /aʊ/ + schwa + l",
     "how": "‘ow’ then a soft ‘uhl’ — two little beats, ending on the l.",
     "alike": ["off-target → ‘tower’ (ends in r not l)"], "feel": "TOW‑uhl, ends in l."},
    {"word": "tower", "say": "tower", "ipa": "/ˈtaʊər/", "map": "t + /aʊ/ + r",
     "how": "‘ow’ then an r‑colour; tongue floats back, no l at the end.",
     "alike": ["off-target → ‘towel’ (ends in l)"], "feel": "TOW‑er, ends in r."},
    {"word": "wash", "say": "wash", "ipa": "/wɒʃ/", "map": "w + /ɒ/ (hot) + sh",
     "how": "Round lips for w, open ‘o’ as in ‘hot’, finish with ‘sh’.",
     "alike": ["off-target → ‘what's’ (ends /ts/)"], "feel": "w‑o‑sh, soft ending."},
    {"word": "what's", "say": "what's", "ipa": "/wɒts/", "map": "w + /ɒ/ + /ts/",
     "how": "Same start as wash, but end crisp with t+s.",
     "alike": ["off-target → ‘wash’ (/ts/ softens to /ʃ/)"], "feel": "Ends in a sharp ‘ts’."},
    {"word": "dinner", "say": "dinner", "ipa": "/ˈdɪnər/", "map": "DIN‑ner  ·  short /ɪ/",
     "how": "Quick ‘i’ as in ‘sit’, then ‘ner’; stress the first beat.",
     "alike": ["off-target → ‘dean‑er’ (/ɪ/ goes long)"], "feel": "DIN‑ner, short i."},
    {"word": "dessert", "say": "dessert", "ipa": "/dɪˈzɜːrt/", "map": "stress 2nd  ·  di‑ZERT",
     "how": "Weak first beat, punch the second: di‑ZERT. The sweet one.",
     "alike": ["off-target → ‘desert’ (dry) if you stress beat 1"], "feel": "deSSert = 2 s’s = 2nd stress = sweet."},
    {"word": "desert", "say": "desert", "ipa": "/ˈdezərt/", "map": "stress 1st  ·  DEZ‑ert",
     "how": "Punch the first beat: DEZ‑ert. The dry sand one.",
     "alike": ["off-target → ‘dessert’ (sweet) if you stress beat 2"], "feel": "DEsert = dry = 1st stress."},
    # --- Notebook page 2 (Edward the lad) ---
    {"word": "show", "say": "show", "ipa": "/ʃəʊ/", "map": "sh + /əʊ/ (oh)",
     "how": "‘sh’ then a clean ‘oh’ glide.",
     "alike": ["off-target → ‘shoe’ (vowel → /uː/)"], "feel": "Rhymes with ‘go’."},
    {"word": "shoe", "say": "shoe", "ipa": "/ʃuː/", "map": "sh + /uː/ (oo)",
     "how": "‘sh’ then long ‘oo’, lips rounded.",
     "alike": ["off-target → ‘show’ (vowel → /əʊ/)"], "feel": "Rhymes with ‘two’."},
    {"word": "advert", "say": "advert", "ipa": "/ˈædvɜːrt/", "map": "AD‑vert  ·  /æ/ + /ɜː/",
     "how": "Stress beat 1 with ‘a’ as in cat; beat 2 is an r‑vowel.",
     "alike": ["off-target → ‘Edward’ (beat 1 → /e/)"], "feel": "AD‑vert (the ad)."},
    {"word": "Edward", "say": "Edward", "ipa": "/ˈedwərd/", "map": "ED‑wed  ·  /e/ + schwa",
     "how": "Stress beat 1 with ‘e’ as in bed; soft ‘wud’ second beat.",
     "alike": ["off-target → ‘advert’ (beat 1 → /æ/)"], "feel": "ED‑ward (the name)."},
    {"word": "word", "say": "word", "ipa": "/wɜːrd/", "map": "w + /ɜː/ + d  ·  no l",
     "how": "Round lips for w, r‑coloured nurse vowel, end d — no l.",
     "alike": ["off-target → ‘world’ (added dark l)", "off-target → ‘ward’ (vowel opens)"],
     "feel": "No l: w‑er‑d."},
    {"word": "turn", "say": "turn", "ipa": "/tɜːrn/", "map": "t + /ɜː/ + n",
     "how": "r‑coloured vowel (same as bird), end with n.",
     "alike": ["off-target → ‘ten’ (vowel → short /e/)"], "feel": "Long er, not ‘ten’."},
    {"word": "ten", "say": "ten", "ipa": "/ten/", "map": "t + short /e/ + n",
     "how": "Clean short ‘e’ as in bed, quick.",
     "alike": ["off-target → ‘turn’ (vowel r‑colours)"], "feel": "Short and flat: ten."},
    {"word": "red", "say": "red", "ipa": "/red/", "map": "r (float) + /e/ + d",
     "how": "True /r/ (tongue floats, no contact), short e, voiced d.",
     "alike": ["off-target → ‘led’ (/r/ becomes /l/)"], "feel": "r floats; l touches."},
    {"word": "led", "say": "led", "ipa": "/led/", "map": "l (touch) + /e/ + d",
     "how": "Tongue TIP touches the ridge for l, short e.",
     "alike": ["off-target → ‘red’ (l floats into r)", "off-target → ‘lad’ (vowel → /æ/)"],
     "feel": "l = tongue touches."},
    {"word": "lad", "say": "lad", "ipa": "/læd/", "map": "l + /æ/ (wide) + d",
     "how": "l touches, then open the jaw wide for ‘a’ as in cat.",
     "alike": ["off-target → ‘led’ (vowel → /e/)"], "feel": "Wide a: lad."},
    {"word": "celery", "say": "celery", "ipa": "/ˈseləri/", "map": "SEL‑er‑ee  ·  /e/",
     "how": "Beat 1 = ‘e’ as in bed; ‘sel‑uh‑ree’.",
     "alike": ["off-target → ‘salary’ (beat 1 → /æ/)"], "feel": "cELery = e = bed."},
    {"word": "salary", "say": "salary", "ipa": "/ˈsæləri/", "map": "SAL‑er‑ee  ·  /æ/",
     "how": "Beat 1 = ‘a’ as in cat; ‘sal‑uh‑ree’.",
     "alike": ["off-target → ‘celery’ (beat 1 → /e/)"], "feel": "sALary = a = cat (money)."},
    {"word": "are", "say": "are", "ipa": "/ɑːr/", "map": "/ɑː/ + r",
     "how": "Open ‘ah’ with an r‑colour; longer than bare ‘ah’.",
     "alike": ["off-target → ‘ah’ (r dropped)"], "feel": "are = ah + r."},
    {"word": "ah", "say": "ah", "ipa": "/ɑː/", "map": "/ɑː/ (open ah)",
     "how": "Jaw open, tongue low and back; no r, no ending.",
     "alike": ["off-target → ‘are’ (an r sneaks in)"], "feel": "Pure open ah."},
    {"word": "lamb", "say": "lamb", "ipa": "/læm/", "map": "l + /æ/ + m  ·  silent b",
     "how": "l touches, open ‘a’, close the lips for m; the b is silent.",
     "alike": ["off-target → ‘lamp’ (a /p/ appears)"], "feel": "No b, no p: lam."},
    {"word": "lamp", "say": "lamp", "ipa": "/læmp/", "map": "l + /æ/ + m + p",
     "how": "Same as lamb but add a crisp /p/ at the end.",
     "alike": ["off-target → ‘lamb’ (p dropped)"], "feel": "Ends in p."},
    {"word": "them", "say": "them", "ipa": "/ðem/", "map": "/ð/ + /e/ + m",
     "how": "Voiced ‘th’ (tongue peeks, throat buzzes), end with the lips together for m.",
     "alike": ["off-target → ‘then’ (ends /n/)", "off-target → ‘dem’ (/ð/ → /d/)"],
     "feel": "Ends with lips: m."},
    {"word": "then", "say": "then", "ipa": "/ðen/", "map": "/ð/ + /e/ + n",
     "how": "Voiced ‘th’, short e, end with the tongue on the ridge for n.",
     "alike": ["off-target → ‘them’ (ends /m/)", "off-target → ‘den’ (/ð/ → /d/)"],
     "feel": "Ends with tongue: n."},
    {"word": "dyed", "say": "dyed", "ipa": "/daɪd/", "map": "d + /aɪ/ + d  ·  voiced start",
     "how": "Voiced d at both ends (throat buzzes); ‘eye’ in the middle.",
     "alike": ["off-target → ‘tied’ (first d hardens to /t/)"], "feel": "d starts — throat buzzes."},
    {"word": "tied", "say": "tied", "ipa": "/taɪd/", "map": "t + /aɪ/ + d  ·  voiceless start",
     "how": "Voiceless t to start (just air), ‘eye’, voiced d.",
     "alike": ["off-target → ‘dyed’ (/t/ softens to /d/)"], "feel": "t starts — air only."},
    {"word": "brown", "say": "brown", "ipa": "/braʊn/", "map": "b + r + /aʊ/ + n",
     "how": "b then a floating /r/, ‘ow’ glide, end n.",
     "alike": ["off-target → ‘belong’ (/br/ → /bl/)"], "feel": "br, not bl."},
    {"word": "belong", "say": "belong", "ipa": "/bɪˈlɒŋ/", "map": "b + /ɪ/ + l + /ɒŋ/",
     "how": "Weak ‘bi’, then ‘long’ with tongue‑tip l and the ‑ng hum.",
     "alike": ["off-target → ‘brown’ (/bl/ → /br/)"], "feel": "bl + long."},
    {"word": "perm", "say": "perm", "ipa": "/pɜːrm/", "map": "p + /ɜː/ + m",
     "how": "p, r‑coloured nurse vowel (like bird), close the lips for m.",
     "alike": ["off-target → ‘pan’ (vowel → /æ/)", "off-target → ‘pam’ (r dropped)"],
     "feel": "perm = bird vowel + m."},
    {"word": "pan", "say": "pan", "ipa": "/pæn/", "map": "p + /æ/ (wide) + n",
     "how": "p, wide open ‘a’ as in cat, end n.",
     "alike": ["off-target → ‘perm’ (vowel r‑colours)", "off-target → ‘pen’ (vowel → /e/)"],
     "feel": "Wide a: pan."},
    {"word": "live", "say": "live", "ipa": "/lɪv/", "map": "l + short /ɪ/ + v  (verb)",
     "how": "Short ‘i’ as in ‘sit’, end with a buzzing /v/.",
     "alike": ["off-target → ‘leave’ (vowel → long /iː/)"], "feel": "Short i: ‘live here’."},
    {"word": "leave", "say": "leave", "ipa": "/liːv/", "map": "l + long /iː/ + v",
     "how": "Long ‘ee’, smiling, end /v/.",
     "alike": ["off-target → ‘live’ (vowel shortens)"], "feel": "Long ee: ‘leave now’."},
    {"word": "all", "say": "all", "ipa": "/ɔːl/", "map": "/ɔː/ (aw) + dark l",
     "how": "Rounded ‘aw’, then curl the tongue up for a dark l.",
     "alike": ["off-target → ‘or’ (l becomes r)"], "feel": "aw + l."},
    {"word": "or", "say": "or", "ipa": "/ɔːr/", "map": "/ɔː/ (aw) + r",
     "how": "Rounded ‘aw’, then r‑colour; tongue floats, no l.",
     "alike": ["off-target → ‘all’ (an l appears)"], "feel": "aw + r."},
]

_HOWTO_JS = r"""
(function(){var S=window.SkillStore; var CARDS=window.HOWTO_CARDS||[];
 var i=0, flip=false;
 function removed(){ return S.get('hw_removed',[])||[]; }
 function activeCards(){ var r=removed(); return CARDS.filter(function(c){return r.indexOf(c.word)<0;}); }
 var LIST=activeCards(), order=LIST.map(function(_,k){return k;});
 function rebuild(){ LIST=activeCards(); order=LIST.map(function(_,k){return k;}); if(i>=LIST.length)i=Math.max(0,LIST.length-1); }
 function cur(){ return LIST[order[i]]; }
 function body(){ return document.getElementById('hw-body'); }
 function render(){
   var rc=removed().length;
   var restore=rc?"<button class='btn small' onclick='HW.restore()' style='background:#1f3542'>↺ Restore "+rc+" mastered</button>":"";
   if(!LIST.length){ body().innerHTML="<div class='hwcard' style='text-align:center'><div style='font-size:32px;font-weight:800'>🎉 All mastered!</div><p class='hint' style='margin-top:10px'>You've cleared every tricky word.</p></div><div class='hwctl'>"+restore+"</div>"; return; }
   var c=cur(); if(!c){ body().innerHTML=''; return; }
   var actions="<div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap'>"+
     "<button class='btn' data-w=\""+S.esc(c.say)+"\" onclick=\"event.stopPropagation();window.SkillStore.speak(this.getAttribute('data-w'),0.95)\">🔊 Hear</button>"+
     "<button class='btn rec' data-w=\""+S.esc(c.say)+"\" onclick='event.stopPropagation();HW.score(this)'>● Score</button></div>"+
     "<div class='hw-score hint' style='margin-top:10px;min-height:18px'></div>";
   var front="<div style='text-align:center'>"+
     "<div style='font-size:46px;font-weight:800;line-height:1.1'>"+S.esc(c.word)+"</div>"+
     "<div style='font-family:ui-monospace,monospace;color:var(--accent);font-size:22px;margin:8px 0'>"+S.esc(c.ipa)+"</div>"+
     "<div class='hint' style='font-size:16px;margin:6px 0 14px'>"+S.esc(c.map)+"</div>"+
     actions+
     "<p class='hint' style='margin-top:14px;font-size:12px'>tap the card to flip</p></div>";
   var alike=(c.alike||[]).map(function(a){return "<div style='color:var(--warn);margin:5px 0'>⚠️ "+S.esc(a)+"</div>";}).join('');
   var back="<div><div style='font-size:28px;font-weight:800'>"+S.esc(c.word)+
     " <span style='color:var(--accent);font-family:ui-monospace,monospace;font-size:18px'>"+S.esc(c.ipa)+"</span></div>"+
     "<p style='margin-top:12px;font-size:16px'>"+S.esc(c.how)+"</p>"+alike+
     "<div style='background:rgba(67,197,158,.12);border-left:3px solid var(--good);padding:9px 13px;border-radius:8px;margin-top:12px'>Feel it: "+S.esc(c.feel)+"</div></div>";
   body().innerHTML="<div class='hwcard' onclick='HW.flip()'>"+(flip?back:front)+"</div>"+
     "<div class='hwctl'>"+
       "<button class='btn small' onclick='HW.prev()'>◀ Prev</button>"+
       "<button class='btn' onclick='HW.flip()'>Flip</button>"+
       "<button class='btn small' onclick='HW.next()'>Next ▶</button>"+
       "<span style='flex:1'></span>"+
       "<span class='hint'>"+(i+1)+" / "+LIST.length+"</span>"+
       "<button class='btn small' onclick='HW.shuffle()'>🔀 Shuffle</button>"+
       "<button class='btn small' onclick='HW.master()' style='background:rgba(67,197,158,.18);border-color:var(--good);color:var(--good)'>✓ Mastered — remove</button>"+
       restore+"</div>";
 }
 window.HW={ flip:function(){ flip=!flip; render(); },
   next:function(){ if(LIST.length){i=(i+1)%LIST.length;} flip=false; render(); },
   prev:function(){ if(LIST.length){i=(i-1+LIST.length)%LIST.length;} flip=false; render(); },
   shuffle:function(){ for(var k=order.length-1;k>0;k--){var j=Math.floor(Math.random()*(k+1));var t=order[k];order[k]=order[j];order[j]=t;} i=0; flip=false; render(); },
   master:function(){ var c=cur(); if(!c)return; var r=removed(); if(r.indexOf(c.word)<0)r.push(c.word); S.set('hw_removed',r); rebuild(); flip=false; render(); },
   restore:function(){ S.set('hw_removed',[]); rebuild(); i=0; flip=false; render(); },
   score:function(btn){
     var word=(btn.getAttribute('data-w')||'').trim();
     var out=document.querySelector('.hw-score');
     if(btn.__mr){ try{btn.__mr.stop();}catch(e){} return; }
     if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){ if(out)out.textContent='Recording not supported in this browser.'; return; }
     navigator.mediaDevices.getUserMedia({audio:true}).then(function(stream){
       var mr,ch=[]; try{mr=new MediaRecorder(stream);}catch(e){ if(out)out.textContent='Recorder error.'; stream.getTracks().forEach(function(t){t.stop();}); return; }
       btn.__mr=mr; btn.__t0=Date.now(); btn.textContent='■ Stop'; btn.classList.add('on');
       if(out)out.innerHTML='<span style="color:var(--bad)">● recording…</span> say “'+S.esc(word)+'”, then Stop';
       mr.ondataavailable=function(e){ if(e.data&&e.data.size)ch.push(e.data); };
       mr.onstop=function(){
         stream.getTracks().forEach(function(t){t.stop();}); btn.__mr=null; btn.textContent='● Score'; btn.classList.remove('on');
         var blob=new Blob(ch,{type:'audio/webm'});
         if(blob.size<1200 || (Date.now()-btn.__t0)<500){ if(out)out.innerHTML='<span style="color:var(--warn)">Too short — hold the button, say the word, then Stop.</span>'; return; }
         if(out)out.textContent='scoring…';
         var fd=new FormData();
         fd.append('word',word); fd.append('audio',blob,'clip.webm');
         fetch('/practice',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(j){
           if(!out)return;
           if(j.error){ out.textContent='Scoring unavailable: '+j.error; return; }
           var sc=(j.score!=null?Math.max(0,Math.min(100,Math.round(j.score))):null);
           if(sc==null){ out.textContent='No score returned.'; return; }
           S.logScore('word:'+word.toLowerCase(), sc);
           var col=sc>=85?'var(--good)':sc>=70?'var(--warn)':'var(--bad)';
           var extra=(j.accuracy!=null)?' · accuracy '+Math.round(j.accuracy):'';
           out.innerHTML='<b style="color:'+col+'">'+sc+'/100</b>'+extra+' <span class="hint">· saved to Practice</span>';
         }).catch(function(){ if(out)out.textContent='Could not reach the scoring server.'; });
       };
       mr.start(200);
     }).catch(function(){ if(out)out.textContent='Microphone permission denied.'; });
   } };
 if(document.getElementById('hw-body')) render();
})();
"""


def _howto_panel():
    payload = json.dumps(_HOWTO_CARDS, ensure_ascii=False).replace("</", "<\\/")
    return ("<section id='howto' class='tabpanel hidden'>"
            "<h1>Tricky-word flashcards</h1>"
            "<p class='sub'>Front: the word, its IPA and the key sound. Tap to flip for how to make it "
            "and what it turns into if you're off. 🔊 hears it, ● Score records you and grades it "
            "(saved to Practice), 🔀 shuffles, ✓ Mastered removes a word you've nailed (↺ restore any time).</p>"
            "<div id='hw-body'></div></section>"
            "<script>window.HOWTO_CARDS=%s;%s</script>" % (payload, _HOWTO_JS))


def _pronscore_panel():
    """Embedded reference for how the Azure pronunciation score is built."""
    W = "#ffb454"  # warn/yellow (dashboard :root has no --warn)
    fm = ("background:#11152b;border:1px solid var(--line);border-radius:10px;"
          "padding:11px 15px;margin:7px 0;font-family:ui-monospace,Menlo,monospace;font-size:14px")
    box = ("background:var(--card);border-radius:12px;padding:14px 18px;margin:12px 0")
    key = ("background:rgba(70,179,201,.12);border-left:3px solid var(--accent);"
           "padding:9px 13px;border-radius:8px;margin:10px 0")
    flag = ("background:rgba(255,180,84,.12);border-left:3px solid %s;"
            "padding:9px 13px;border-radius:8px;margin:10px 0" % W)
    th = "text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);color:var(--mut)"
    td = "text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top"
    h2 = "font-size:18px;margin:24px 0 8px;border-left:3px solid var(--accent);padding-left:10px"
    pill_on = ("display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px;"
               "font-weight:600;background:rgba(67,197,158,.18);color:var(--good)")
    pill_off = ("display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px;"
                "font-weight:600;background:rgba(255,107,107,.18);color:var(--bad)")
    return (
        "<section id='pronscore' class='tabpanel hidden'>"
        "<h1>PronScore — how the score is built</h1>"
        "<p class='sub'>Azure Pronunciation Assessment reference. "
        "Source: Microsoft Learn, “Use pronunciation assessment”.</p>"

        "<h2 style='%s'>1 · Four sub-scores</h2>" % h2 +
        "<div style='%s'><table style='border-collapse:collapse;width:100%%;font-size:14px'>" % box +
        "<tr><th style='%s'>Sub-score</th><th style='%s'>What it measures</th></tr>" % (th, th) +
        "<tr><td style='%s'><b>Accuracy</b></td><td style='%s'>How closely each <b>phoneme</b> matches a native speaker; word/full-text accuracy is aggregated up from phonemes.</td></tr>" % (td, td) +
        "<tr><td style='%s'><b>Fluency</b></td><td style='%s'>How native-like your <b>silent breaks / pausing</b> between words are.</td></tr>" % (td, td) +
        "<tr><td style='%s'><b>Completeness</b></td><td style='%s'>Ratio of words you <b>said</b> ÷ words in the reference text.</td></tr>" % (td, td) +
        "<tr><td style='%s'><b>Prosody</b></td><td style='%s'>Naturalness: <b>stress, intonation, speed, rhythm</b>. Drives Monotone / break tags.</td></tr>" % (td, td) +
        "</table><p class='hint' style='margin:8px 0 0'>A word is flagged <b>Mispronunciation</b> when its accuracy is below <b>60</b>.</p></div>"

        "<h2 style='%s'>2 · The formula</h2>" % h2 +
        "<p class='sub'>Sort the available sub-scores <b>low → high</b> as "
        "<span style='font-family:ui-monospace,monospace'>s0</span> (lowest) … "
        "<span style='font-family:ui-monospace,monospace'>s3</span> (highest):</p>"
        "<div style='%s'><b>Reading a script (this app):</b>" % box +
        "<div style='%s'>with prosody:&nbsp;&nbsp;&nbsp;PronScore = <span style='color:var(--accent)'>0.4·s0</span> + 0.2·s1 + 0.2·s2 + 0.2·s3</div>" % fm +
        "<div style='%s'>without prosody: PronScore = <span style='color:var(--accent)'>0.6·s0</span> + 0.2·s1 + 0.2·s2</div>" % fm +
        "<b style='display:block;margin-top:10px'>Speaking freely (no completeness):</b>" +
        "<div style='%s'>with prosody:&nbsp;&nbsp;&nbsp;PronScore = <span style='color:var(--accent)'>0.6·s0</span> + 0.2·s1 + 0.2·s2</div>" % fm +
        "<div style='%s'>without prosody: PronScore = <span style='color:var(--accent)'>0.6·s0</span> + 0.4·s1</div>" % fm +
        "</div>"
        "<div style='%s'><b>Remember:</b> <span style='color:var(--accent)'>s0 is your <u>worst</u> sub-score and carries 40–60%% of the weight.</span> "
        "Your weakest dimension dominates the number — by design. One floor-level area (e.g. prosody on a single word) drags the whole score down.</div>" % key +

        "<h2 style='%s'>3 · Worked example: “ah” capped at ~68–73</h2>" % h2 +
        "<div style='%s'>Prosody on, one word. Accuracy 96, Fluency 80, Prosody 55 → sorted s0=55, s1=80, s2=96:" % box +
        "<div style='%s'>0.6·55 + 0.2·80 + 0.2·96 = 33 + 16 + 19.2 = <b style='color:%s'>68.2</b></div>" % (fm, W) +
        "Your 96 accuracy barely counted because weak prosody was s0. Fix: don’t score prosody on one word.</div>"

        "<h2 style='%s'>4 · How English Coach is configured</h2>" % h2 +
        "<div style='%s'><table style='border-collapse:collapse;width:100%%;font-size:14px'>" % box +
        "<tr><th style='%s'>Setting</th><th style='%s'>Single-word drill</th><th style='%s'>Full-story analysis</th></tr>" % (th, th, th) +
        "<tr><td style='%s'>Locale (accent)</td><td style='%s'><span style='color:var(--accent)'>en-US</span></td><td style='%s'><span style='color:var(--accent)'>en-US</span></td></tr>" % (td, td, td) +
        "<tr><td style='%s'>Grading</td><td style='%s'>HundredMark (0–100)</td><td style='%s'>HundredMark</td></tr>" % (td, td, td) +
        "<tr><td style='%s'>Granularity</td><td style='%s'>Phoneme</td><td style='%s'>Phoneme</td></tr>" % (td, td, td) +
        "<tr><td style='%s'>Miscue (word match)</td><td style='%s'><span style='%s'>OFF</span> homophones (tied/tide) not docked</td><td style='%s'><span style='%s'>ON</span> catches skipped/inserted words</td></tr>" % (td, td, pill_off, td, pill_on) +
        "<tr><td style='%s'>Prosody</td><td style='%s'><span style='%s'>OFF</span> meaningless for one word</td><td style='%s'><span style='%s'>ON</span> rhythm / monotone feedback</td></tr>" % (td, td, pill_off, td, pill_on) +
        "<tr><td style='%s'>Number shown</td><td style='%s'><b>Accuracy</b> only</td><td style='%s'>blended <b>PronScore</b></td></tr>" % (td, td, td) +
        "</table></div>"

        "<h2 style='%s'>5 · Accent locale</h2>" % h2 +
        "<div style='%s'>Both single words and stories grade against <b>en-US</b> (American target). "
        "This keeps every Azure feature available (prosody, syllable breakdown, spoken-phoneme candidates, IPA names are all en-US only). "
        "en-GB was tried for single words but graded British targets too harshly on an American-leaning voice, so it was reverted. "
        "Single words still turn <b>prosody off</b> (meaningless for one word) and grade on <b>accuracy</b>; stories keep the full blended PronScore.</div>" % flag +

        "<h2 style='%s'>6 · The app’s own colour &amp; mastery rules</h2>" % h2 +
        "<div style='%s'>Trend dots: <span style='color:var(--good)'>■ green ≥ 85</span> &nbsp; "
        "<span style='color:%s'>■ yellow 75–84</span> &nbsp; <span style='color:var(--bad)'>■ red &lt; 75</span>.<br>"
        "<b>Mastered</b> (Practice filter) = <b>last 3 attempts all ≥ 85</b>.<br>"
        "No hidden strictness curve on practice scores — 85 means a raw Azure 85.</div>" % (box, W) +
        "</section>")


# ---------------------------------------------------------------------------
# Vocabulary reports — what you actually produce vs what you actually take in
# ---------------------------------------------------------------------------
# Function words carry no vocabulary signal: everyone uses "the" constantly, so
# they swamp any frequency view. They stay in the totals (they ARE words you
# said) but can be filtered out of the tables.
_FUNCTION_WORDS = set("""
a an the and or but so because if then than that this these those there here
i me my mine myself you your yours we us our ours they them their theirs
he him his she her hers it its
am is are was were be been being do does did doing done have has had having
will would shall should can could may might must
of in on at to for with from by about into over under between through during
as up down out off again just very too also only even still yet
what which who whom whose when where why how
not no nor none some any all both each few more most other another such own
same s t don didn doesn isn aren wasn weren won wouldn couldn shouldn
one two three
""".split())


def _tokenize(text):
    """Words for vocabulary counting: lowercase, apostrophes kept.

    Keeps contractions whole ("don't" is one word you either know or don't) and
    drops digits and CJK, which aren't English vocabulary.
    """
    out = []
    for w in re.findall(r"[A-Za-z][A-Za-z']*", text or ""):
        w = w.lower().strip("'")
        if len(w) > 1 or w in ("a", "i"):
            out.append(w)
    return out


def load_transcripts(library=None, min_words=5):
    """Every recording's transcript — the words you actually said.

    Read from <stem>.txt rather than result.json, which doesn't keep the
    transcript. Recordings below `min_words` are placeholders or non-English
    clips and would only add noise.

    Walks the whole library rather than just its top level: recordings can be
    filed into subfolders (a month, a topic), and a grouping folder must never
    make a recording invisible to this report.
    """
    lib = library or library_dir()
    out = []
    if not os.path.isdir(lib):
        return out
    dirs = []
    for dirpath, _subdirs, _files in os.walk(lib):
        if dirpath != lib:
            dirs.append(dirpath)
    for d in sorted(dirs):
        name = os.path.basename(d)
        txt = os.path.join(d, name + ".txt")
        if not os.path.exists(txt):
            continue
        try:
            with open(txt, encoding="utf-8") as f:
                text = f.read().strip()
        except Exception:
            continue
        words = _tokenize(text)
        if len(words) < min_words:
            continue
        date, title, analyzed = None, name, False
        res = os.path.join(d, name + ".result.json")
        if os.path.exists(res):
            analyzed = True
            try:
                with open(res, encoding="utf-8") as f:
                    r = json.load(f)
                date = r.get("date")
                title = r.get("title") or name
            except Exception:
                pass
        if not date:
            at = _timestamp_from_name(name)
            if at:
                date = at.date().isoformat()
            else:
                try:
                    import datetime
                    date = datetime.date.fromtimestamp(
                        os.path.getmtime(txt)).isoformat()
                except OSError:
                    date = ""
        out.append({"stem": name, "title": title, "date": date or "",
                    "text": text, "words": words, "analyzed": analyzed})
    out.sort(key=lambda r: (r["date"], r["stem"]))
    return out


def speaking_vocabulary(transcripts=None):
    """Vocabulary statistics across everything you've recorded."""
    from collections import Counter
    rows = transcripts if transcripts is not None else load_transcripts()
    counts = Counter()
    first_seen = {}
    sessions = []
    seen = set()
    for r in rows:
        before = len(seen)
        counts.update(r["words"])
        for w in r["words"]:
            if w not in first_seen:
                first_seen[w] = r["date"] or r["title"]
        seen.update(r["words"])
        sessions.append({
            "date": r["date"], "title": r["title"],
            "tokens": len(r["words"]), "types": len(set(r["words"])),
            "new": len(seen) - before, "cumulative": len(seen),
        })
    tokens = sum(counts.values())
    types = len(counts)
    hapax = sum(1 for _w, n in counts.items() if n == 1)
    return {
        "counts": counts, "first_seen": first_seen, "sessions": sessions,
        "tokens": tokens, "types": types, "hapax": hapax,
        "recordings": len(rows),
        "ttr": round(types / tokens, 3) if tokens else 0.0,
    }


def _vocab_growth_svg(sessions):
    """Cumulative distinct words over time — the line that should keep rising."""
    pts = [s for s in sessions if s.get("cumulative")]
    if len(pts) < 2:
        return ""
    W, H = 820, 260
    L, R, T, B = 56, 16, 18, 42
    iw, ih = W - L - R, H - T - B
    n = len(pts)
    ymax = max(s["cumulative"] for s in pts)
    ymax = int(-(-ymax // 100) * 100) or 100

    def x(i):
        return L + (iw * i / (n - 1) if n > 1 else iw / 2)

    def y(v):
        return T + ih * (1 - v / float(ymax))

    grid = ""
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        gy = T + ih * (1 - f)
        grid += ("<line x1='%s' y1='%.1f' x2='%s' y2='%.1f' stroke='#24404c'/>"
                 "<text x='%s' y='%.1f' fill='#9aa3bf' font-size='11' "
                 "text-anchor='end'>%d</text>"
                 % (L, gy, W - R, gy, L - 8, gy + 4, round(ymax * f)))
    poly = " ".join("%.1f,%.1f" % (x(i), y(s["cumulative"])) for i, s in enumerate(pts))
    dots = "".join("<circle cx='%.1f' cy='%.1f' r='3.2' fill='var(--accent)'>"
                   "<title>%s — %d distinct words (+%d new)</title></circle>"
                   % (x(i), y(s["cumulative"]), _esc(s["date"] or s["title"]),
                      s["cumulative"], s["new"])
                   for i, s in enumerate(pts))
    bars = "".join(
        "<rect x='%.1f' y='%.1f' width='6' height='%.1f' rx='2' fill='var(--good)' "
        "opacity='.55'><title>%s: +%d new words</title></rect>"
        % (x(i) - 3, T + ih - (ih * s["new"] / float(ymax)),
           max(0.0, ih * s["new"] / float(ymax)), _esc(s["date"] or s["title"]), s["new"])
        for i, s in enumerate(pts))
    labels = ""
    step = max(1, -(-n // 8))
    for i, s in enumerate(pts):
        if i % step and i != n - 1:
            continue
        lbl = (s["date"] or "")[5:] or str(i + 1)
        labels += ("<text x='%.1f' y='%s' fill='#9aa3bf' font-size='11' "
                   "text-anchor='middle'>%s</text>" % (x(i), H - 14, _esc(lbl)))
    return ("<div class='chartcard'>"
            "<svg viewBox='0 0 %s %s' width='100%%' style='max-width:%spx'>%s"
            "<polyline fill='none' stroke='var(--accent)' stroke-width='2.5' points='%s'/>"
            "%s%s%s</svg>"
            "<div class='legend'><span class='lg'><i style='background:var(--accent)'></i>"
            "Distinct words so far</span><span class='lg'>"
            "<i style='background:var(--good)'></i>New words that session</span></div></div>"
            % (W, H, W, grid, poly, bars, dots, labels))


_VOCAB_TABLE_JS = r"""
// A word-frequency table, instantiated once per report. Both vocabulary panels
// live on the same page, so this has to be a component with its own container
// and state rather than a singleton bound to a fixed element id.
window.VocabTable=function(bodyId, getRows){
 var S=window.SkillStore;
 var ST={sort:'n', dir:'desc', q:'', content:false, page:0, per:150};
 function DATA(){ return getRows()||[]; }
 function body(){ return document.getElementById(bodyId); }
 function rows(){
   var q=ST.q.toLowerCase();
   var r=DATA().filter(function(x){
     if(ST.content && x.f) return false;
     return !q || x.w.indexOf(q)>=0;
   });
   r.sort(function(a,b){
     var va = ST.sort==='w' ? a.w : (ST.sort==='n' ? a.n : (a.d||''));
     var vb = ST.sort==='w' ? b.w : (ST.sort==='n' ? b.n : (b.d||''));
     if(va<vb) return ST.dir==='asc'?-1:1;
     if(va>vb) return ST.dir==='asc'?1:-1;
     return a.w<b.w?-1:1;
   });
   return r;
 }
 function render(){
   var el=body(); if(!el) return;
   var r=rows(), total=DATA().length;
   var start=ST.page*ST.per, page=r.slice(start, start+ST.per);
   function hd(k,label,extra){
     var ar = ST.sort===k ? (ST.dir==='asc'?' ▲':' ▼') : '';
     return "<th class='vsort' data-k='"+k+"' style='cursor:pointer"+(extra||'')+"'>"+label+ar+"</th>";
   }
   var bar="<div style='display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 2px'>"+
     "<input class='vq' placeholder='find a word…' value='"+S.esc(ST.q)+"' style='max-width:200px'>"+
     "<button class='btn small vtoggle'"+(ST.content?" style='background:var(--accent);color:#08222b'":"")+
       ">"+(ST.content?'☑':'☐')+" Content words only</button>"+
     "<span class='hint'>showing "+(r.length?start+1:0)+"–"+Math.min(start+ST.per,r.length)+
       " of "+r.length+(r.length!==total?(" (filtered from "+total+")"):"")+"</span>"+
     "<span style='flex:1'></span>"+
     "<button class='btn small vpage' data-d='-1'"+(ST.page<=0?" disabled":"")+">← prev</button>"+
     "<button class='btn small vpage' data-d='1'"+(start+ST.per>=r.length?" disabled":"")+">next →</button>"+
     "</div>";
   if(!r.length){ el.innerHTML=bar+"<div class='card'>No words match.</div>"; return; }
   var max=r[0]&&ST.sort==='n'&&ST.dir==='desc'?r[0].n:Math.max.apply(null,r.map(function(x){return x.n;}));
   var body_=page.map(function(x){
     var w=Math.max(2, Math.round(100*x.n/(max||1)));
     return "<tr><td><b>"+S.esc(x.w)+"</b>"+(x.f?" <span class='hint'>fn</span>":"")+"</td>"+
       "<td style='text-align:right'>"+x.n+"</td>"+
       "<td style='width:40%'><div class='sb-t' style='height:7px'><div class='sb-f' style='width:"+w+"%'></div></div></td>"+
       "<td class='hint'>"+S.esc(x.d||'')+"</td>"+
       "<td style='text-align:right'><button class='btn small' data-say=\""+S.esc(x.w)+"\">🔊</button></td></tr>";
   }).join('');
   el.innerHTML=bar+"<table class='pwt'><tr>"+hd('w','Word')+hd('n','Times used',';text-align:right')+
     "<th></th>"+hd('d','First used')+"<th></th></tr>"+body_+"</table>";
 }
 // Listeners go on the container, not the document: innerHTML is replaced on
 // every render but the container itself survives, so they bind once and each
 // instance only ever sees its own events.
 function wire(){
   var el=body(); if(!el || el.__wired) return; el.__wired=true;
   el.addEventListener('input', function(e){
     if(e.target && e.target.classList && e.target.classList.contains('vq')){
       ST.q=e.target.value||''; ST.page=0; render();
       var i=body().querySelector('.vq');
       if(i){ i.focus(); i.setSelectionRange(i.value.length, i.value.length); }
     }
   });
   el.addEventListener('click', function(e){
     var t=e.target; if(!t||!t.closest) return;
     var s=t.closest('.vsort');
     if(s){ var k=s.getAttribute('data-k');
       if(ST.sort===k){ ST.dir = ST.dir==='asc'?'desc':'asc'; }
       else { ST.sort=k; ST.dir = (k==='w')?'asc':'desc'; }
       ST.page=0; render(); return; }
     if(t.closest('.vtoggle')){ ST.content=!ST.content; ST.page=0; render(); return; }
     var p=t.closest('.vpage');
     if(p && !p.disabled){ ST.page=Math.max(0, ST.page+parseInt(p.getAttribute('data-d'),10)); render(); return; }
   });
 }
 return {render:function(){ wire(); render(); }};
};
"""


_VOCAB_BARS_JS = r"""
// A word-frequency bar list. Content words are the useful default; function
// words remain available for fluency and sentence-pattern inspection.
window.VocabBars=function(bodyId, getRows){
 var S=window.SkillStore, ST={mode:'content', page:0}, PAGE_SIZE=30;
 // Skip list — per panel (bodyId), server-synced like everything else now.
 // For names and other one-off words that dominate the count but aren't
 // vocabulary worth tracking (a person's name said 17 times isn't a word
 // you're learning).
 var SKKEY='skipwords:'+bodyId;
 function skipList(){ return S.get(SKKEY,[])||[]; }
 function isSkipped(w){ var lw=w.toLowerCase(); return skipList().some(function(s){return s.toLowerCase()===lw;}); }
 function addSkip(w){ var l=skipList(); if(!isSkipped(w)){ l.push(w); S.set(SKKEY,l); } }
 function removeSkip(w){ var lw=w.toLowerCase(); S.set(SKKEY, skipList().filter(function(s){return s.toLowerCase()!==lw;})); }
 function data(){ return getRows()||[]; }
 function body(){ return document.getElementById(bodyId); }
 function rows(){
   var r=data().filter(function(x){ return (ST.mode==='all' || (ST.mode==='content' ? !x.f : x.f)) && !isSkipped(x.w); });
   return r.sort(function(a,b){ return b.n-a.n || (a.w<b.w?-1:1); });
 }
 function render(){
   var el=body(); if(!el) return;
   var all=data(), r=rows(), counts={all:all.length,content:0,function:0};
   all.forEach(function(x){ counts[x.f?'function':'content']++; });
   function filter(mode,label){ return "<button class='btn small vfilter' data-mode='"+mode+"' aria-pressed='"+(ST.mode===mode?'true':'false')+"'"+
     (ST.mode===mode?" style='background:var(--accent);color:#08222b'":"")+">"+label+" <span class='hint'>"+counts[mode]+"</span></button>"; }
   var skip=skipList();
   var skipBar=skip.length?("<div style='margin:6px 2px' class='hint'>Skip words: "+
     skip.map(function(w){return "<span class='chip vskip' data-w=\""+S.esc(w)+"\" style='cursor:pointer;margin-right:4px' title='Click to unskip'>"+S.esc(w)+" ✕</span>";}).join(' ')+
     "</div>") : "";
   var controls="<div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 2px'>"+
     filter('content','Content words')+filter('function','Function words')+filter('all','All words')+
     "<span class='hint'>Click a word to hear it.</span></div>"+skipBar;
   if(!r.length){ el.innerHTML=controls+"<div class='card'>No words in this group yet.</div>"; return; }
   var pages=Math.max(1,Math.ceil(r.length/PAGE_SIZE));
   if(ST.page>=pages) ST.page=pages-1;
   if(ST.page<0) ST.page=0;
   var start=ST.page*PAGE_SIZE, slice=r.slice(start,start+PAGE_SIZE);
   var max=Math.max.apply(null,r.map(function(x){ return x.n; }));
   var list=slice.map(function(x){
     var pct=Math.max(2,Math.round(x.n/max*100));
     var tone=x.f?'var(--mut)':'var(--accent)';
     return "<div class='vbar' style='display:flex;align-items:center;gap:10px;padding:5px 2px' aria-label='"+S.esc(x.w)+", used "+x.n+" times'>"+
       "<span data-say=\""+S.esc(x.w)+"\" style='min-width:140px;font-weight:600;color:"+tone+";cursor:pointer'>"+S.esc(x.w)+"</span>"+
       "<span style='flex:1;background:var(--card);border-radius:4px;overflow:hidden;height:14px'>"+
         "<span style='display:block;height:100%;width:"+pct+"%;background:"+tone+"'></span></span>"+
       "<span class='hint' style='min-width:36px;text-align:right'>"+x.n+"</span>"+
       "<button class='btn small vbar-skip' data-w=\""+S.esc(x.w)+"\" title='Skip this word' style='padding:2px 7px'>🚫</button></div>";
   }).join('');
   var pager=pages>1 ? "<div style='display:flex;gap:10px;align-items:center;justify-content:center;margin:10px 2px'>"+
     "<button class='btn small vpage' data-d='-1'"+(ST.page<=0?' disabled':'')+">◀ Prev</button>"+
     "<span class='hint'>page "+(ST.page+1)+" of "+pages+" · "+r.length+" words</span>"+
     "<button class='btn small vpage' data-d='1'"+(ST.page>=pages-1?' disabled':'')+">Next ▶</button></div>" : "";
   el.innerHTML=controls+"<div style='padding:6px 0'>"+list+"</div>"+pager;
 }
 function wire(){ var el=body(); if(!el || el.__wired) return; el.__wired=true;
   el.addEventListener('click',function(e){
     var f=e.target.closest&&e.target.closest('.vfilter');
     if(f){ ST.mode=f.getAttribute('data-mode'); ST.page=0; render(); return; }
     var p=e.target.closest&&e.target.closest('.vpage');
     if(p && !p.disabled){ ST.page+=parseInt(p.getAttribute('data-d'),10); render(); return; }
     var sk=e.target.closest&&e.target.closest('.vbar-skip');
     if(sk){ addSkip(sk.getAttribute('data-w')); render(); return; }
     var un=e.target.closest&&e.target.closest('.vskip');
     if(un){ removeSkip(un.getAttribute('data-w')); render(); return; } }); }
 return {render:function(){ wire(); render(); }};
};
"""


def _stat(big, small, color=""):
    return ("<div class='m' style='min-width:120px'><span%s>%s</span>%s</div>"
            % ((" style='color:%s'" % color) if color else "", big, small))


def _speaking_vocab_panel(hidden=True):
    """How much distinct vocabulary you actually produce when speaking."""
    cls = "tabpanel hidden" if hidden else "tabpanel"
    v = speaking_vocabulary()
    if not v["tokens"]:
        return ("<section id='vocabspeak' class='%s'>"
                "<h1>Speaking vocabulary</h1><div class='card'>No transcripts yet — "
                "analyze a recording and its transcript will feed this report."
                "</div></section>" % cls)
    counts = v["counts"]
    rows = [{"w": w, "n": n, "d": v["first_seen"].get(w, ""),
             "f": 1 if w in _FUNCTION_WORDS else 0}
            for w, n in counts.most_common()]
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    content_types = sum(1 for w in counts if w not in _FUNCTION_WORDS)

    stats = ("<div class='metrics'>"
             + _stat(v["types"], "distinct words", "var(--accent)")
             + _stat(content_types, "excluding function words")
             + _stat("{:,}".format(v["tokens"]), "words spoken in total")
             + _stat(v["hapax"], "used only once", "var(--warn)")
             + _stat(v["ttr"], "type-token ratio")
             + _stat(v["recordings"], "recordings")
             + "</div>")

    top = counts.most_common(12)
    top_html = "".join(
        "<span class='chip'>%s <b>%d</b></span>" % (_esc(w), n) for w, n in top)
    once = [w for w, n in counts.most_common() if n == 1][:40]
    once_html = " ".join("<span class='chip'>%s</span>" % _esc(w) for w in once)

    body = (
        "<h1>Speaking vocabulary</h1>"
        "<p class='sub'>Every word you've actually said, counted from the "
        "transcripts of your %d recordings. Distinct words is the number that "
        "matters — total words just measures how much you talked.</p>"
        "%s"
        "<h2>Vocabulary growth</h2>"
        "<p class='sub'>The line is your running total of distinct words; the bars "
        "are how many were new that session. Bars shrinking toward zero means "
        "you're recycling the same vocabulary — time to deliberately reach for "
        "unfamiliar words.</p>%s"
        "<h2>Most used</h2><div class='chips'>%s</div>"
        "<h2>Used only once</h2>"
        "<p class='sub'>Words you produced a single time. These are the edge of "
        "your active vocabulary — the ones most likely to slip away unless you "
        "use them again.</p><div class='chips'>%s</div>"
        "<h2>Word frequency</h2>"
        "<p class='sub'>Content words are shown first because they reveal the ideas you can express. "
        "Switch to function words when you want to inspect fluency and sentence-building habits.</p>"
        "<div id='vt-speak'></div>"
        "<div id='vt-overlap'></div>"
        % (v["recordings"], stats, _vocab_growth_svg(v["sessions"]),
           top_html, once_html))

    overlap_js = r"""
    (function(){
      // Words you've MET while listening but never produced. Same origin as the
      // listening module, so its localStorage is readable from here.
      var S=window.SkillStore; if(!S) return;
      var srs=S.get('dict_srs',{}), sc=S.get('ec_scores',{}), heardIds={};
      Object.keys(srs).forEach(function(k){ heardIds[k]=1; });
      Object.keys(sc).forEach(function(k){ if(k.indexOf('dict:')===0) heardIds[k.slice(5)]=1; });
      var clips=window.LISTEN_TEXTS||{}, heard={};
      Object.keys(heardIds).forEach(function(id){
        (clips[id]||'').toLowerCase().replace(/[^a-z' ]+/g,' ').split(/\s+/).forEach(function(w){
          w=w.replace(/^'+|'+$/g,''); if(w.length>1||w==='a'||w==='i') heard[w]=1; });
      });
      var n=Object.keys(heard).length;
      var el=document.getElementById('vt-overlap'); if(!el) return;
      if(!n){ el.innerHTML="<h2>Spoken vs heard</h2><p class='sub'>Practise some "+
        "dictation and this will compare the two vocabularies.</p>"; return; }
      var spoken={}; (window.VOCAB_SPEAK||[]).forEach(function(r){ spoken[r.w]=1; });
      var both=0, onlyHeard=[];
      Object.keys(heard).forEach(function(w){ if(spoken[w]) both++; else onlyHeard.push(w); });
      onlyHeard.sort();
      el.innerHTML="<h2>Spoken vs heard</h2>"+
        "<p class='sub'>Comparing this report with the words in the clips you've "+
        "practised. Words you understand but never say are your nearest reachable gain.</p>"+
        "<div class='metrics'>"+
        "<div class='m'><span>"+n+"</span>distinct words heard</div>"+
        "<div class='m'><span style='color:var(--good)'>"+both+"</span>heard AND spoken</div>"+
        "<div class='m'><span style='color:var(--warn)'>"+onlyHeard.length+"</span>heard but never spoken</div>"+
        "</div><div class='chips'>"+onlyHeard.slice(0,60).map(function(w){
          return "<span class='chip'>"+S.esc(w)+"</span>"; }).join('')+"</div>";
    })();
    """
    # LISTEN_TEXTS is embedded once, by the listening-vocabulary panel, and
    # shared — both reports need the clip texts and they are the bulk of the
    # payload.
    return ("<section id='vocabspeak' class='%s'>%s</section>"
            "<script>window.VOCAB_SPEAK=%s;\n"
            "window.addEventListener('load',function(){"
            "  if(document.getElementById('vt-speak'))"
            "    window.VocabBars('vt-speak',function(){return window.VOCAB_SPEAK;}).render();"
            "});\n%s</script>"
            % (cls, body, payload, overlap_js))


_LISTEN_VOCAB_JS = r"""
(function(){
 var S=window.SkillStore;
 function practisedIds(){
   var srs=S.get('dict_srs',{}), sc=S.get('ec_scores',{}), ids={};
   Object.keys(srs).forEach(function(k){ ids[k]=1; });
   Object.keys(sc).forEach(function(k){ if(k.indexOf('dict:')===0) ids[k.slice(5)]=1; });
   return ids;
 }
 function words(text){
   var out=[];
   (text||'').toLowerCase().replace(/[^a-z' ]+/g,' ').split(/\s+/).forEach(function(w){
     w=w.replace(/^'+|'+$/g,''); if(w.length>1||w==='a'||w==='i') out.push(w); });
   return out;
 }
 window.addEventListener('load', function(){
   var el=document.getElementById('lv-body'); if(!el) return;
   var clips=window.LISTEN_TEXTS||{}, done=practisedIds();
   var counts={}, tokens=0, nclips=0;
   Object.keys(done).forEach(function(id){
     var t=clips[id]; if(t===undefined) return;
     nclips++;
     words(t).forEach(function(w){ counts[w]=(counts[w]||0)+1; tokens++; });
   });
   var types=Object.keys(counts).length;
   if(!types){
     el.innerHTML="<div class='card'><h4>Nothing practised yet</h4><p>Do some "+
       "dictation and this report fills in — it counts the words in the clips you "+
       "have actually worked through, not the whole library.</p>"+
       "<p><button class='btn' onclick=\"showPanel('dictation')\">Go to dictation practice</button></p></div>";
     return;
   }
   // the library at large, for "available but not yet met"
   var allW={}, allTok=0;
   Object.keys(clips).forEach(function(id){ words(clips[id]).forEach(function(w){ allW[w]=1; allTok++; }); });
   var unmet=Object.keys(allW).filter(function(w){ return !counts[w]; });

   var spoken={}; (window.SPOKEN_WORDS||[]).forEach(function(w){ spoken[w]=1; });
   var both=0, onlyHeard=[];
   Object.keys(counts).forEach(function(w){ if(spoken[w]) both++; else onlyHeard.push(w); });
   var spokenOnly=Object.keys(spoken).filter(function(w){ return !counts[w]; });
   onlyHeard.sort(); unmet.sort();

   // words you MISSED while listening, from the dictation error log
   var errs=S.get('dict_errors',{});
   var missed=Object.keys(errs).sort(function(a,b){ return errs[b].n-errs[a].n; });

   var hapax=0; Object.keys(counts).forEach(function(w){ if(counts[w]===1) hapax++; });
   var stats="<div class='metrics'>"+
     "<div class='m'><span style='color:var(--accent)'>"+types+"</span>distinct words heard</div>"+
     "<div class='m'><span>"+tokens.toLocaleString()+"</span>words heard in total</div>"+
     "<div class='m'><span>"+nclips+"</span>clips practised</div>"+
     "<div class='m'><span style='color:var(--warn)'>"+hapax+"</span>met only once</div>"+
     "<div class='m'><span>"+(tokens?(Math.round(types/tokens*1000)/1000):0)+"</span>type-token ratio</div>"+
     "</div>";

   var rows=Object.keys(counts).map(function(w){
     return {w:w, n:counts[w], d:'', f:(window.VOCAB_FUNCTION_WORDS||[]).indexOf(w)>=0?1:0, said:!!spoken[w], miss:(errs[w]||{}).n||0};
   }).sort(function(a,b){ return b.n-a.n || (a.w<b.w?-1:1); });
   window.VOCAB_LISTEN=rows.map(function(r){
     return {w:r.w, n:r.n, f:r.f, d:(r.miss?('missed '+r.miss+'×'):(r.said?'also spoken':''))};
   });

   el.innerHTML=stats+
     "<h2>Heard vs spoken</h2>"+
     "<p class='sub'>Words you have understood in real speech but never produced "+
     "yourself. This is the most reachable vocabulary you have — recognition is "+
     "already there, only production is missing.</p>"+
     "<div class='metrics'>"+
       "<div class='m'><span style='color:var(--good)'>"+both+"</span>heard and spoken</div>"+
       "<div class='m'><span style='color:var(--warn)'>"+onlyHeard.length+"</span>heard but never spoken</div>"+
       "<div class='m'><span class='hint' style='font-size:28px'>"+spokenOnly.length+"</span>spoken but never heard</div>"+
     "</div>"+
     "<div class='chips'>"+onlyHeard.slice(0,80).map(function(w){
        return "<span class='chip'>"+S.esc(w)+"</span>"; }).join('')+"</div>"+
     (missed.length ? ("<h2>Words you mis-heard</h2><p class='sub'>From the listening "+
        "error log — heard, but not correctly.</p><div class='chips'>"+
        missed.slice(0,50).map(function(w){
          return "<span class='chip down'>"+S.esc(w)+" <b>"+errs[w].n+"</b></span>"; }).join('')+
        "</div>") : "")+
     "<h2>Word frequency</h2><p class='sub'>Content words are shown first; switch to function words to examine "
       +"the glue words that make fast connected speech difficult.</p><div id='vt-listen'></div>"+
     "<h2>Waiting in the library</h2>"+
     "<p class='sub'><b>"+unmet.length+"</b> distinct words appear in clips you "+
     "haven't practised yet.</p><div class='chips'>"+
       unmet.slice(0,60).map(function(w){ return "<span class='chip'>"+S.esc(w)+"</span>"; }).join('')+
     "</div>";
   // the table markup only exists now, so build it after the panel is written
   window.VocabBars('vt-listen', function(){ return window.VOCAB_LISTEN; }).render();
 });
})();
"""


def _listening_vocab_panel(hidden=True):
    """How much distinct vocabulary you've actually taken in by ear."""
    cls = "tabpanel hidden" if hidden else "tabpanel"
    clips = load_listening_library()
    texts = json.dumps({c["id"]: c["text"] for c in clips},
                       ensure_ascii=False).replace("</", "<\\/")
    spoken = json.dumps(sorted(speaking_vocabulary()["counts"].keys()),
                        ensure_ascii=False).replace("</", "<\\/")
    body = ("<h1>Listening vocabulary</h1>"
            "<p class='sub'>Counted from the clips you have actually practised — "
            "not the whole library. Hearing a word in real connected speech and "
            "recognising it is a different skill from reading it, so this is "
            "tracked separately from what you can say.</p>"
            "<div id='lv-body'></div>")
    function_words = json.dumps(sorted(_FUNCTION_WORDS), ensure_ascii=False).replace("</", "<\\/")
    return ("<section id='vocablisten' class='%s'>%s</section>"
            "<script>window.LISTEN_TEXTS=%s;window.SPOKEN_WORDS=%s;window.VOCAB_FUNCTION_WORDS=%s;\n%s</script>"
            % (cls, body, texts, spoken, function_words, _LISTEN_VOCAB_JS))


def _skill_panels(items):
    return (_SKILLS_UTIL_JS + _vocab_panel()
            + _photo_desc_panel()
            + _grammar_panel(items)
            + _dictation_panel() + _listening_log_panel()
            # the shared table component, defined once — the speaking panel
            # renders an empty state when there are no transcripts, so it can't
            # be the thing that provides it
            + ("<script>%s</script>" % _VOCAB_BARS_JS)
            + _listening_vocab_panel() + _speaking_vocab_panel()
            + _listening_panel() + _register_panel() + _mandarin_panel()
            + _reading_panel(items) + _knowledge_panel() + _howto_panel()
            + _pronscore_panel())


_TS_PATTERNS = (
    # Recording 20260726-234737 yehu_3d_r2.webm
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[-_ T](\d{2})(\d{2})(\d{2})(?!\d)"),
    # yehu_3d_r2_2026-07-26_224624.webm
    re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})[-_ T](\d{2})(\d{2})(\d{2})(?!\d)"),
)


def _timestamp_from_name(name):
    """Pull the recording time out of a filename, or None."""
    import datetime
    for rx in _TS_PATTERNS:
        m = rx.search(name or "")
        if m:
            try:
                return datetime.datetime(*(int(g) for g in m.groups()))
            except ValueError:
                continue
    return None


def _recorded_at(d):
    """Best datetime available for a recording, or None.

    Prefers the timestamp embedded in the filename: it is the moment the audio
    was captured, and unlike mtime it doesn't move when the file is re-saved by
    a re-analysis. Then `recorded_at`, stamped into result.json from the media
    file while it still existed — this is what keeps the ordering correct after
    the audio is pruned. Falls back to mtime, then the date.
    """
    import datetime
    ts = _timestamp_from_name(str(d.get("title", "")))
    if ts:
        return ts
    ra = d.get("recorded_at")
    if isinstance(ra, str) and ra:
        try:
            return datetime.datetime.fromisoformat(ra)
        except ValueError:
            pass
    mt = d.get("_mtime")
    if isinstance(mt, (int, float)) and mt > 0:
        try:
            return datetime.datetime.fromtimestamp(mt)
        except (ValueError, OSError):
            pass
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", str(d.get("date", "")))
    if m:
        try:
            return datetime.datetime(*(int(g) for g in m.groups()))
        except ValueError:
            pass
    return None


def _sort_ts(d):
    """Sort key for recordings — oldest first. Date alone is too coarse: several
    sessions a day is normal, and they'd otherwise fall back to whatever order
    the filesystem walk produced."""
    at = _recorded_at(d)
    return at.timestamp() if at else 0.0


def _when_label(d):
    """Sidebar timestamp — includes the time when we know it, so same-day
    recordings are distinguishable and the ordering is verifiable."""
    at = _recorded_at(d)
    if at and (_timestamp_from_name(str(d.get("title", "")))
               or d.get("recorded_at") or d.get("_mtime")):
        return at.strftime("%Y-%m-%d %H:%M")
    return str(d.get("date", ""))


def _plan_word(items):
    """Return the most useful word-level pronunciation target, if we have one.

    A word must recur before it becomes a plan item. This prevents one noisy
    Azure result from displacing a pattern the learner has actually met again.
    """
    from collections import Counter
    words = Counter()
    for item in items or []:
        for word in (item.get("azure") or {}).get("words", []):
            if word.get("error") == "Mispronunciation":
                token = (word.get("word") or "").lower().strip(".,!?;:\"'()")
                if token:
                    words[token] += 1
    return words.most_common(1)[0] if words else (None, 0)


def _plan_grammar(items):
    """Return a stable display label for the most repeated grammar finding."""
    from collections import Counter
    findings = Counter()
    for item in items or []:
        for finding in item.get("grammar", []) or []:
            label = (finding.get("rule") or finding.get("type") or "").strip()
            if label:
                findings[label] += 1
    return findings.most_common(1)[0] if findings else (None, 0)


def _today_panel(items, history=None):
    """A short, deliberately finite next-practice loop.

    Reports contain much more detail, but the front page should answer just one
    question: what should I do now? The tasks point into existing panels, so no
    duplicate scheduling store is needed.
    """
    items = list(items or [])
    word, word_count = _plan_word(items)
    grammar, grammar_count = _plan_grammar(items)
    tasks = []
    if word:
        repeat = " · seen %d times" % word_count if word_count > 1 else ""
        tasks.append(("1", "Say <b>%s</b> three times" % _esc(word),
                      "Record it in Single-word practice%s." % repeat, "practice",
                      "Open word practice"))
    if grammar:
        repeat = " · recurring in %d sessions" % grammar_count if grammar_count > 1 else ""
        tasks.append((str(len(tasks) + 1), "Review <b>%s</b>" % _esc(grammar),
                      "Read one correction, then say a new sentence using it%s." % repeat,
                      "grammar", "Open error log"))
    tasks.append((str(len(tasks) + 1), "Do one listening dictation", 
                  "Listen once before typing. Missed words will reappear for review.",
                  "dictation", "Start dictation"))
    if not items:
        tasks.insert(0, ("1", "Start your first speaking analysis",
                         "A short 30–60 second recording is enough to create a personal plan.",
                         "newrec", "New Speaking Analysis"))

    cards = ""
    for num, title, detail, panel, button in tasks[:3]:
        cards += ("<div class='card' style='display:flex;gap:14px;align-items:flex-start'>"
                  "<b style='display:grid;place-items:center;flex:0 0 30px;height:30px;"
                  "border-radius:50%%;background:var(--accent);color:#08222b'>%s</b>"
                  "<div style='flex:1'><b>%s</b><div class='hint' style='margin-top:4px'>%s</div>"
                  "<button class='btn small' style='margin-top:9px' onclick=\"showPanel('%s')\">%s</button>"
                  "</div></div>" % (num, title, detail, panel, button))
    count = len(items)
    context = ("Built from %d analyzed recording%s. Keep this loop small; finish it, then stop."
               % (count, "s" if count != 1 else "")) if count else (
               "Start with one recording. The plan will become personal once the app has evidence.")
    return ("<h1>Today’s plan</h1><p class='sub'>%s</p>"
            "<div style='max-width:720px'>%s</div>"
            "<p class='hint'>Tip: a useful practice session is usually 10–15 minutes, not an hour of browsing modules.</p>"
            % (context, cards))


def _history_day(row):
    """Extract a calendar day from a history row without rejecting old data."""
    value = str(row.get("date", ""))[:10]
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


_WEEKLY_PRACTICE_JS = r"""
(function(){
 function iso(d){ return d.toISOString().slice(0,10); }
 function esc(s){ return (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;'); }
 function mean(rows){ return rows.length ? Math.round(rows.reduce(function(n,x){return n+x.s;},0)/rows.length) : null; }
 var MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
 function pretty(d){ var p=String(d).split('-'); return MON[+p[1]-1]+' '+(+p[2]); }
 // The four dates that define "this week" and the week before it.
 function windows(anchor){
   var d=new Date(anchor+'T12:00:00Z'); d.setUTCDate(d.getUTCDate()-6); var start=iso(d);
   d.setUTCDate(d.getUTCDate()-7); var prevStart=iso(d);
   var e=new Date(start+'T12:00:00Z'); e.setUTCDate(e.getUTCDate()-1); var prevEnd=iso(e);
   return {start:start, anchor:anchor, prevStart:prevStart, prevEnd:prevEnd};
 }
 // Latest day on which anything under these key prefixes was practised.
 function lastPractised(scores, prefixes){
   var m='';
   Object.keys(scores||{}).forEach(function(key){
     var hit=false;
     for(var i=0;i<prefixes.length;i++){ if(key.indexOf(prefixes[i])===0){ hit=true; break; } }
     if(!hit) return;
     (scores[key]||[]).forEach(function(x){ if(x && x.d && x.d>m) m=x.d; });
   });
   return m;
 }
 function report(prefix, label, noun, scores, w){
   var start=w.start, anchor=w.anchor, prevStart=w.prevStart, prevEnd=w.prevEnd;
   var rows=[], previous=[], ids={};
   Object.keys(scores||{}).forEach(function(key){
     if(key.indexOf(prefix)!==0) return;
     (scores[key]||[]).forEach(function(x){
       if(!x || typeof x.s!=='number' || !x.d) return;
       if(x.d>=start && x.d<=anchor){ rows.push({s:x.s,key:key}); ids[key]=1; }
       if(x.d>=prevStart && x.d<=prevEnd) previous.push({s:x.s,key:key});
     });
   });
   var range="<div class='hint' style='margin-top:8px'>"+pretty(start)+" – "+pretty(anchor)+"</div>";
   if(!rows.length) return "<div class='card'><b>"+label+"</b><p class='hint'>No "+noun+
     " practised between "+pretty(start)+" and "+pretty(anchor)+".</p></div>";
   var avg=mean(rows), old=mean(previous), delta=old==null?'—':(avg-old>0?'+':'')+(avg-old);
   var best=Math.max.apply(null,rows.map(function(x){return x.s;}));
   return "<div class='card'><b>"+label+"</b><div style='display:flex;gap:18px;flex-wrap:wrap;margin-top:10px'>"+
     "<span><b style='font-size:23px'>"+rows.length+"</b><span class='hint'> attempts</span></span>"+
     "<span><b style='font-size:23px'>"+avg+"</b><span class='hint'> average</span></span>"+
     "<span><b style='font-size:23px'>"+best+"</b><span class='hint'> best</span></span>"+
     "<span><b style='font-size:23px'>"+Object.keys(ids).length+"</b><span class='hint'> "+noun+" this week</span></span>"+
     "<span><b style='font-size:23px'>"+delta+"</b><span class='hint'> vs. previous week</span></span></div>"+
     range+"</div>";
 }
 window.addEventListener('load',function(){
   var el=document.getElementById('weekly-practice-report'); if(!el)return;
   var anchor=window.WEEKLY_ANCHOR; if(!anchor)return;
   var scores=(window.SkillStore&&window.SkillStore.get('ec_scores',{}))||{};
   // WEEKLY_ANCHOR is the date of your last speaking RECORDING, which is the
   // right frame for the speaking sections above — but dictation and reading
   // are practised independently of recording yourself, so anchoring them
   // there hid every clip done since the last recording. (Concretely: 40 clips
   // practised, only 30 counted, because the newest session postdated the
   // anchor.) Run these two off whichever is more recent instead.
   var last=lastPractised(scores,['dict:','reading:','word:']);
   var w=windows(last>anchor?last:anchor);
   el.innerHTML=report('dict:','Listening','clips',scores,w)+
                report('reading:','Reading Passage','passages',scores,w)+
                report('word:','Reading Single Word','words',scores,w);
 });
})();
"""


def _weekly_review_panel(history, items=None, embedded=False):
    """Compare the most recent active seven-day window to its predecessor."""
    history = list(history or [])
    dated = [(row, _history_day(row)) for row in history]
    dated = [(row, day) for row, day in dated if day]
    if not dated:
        anchor = date.today()
        heading = "<h2>Weekly review</h2>" if embedded else "<h1>Weekly review</h1>"
        return (heading + "<p class='sub'>No speaking recordings this week yet.</p>"
                "<h2>Speaking</h2><div class='card'>Analyze a recording to start your speaking review.</div>"
                "<h2>Listening &amp; Reading</h2><div id='weekly-practice-report' class='metrics'></div>"
                "<script>window.WEEKLY_ANCHOR=%s;%s</script>"
                % (json.dumps(anchor.isoformat()), _WEEKLY_PRACTICE_JS))
    anchor = max(day for _row, day in dated)
    current_start = anchor - timedelta(days=6)
    previous_start, previous_end = current_start - timedelta(days=7), current_start - timedelta(days=1)
    current = [row for row, day in dated if current_start <= day <= anchor]
    previous = [row for row, day in dated if previous_start <= day <= previous_end]

    def duration(rows):
        return sum(r.get("duration_sec", 0) for r in rows
                   if isinstance(r.get("duration_sec"), (int, float)))

    cur_sec, prev_sec = duration(current), duration(previous)
    heading = "<h2>Weekly review</h2>" if embedded else "<h1>Weekly review</h1>"
    return (heading +
            "<p class='sub'>Week ending <b>%s</b>. How much you practised, and how "
            "each kind of practice scored.</p>"
            "<h2>Speaking</h2>"
            "<div class='metrics'><div class='card'><b style='font-size:24px'>%d</b><div class='hint'>recordings this week</div></div>"
            "<div class='card'><b style='font-size:24px'>%s</b><div class='hint'>recorded practice</div></div>"
            "<div class='card'><b style='font-size:24px'>%s</b><div class='hint'>previous week</div></div></div>"
            "<h2>Listening &amp; Reading</h2><div id='weekly-practice-report' class='metrics'></div>"
            "<script>window.WEEKLY_ANCHOR=%s;%s</script>"
            % (anchor.strftime("%b %-d, %Y"),
               len(current), _fmt_total(cur_sec) if cur_sec else "—", _fmt_total(prev_sec) if prev_sec else "—",
               json.dumps(anchor.isoformat()), _WEEKLY_PRACTICE_JS))


def generate_dashboard_html(items, history=None, extra_nav="", extra_panels="",
                            active="summary"):
    """One self-contained page: a sidebar to switch between every recording,
    plus a Summary tab with the improvement-over-time curve.

    items       : list of analysis dicts (one per recording)
    history     : list of per-session metric dicts (defaults derived from items)
    extra_nav   : extra sidebar HTML inserted at the top (e.g. a "New analysis" link)
    extra_panels: extra <section class='tabpanel'> HTML (e.g. an upload form)
    active      : which panel id starts visible
    """
    items = list(items or [])
    # newest first in the sidebar; summary curve wants oldest->newest
    items_sorted = sorted(items, key=_sort_ts)
    if history is None:
        history = [_metrics_from_data(d) for d in items_sorted]

    def acls(pid):
        return " class='active'" if pid == active else ""

    nav = extra_nav
    # Summary sits under the action-oriented views (or comes from the web nav).
    if "data-panel='summary'" not in extra_nav:
        nav += "<a data-panel='summary'%s>📈 Summary &amp; progress</a>" % acls("summary")
    nav += "<a data-panel='grammar'>📋 Speaking error log</a>"
    # The daily loop: what you produce, listening practice, what you missed,
    # what you took in. Kept above the first section header so they stay
    # top-level and can never be collapsed out of sight.
    nav += ("<a data-panel='vocabspeak'>🗣️ Speaking vocabulary"
            "<small>what you produce</small></a>")
    nav += "<a data-panel='dictation'>🎧 Listening — dictation</a>"
    nav += "<a data-panel='listenlog'>📋 Listening error log</a>"
    nav += ("<a data-panel='vocablisten'>🎧 Listening vocabulary"
            "<small>what you take in</small></a>")
    nav += "<a data-panel='stories'>📖 Reading</a>"
    nav += ("<a data-panel='vocab'>🧭 Surrounding vocabulary"
            "<small>photos and captures</small></a>")
    nav += ("<a data-panel='photodesc'>🖼 Describe a photo"
            "<small>say what you see, then check</small></a>")
    nav += "<p class='navsec'>Train — ear &amp; sound</p>"
    nav += "<a data-panel='howto'>📣 How-to: tricky words</a>"
    nav += "<a data-panel='listening'>🔉 Listening (ear training)</a>"
    nav += "<a data-panel='drills'%s>🎧 Sound system</a>" % acls("drills")
    nav += "<a data-panel='mandarin'>🗣️ 中→EN Pronunciation</a>"
    nav += "<p class='navsec'>Train — words &amp; usage</p>"
    nav += "<a data-panel='register'>🎭 Register</a>"
    nav += "<p class='navsec'>Reference — read &amp; understand</p>"
    nav += "<a data-panel='knowledge'>📕 Vowels &amp; consonants 101</a>"
    nav += "<a data-panel='pronscore'>🧮 How scoring works</a>"
    nav += "<p class='navsec'>Recordings</p>"
    # map each recording (date, title) to its report panel id, so the Summary
    # table rows can link straight to the detailed report
    rec_ids = {}
    for _i, _d in enumerate(reversed(items_sorted)):
        rec_ids.setdefault(
            (str(_d.get("date", "")), str(_d.get("title", ""))), "rec%d" % _i)
    panels = extra_panels
    panels += ("<section id='summary' class='tabpanel%s'>"
               % ("" if active == "summary" else " hidden")
               + _summary_panel(history, items_sorted, rec_ids) + "</section>")
    panels += _sound_panel()
    panels += _skill_panels(items_sorted)

    if not items_sorted:
        nav += "<a style='cursor:default;color:#5a6080'>No recordings yet</a>"
    for i, d in enumerate(reversed(items_sorted)):  # newest first in list
        pid = "rec%d" % i
        title = _esc(d.get("title") or ("Recording %d" % (i + 1)))
        when = _esc(_when_label(d))
        nav += "<a data-panel='%s'%s>%s<small>%s</small></a>" % (pid, acls(pid), title, when)
        panels += ("<section id='%s' class='tabpanel%s'>%s</section>"
                   % (pid, "" if pid == active else " hidden", _report_body(d)))

    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>English Coach — Dashboard</title><style>" + _DASHBOARD_CSS
            + "</style></head><body>"
            "<button class='navtoggle' onclick='toggleNav()'>☰ Menu</button>"
            "<div class='navbackdrop' onclick='toggleNav()'></div>"
            "<nav class='sidenav'><p class='brand'>English Coach</p>" + nav + "</nav>"
            "<main class='content'><div class='wrap'>" + panels
            + "<footer>Generated by English Coach. Pick a recording on the left, "
              "or open Summary to see your progress over time.</footer>"
            "</div></main><script>" + _DASHBOARD_JS + "</script></body></html>")


def generate_html_report(d):
    """Backward-compatible single-recording page (renders as a 1-item dashboard)."""
    return generate_dashboard_html([d])


# ---------------------------------------------------------------------------
# Library layout: one subfolder per recording, named after the audio file.
#   VideoAudioFiles/
#     history.json
#     <stem>/<stem>.{mp4,txt,json,result.json,polished.txt}
# ---------------------------------------------------------------------------
LIBRARY_NAME = "VideoAudioFiles"


def library_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), LIBRARY_NAME)


def rec_dir_for(stem, library=None, create=True):
    """Return (and optionally create) the per-recording subfolder for `stem`."""
    d = os.path.join(library or library_dir(), stem)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


# Every media extension the app treats as a recording. Module-level because
# both the dashboard loader and the storage-pruning helpers need it.
_AV = (".m4a", ".mp3", ".wav", ".mp4", ".mov", ".aac", ".flac", ".ogg", ".webm")


def build_dashboard_for_dir(out_dir):
    """Aggregate every <name>.result.json found anywhere under a folder (so each
    recording can live in its own subfolder) into one dashboard, using
    history.json at the folder root for the Summary curve."""
    project_root = os.path.dirname(os.path.abspath(out_dir))  # dashboard.html lives here
    items = []
    paths = []
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            if fn.endswith(".result.json"):
                paths.append(os.path.join(root, fn))
    for p in sorted(paths):
        try:
            with open(p, encoding="utf-8") as f:
                item = json.load(f)
        except Exception:
            continue
        # locate this recording's audio file (same folder) for click-to-play
        d = os.path.dirname(p)
        item["has_audio"] = False
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(_AV):
                item["audio_rel"] = os.path.relpath(os.path.join(d, fn), project_root)
                item["audio_abs"] = os.path.join(d, fn)
                item["has_audio"] = True
                try:
                    item["_mtime"] = os.path.getmtime(item["audio_abs"])
                except OSError:
                    pass
                break
        _stamp_recording_meta(item, p)   # survive the media being pruned later
        _backfill_prosody(item, p)   # one-time prosody for older recordings
        items.append(item)
    history = None
    hp = os.path.join(out_dir, "history.json")
    if os.path.exists(hp):
        try:
            with open(hp, encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = None
    return generate_dashboard_html(items, history)


# ---------------------------------------------------------------------------
# 5b. Progress tracking — log every session and draw the improvement curve
# ---------------------------------------------------------------------------
def _audio_duration(path):
    """Length of an audio/video file in seconds (via PyAV), or None."""
    try:
        import av
        with av.open(path) as c:
            if c.duration:
                return round(c.duration / 1_000_000, 1)
            for s in c.streams:
                if s.type == "audio" and s.duration and s.time_base:
                    return round(float(s.duration * s.time_base), 1)
    except Exception:
        return None
    return None


def analyze_prosody(audio_path):
    """Statistical prosody metrics from the audio, in pure NumPy (no librosa).

    Returns pitch variation (a 'monotone index'), pitch range, speaking rate,
    pause ratio, and an nPVI rhythm score, plus a downsampled pitch contour for
    plotting. None if the audio is too short or can't be read.
    """
    try:
        import numpy as np
        import wave
    except Exception:
        return None
    try:
        wav = _to_wav_16k_mono(audio_path)          # 16 kHz mono PCM via PyAV
        with wave.open(wav, "rb") as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
    except Exception:
        return None
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if x.size < sr:                                  # need at least ~1s
        return None
    x = x / (np.max(np.abs(x)) + 1e-9)

    frame, hop = int(0.04 * sr), int(0.01 * sr)
    win = np.hanning(frame)
    lo, hi = int(sr / 400.0), int(sr / 75.0)         # F0 search 75–400 Hz
    nfft = 1
    while nfft < 2 * frame:
        nfft *= 2

    rms, f0, conf = [], [], []
    for i in range(0, len(x) - frame, hop):
        seg = x[i:i + frame]
        rms.append(np.sqrt(np.mean(seg * seg)))
        spec = np.fft.rfft(seg * win, nfft)
        ac = np.fft.irfft(spec * np.conj(spec))[:frame]
        if ac[0] <= 0 or hi >= frame:
            f0.append(0.0); conf.append(0.0); continue
        k = lo + int(np.argmax(ac[lo:hi]))
        conf.append(ac[k] / ac[0])
        f0.append(sr / float(k) if k > 0 else 0.0)
    rms, f0, conf = np.array(rms), np.array(f0), np.array(conf)
    if rms.size == 0:
        return None

    ethr = max(0.02, 0.15 * np.percentile(rms, 90))
    speech = rms > ethr
    voiced = speech & (conf > 0.3) & (f0 >= 75) & (f0 <= 400)
    total_t = len(rms) * hop / sr
    speech_t = float(speech.sum()) * hop / sr
    pause_t = total_t - speech_t

    vf = f0[voiced]
    if vf.size < 5:
        return None
    lo5, hi95 = np.percentile(vf, [5, 95])           # trim octave-jump outliers
    vfc = vf[(vf >= lo5) & (vf <= hi95)]
    if vfc.size < 3:
        vfc = vf
    med = np.median(vfc)
    semis = 12 * np.log2(vfc / med)
    pitch_var_st = float(np.std(semis))
    pitch_range_st = float(12 * np.log2(vfc.max() / vfc.min()))

    # syllable proxy: peaks in a smoothed energy envelope, ≥120 ms apart
    ksz = max(1, int(0.05 * sr / hop))
    envs = np.convolve(rms, np.ones(ksz) / ksz, "same")
    pthr = max(ethr, 0.3 * np.percentile(envs, 90))
    mind = int(0.12 * sr / hop)
    peaks, last = [], -10 ** 9
    for j in range(1, len(envs) - 1):
        if envs[j] > pthr and envs[j] >= envs[j - 1] and envs[j] > envs[j + 1] and (j - last) >= mind:
            peaks.append(j); last = j
    rate = (len(peaks) / speech_t) if speech_t > 0 else 0.0

    npvi = None
    if len(peaks) >= 3:
        d = np.diff(np.array(peaks)) * hop / sr
        npvi = float(100 * np.mean(np.abs(d[:-1] - d[1:]) / ((d[:-1] + d[1:]) / 2 + 1e-9)))

    N = 120
    idxs = np.linspace(0, len(f0) - 1, min(N, len(f0))).astype(int)
    contour = [round(float(f0[k]), 1) if voiced[k] else None for k in idxs]

    return {
        "pitch_mean_hz": round(float(med), 1),
        "pitch_min_hz": round(float(vfc.min()), 1),
        "pitch_max_hz": round(float(vfc.max()), 1),
        "pitch_range_st": round(pitch_range_st, 1),
        "pitch_var_st": round(pitch_var_st, 2),
        "speech_rate_syl_s": round(float(rate), 2),
        "pause_ratio_pct": int(round(100 * pause_t / total_t)) if total_t > 0 else 0,
        "npvi": int(round(npvi)) if npvi is not None else None,
        "voiced_pct": int(round(100 * voiced.sum() / max(1, speech.sum()))),
        "dur_s": round(total_t, 1),
        "contour": contour,
    }


# Fields attached at load time from the filesystem — never written back into
# result.json, because they describe where the media is, not what was said.
_RUNTIME_KEYS = ("audio_abs", "audio_rel", "_mtime", "has_audio")


def _save_result(item, result_path):
    """Write an item back to its result.json, minus the runtime-only fields."""
    try:
        save = {k: v for k, v in item.items() if k not in _RUNTIME_KEYS}
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(save, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _stamp_recording_meta(item, result_path):
    """Persist the facts that would otherwise only exist on the media file.

    When the audio is deleted to reclaim space, `_mtime` goes with it — and a
    recording whose title carries no timestamp then falls back to a date-only
    sort key, which silently reshuffles same-day sessions and blanks the time
    in the sidebar. Recording length has the same problem: it is measured from
    the file. Writing both into result.json once, while the media is still
    here, makes every report independent of whether the audio survives.
    """
    changed = False
    if not item.get("recorded_at"):
        at = _recorded_at(item)
        if at:
            item["recorded_at"] = at.isoformat(timespec="seconds")
            changed = True
    if item.get("duration_sec") is None:
        ap = item.get("audio_abs")
        if ap and os.path.exists(ap):
            try:
                dur = _audio_duration(ap)
            except Exception:
                dur = None
            if dur:
                item["duration_sec"] = round(float(dur), 1)
                changed = True
    if changed:
        _save_result(item, result_path)
    return changed


def _backfill_prosody(item, result_path):
    """Compute prosody for an already-analyzed recording that predates the
    feature, and persist it into its result.json so it's only computed once."""
    if item.get("prosody_metrics"):
        return
    ap = item.get("audio_abs")
    if not ap or not os.path.exists(ap):
        return
    try:
        pm = analyze_prosody(ap)
    except Exception:
        pm = None
    if not pm:
        return
    item["prosody_metrics"] = pm
    _save_result(item, result_path)


def _fmt_dur(sec):
    """Short m:ss for a single recording."""
    sec = int(round(sec))
    return "%d:%02d" % divmod(sec, 60)


def _fmt_total(sec):
    """Human total like '1h 05m' / '12m 30s' / '45s'."""
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh %02dm" % (h, m)
    if m:
        return "%dm %02ds" % (m, s)
    return "%ds" % s


def _metrics_from_data(data):
    """Pull the per-session metrics we track over time out of an analysis dict."""
    rec = {
        "date": data.get("date", str(date.today())),
        "title": data.get("title", ""),
        "grammar_errors": len(data.get("grammar", [])),
        "word_choice_issues": len(data.get("word_choice", [])),
    }
    dur = data.get("duration_sec")
    if dur is None:
        m = re.match(r"\s*(\d+(?:\.\d+)?)", str(data.get("duration", "")))
        if m:
            dur = float(m.group(1))
    if dur is not None:
        rec["duration_sec"] = dur
    pm = data.get("prosody_metrics")
    if isinstance(pm, dict):
        if pm.get("pitch_var_st") is not None:
            rec["pitch_var_st"] = pm["pitch_var_st"]
        if pm.get("speech_rate_syl_s") is not None:
            rec["speech_rate"] = pm["speech_rate_syl_s"]
    az = data.get("azure")
    if az:
        rec.update({
            "pron_score": az.get("pron_score"),
            "accuracy": az.get("accuracy"),
            "fluency_score": az.get("fluency"),
            "completeness": az.get("completeness"),
            "prosody": az.get("prosody"),
            "pron_errors": sum(az.get("error_counts", {}).values()),
        })
    fl = data.get("fluency")
    if isinstance(fl, dict):
        rec["wpm"] = fl.get("wpm")
    return rec


def log_session(data, history_path):
    """Append this session's key metrics to history.json and return the list."""
    rec = _metrics_from_data(data)
    hist = []
    if os.path.exists(history_path):
        try:
            hist = json.load(open(history_path, encoding="utf-8"))
        except Exception:
            hist = []
    hist.append(rec)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)
    return hist


_SERIES = [
    ("pron_score", "Overall", "#46b3c9"),
    ("accuracy", "Accuracy", "#43c59e"),
    ("fluency_score", "Fluency", "#ffb454"),
    ("prosody", "Prosody", "#b58cff"),
]


_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def _bucket_practice(rows, mode):
    """Group (date, seconds) pairs into day / week / month buckets.

    Returns an ordered list of (label, seconds), with empty periods filled in —
    a gap is information: it's a day you didn't practise.
    """
    import datetime
    agg = {}
    for ds, sec in rows:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(ds))
        if not m:
            continue
        try:
            d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        if mode == "day":
            key = d
        elif mode == "week":
            key = d - datetime.timedelta(days=d.weekday())   # Monday
        else:
            key = d.replace(day=1)
        agg[key] = agg.get(key, 0.0) + (sec or 0.0)
    if not agg:
        return []
    lo, hi = min(agg), max(agg)
    out = []
    if mode == "month":
        y, mo = lo.year, lo.month
        while (y, mo) <= (hi.year, hi.month):
            k = datetime.date(y, mo, 1)
            # full year, so a month bucket can't be misread as a day-of-month
            out.append(("%s %d" % (_MONTHS[mo - 1], y), agg.get(k, 0.0)))
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
        out = out[-12:]
    else:
        stepdays = 1 if mode == "day" else 7
        cap = 30 if mode == "day" else 26
        k = lo
        while k <= hi:
            lbl = "%s %d" % (_MONTHS[k.month - 1], k.day)
            out.append((lbl, agg.get(k, 0.0)))
            k += datetime.timedelta(days=stepdays)
        out = out[-cap:]
    return out


def _practice_bars(buckets, elid, visible):
    """A bar chart of practice time per period, rendered as static SVG."""
    W, H = 820, 250
    L, R, T, B = 52, 16, 18, 40
    iw, ih = W - L - R, H - T - B
    if not buckets:
        return ""
    peak = max(s for _, s in buckets) or 1.0
    # round the axis up to a friendly minute value
    top_min = max(1, int(-(-peak // 60)))
    for nice in (1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480):
        if top_min <= nice:
            top_min = nice
            break
    top = top_min * 60.0

    grid = ""
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        gy = T + ih * (1 - f)
        mins = top_min * f
        lbl = ("%dh%02d" % (int(mins // 60), int(mins % 60))) if mins >= 60 else "%dm" % mins
        grid += ("<line x1='%s' y1='%.1f' x2='%s' y2='%.1f' stroke='#24404c'/>"
                 "<text x='%s' y='%.1f' fill='#9aa3bf' font-size='11' "
                 "text-anchor='end'>%s</text>"
                 % (L, gy, W - R, gy, L - 8, gy + 4, lbl))

    n = len(buckets)
    slot = iw / float(n)
    bw = max(3.0, min(38.0, slot * 0.68))
    bars, labels = "", ""
    label_every = max(1, -(-n // 12))
    for i, (lbl, sec) in enumerate(buckets):
        cx = L + slot * (i + 0.5)
        bh = ih * (sec / top) if sec else 0
        if sec:
            bars += ("<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' rx='3' "
                     "fill='var(--accent)'><title>%s — %s</title></rect>"
                     % (cx - bw / 2, T + ih - bh, bw, bh, _esc(lbl), _fmt_total(sec)))
            if n <= 14:
                bars += ("<text x='%.1f' y='%.1f' fill='#9aa3bf' font-size='10' "
                         "text-anchor='middle'>%s</text>"
                         % (cx, T + ih - bh - 4, _esc(_fmt_total(sec))))
        else:
            # a visible stub for "no practice", so gaps read as zero not missing
            bars += ("<rect x='%.1f' y='%.1f' width='%.1f' height='2' rx='1' "
                     "fill='#24404c'><title>%s — no practice</title></rect>"
                     % (cx - bw / 2, T + ih - 2, bw, _esc(lbl)))
        if i % label_every == 0 or i == n - 1:
            labels += ("<text x='%.1f' y='%s' fill='#9aa3bf' font-size='11' "
                       "text-anchor='middle'>%s</text>" % (cx, H - 14, _esc(lbl)))
    return ("<div id='%s' style='%s'>"
            "<svg viewBox='0 0 %s %s' width='100%%' style='max-width:%spx'>"
            "%s%s%s</svg></div>"
            % (elid, "" if visible else "display:none", W, H, W, grid, bars, labels))


_WEEKLY_PRON_JS = r"""
(function(){
 var MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
 // MIN_WEEK is a floor for OPENING the chart, not for plotting: a week resting
 // on one or two attempts is a coin flip, and putting it first makes the whole
 // trend start from noise. Interior thin weeks are kept — a dip between two
 // real measurements says something; a lone dot before you started doesn't.
 var WEEKS=12, GOAL=%d, MIN_WEEK=5, TAIL=4;
 // Every source of an Azure pronunciation score. 'rec' comes from the server
 // (history.json); the other two are score-history prefixes in SkillStore.
 var SERIES=[['rec','Speaking','#46b3c9'],
             ['reading:','Reading passage','#43c59e'],
             ['word:','Single word','#ffb454']];
 function monday(ds){
   var d=new Date(ds+'T12:00:00Z'); d.setUTCDate(d.getUTCDate()-((d.getUTCDay()+6)%%7));
   return d.toISOString().slice(0,10);
 }
 function label(ds){ var p=ds.split('-'); return MON[+p[1]-1]+' '+(+p[2]); }
 // Local date, not toISOString(): UTC is a day behind here for most of the
 // evening, which would put "this week" in the previous one every Monday.
 function today(){
   var d=new Date(), p=function(n){ return (n<10?'0':'')+n; };
   return d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate());
 }
 function span(wk){                       // "Aug 3 – 9" / "Jul 27 – Aug 2"
   var d=new Date(wk+'T12:00:00Z'); d.setUTCDate(d.getUTCDate()+6);
   var end=d.toISOString().slice(0,10);
   return label(wk)+' – '+(end.slice(5,7)===wk.slice(5,7) ? +end.slice(8) : label(end));
 }
 function addWeeks(wk,n){
   var d=new Date(wk+'T12:00:00Z'); d.setUTCDate(d.getUTCDate()+7*n);
   return d.toISOString().slice(0,10);
 }
 function weeksApart(a,b){
   return Math.round((new Date(b+'T12:00:00Z')-new Date(a+'T12:00:00Z'))/6048e5);
 }
 function pairs(src, scores){
   if(src==='rec') return (window.PRON_WEEKLY_REC||[]).slice();
   var out=[];
   Object.keys(scores||{}).forEach(function(key){
     if(key.indexOf(src)!==0) return;
     (scores[key]||[]).forEach(function(x){
       if(x && typeof x.s==='number' && x.d) out.push([x.d, x.s]);
     });
   });
   return out;
 }
 window.addEventListener('load',function(){
   var host=document.getElementById('pron-weekly'); if(!host)return;
   var scores=(window.SkillStore&&window.SkillStore.get('ec_scores',{}))||{};
   // bucket every series by the Monday of its week
   var data=[], newest='', busy={};
   SERIES.forEach(function(s){
     var by={};
     pairs(s[0],scores).forEach(function(p){
       var wk=monday(p[0]); (by[wk]=by[wk]||[]).push(p[1]);
       if(wk>newest) newest=wk;
     });
     var avg={};
     Object.keys(by).forEach(function(wk){
       var v=by[wk], m=v.reduce(function(n,x){return n+x;},0)/v.length;
       avg[wk]={v:Math.round(m*10)/10, n:v.length};
       busy[wk]=(busy[wk]||0)+v.length;      // attempts that week, all series
     });
     data.push({key:s[0], label:s[1], color:s[2], avg:avg});
   });
   if(!newest){
     host.innerHTML="<p class='hint'>No pronunciation scores logged yet — analyze a "+
       "recording, or score a word in Practice single word, to start this chart.</p>";
     return;
   }

   // Right edge is the week you are IN, not the newest week with data, so a
   // silent week reads as a silent week instead of vanishing off the axis.
   // Past TAIL weeks that stops paying: an absence long enough to fill the
   // chart with blanks would push the history it exists to show off the left
   // edge, so cap the empty tail and put the real length of it in words.
   var CUR=monday(today()), gap=weeksApart(newest,CUR), last=CUR;
   if(gap>TAIL) last=addWeeks(newest,TAIL);
   if(newest>CUR) last=newest;              // future-dated rows / clock skew
   // Left edge is the first week solid enough to average. Leading thin weeks
   // are dropped; how many attempts went with them is reported, not hidden.
   var have=Object.keys(busy).sort(), first=null, dropped=0;
   for(var i=0;i<have.length;i++){ if(busy[have[i]]>=MIN_WEEK){ first=have[i]; break; } }
   if(first===null) first=have[0];          // nothing solid yet — show it all
   have.forEach(function(wk){ if(wk<first) dropped+=busy[wk]; });
   var weeks=[], d=new Date(first+'T12:00:00Z'), stop=new Date(last+'T12:00:00Z');
   while(d<=stop && weeks.length<WEEKS*4){
     weeks.push(d.toISOString().slice(0,10)); d.setUTCDate(d.getUTCDate()+7);
   }
   var capped=Math.max(0, weeks.length-WEEKS);
   if(capped) weeks=weeks.slice(-WEEKS);
   var shown={}; weeks.forEach(function(wk){ shown[wk]=1; });

   // the y-range must follow what is drawn, not what was trimmed away
   var lo=GOAL;
   data.forEach(function(s){ weeks.forEach(function(wk){
     if(s.avg[wk] && s.avg[wk].v<lo) lo=s.avg[wk].v; }); });

   var W=920,H=300,L=44,R=16,T=18,B=46, iw=W-L-R, ih=H-T-B;
   // Scores cluster high, so a 0-100 axis would flatten the trend. Zoom to the
   // data instead, floored at 50 and always wide enough to show the goal line.
   var top=100, bot=Math.min(50, Math.max(0, Math.floor((Math.min(lo,GOAL)-5)/5)*5));
   function y(v){ return T+ih*(1-(v-bot)/(top-bot)); }
   function x(i){ return L+(weeks.length<2?iw/2:iw*i/(weeks.length-1)); }

   var grid='';
   for(var g=bot; g<=top; g+=Math.max(5,Math.round((top-bot)/5/5)*5)){
     grid+="<line x1='"+L+"' y1='"+y(g).toFixed(1)+"' x2='"+(W-R)+"' y2='"+y(g).toFixed(1)+
           "' stroke='#24404c'/><text x='"+(L-8)+"' y='"+(y(g)+4).toFixed(1)+
           "' fill='#9aa3bf' font-size='11' text-anchor='end'>"+g+"</text>";
   }
   if(GOAL>bot && GOAL<top){
     grid+="<line x1='"+L+"' y1='"+y(GOAL).toFixed(1)+"' x2='"+(W-R)+"' y2='"+y(GOAL).toFixed(1)+
           "' stroke='#ff9db0' stroke-width='1.5' stroke-dasharray='5 4'/>"+
           // left-aligned: the newest week is always pinned to the right edge,
           // which is exactly where a right-aligned label would collide with it
           "<text x='"+(L+6)+"' y='"+(y(GOAL)-6).toFixed(1)+"' fill='#ff9db0' font-size='10' "+
           "text-anchor='start'>goal "+GOAL+"</text>";
   }
   // The week in progress is a partial average, so say so on the chart rather
   // than letting a half-done week look like a finished one that dipped.
   // A band, not a line: the current week is always the last column, so a rule
   // drawn on it lands on the plot's own right edge and reads as a border.
   if(shown[CUR]){
     var half=(weeks.length>1 ? (x(1)-x(0))/2 : iw/2),
         bx=Math.max(L, x(weeks.length-1)-half);
     grid+="<rect x='"+bx.toFixed(1)+"' y='"+T+"' width='"+(W-R-bx).toFixed(1)+
           "' height='"+ih+"' fill='#9aa3bf' opacity='.07'/>"+
           "<line x1='"+bx.toFixed(1)+"' y1='"+T+"' x2='"+bx.toFixed(1)+"' y2='"+(T+ih)+
           "' stroke='#9aa3bf' stroke-width='1' stroke-dasharray='3 4' opacity='.4'/>"+
           "<text x='"+(W-R-5)+"' y='"+(T+12)+"' fill='#9aa3bf' font-size='10' "+
           "text-anchor='end' opacity='.85'>this week so far</text>";
   }
   var xlab='', every=Math.max(1,Math.ceil(weeks.length/12));
   weeks.forEach(function(wk,i){
     if(i%%every===0||i===weeks.length-1)
       xlab+="<text x='"+x(i).toFixed(1)+"' y='"+(H-16)+"' fill='#9aa3bf' font-size='11' "+
             "text-anchor='middle'>"+label(wk)+"</text>";
   });

   var svg='', legend='';
   data.forEach(function(s){
     var pts=[];
     weeks.forEach(function(wk,i){ if(s.avg[wk]) pts.push([x(i),y(s.avg[wk].v),wk,s.avg[wk]]); });
     if(!pts.length) return;
     if(pts.length>1)
       svg+="<polyline fill='none' stroke='"+s.color+"' stroke-width='2.5' points='"+
            pts.map(function(p){return p[0].toFixed(1)+','+p[1].toFixed(1);}).join(' ')+"'/>";
     pts.forEach(function(p){
       svg+="<circle cx='"+p[0].toFixed(1)+"' cy='"+p[1].toFixed(1)+"' r='4' fill='"+s.color+
            "'><title>"+s.label+" — "+span(p[2])+": "+p[3].v+" avg over "+
            p[3].n+" attempt"+(p[3].n===1?'':'s')+
            (p[2]===CUR?" so far — this week isn't over":"")+"</title></circle>";
     });
     legend+="<span class='lg'><i style='background:"+s.color+"'></i>"+s.label+"</span>";
   });

   // Hover target: a whole column, not the 4px dot. The dot is what you aim
   // at, but three series in one week can sit two pixels apart — asking for a
   // dot-sized hit box means the number you want is the number you can't get.
   // Landing anywhere in the week shows every series in it at once, which is
   // also the comparison actually worth making.
   var step=(weeks.length>1 ? x(1)-x(0) : iw), bands='';
   weeks.forEach(function(wk,i){
     // Clamp both edges, not just the left one: moving the start of the end
     // columns inward while keeping the full width makes them overlap their
     // neighbour, and the band painted later silently wins the hover.
     var bx=Math.max(L, x(i)-step/2), bx2=Math.min(W-R, x(i)+step/2);
     bands+="<rect class='pw-band' data-i='"+i+"' x='"+bx.toFixed(1)+"' y='"+T+"' width='"+
            (bx2-bx).toFixed(1)+"' height='"+ih+"' fill='transparent' aria-hidden='true'/>";
   });
   // Drawn in SVG units so it needs no rescaling when the chart is responsive.
   var hov="<line id='pw-cross' y1='"+T+"' y2='"+(T+ih)+"' x1='0' x2='0' stroke='#e8eef5' "+
           "stroke-width='1' stroke-dasharray='2 3' opacity='0' pointer-events='none'/>"+
           "<g id='pw-focus' pointer-events='none'></g>";

   host.innerHTML="<div id='pw-wrap' style='position:relative'>"+
     "<svg viewBox='0 0 "+W+" "+H+"' width='100%%' style='max-width:"+W+"px;display:block'>"+
     grid+svg+xlab+hov+bands+"</svg>"+
     // No fade: a transition only advances while the page is actually being
     // painted, so a throttled or backgrounded tab leaves the tooltip stuck at
     // opacity 0 with all the right content in it. Not worth 80ms of polish.
     "<div id='pw-tip' role='status' style='position:absolute;left:0;top:0;pointer-events:none;"+
       "opacity:0;background:#0d1a22;border:1px solid #2f5364;"+
       "border-radius:9px;padding:8px 11px;font-size:12.5px;line-height:1.55;color:#e8eef5;"+
       "box-shadow:0 8px 24px rgba(0,0,0,.5);white-space:nowrap;z-index:5'></div></div>";

   var wrap=document.getElementById('pw-wrap'), tip=document.getElementById('pw-tip'),
       cross=document.getElementById('pw-cross'), focus=document.getElementById('pw-focus');
   function place(ev){
     var r=wrap.getBoundingClientRect(), px=ev.clientX-r.left, py=ev.clientY-r.top;
     var tw=tip.offsetWidth, th=tip.offsetHeight;
     var lx=px+16; if(lx+tw > r.width-4) lx=px-tw-16; if(lx<4) lx=4;
     var ly=py-th-14; if(ly<4) ly=py+18;
     tip.style.left=lx.toFixed(0)+'px'; tip.style.top=ly.toFixed(0)+'px';
   }
   function show(i, ev){
     var wk=weeks[i], rows='', rings='';
     data.forEach(function(s){
       var a=s.avg[wk]; if(!a) return;
       rows+="<div style='margin-top:4px'><span style='display:inline-block;width:9px;height:9px;"+
             "border-radius:2px;background:"+s.color+";margin-right:7px'></span>"+s.label+
             "  <b>"+a.v+"</b> <span style='opacity:.6'>· "+a.n+" attempt"+(a.n===1?'':'s')+
             "</span></div>";
       rings+="<circle cx='"+x(i).toFixed(1)+"' cy='"+y(a.v).toFixed(1)+"' r='7' fill='none' "+
              "stroke='"+s.color+"' stroke-width='2'/>";
     });
     // An empty week is a real answer to "what happened here?", so say it.
     if(!rows) rows="<div style='margin-top:4px;opacity:.65'>nothing scored this week</div>";
     tip.innerHTML="<div style='font-weight:700'>"+span(wk)+
       (wk===CUR?" <span style='opacity:.6;font-weight:400'>· so far</span>":"")+"</div>"+rows;
     focus.innerHTML=rings;
     cross.setAttribute('x1',x(i).toFixed(1)); cross.setAttribute('x2',x(i).toFixed(1));
     cross.setAttribute('opacity','.3');
     tip.style.opacity='1';
     place(ev);
   }
   function hide(){ tip.style.opacity='0'; cross.setAttribute('opacity','0'); focus.innerHTML=''; }
   Array.prototype.forEach.call(host.querySelectorAll('.pw-band'), function(b){
     var i=+b.getAttribute('data-i');
     b.addEventListener('mouseenter', function(ev){ show(i,ev); });
     b.addEventListener('mousemove', function(ev){ place(ev); });
   });
   wrap.addEventListener('mouseleave', hide);
   // A tap on a phone fires mouseenter with no reliable mouseleave to follow,
   // which would leave the tooltip parked over a chart that is mostly tooltip
   // at that width. Tapping anywhere else clears it.
   document.addEventListener('pointerdown', function(ev){
     if(!wrap.contains(ev.target)) hide();
   }, true);

   var lg=document.getElementById('pron-weekly-legend'); if(lg) lg.innerHTML=legend;
   var nt=document.getElementById('pron-weekly-note');
   if(nt){
     var msg = gap>0
       ? " Nothing scored yet in the week you're in (" + span(CUR) + ")" +
         (gap>TAIL ? ", and nothing since " + span(newest) + " — " + gap + " weeks ago." : ".")
       : " The last column is the week you're in (" + span(CUR) +
         "), averaged over what you've done so far.";
     if(dropped)
       msg += " " + dropped + " earlier attempt" + (dropped===1?"":"s") + ", before " +
              label(first) + ", rest on too few tries to average and are left out.";
     if(capped) msg += " Only the last " + WEEKS + " weeks are shown.";
     nt.textContent = msg;
   }
 });
})();
"""


def _weekly_pron_section(history):
    """Average pronunciation score per week — the headline view.

    The improvement curve further down plots one point per recording, so a
    short or hard session reads as a slump. Averaging by week, across every
    Azure-scored attempt (recordings and drills alike), is what actually shows
    the trend — which is why this leads the page and the old one-week-vs-last
    tables no longer do.
    """
    rec = []
    for row in (history or []):
        day, score = _history_day(row), row.get("pron_score")
        if day and isinstance(score, (int, float)):
            rec.append([day.isoformat(), round(float(score), 1)])
    return ("<h2>Average pronunciation score by week</h2>"
            "<p class='sub'>Every Azure-scored attempt, averaged per week. "
            "<b>Speaking</b> comes from your analyzed recordings, <b>Reading passage</b> "
            "and <b>Single word</b> from those practice panels. A missing dot is a week "
            "with no attempts. Hover anywhere in a week's column for every score in it "
            "and how many attempts each one rests on.</p>"
            "<div class='chartcard'><div id='pron-weekly'></div>"
            "<div class='legend' id='pron-weekly-legend'></div></div>"
            "<p class='hint'>Read the slope, not the wobble: recording length and "
            "task difficulty move any single week, and a dot resting on 3 attempts "
            "says far less than one resting on 300."
            "<span id='pron-weekly-note'></span></p>"
            "<script>window.PRON_WEEKLY_REC=%s;%s</script>"
            % (json.dumps(rec), _WEEKLY_PRON_JS % PRON_GOAL))


def _practice_time_section(rows):
    """'How much have I actually practised?' — total time by day / week / month."""
    import datetime
    if not rows:
        return ""
    per_day = {}
    for ds, sec in rows:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(ds))
        if m and sec:
            per_day[m.group(0)] = per_day.get(m.group(0), 0.0) + sec
    if not per_day:
        return ""
    total = sum(per_day.values())
    active = len(per_day)
    best_day, best_sec = max(per_day.items(), key=lambda kv: kv[1])

    # current streak: consecutive days with practice, ending at the most recent
    days = sorted(datetime.date(*(int(x) for x in d.split("-"))) for d in per_day)
    streak, cur = 1, 1
    for a, b in zip(days, days[1:]):
        cur = cur + 1 if (b - a).days == 1 else 1
        streak = max(streak, cur)
    run = 1
    for a, b in zip(reversed(days), list(reversed(days))[1:]):
        if (a - b).days == 1:
            run += 1
        else:
            break

    def stat(big, small):
        return ("<div class='m' style='min-width:110px'><span>%s</span>%s</div>"
                % (big, small))

    stats = ("<div class='metrics'>%s%s%s%s</div>" % (
        stat(_fmt_total(total), "total practice"),
        stat(active, "days practised"),
        stat(_fmt_total(total / active), "average per active day"),
        stat("%d" % run, "day streak (best %d)" % streak)))

    charts = ""
    for mode, vis in (("day", True), ("week", False), ("month", False)):
        charts += _practice_bars(_bucket_practice(rows, mode), "ptime-" + mode, vis)

    btns = "".join(
        "<button class='btn small ptimebtn%s' data-range='%s'>%s</button>"
        % (" active" if mode == "day" else "", mode, label)
        for mode, label in (("day", "Daily"), ("week", "Weekly"), ("month", "Monthly")))

    return ("<h2>Time spent on Speaking</h2>"
            "<p class='sub'>Measured from the length of every recording you've "
            "analyzed. Daily shows the last 30 days, weekly the last 26 weeks, "
            "monthly the last 12 months — empty slots are days you didn't practise.</p>"
            "%s<div class='chartcard'><div class='drillnav'>%s</div>%s"
            "<p class='hint'>Best day: %s (%s). Hover a bar for its exact total.</p>"
            "</div>"
            % (stats, btns, charts, _esc(best_day), _fmt_total(best_sec)))


def _progress_body(history, items=None, rec_ids=None):
    """Inner HTML: improvement curve (SVG) + delta chips + session table."""
    n = len(history)
    # plot geometry
    W, H = 820, 360
    L, R, T, B = 56, 24, 24, 48
    iw, ih = W - L - R, H - T - B

    # y-range: zoom to the data but keep 0-100 ceiling
    vals = [h[k] for h in history for k, _, _ in _SERIES if h.get(k) is not None]
    ymin = max(0, (min(vals) // 10) * 10 - 5) if vals else 0
    ymax = 100

    def x(i):
        return L + (iw if n == 1 else iw * i / (n - 1)) * (0.5 if n == 1 else 1)

    def y(v):
        return T + ih * (1 - (v - ymin) / (ymax - ymin))

    # gridlines + y labels
    grid = ""
    step = 10 if (ymax - ymin) <= 60 else 20
    g = ymin
    while g <= ymax:
        gy = y(g)
        grid += ("<line x1='%s' y1='%s' x2='%s' y2='%s' stroke='#24404c'/>"
                 "<text x='%s' y='%s' fill='#9aa3bf' font-size='11' "
                 "text-anchor='end'>%s</text>") % (L, gy, W - R, gy, L - 8, gy + 4, g)
        g += step

    # x labels — a full ISO date per session overlaps badly once there are more
    # than a handful, so tick every session but LABEL only a readable subset,
    # always including the first and last.
    xlabels = ""
    label_every = max(1, -(-n // 9))          # ceil(n/9) -> at most ~9 labels
    years = {str(h.get("date", ""))[:4] for h in history if h.get("date")}
    multiyear = len(years) > 1
    # decide which indices get a text label, then drop any that would collide
    # with the final one (which is always shown — it's the most recent session)
    labelled = [i for i in range(n) if i % label_every == 0]
    if n and labelled and labelled[-1] != n - 1:
        if n - 1 - labelled[-1] < label_every / 2:
            labelled.pop()
        labelled.append(n - 1)
    labelled = set(labelled)
    for i, h in enumerate(history):
        xi = x(i)
        xlabels += ("<line x1='%.1f' y1='%s' x2='%.1f' y2='%s' stroke='#24404c'/>"
                    % (xi, T + ih, xi, T + ih + 4))
        if i not in labelled:
            continue
        raw = str(h.get("date", "") or (i + 1))
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
        if m:
            mon = ("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec"
                   .split()[int(m.group(2)) - 1])
            lbl = "%s %s" % (mon, m.group(3).lstrip("0"))
            if multiyear:
                lbl += " '%s" % m.group(1)[2:]
        else:
            lbl = raw
        xlabels += ("<text x='%.1f' y='%s' fill='#9aa3bf' font-size='11' "
                    "text-anchor='middle'>%s</text>") % (xi, H - 18, _esc(lbl))
    if not multiyear and years:
        xlabels += ("<text x='%s' y='%s' fill='#5f7b84' font-size='10' "
                    "text-anchor='end'>%s</text>"
                    % (W - R, H - 4, _esc(sorted(years)[0])))

    # series polylines + dots
    series_svg, legend = "", ""
    for key, label, color in _SERIES:
        pts = [(x(i), y(h[key])) for i, h in enumerate(history) if h.get(key) is not None]
        if not pts:
            continue
        poly = " ".join("%s,%s" % (round(px, 1), round(py, 1)) for px, py in pts)
        series_svg += "<polyline fill='none' stroke='%s' stroke-width='2.5' points='%s'/>" % (color, poly)
        for px, py in pts:
            series_svg += "<circle cx='%s' cy='%s' r='3.5' fill='%s'/>" % (round(px, 1), round(py, 1), color)
        legend += ("<span class='lg'><i style='background:%s'></i>%s</span>" % (color, label))

    if not series_svg:
        series_svg = ("<text x='%s' y='%s' fill='#9aa3bf' font-size='14' "
                      "text-anchor='middle'>No Azure pronunciation scores logged yet — "
                      "run a session with Azure to start the curve.</text>"
                      % (W / 2, H / 2))

    chart = ("<svg viewBox='0 0 %s %s' width='100%%' style='max-width:%spx'>"
             "%s%s%s%s</svg>" % (W, H, W, grid, series_svg, xlabels,
                                 ""))

    # latest deltas (vs previous session)
    deltas = ""
    if n >= 2:
        cur, prev = history[-1], history[-2]
        chips = []
        for key, label, _ in _SERIES + [("grammar_errors", "Grammar errors", "")]:
            if cur.get(key) is not None and prev.get(key) is not None:
                d = cur[key] - prev[key]
                good = d < 0 if key == "grammar_errors" else d > 0
                arrow = "▲" if d > 0 else ("▼" if d < 0 else "—")
                cls = "up" if good else ("down" if d != 0 else "flat")
                chips.append("<span class='chip %s'>%s %s%+d</span>" % (cls, _esc(label), arrow, d))
        deltas = "<div class='chips'>%s</div>" % "".join(chips)

    # backfill recording length from the analysis items when a history row
    # (logged before we tracked duration) doesn't carry it
    durmap, durmap_t = {}, {}
    for it in (items or []):
        d = _metrics_from_data(it).get("duration_sec")
        if d is None:
            ap = it.get("audio_abs")
            if ap and os.path.exists(ap):
                d = _audio_duration(ap)   # measure the real audio file
        if d is not None:
            durmap[(str(it.get("date", "")), str(it.get("title", "")))] = d
            durmap_t[str(it.get("title", ""))] = d

    # session table (Length = how long that recording ran)
    cols = [("date", "Date"), ("title", "Session"), ("pron_score", "Overall"),
            ("accuracy", "Accuracy"), ("fluency_score", "Fluency"),
            ("prosody", "Prosody"), ("grammar_errors", "Grammar errs"),
            ("wpm", "WPM"), ("duration_sec", "Length")]
    head = "".join("<th>%s</th>" % c[1] for c in cols) + "<th></th>"
    body = ""
    total_sec, counted = 0.0, 0
    practice_rows = []      # (date, seconds) for the time-spent chart
    for idx in range(len(history) - 1, -1, -1):   # newest first, keep real index
        h = history[idx]
        cells = ""
        for key, _label in cols:
            if key == "duration_sec":
                dur = h.get("duration_sec")
                if dur is None:
                    dur = (durmap.get((str(h.get("date", "")), str(h.get("title", ""))))
                           or durmap_t.get(str(h.get("title", ""))))
                if dur is not None:
                    total_sec += dur
                    counted += 1
                    practice_rows.append((str(h.get("date", "")), dur))
                    cells += "<td>%s</td>" % _fmt_dur(dur)
                else:
                    cells += "<td>—</td>"
            elif key == "title":
                v = h.get("title")
                rid = (rec_ids or {}).get(
                    (str(h.get("date", "")), str(h.get("title", ""))))
                if rid and v:
                    cells += ("<td><a onclick=\"showPanel('%s')\" style='color:var(--accent);"
                              "cursor:pointer' title='Open the detailed report'>%s ↗</a></td>"
                              % (rid, _esc(v)))
                else:
                    cells += "<td>%s</td>" % _esc(v if v is not None else "—")
            else:
                v = h.get(key)
                cells += "<td>%s</td>" % _esc(v if v is not None else "—")
        cells += ("<td style='text-align:right'><button class='btn small' "
                  "onclick='deleteSession(%d)' title='Remove this row from history' "
                  "style='background:#3a2029;color:#ff9db0'>&#10005;</button></td>" % idx)
        body += "<tr>" + cells + "</tr>"

    total_line = ""
    if total_sec > 0:
        total_line = ("<p class='summary' style='border-left:4px solid var(--accent)'>"
                      "⏱ Total recorded practice: <b>%s</b> across %d recording(s)."
                      "</p>" % (_fmt_total(total_sec), counted))

    return ("<h1>Your Speaking improvement curve</h1>"
            "<p class='sub'>One point per analyzed recording, oldest first — %s "
            "logged. Higher is better for scores; lower is better for grammar "
            "errors. For the smoothed week-by-week view, see the chart at the "
            "top of this page.</p>"
            "<div class='chartcard'>%s<div class='legend'>%s</div></div>%s"
            "%s"                              # time-spent charts
            "%s<table><tr>%s</tr>%s</table>"
            % (n, chart, legend, deltas,
               _practice_time_section(practice_rows), total_line, head, body))


# Target overall pronunciation score. No longer rendered as its own progress
# bar — it's the dashed reference line on the weekly average chart, where a
# single number is far easier to read against an actual trend.
PRON_GOAL = 85


def _blind_spots(items):
    """Top recurring errors across every recording — turns data into a plan."""
    from collections import Counter
    words, rules, choices = Counter(), Counter(), Counter()
    for d in items or []:
        az = d.get("azure") or {}
        for w in az.get("words", []):
            if w.get("error") == "Mispronunciation":
                tok = (w.get("word", "") or "").lower().strip(".,!?;:")
                if tok:
                    words[tok] += 1
        for g in d.get("grammar", []):
            if g.get("rule"):
                rules[g["rule"]] += 1
        for c in d.get("word_choice", []):
            if c.get("note"):
                choices[c["note"]] += 1

    def chips(counter, color, kind):
        if not counter:
            return "<p class='hint'>Nothing recurring yet — analyze a few recordings.</p>"
        out = []
        # Word-choice chips still land on a row in the Word choice table. Word
        # chips used to land on a pill in the per-recording word list, and that
        # list is gone — so they no longer offer a jump rather than offering
        # one that quietly does nothing. The ✓ still works on both.
        jump = kind != "word"
        for k, n in counter.most_common(8):
            cid = kind + ":" + k
            out.append(
                "<span class='chip bschip%s' data-kind='%s' data-key=\"%s\" data-id=\"%s\" "
                "%sstyle='color:%s'>"
                "%s · %d×<button class='masterbtn' title='Toggle mastered' "
                "onclick='toggleMaster(event,this)'>✓</button></span>"
                % ("" if jump else " nojump", kind, _attr(k), _attr(cid),
                   "title='Click to find it in a recording' " if jump else "",
                   color, _esc(k), n))
        return "".join(out)
    # Grammar mistakes are consolidated in the Grammar module (the error log),
    # so we no longer duplicate them here — just point to it.
    if not (words or choices):
        return ""
    grammar_link = ("<div class='card'><b>Grammar mistakes</b><p class='hint' "
                    "style='margin:6px 0 0'>Now consolidated in the "
                    "<a onclick=\"showPanel('grammar')\" style='color:var(--accent);"
                    "cursor:pointer'>Grammar module</a> — your recurring grammar "
                    "errors with corrections and sources live there.</p></div>")
    cards = ("<div class='card'><b>Most-mispronounced words</b><div class='chips'>%s</div></div>"
             "<div class='card'><b>Word-choice habits</b><div class='chips'>%s</div></div>"
             % (chips(words, "#ff6b6b", "word"), chips(choices, "#ffb454", "note")))
    return ("<h2>Your recurring blind spots</h2>"
            "<p class='sub'>Sound and word-choice patterns from your recordings — "
            "mark one mastered with ✓, or click a word-choice chip to jump to "
            "where it happened.</p>"
            "<div class='bsfilter'>"
            "<button class='btn small active' data-f='all' onclick='bsFilter(this)'>All</button>"
            "<button class='btn small' data-f='active' onclick='bsFilter(this)'>Active</button>"
            "<button class='btn small' data-f='mastered' onclick='bsFilter(this)'>Mastered ✓</button>"
            "</div>" + cards + grammar_link)


# ---------------------------------------------------------------------------
# Project backup — the code and data, without the media
# ---------------------------------------------------------------------------
# Mirrors .gitignore: the things worth keeping are small and hand-made, the
# things left out are large and either re-recordable, re-importable or
# regenerated on the next run.
_BACKUP_SKIP_DIRS = {
    "models", "__pycache__", ".git", "debug", "backups",
    ".venv", "venv", "node_modules", ".idea", ".vscode", ".pytest_cache",
}
# The recordings folder is NOT skipped wholesale: the audio is 157 MB and is
# excluded by extension, but the transcripts, per-recording analyses and
# history.json beside it are ~800 KB of irreplaceable text. Without them a
# restore has no progress curve, no vocabulary report and no past reports.
_BACKUP_SKIP_FILES = {
    ".DS_Store", "dashboard.html", "demo_report.html", "report.html",
    "progress.html", "sample_report.html", ".english_coach.json",
}
# audio/video anywhere is recorded or downloaded material, never source
_BACKUP_SKIP_EXT = {".pyc", ".pyo", ".zip", ".bz2", ".gz", ".tar",
                    ".webm", ".mp3", ".m4a", ".wav", ".mp4", ".mov",
                    ".aac", ".flac", ".ogg"}
# inside listening/ only these are kept: library.json preserves the clip ids
# your practice history is keyed on, so restoring it keeps that history meaningful
_BACKUP_LISTENING_KEEP = {"README.md", "library.json"}


def _backup_members(root=None):
    """(absolute path, name inside the zip) for every file that belongs in a backup."""
    root = os.path.abspath(root or os.path.dirname(os.path.abspath(__file__)))
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames)
                       if d not in _BACKUP_SKIP_DIRS and not d.startswith(".")]
        rel_dir = os.path.relpath(dirpath, root)
        top = "" if rel_dir == "." else rel_dir.split(os.sep)[0]
        for fn in sorted(filenames):
            if fn in _BACKUP_SKIP_FILES or fn.startswith("._"):
                continue
            if os.path.splitext(fn)[1].lower() in _BACKUP_SKIP_EXT:
                continue
            if top == LISTENING_DIR and fn not in _BACKUP_LISTENING_KEEP:
                continue
            # The recordings library is media by definition. Take only the
            # analysis text out of it — an allowlist, so a cover image or any
            # future file type can't slip past an extension blacklist.
            if top == LIBRARY_NAME and os.path.splitext(fn)[1].lower() not in (".txt", ".json"):
                continue
            p = os.path.join(dirpath, fn)
            if os.path.isfile(p):
                out.append((p, os.path.relpath(p, root)))
    return out


def backup_summary(root=None):
    """What a backup would contain, for showing before the user commits to it."""
    members = _backup_members(root)
    size = 0
    for p, _a in members:
        try:
            size += os.path.getsize(p)
        except OSError:
            pass
    return {"files": len(members), "bytes": size}


_RESTORE_TEMPLATE = """# Restoring English Coach

Backup taken %(when)s — %(files)d files.

## What's in here

- All source, data files, notes and practice scripts
- Your transcripts, per-recording analyses and `history.json` (%(analysis)d files)
  — so the progress curve, the vocabulary reports and every past report come back
- `VideoAudioFiles/progress.json` — scores, review schedules and error logs
  (Vocabulary, Practice words, Grammar log, Listening, ear training). Server-side
  now, so it's just another data file here, not something only a browser has.

## What is NOT in here, and why

| Missing | Why | How to get it back |
|---|---|---|
| Audio recordings (~157 MB) | Too large, and the analysis of each is preserved | Copy `VideoAudioFiles/` across by hand if you want playback |
| Imported listening clips | Third-party audio, re-downloadable | `python3 listening_import.py tatoeba --count 200` |
| Speech models (multi-GB) | Shared across projects, downloaded separately | Put them in `~/Desktop/models/` — see README |
| API keys | Stored in `~/.english_coach.json`, outside the project | Re-enter them in the New analysis form once |

Reports and scores work without any of the above. Playback, transcription and
new analyses need the corresponding piece.

## Steps

1. Unzip somewhere sensible, e.g. `~/Desktop/English Coach`
2. `pip install -r requirements.txt`
3. `python english_coach_web.py` → http://localhost:8000
4. Restore your practice history: open **Summary & progress**, click
   **Restore practice data**, and choose `VideoAudioFiles/progress.json` from
   this zip. This pushes it to the server (`POST /api/progress`), replacing
   whatever's already there — it's no longer trapped in one browser's storage.
5. Optional: re-enter your Azure and DeepSeek keys in **New analysis** to run
   new recordings.

## Sanity check

    python3 english_coach.py --demo --out demo.html   # renders with no keys or models
    python3 test_dictation.py                          # should report 38 passed
"""


def build_project_backup(root=None):
    """Zip the project in memory. Returns (bytes, filename, member count).

    Scores/schedules/error logs live in VideoAudioFiles/progress.json now, so
    _backup_members() already picks it up like any other data file — nothing
    special to collect here.
    """
    import datetime
    import io
    import zipfile
    root = os.path.abspath(root or os.path.dirname(os.path.abspath(__file__)))
    members = _backup_members(root)
    now = datetime.datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    folder = os.path.basename(root).replace(" ", "-") or "english-coach"
    name = "%s-backup-%s.zip" % (folder, stamp)

    analysis = sum(1 for _p, a in members if a.split(os.sep)[0] == LIBRARY_NAME)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path, arc in members:
            try:
                z.write(path, os.path.join(folder, arc))
            except OSError:
                continue
        z.writestr(os.path.join(folder, "RESTORE.md"), _RESTORE_TEMPLATE % {
            "when": now.strftime("%Y-%m-%d %H:%M"),
            "files": len(members),
            "analysis": analysis,
        })
    return buf.getvalue(), name, len(members)


def prunable_media(root=None, older_than_days=0):
    """Media files that can be deleted without losing anything.

    The rule is deliberately strict: a media file is prunable only if a
    `.result.json` sits in the same folder. That result is the whole reason
    deletion is safe — it holds the scores, the transcript and the word-level
    timings, so everything the reports render survives. Audio with no result
    beside it has never been analyzed; it is the only copy of that material and
    is never touched, no matter how old.

    Returns (rows, total_bytes) where each row is
    (path, bytes, mtime, recording folder name).
    """
    import time
    lib = root or library_dir()
    cutoff = time.time() - older_than_days * 86400 if older_than_days else None
    rows, total = [], 0
    if not os.path.isdir(lib):
        return rows, total
    for dirpath, _dirs, files in os.walk(lib):
        if not any(f.endswith(".result.json") for f in files):
            continue
        for fn in sorted(files):
            if not fn.lower().endswith(_AV):
                continue
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if cutoff is not None and st.st_mtime > cutoff:
                continue
            rows.append((p, st.st_size, st.st_mtime,
                         os.path.relpath(dirpath, lib)))
            total += st.st_size
    rows.sort(key=lambda r: r[2])          # oldest first
    return rows, total


def prune_media(root=None, older_than_days=0):
    """Delete prunable media. Returns (deleted_count, bytes_freed, errors).

    Every affected recording is re-stamped first, so `recorded_at` and the
    duration are in result.json before the file they were derived from goes
    away. Without that the reports would still render, but same-day sessions
    would reshuffle and the Length column would go blank.
    """
    lib = root or library_dir()
    rows, _total = prunable_media(lib, older_than_days)
    by_dir = {}
    for path, _size, _mt, _name in rows:
        by_dir.setdefault(os.path.dirname(path), []).append(path)

    deleted, freed, errors = 0, 0, []
    for dirpath, paths in by_dir.items():
        # stamp first: after this, the result no longer needs the media
        for fn in sorted(os.listdir(dirpath)):
            if not fn.endswith(".result.json"):
                continue
            rp = os.path.join(dirpath, fn)
            try:
                with open(rp, encoding="utf-8") as f:
                    item = json.load(f)
            except Exception as e:
                errors.append("%s: %s" % (fn, e))
                continue
            item["audio_abs"] = paths[0]
            try:
                item["_mtime"] = os.path.getmtime(paths[0])
            except OSError:
                pass
            _stamp_recording_meta(item, rp)
        for p in paths:
            try:
                size = os.path.getsize(p)
                os.remove(p)
                deleted += 1
                freed += size
            except OSError as e:
                errors.append("%s: %s" % (os.path.basename(p), e))
    return deleted, freed, errors


def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.0f %s" % (n, unit) if unit == "B" else "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f GB" % n


def _backup_card():
    s = backup_summary()
    return ("<h2>Back up the project</h2>"
            "<p class='sub'>Downloads a zip of the project itself — the code, the "
            "data files, your notes and practice scripts. Recordings and imported "
            "listening audio are left out: they're the bulk of the folder and "
            "neither is part of the project.</p>"
            "<div class='card'>"
            "<button class='btn' id='backup-btn'>&#11015; Download backup</button>"
            "<span class='hint' id='backup-msg' style='margin-left:10px'>"
            "%d files &middot; about %s, plus your practice history</span>"
            "<div style='margin-top:14px;border-top:1px solid var(--line);padding-top:12px'>"
            "<button class='btn small' id='restore-btn'>&#8635; Restore practice data</button>"
            "<input type='file' id='restore-file' accept='.json' style='display:none'>"
            "<span class='hint' id='restore-msg' style='margin-left:10px'>"
            "Load <code>VideoAudioFiles/progress.json</code> from a backup to push "
            "scores, review schedules and error logs to the server.</span></div>"
            "</div>" % (s["files"], _fmt_bytes(s["bytes"])))


def _summary_panel(history, items=None, rec_ids=None):
    """Combined weekly review and long-term progress; no separate report page.

    The weekly pronunciation trend leads: it is the one chart that answers
    "am I getting better?" across every kind of practice, so it belongs above
    the this-week detail rather than buried under it.
    """
    return ("<h1>Summary &amp; progress</h1>"
            + _weekly_pron_section(history)
            + _weekly_review_panel(history, items, embedded=True)
            + "<hr style='border:0;border-top:1px solid var(--line);margin:36px 0'>"
            + _progress_body(history, items, rec_ids=rec_ids))


def generate_progress_html(history):
    """Stand-alone progress page (CLI --progress). Same look as the dashboard."""
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>English Coach — Progress</title><style>" + _DASHBOARD_CSS
            + "</style></head><body><main class='content'><div class='wrap'>"
            + _weekly_pron_section(history)
            + _progress_body(history)
            + "<footer>Generated by English Coach — history.json drives this view.</footer>"
            "</div></main></body></html>")


# ---------------------------------------------------------------------------
# 6. Built-in demo data
#
# A SYNTHETIC sample session — an invented learner ("Mei Lin", Northwind
# Analytics, Riverside University) with realistic Mandarin-L1 error patterns.
# It exists so `--demo` renders a complete, representative report with no
# models, no API keys, and no real personal data. Keep it fictional.
# ---------------------------------------------------------------------------
DEMO_DATA = {
    "title": "Interview self-introduction (sample)",
    "date": str(date.today()),
    "duration": "≈ 2 min",
    "duration_sec": 118.0,
    "level_estimate": "B2 (reaching C1)",
    "overall_summary": (
        "Strong content and a clear narrative arc (background → track record "
        "→ flagship project → communication skills → fit). The main work "
        "is grammar polish, a few translated-sounding word choices, and three "
        "repeatable pronunciation patterns visible in the transcript."
    ),
    "top_fixes": [
        {"issue": "Finish your final consonants",
         "example": "my core strength lies in blendin'",
         "better": "my core strength lies in blendinG",
         "why": "Dropped endings are the single biggest hit to intelligibility — "
                "a listener loses the grammatical signal (-s, -ed, -ing) along "
                "with the sound."},
        {"issue": "Articles before institutions and industries",
         "example": "I worked in fintech industry",
         "better": "I worked in THE fintech industry",
         "why": "Mandarin has no direct equivalent of a/the, so these drop out "
                "under speaking pressure. It reads as non-native immediately, "
                "even though the meaning is clear."},
        {"issue": "Keep weight on the end of the sentence",
         "example": "…which makes me a strong fit for this role (trailing off)",
         "better": "…which makes me a STRONG FIT for this ROLE.",
         "why": "Your best point is the one you say most quietly. English is "
                "stress-timed; landing the final phrase is what makes an answer "
                "sound confident rather than uncertain."},
    ],
    "blind_spots": [
        {"pattern": "Tense drift when describing past work",
         "kind": "grammar",
         "evidence": ["which would took hours", "this position required the candidate"],
         "fix": "Pick the tense before you start the sentence. Drill: narrate "
                "yesterday's work for 60 seconds, past tense only."},
        {"pattern": "Word-final consonants dropped in multi-syllable words",
         "kind": "pronunciation",
         "evidence": ["cutting-edge → 'cuttage'", "vendors → 'vendor'"],
         "fix": "Over-articulate endings in isolation, then in the full phrase: "
                "edge, scores, vendors, departments."},
    ],
    "pronunciation_patterns": [
        {"name": "/v/ vs /w/  (you soften /v/ into /w/)",
         "desc": "Bite the lower lip lightly for /v/; round the lips with no teeth for /w/.",
         "examples": ["external vendors → 'extra windows'", "vendor's core system → 'windows core system'"],
         "drill": "vine/wine, vest/west, very/wary, van/wan"},
        {"name": "/l/ vs /r/",
         "desc": "Tongue tip touches the ridge behind the teeth for /l/; pulls back, no contact, for /r/.",
         "examples": ["blending → 'branding'", "relevant → 'really went'", "Legal → 'Luis'"],
         "drill": "light/right, blend/brand, lay/ray, glass/grass"},
        {"name": "Dropped endings & word stress",
         "desc": "Finish final consonants and put stress on the right syllable.",
         "examples": ["cutting-edge → 'cuttage'", "Riverside → 'river side'", "PRD → 'PID'"],
         "drill": "exaggerate endings: edge, live, scores, vendors, departments"},
    ],
    "grammar": [
        {"said": "10 years experience", "correction": "10 years of experience", "rule": "missing 'of'"},
        {"said": "in fintech industry", "correction": "in the fintech industry", "rule": "article"},
        {"said": "lies on blending", "correction": "lies in blending", "rule": "preposition"},
        {"said": "master's degree of data science", "correction": "master's degree in data science", "rule": "fixed phrase"},
        {"said": "in Riverside University", "correction": "at Riverside University", "rule": "'at' for institutions"},
        {"said": "I want to pass the decade", "correction": "Over the past decade", "rule": "phrasing"},
        {"said": "foundation on machine learning", "correction": "foundation in machine learning", "rule": "preposition"},
        {"said": "the final goes live process", "correction": "the final go-live process", "rule": "compound noun"},
        {"said": "managers are enabled to retrive", "correction": "managers can retrieve", "rule": "wordy + spelling"},
        {"said": "which would took hours", "correction": "which would take hours", "rule": "tense"},
        {"said": "this position required the candidate", "correction": "this position requires the candidate", "rule": "present tense"},
        {"said": "I have really went experienced", "correction": "I have relevant experience", "rule": "phrasing"},
        {"said": "let's about me about my brief introduction", "correction": "that's my brief introduction", "rule": "phrasing"},
    ],
    "word_choice": [
        {"said": "Good morning, interviewer", "suggestion": "Good morning, and thank you for having me", "note": "'interviewer' as a greeting is unnatural"},
        {"said": "My core competitiveness", "suggestion": "My core strength / What sets me apart", "note": "'competitiveness' sounds translated"},
        {"said": "managers are enabled to", "suggestion": "managers can", "note": "simpler and stronger"},
        {"said": "proved my capability", "suggestion": "demonstrates my ability", "note": "more idiomatic"},
        {"said": "Make me a good fit", "suggestion": "make me a strong fit", "note": "stronger collocation"},
    ],
    "azure": {
        "pron_score": 88, "accuracy": 92, "fluency": 84,
        "completeness": 93, "prosody": 87,
        "error_counts": {"Mispronunciation": 1, "Omission": 2, "Insertion": 1,
                         "UnexpectedBreak": 1, "MissingBreak": 0, "Monotone": 0},
        "words": [
            {"word": "today", "accuracy": 99, "error": ""},
            {"word": "was", "accuracy": 98, "error": ""},
            {"word": "a", "accuracy": 95, "error": ""},
            {"word": "beautiful", "accuracy": 96, "error": ""},
            {"word": "day", "accuracy": 99, "error": ""},
            {"word": "long", "accuracy": 40, "error": "Insertion"},
            {"word": "walk", "accuracy": 94, "error": ""},
            {"word": "outside", "accuracy": 0, "error": "Omission"},
            {"word": "countryside", "accuracy": 62, "error": "Mispronunciation"},
            {"word": "the", "accuracy": 0, "error": "Omission"},
            {"word": "forecasting", "accuracy": 90, "error": ""},
            {"word": "rain", "accuracy": 97, "error": ""},
        ],
    },
    "pron_diff": {
        "score": 78,
        "flags": [
            {"expected": "core strength", "heard_as": "co competitive", "type": "replace"},
            {"expected": "blending", "heard_as": "branding", "type": "replace"},
            {"expected": "cutting-edge", "heard_as": "cuttage", "type": "replace"},
            {"expected": "riverside", "heard_as": "river side", "type": "replace"},
            {"expected": "relevant", "heard_as": "really went", "type": "replace"},
            {"expected": "external vendors", "heard_as": "extra windows", "type": "replace"},
            {"expected": "vendor's", "heard_as": "windows", "type": "replace"},
        ],
    },
    "fluency": {"wpm": 132, "fillers": 2, "long_pauses": 3,
                "notes": "pace is in a natural range; two 'uh' fillers — swap for a brief pause; "
                         "'AI agent application' repeats ~5 times, vary it (the agent, the system, this solution)"},
    "polished": (
        "Good morning, and thank you for having me today. My name is Mei Lin, and I'm a "
        "data scientist with about ten years of experience in the fintech industry. My core "
        "strength lies in blending traditional financial modelling with cutting-edge machine learning.\n\n"
        "I hold a master's degree in data science from Riverside University. Over the past decade, "
        "working at Northwind Analytics, I've built a solid foundation in machine learning and "
        "data analysis. I have extensive hands-on experience across the full model-building lifecycle — "
        "from feature-table design and feature mining, to model training, and the final go-live process.\n\n"
        "Recently, I've shifted my focus to automated decision systems. I designed and deployed a "
        "workflow-orchestration service to manage customer tier transitions. With it, account managers "
        "can retrieve a customer's key information, predictive scores, and recommended next steps "
        "in minutes — work that used to take hours or even days. This project demonstrates my "
        "ability to design, build, and optimize production data systems.\n\n"
        "I also noticed this position calls for strong cross-functional communication. I gained relevant "
        "experience on a vendor-procurement project, where I was the "
        "primary owner of the system's requirements documentation. I collaborated closely with internal "
        "teams — Legal, Product, and Finance — as well as external vendors, to define customized features "
        "and modules on top of the vendor's core system.\n\n"
        "So, in short: my background in data science, my track record in machine learning, "
        "and my experience collaborating across teams and with external vendors make "
        "me a strong fit for this role. That's a brief introduction of myself, and I'm happy to take your "
        "questions. Thank you."
    ),
    "prosody_metrics": {
        "pitch_mean_hz": 126.4, "pitch_min_hz": 105.8, "pitch_max_hz": 152.4,
        "pitch_range_st": 6.3, "pitch_var_st": 2.4,
        "speech_rate_syl_s": 4.1, "pause_ratio_pct": 22, "npvi": 43,
        "voiced_pct": 78, "dur_s": 118.0,
        "contour": [136.6, 137.6, 143.7, 140.6, 145.0, 143.4, 139.7, 141.1, 134.1,
                    133.4, 126.1, 121.7, 119.9, 119.0, 109.9, 108.1, 109.7, 111.7,
                    109.1, 108.9, None, None, None, 122.5, 142.8, 150.3, 145.9,
                    143.9, 141.7, 140.4, 140.8, 131.5, 130.1, 126.0, 119.6, 117.3,
                    110.5, 108.5, 108.7, 112.5, 111.5, 112.3, 116.6, 118.0, None,
                    None, None, 122.9, 152.4, 150.1, 143.9, 143.1, 138.7, 137.0,
                    131.3, 123.3, 124.9, 114.7, None, None, 113.1, 116.5, 113.1,
                    117.9, 116.7, 124.1, 127.0, 127.1, None, None, None, 126.8,
                    145.2, 144.4, 139.3, 134.6, 129.1, 128.0, 125.3, 118.8, 118.7,
                    113.1, 118.6, 119.3, 124.0, 124.9, 123.0, 126.1, None, 131.3,
                    126.4, 129.1, None, None, None, 114.4, 135.4, 130.4, 131.7,
                    122.8, 120.8, 119.8, 122.6, 116.3, 120.1, 122.5, 127.4, 129.3,
                    132.0, 129.3, 131.8, 131.9, 135.7, 134.9, 126.0, 122.9, None,
                    None, None, 105.8],
    },
}


# ---------------------------------------------------------------------------
# 6b. Shared analysis pipeline (used by both the CLI and the GUI)
# ---------------------------------------------------------------------------
def analyze_recording(audio_path, reference_text=None, do_llm=True,
                      do_azure=False, base_data=None, strictness="strict",
                      progress=lambda m: None):
    """Run the chosen analyses and return the report-data dict.

    base_data: a pre-made grammar/word-choice/fluency analysis (e.g. a JSON
        file produced by Claude in chat). When given, the local Whisper+Claude
        step is skipped and these contents are used instead — only Azure (if
        requested) is run and merged in. This avoids needing an API key.

    progress(msg) is called with human-readable status strings so a GUI can
    show what's happening.
    """
    data, transcript = {}, ""
    whisper_hyp = None   # only set when we actually run Whisper
    if base_data:
        progress("Using provided grammar analysis…")
        data = dict(base_data)
    elif do_llm:
        # A grammar-analysis failure must NOT sink the whole run: pronunciation
        # scoring is a paid API call and prosody is free signal, and both are
        # still worth having. Record the error and carry on.
        if reference_text:
            # Use the transcript the user already provided — no need to re-run
            # Whisper (which downloads a model from HuggingFace that may be
            # unreachable). This is also more accurate: we analyze exactly what
            # they typed.
            progress("Analyzing grammar & word choice (%s)…" % _provider_label())
            transcript = reference_text
            try:
                data = llm_analyze(transcript)
            except Exception as e:
                progress("⚠️ Grammar analysis failed — continuing without it…")
                data = {"analysis_error": str(e)}
        else:
            progress("Transcribing audio (Whisper)…")
            whisper_hyp, words, duration = transcribe(audio_path)
            transcript = whisper_hyp
            progress("Analyzing grammar & word choice (%s)…" % _provider_label())
            try:
                data = llm_analyze(transcript)
            except Exception as e:
                progress("⚠️ Grammar analysis failed — continuing without it…")
                data = {"analysis_error": str(e)}
            data["fluency"] = fluency_metrics(words, duration)
            data["duration"] = "%d sec" % duration
    data.setdefault("title", os.path.basename(audio_path))
    data.setdefault("date", str(date.today()))
    # record how long the clip is, so total practice time can be tracked
    if data.get("duration_sec") is None:
        ds = _audio_duration(audio_path)
        if ds:
            data["duration_sec"] = ds
            data.setdefault("duration", "%d sec" % round(ds))
    # statistical prosody metrics (pitch variation, rhythm, pace) from the audio
    if data.get("prosody_metrics") is None:
        try:
            progress("Measuring prosody (pitch & rhythm)…")
            pm = analyze_prosody(audio_path)
            if pm:
                data["prosody_metrics"] = pm
        except Exception:
            pass
    # the script-vs-heard diff only makes sense when Whisper produced a separate
    # hypothesis; skip it when the transcript IS the reference text
    if reference_text and whisper_hyp:
        data["pron_diff"] = pronunciation_diff(reference_text, whisper_hyp)
    if do_azure and _looks_non_english(reference_text or ""):
        # Azure scores English pronunciation; a Chinese (or other non-Latin)
        # transcript can't match English audio and would return all zeros.
        progress("Skipping Azure — the transcript isn't English…")
        data["azure_note"] = ("Pronunciation scoring was skipped: the transcript "
                              "isn't English, so Azure can't grade it against your "
                              "audio. Re-transcribe in English (or fix the transcript) "
                              "and re-analyze.")
    elif do_azure:
        if not reference_text:
            raise ValueError("Azure scoring needs the transcript/script you read.")
        progress("Scoring pronunciation (Azure)…")
        az = azure_pronunciation(audio_path, reference_text, progress=progress)
        derived = az.pop("fluency_derived", None)
        if strictness and strictness != "standard":
            progress("Applying %s grading…" % STRICTNESS_LABEL.get(strictness, strictness))
            az = apply_strictness(az, strictness)
        data["azure"] = az
        # fill any missing fluency numbers from Azure's word timings
        if derived:
            fl = dict(data.get("fluency") or {})
            for k in ("wpm", "fillers", "long_pauses"):
                if fl.get(k) in (None, "", "—") and derived.get(k) is not None:
                    fl[k] = derived[k]
            data["fluency"] = fl
    progress("Building report…")
    return data


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="English Speaking Coach — Phase 1")
    ap.add_argument("audio", nargs="?", help="audio/video file to analyze")
    ap.add_argument("--reference", help="text file of the script you read (enables pronunciation diff / Azure)")
    ap.add_argument("--analysis-json", dest="analysis_json",
                    help="grammar/word-choice/fluency analysis from Claude (skips Whisper+Claude, no API key)")
    ap.add_argument("--azure", action="store_true",
                    help="real Azure pronunciation scores (needs --reference + AZURE_SPEECH_KEY/REGION)")
    ap.add_argument("--demo", action="store_true", help="generate a sample report, no models needed")
    ap.add_argument("--progress", action="store_true",
                    help="(re)build the improvement-curve view from history.json and exit")
    ap.add_argument("--out", default="report.html", help="output HTML path")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    history_path = os.path.join(out_dir, "history.json")
    progress_path = os.path.join(out_dir, "progress.html")

    if args.progress:
        hist = json.load(open(history_path, encoding="utf-8")) if os.path.exists(history_path) else []
        with open(progress_path, "w", encoding="utf-8") as f:
            f.write(generate_progress_html(hist))
        print("Wrote %s" % progress_path)
        return

    if args.demo:
        data = DEMO_DATA
    else:
        if not args.audio:
            ap.error("provide an audio file, or use --demo")
        if args.azure and not args.reference:
            ap.error("--azure needs --reference (the script you read aloud)")
        ref = open(args.reference, encoding="utf-8").read() if args.reference else None
        base = None
        if args.analysis_json:
            with open(args.analysis_json, encoding="utf-8") as f:
                base = json.load(f)
        data = analyze_recording(
            args.audio, reference_text=ref, do_llm=not base, do_azure=args.azure,
            base_data=base, progress=lambda m: print(m, file=sys.stderr),
        )

    html_out = generate_html_report(data)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_out)
    print("Wrote %s" % args.out)

    if not args.demo:
        hist = log_session(data, history_path)
        with open(progress_path, "w", encoding="utf-8") as f:
            f.write(generate_progress_html(hist))
        print("Logged session -> %s (%d total). Progress: %s"
              % (history_path, len(hist), progress_path))


if __name__ == "__main__":
    main()
