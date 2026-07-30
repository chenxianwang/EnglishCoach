#!/usr/bin/env python3
"""
Listening library importer
==========================

Populates `listening/library.json` with sentence-level clips of REAL recorded
speech, normalized across sources so the dictation panel can shuffle between
them without knowing where anything came from.

    python3 listening_import.py voa      --count 20
    python3 listening_import.py tatoeba  --count 200
    python3 listening_import.py sbc      --dir ~/Downloads/sbc
    python3 listening_import.py local    --dir ~/my-audio
    python3 listening_import.py list

Sources and licences
--------------------
voa       Voice of America "Learning English". Text, audio and video produced
          exclusively by VOA are PUBLIC DOMAIN and may be reused, including
          commercially, with credit to learningenglish.voanews.com. Graded,
          slower-than-native delivery — the gentlest starting point.
          https://learningenglish.voanews.com/p/6861.html

sbc       Santa Barbara Corpus of Spoken American English (UC Santa Barbara).
          Genuine unscripted conversation — by far the most authentic material
          here — distributed under CC BY-ND 3.0 US, with transcripts already
          timestamped at the intonation-unit level, so no alignment is needed.
          NoDerivatives: fine for your own listening practice, but do not
          redistribute a re-segmented copy.
          https://www.linguistics.ucsb.edu/research/santa-barbara-corpus-spoken-american-english

tatoeba   Sentences with native-speaker audio. Sentence text is CC-BY 2.0 FR;
          EACH RECORDING carries its own licence, given in the metadata. Tatoeba
          is explicit that an empty licence field means you may not reuse that
          audio outside Tatoeba, so this importer skips those recordings.
          https://tatoeba.org/en/downloads

local     Any folder of your own: pairs of `name.mp3` + `name.txt`, or a single
          audio file plus a transcript to be aligned with Whisper.

Why alignment is needed
-----------------------
VOA gives you one MP3 per article and one transcript — no per-sentence timings,
so you cannot loop sentence 7. faster-whisper produces word-level timestamps,
and difflib aligns its (imperfect) hypothesis against the (authoritative)
transcript. Result: sentence boundaries with real start/end times, using the
official wording rather than Whisper's guess.
"""

import argparse
import difflib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LISTEN_DIR = os.path.join(HERE, "listening")
LIB_PATH = os.path.join(LISTEN_DIR, "library.json")

VOA_CREDIT = "Public domain (VOA) — credit learningenglish.voanews.com"
SBC_LICENSE = "CC BY-ND 3.0 US — UC Santa Barbara"


# --------------------------------------------------------------------------- #
# library helpers                                                             #
# --------------------------------------------------------------------------- #
def load_library():
    try:
        with open(LIB_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data.get("clips", []) if isinstance(data, dict) else (data or [])


def load_meta():
    """Per-source bookkeeping (how much material exists vs how much is imported)."""
    try:
        with open(LIB_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("sources", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_library(clips, meta=None):
    os.makedirs(LISTEN_DIR, exist_ok=True)
    by_id = {}
    for c in clips:
        by_id[c["id"]] = c              # later import wins on id collision
    out = sorted(by_id.values(), key=lambda c: (c.get("source", ""), c["id"]))
    sources = load_meta()
    sources.update(meta or {})
    with open(LIB_PATH, "w", encoding="utf-8") as f:
        json.dump({"clips": out, "sources": sources}, f, ensure_ascii=False, indent=1)
    return out


def add(clips, new, meta=None):
    merged = save_library(load_library() + new, meta)
    print("Added %d clip(s). Library now holds %d." % (len(new), len(merged)))
    counts = {}
    for c in merged:
        counts[c.get("source", "?")] = counts.get(c.get("source", "?"), 0) + 1
    for s, n in sorted(counts.items()):
        print("   %-10s %d" % (s, n))


def _slug(s, n=40):
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:n] or "clip"


def _today():
    import datetime
    return datetime.date.today().isoformat()


# --------------------------------------------------------------------------- #
# sentence splitting + Whisper alignment                                      #
# --------------------------------------------------------------------------- #
def split_sentences(text):
    """Split a transcript into sentences, keeping them a usable dictation size."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'“])", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # a very long sentence is a poor dictation target — split on clause commas
        if len(p.split()) > 28:
            chunks = re.split(r"(?<=,)\s+", p)
            buf = ""
            for ch in chunks:
                if len((buf + " " + ch).split()) > 24 and buf:
                    out.append(buf.strip())
                    buf = ch
                else:
                    buf = (buf + " " + ch).strip()
            if buf:
                out.append(buf)
        else:
            out.append(p)
    return [s for s in out if len(s.split()) >= 3]


def _norm_words(s):
    return re.sub(r"[^a-z0-9']+", " ", (s or "").lower()).split()


def align_sentences(audio_path, transcript, model="medium", language="en"):
    """Return [(sentence, start, end)] by aligning `transcript` to Whisper's
    word timings for `audio_path`.

    Whisper supplies the CLOCK; the transcript supplies the WORDS. We never use
    Whisper's wording — it mishears, and a wrong reference would be graded as
    your mistake.
    """
    import english_coach as ec
    print("  transcribing for timings (%s)…" % model)
    _text, words, _dur = ec.transcribe(audio_path, language=language,
                                       model_name=model,
                                       progress=lambda m: None)
    if not words:
        raise RuntimeError("Whisper returned no word timings for %s" % audio_path)

    hyp = [_norm_words(w["w"])[0] if _norm_words(w["w"]) else "" for w in words]
    sents = split_sentences(transcript)
    ref, spans = [], []
    for si, s in enumerate(sents):
        ws = _norm_words(s)
        spans.append((len(ref), len(ref) + len(ws), si))
        ref += ws

    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    # map every reference-word index onto a hypothesis index where they matched
    ref2hyp = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ref2hyp[i1 + k] = j1 + k

    out = []
    for a, b, si in spans:
        hits = [ref2hyp[i] for i in range(a, b) if i in ref2hyp]
        if len(hits) < max(2, (b - a) // 3):
            continue                      # too little anchoring to trust the timing
        start = words[min(hits)]["start"]
        end = words[max(hits)]["end"]
        if end <= start:
            continue
        out.append((sents[si], round(max(0.0, start - 0.15), 2), round(end + 0.25, 2)))
    print("  aligned %d/%d sentences" % (len(out), len(sents)))
    return out


# --------------------------------------------------------------------------- #
# adapters                                                                    #
# --------------------------------------------------------------------------- #
def import_local(args):
    """A folder of your own audio. Either name.mp3 + name.txt pairs (one clip
    per file), or --align to split a long recording into sentences."""
    src = os.path.expanduser(args.dir)
    if not os.path.isdir(src):
        sys.exit("No such folder: %s" % src)
    exts = (".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mp4")
    dest = os.path.join(LISTEN_DIR, "local")
    os.makedirs(dest, exist_ok=True)
    import shutil
    clips = []
    for fn in sorted(os.listdir(src)):
        if not fn.lower().endswith(exts):
            continue
        stem = os.path.splitext(fn)[0]
        txt = os.path.join(src, stem + ".txt")
        if not os.path.exists(txt):
            print("  skip %s — no matching %s.txt" % (fn, stem))
            continue
        with open(txt, encoding="utf-8") as f:
            transcript = f.read().strip()
        if not transcript:
            continue
        rel = os.path.join("local", fn)
        if not os.path.exists(os.path.join(dest, fn)):
            shutil.copy2(os.path.join(src, fn), os.path.join(dest, fn))
        if args.align:
            for i, (s, a, b) in enumerate(
                    align_sentences(os.path.join(dest, fn), transcript, args.model)):
                clips.append({"id": "local-%s-%03d" % (_slug(stem), i), "source": "Local",
                              "license": "your own material", "audio": rel,
                              "start": a, "end": b, "text": s})
        else:
            clips.append({"id": "local-%s" % _slug(stem), "source": "Local",
                          "license": "your own material", "audio": rel,
                          "text": transcript})
    if not clips:
        sys.exit("Nothing imported. Expected name%s + name.txt pairs in %s"
                 % (exts[0], src))
    add(clips, clips)


def import_sbc(args):
    """Santa Barbara Corpus: transcripts already carry intonation-unit times.

    Download the .wav/.mp3 files and their .trn transcripts from UCSB, put them
    in one folder, and point --dir at it.
    """
    src = os.path.expanduser(args.dir)
    if not os.path.isdir(src):
        sys.exit("No such folder: %s\nDownload the corpus first:\n  %s"
                 % (src, "https://www.linguistics.ucsb.edu/research/"
                         "santa-barbara-corpus-spoken-american-english"))
    dest = os.path.join(LISTEN_DIR, "sbc")
    os.makedirs(dest, exist_ok=True)
    import shutil
    clips = []
    for fn in sorted(os.listdir(src)):
        if not fn.lower().endswith(".trn"):
            continue
        stem = os.path.splitext(fn)[0]
        audio = None
        for ext in (".wav", ".mp3"):
            for cand in (stem + ext, stem.upper() + ext, stem.lower() + ext):
                if os.path.exists(os.path.join(src, cand)):
                    audio = cand
                    break
            if audio:
                break
        if not audio:
            print("  skip %s — no matching audio" % fn)
            continue
        if not os.path.exists(os.path.join(dest, audio)):
            shutil.copy2(os.path.join(src, audio), os.path.join(dest, audio))
        with open(os.path.join(src, fn), encoding="utf-8", errors="replace") as f:
            units = parse_sbc_trn(f.read())
        for i, (a, b, spk, text) in enumerate(units):
            if len(text.split()) < 4 or (b - a) < 0.8:
                continue
            clips.append({
                "id": "sbc-%s-%04d" % (_slug(stem), i), "source": "Santa Barbara Corpus",
                "license": SBC_LICENSE, "accent": "US",
                "source_url": "https://www.linguistics.ucsb.edu/research/"
                              "santa-barbara-corpus-spoken-american-english",
                "audio": os.path.join("sbc", audio), "start": a, "end": b,
                "text": text, "speaker": spk,
            })
        if args.count and len(clips) >= args.count:
            break
    if not clips:
        sys.exit("No usable intonation units found in %s" % src)
    if args.count:
        clips = clips[:args.count]
    add(clips, clips)


def parse_sbc_trn(raw):
    """Parse a Santa Barbara .trn transcript into (start, end, speaker, text).

    Lines look like:   0.940 2.130   LENORE:     ... So what did you do
    Transcription marks (.. pauses, (H) breath, [overlap], =lengthening) are
    stripped — you're dictating words, not prosodic notation.
    """
    out = []
    last_spk = ""
    for line in (raw or "").splitlines():
        m = re.match(r"\s*([\d.]+)\s+([\d.]+)\s+(?:([A-Z][A-Z0-9_>\s]*?):)?\s*(.*)$",
                     line)
        if not m:
            continue
        try:
            a, b = float(m.group(1)), float(m.group(2))
        except ValueError:
            continue
        # a blank speaker field means the previous speaker is still talking
        spk = (m.group(3) or "").strip() or last_spk
        last_spk = spk
        text = m.group(4) or ""
        text = re.sub(r"\[\d*|\]\d*", " ", text)        # overlap brackets
        text = re.sub(r"\(\s*[A-Z@#]+\s*\)", " ", text)  # (H) (TSK) …
        text = re.sub(r"[<>@%$&#]", " ", text)
        text = re.sub(r"\.\.+", " ", text)               # pause dots
        text = re.sub(r"=+", "", text)                   # lengthening
        text = re.sub(r"--+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" ,-")
        if text and b > a:
            out.append((round(a, 2), round(b, 2), spk, text))
    # merge consecutive units from the same speaker into sentence-ish chunks
    merged = []
    for a, b, spk, text in out:
        if (merged and merged[-1][2] == spk and (a - merged[-1][1]) < 0.6
                and len(merged[-1][3].split()) < 16
                and not re.search(r"[.!?]$", merged[-1][3])):
            pa, _pb, pspk, ptext = merged[-1]
            merged[-1] = (pa, b, pspk, (ptext + " " + text).strip())
        else:
            merged.append((a, b, spk, text))
    return merged


def _open_text(path):
    """Open a Tatoeba export as text, whatever wrapper it arrived in.

    The weekly exports come three different ways and they are NOT the same:
      * eng_sentences.tsv.bz2        plain bz2 around a TSV
      * sentences_with_audio.tar.bz2 a TAR archive, bz2-compressed
      * an already-extracted .csv/.tsv
    Reading the tar one with bz2 alone yields tar headers interleaved with the
    data, which parses as garbage rather than failing loudly — so check for tar
    first.
    """
    base = os.path.basename(path)
    if ".tar." in base or base.endswith(".tar"):
        import tarfile
        import io
        tf = tarfile.open(path, "r:*")
        members = [m for m in tf.getmembers()
                   if m.isfile() and not os.path.basename(m.name).startswith("._")]
        if not members:
            raise RuntimeError("%s contains no files" % base)
        m = max(members, key=lambda x: x.size)      # the data file, not a README
        wrapper = io.TextIOWrapper(tf.extractfile(m), encoding="utf-8",
                                   errors="replace")
        wrapper._tarfile = tf                       # keep the archive alive
        return wrapper
    if path.endswith(".bz2"):
        import bz2
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _find_export(explicit, *patterns):
    """Locate a Tatoeba export by explicit path, or by searching listening/
    recursively — the files are just as likely to be dropped in a subfolder."""
    import glob as _glob
    if explicit:
        p = os.path.expanduser(explicit)
        return p if os.path.exists(p) else None
    for pat in patterns:
        hits = sorted(_glob.glob(os.path.join(LISTEN_DIR, "**", pat),
                                 recursive=True))
        if hits:
            return hits[0]
    return None


def import_tatoeba(args):
    """Tatoeba: one sentence, one recording — no alignment needed.

    Needs two weekly exports from https://tatoeba.org/en/downloads:
      * "Sentences with audio"  — sentenceId, audioId, username, licence, url
      * "Sentences" (English)   — sentenceId, lang, text
    Either the .tar.bz2 or the extracted file works. Audio is fetched per id.
    """
    import urllib.request
    meta = _find_export(args.meta, "sentences_with_audio*")
    sents = _find_export(args.sentences, "eng_sentences*", "sentences.csv",
                         "sentences.tsv", "sentences*")
    if not meta or not sents:
        import glob as _g
        here = [os.path.relpath(p, LISTEN_DIR)
                for p in _g.glob(os.path.join(LISTEN_DIR, "**", "*"), recursive=True)
                if os.path.isfile(p)]
        sys.exit(
            "Missing a Tatoeba export.\n\n"
            "  1. Go to https://tatoeba.org/en/downloads\n"
            "  2. Download 'Sentences with audio'  (all languages)\n"
            "     -> sentences_with_audio.tar.bz2\n"
            "  3. Download 'Sentences' with language = English\n"
            "     -> eng_sentences.tsv.bz2\n"
            "  4. Put both anywhere under %s  (leave them compressed)\n"
            "  5. Re-run this command\n\n"
            "  looking for : sentences_with_audio*  and  eng_sentences*\n"
            "  found       : audio metadata=%s, sentences=%s\n"
            "  files in %s:\n%s"
            % (LISTEN_DIR, meta or "MISSING", sents or "MISSING", LISTEN_DIR,
               ("\n".join("    " + f for f in sorted(here)[:20])
                if here else "    (none — the folder is empty)")))
    print("  metadata : %s" % os.path.basename(meta))
    print("  sentences: %s" % os.path.basename(sents))
    text_by_id = {}
    with _open_text(sents) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] == "eng":
                text_by_id[parts[0]] = parts[2]
    if not text_by_id:
        sys.exit("No English sentences found in %s — is it the English export?"
                 % os.path.basename(sents))
    print("  %d English sentences loaded" % len(text_by_id))
    dest = os.path.join(LISTEN_DIR, "tatoeba")
    os.makedirs(dest, exist_ok=True)
    # Anything already imported is skipped, so re-running ADDS new material
    # instead of re-fetching the same first N rows and deduplicating to nothing.
    have = {c["id"] for c in load_library()}
    candidates = []
    with _open_text(meta) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            sid, aid, _user, lic = parts[0], parts[1], parts[2], parts[3]
            text = text_by_id.get(sid)
            # Tatoeba: an empty licence means the audio may not be reused
            # outside their project, so it can't come here.
            if not text or not lic or lic.strip() in ("", "\\N"):
                continue
            if len(text.split()) < 4:
                continue
            if ("tatoeba-%s" % aid) in have:
                continue
            candidates.append((sid, aid, lic.strip(), text))
    eligible_total = len(candidates) + len(have)
    if not candidates:
        save_library(load_library(), {"Tatoeba": {
            "eligible": eligible_total, "checked": _today()}})
        sys.exit("No new licensed English clips left to import "
                 "(library already holds all %d)." % eligible_total)

    # sample at random rather than taking the first N: the export is ordered by
    # sentence id, so the head of the file is the oldest and most repetitive
    # material. Random gives a broader spread of speakers and vocabulary.
    import random
    if not args.in_order:
        random.shuffle(candidates)
    want = args.count or 100
    print("  %d licensed English clips available, %d already imported — fetching %d"
          % (eligible_total, len(have), min(want, len(candidates))))

    clips, n, failed = [], 0, 0
    for sid, aid, lic, text in candidates:
        if n >= want:
            break
        fn = "%s.mp3" % aid
        path = os.path.join(dest, fn)
        if not os.path.exists(path):
            url = "https://tatoeba.org/audio/download/%s" % aid
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print("  skip audio %s (%s)" % (aid, str(e)[:60]))
                if failed == 20:
                    print("  …many downloads failing — check your connection")
                continue
        clips.append({
            "id": "tatoeba-%s" % aid, "source": "Tatoeba", "license": lic,
            "source_url": "https://tatoeba.org/en/sentences/show/%s" % sid,
            "audio": os.path.join("tatoeba", fn), "text": text,
        })
        n += 1
        if n % 25 == 0:
            print("  fetched %d/%d…" % (n, want))
    if not clips:
        sys.exit("Nothing imported — every download failed. Check your connection.")
    if failed:
        print("  (%d download(s) failed and were skipped)" % failed)
    add(clips, clips, {"Tatoeba": {"eligible": eligible_total, "checked": _today()}})


def import_voa(args):
    """VOA Learning English: public-domain audio + transcript, aligned here.

    Give it a folder of already-downloaded article MP3s with matching .txt
    transcripts (same stem). Fetching is left to you deliberately: scraping
    behaviour belongs under your control, and the licence requires crediting
    learningenglish.voanews.com wherever the clips end up.
    """
    src = os.path.expanduser(args.dir) if args.dir else os.path.join(LISTEN_DIR, "voa-src")
    if not os.path.isdir(src):
        sys.exit(
            "Put VOA article audio + transcripts in %s first.\n\n"
            "  1. Open a story on https://learningenglish.voanews.com/\n"
            "  2. Save the MP3 as  <name>.mp3\n"
            "  3. Save the transcript text as  <name>.txt\n"
            "  4. Re-run:  python3 listening_import.py voa --dir %s\n\n"
            "%s" % (src, src, VOA_CREDIT))
    dest = os.path.join(LISTEN_DIR, "voa")
    os.makedirs(dest, exist_ok=True)
    import shutil
    clips = []
    for fn in sorted(os.listdir(src)):
        if not fn.lower().endswith((".mp3", ".m4a", ".wav")):
            continue
        stem = os.path.splitext(fn)[0]
        txt = os.path.join(src, stem + ".txt")
        if not os.path.exists(txt):
            print("  skip %s — no matching %s.txt" % (fn, stem))
            continue
        with open(txt, encoding="utf-8") as f:
            transcript = f.read().strip()
        if not transcript:
            continue
        if not os.path.exists(os.path.join(dest, fn)):
            shutil.copy2(os.path.join(src, fn), os.path.join(dest, fn))
        print("%s:" % stem)
        try:
            aligned = align_sentences(os.path.join(dest, fn), transcript, args.model)
        except Exception as e:
            print("  alignment failed (%s) — importing as one whole clip" % str(e)[:80])
            aligned = [(transcript, None, None)]
        for i, (s, a, b) in enumerate(aligned):
            c = {"id": "voa-%s-%03d" % (_slug(stem), i), "source": "VOA",
                 "license": VOA_CREDIT, "accent": "US",
                 "source_url": "https://learningenglish.voanews.com/",
                 "audio": os.path.join("voa", fn), "text": s}
            if a is not None:
                c["start"], c["end"] = a, b
            clips.append(c)
        if args.count and len(clips) >= args.count:
            break
    if not clips:
        sys.exit("Nothing imported from %s" % src)
    if args.count:
        clips = clips[:args.count]
    add(clips, clips)


def do_list(_args):
    clips = load_library()
    if not clips:
        print("Library is empty. Import something first — see --help.")
        return
    counts, lics = {}, {}
    for c in clips:
        counts[c.get("source", "?")] = counts.get(c.get("source", "?"), 0) + 1
        lics.setdefault(c.get("source", "?"), c.get("license", ""))
    meta = load_meta()
    print("%d clip(s) in %s\n" % (len(clips), LIB_PATH))
    print("  %-24s %6s %10s   %s" % ("source", "have", "available", "licence"))
    for s in sorted(counts):
        avail = (meta.get(s) or {}).get("eligible")
        print("  %-24s %6d %10s   %s"
              % (s, counts[s], avail if avail else "?", lics.get(s, "")))
    tot = sum((meta.get(s) or {}).get("eligible") or 0 for s in counts)
    if tot:
        print("\n  %d of %d available clips imported (%d not yet fetched)"
              % (len(clips), tot, max(0, tot - len(clips))))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_dir=False):
        p.add_argument("--dir", required=need_dir, help="folder to import from")
        p.add_argument("--count", type=int, default=0, help="stop after N clips")
        p.add_argument("--model", default="medium",
                       help="Whisper model used for alignment (default: medium)")

    p = sub.add_parser("voa", help="VOA Learning English (public domain)")
    common(p)
    p.set_defaults(func=import_voa)

    p = sub.add_parser("sbc", help="Santa Barbara Corpus (CC BY-ND, real conversation)")
    common(p, need_dir=True)
    p.set_defaults(func=import_sbc)

    p = sub.add_parser("tatoeba", help="Tatoeba sentences with native audio")
    common(p)
    p.add_argument("--meta", help="path to sentences_with_audio (.tar.bz2 or .csv)")
    p.add_argument("--sentences", help="path to the English sentences export")
    p.add_argument("--in-order", action="store_true",
                   help="take the first N rows instead of a random sample")
    p.set_defaults(func=import_tatoeba)

    p = sub.add_parser("local", help="your own audio + transcripts")
    common(p, need_dir=True)
    p.add_argument("--align", action="store_true",
                   help="split long recordings into sentences with Whisper")
    p.set_defaults(func=import_local)

    p = sub.add_parser("list", help="what's in the library now")
    p.set_defaults(func=do_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
