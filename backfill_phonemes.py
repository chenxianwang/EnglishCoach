#!/usr/bin/env python3
"""Add per-sound accuracy scores to recordings analysed before they were kept.

Why this exists
---------------
The Speaking error log counts how often each English sound goes wrong. Doing
that honestly needs a score for every *phoneme*, and until recently the
analyser threw those away — it kept Azure's verdict for the whole word. So the
report can say "you said /r/ 1286 times and 135 of those words were flagged",
but not which sound in the word was actually at fault.

Azure returns the phoneme scores; nothing else has to change. Re-assessing the
audio you still have on disk upgrades the old recordings from *attributed* to
*exact*, and the error log switches over on its own once the data is there.

What it touches
---------------
Only two keys, and only inside `azure.words`: `phones` and `pacc`. Scores,
grammar findings, transcripts, polish and history are not rewritten — this is
deliberately not a re-analysis. Recordings that already have `pacc` are
skipped, so it is safe to re-run and safe to interrupt.

It costs Azure quota, because it sends the audio again. That is why it does
nothing at all without `--run`.

    python3 backfill_phonemes.py                # what it would do, and the cost
    python3 backfill_phonemes.py --run --limit 2  # try a couple first
    python3 backfill_phonemes.py --run          # the rest
"""

import argparse
import difflib
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import english_coach as ec                                    # noqa: E402

AUDIO_EXT = (".webm", ".wav", ".m4a", ".mp3", ".mp4", ".ogg", ".aac", ".flac")
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".english_coach.json")


def load_credentials():
    """Azure key/region from the environment, else the app's own config file."""
    if os.environ.get("AZURE_SPEECH_KEY") and os.environ.get("AZURE_SPEECH_REGION"):
        return True
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    key, region = cfg.get("azure_key"), cfg.get("azure_region")
    if not (key and region):
        return False
    os.environ["AZURE_SPEECH_KEY"] = key
    os.environ["AZURE_SPEECH_REGION"] = region
    return True


def audio_for(stem):
    for ext in AUDIO_EXT:
        p = stem + ext
        if os.path.exists(p):
            return p
    return None


def reference_for(stem, words):
    """The script Azure was graded against the first time.

    The saved transcript is the real thing. Falling back to re-joining the word
    list is not as good — but it is the same reference Azure itself expanded,
    so the word sequence still lines up, which is all the merge needs.
    """
    p = stem + ".txt"
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                t = f.read().strip()
            if t:
                return t
        except OSError:
            pass
    return " ".join(w.get("word", "") for w in words).strip()


def merge_phonemes(old_words, new_words):
    """Copy phones/pacc onto the stored words. Returns how many were filled.

    Index-for-index would be the obvious move, and usually right — same audio,
    same reference. But "usually" silently mis-attributes every sound after the
    first hiccup, so the two word sequences are aligned properly and only
    genuinely matching runs are copied.
    """
    a = [(w.get("word") or "").lower() for w in old_words]
    b = [(w.get("word") or "").lower() for w in new_words]
    filled = 0
    for i, j, n in difflib.SequenceMatcher(a=a, b=b, autojunk=False)\
                          .get_matching_blocks():
        for k in range(n):
            src, dst = new_words[j + k], old_words[i + k]
            if not src.get("pacc"):
                continue
            dst["phones"] = src.get("phones") or []
            dst["pacc"] = src.get("pacc") or []
            filled += 1
    return filled


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true",
                    help="actually re-assess (without this, only reports)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N recordings")
    ap.add_argument("--library", default=ec.library_dir())
    args = ap.parse_args(argv)

    todo, done, noaudio = [], 0, []
    for path in sorted(glob.glob(os.path.join(args.library, "**", "*.result.json"),
                                 recursive=True)):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        words = ((d.get("azure") or {}).get("words")) or []
        if not words:
            continue
        if any(w.get("pacc") for w in words):
            done += 1
            continue
        stem = path[:-len(".result.json")]
        audio = audio_for(stem)
        if not audio:
            noaudio.append(os.path.basename(stem))
            continue
        todo.append((path, stem, audio, d, words))

    mins = sum(x[3].get("duration_sec") or 0 for x in todo) / 60.0
    print("already exact : %d recordings" % done)
    print("to re-assess  : %d recordings, %.1f min of audio" % (len(todo), mins))
    if noaudio:
        print("no audio found: %d (%s)" % (len(noaudio), ", ".join(noaudio[:3])))
    if not todo:
        print("\nNothing to do.")
        return 0
    if not args.run:
        print("\nThis sends %.1f min of audio to Azure and spends that much quota." % mins)
        print("Re-run with --run to do it (or --run --limit 2 to try a couple first).")
        return 0
    if not load_credentials():
        print("\nNo Azure credentials. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION,")
        print("or save them in the app's Settings panel first.")
        return 1

    if args.limit:
        todo = todo[:args.limit]
    ok = failed = 0
    for n, (path, stem, audio, d, words) in enumerate(todo, 1):
        name = os.path.basename(stem)
        print("[%d/%d] %s" % (n, len(todo), name), flush=True)
        try:
            # Prosody off: it is the one part of pronunciation assessment Azure
            # bills as an add-on (baseline $1.00/h + $0.30/h in eastus), and
            # merge_phonemes() copies nothing but phones/pacc — the prosody
            # score would be computed, charged for, and dropped on the floor.
            # The stored recording keeps the prosody it was first scored with.
            res = ec.azure_pronunciation(
                audio, reference_text=reference_for(stem, words),
                progress=lambda m: None, enable_prosody=False,
                usage_kind="backfill")
        except Exception as e:                        # noqa: BLE001
            print("        failed: %s" % str(e)[:140])
            failed += 1
            continue
        filled = merge_phonemes(words, res.get("words") or [])
        if not filled:
            print("        no phoneme scores returned — left untouched")
            failed += 1
            continue
        # Write via a temp file: a half-written result.json would lose the
        # analysis this is only meant to annotate.
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        ok += 1
        print("        %d/%d words now have per-sound scores" % (filled, len(words)))

    print("\nupgraded %d, failed %d" % (ok, failed))
    if ok:
        print("Rebuilding the dashboard…")
        try:
            with open(os.path.join(HERE, "dashboard.html"), "w", encoding="utf-8") as h:
                h.write(ec.build_dashboard_for_dir(args.library))
            print("Done. The Speaking error log will now say 'exact'.")
        except Exception as e:                        # noqa: BLE001
            print("Dashboard rebuild failed (%s) — it will refresh on your next "
                  "analysis anyway." % str(e)[:100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
