#!/usr/bin/env python3
"""Backfill the banded pause metrics onto every already-analysed recording.

`pause_ratio_pct` counts every silence identically — a 120 ms stop closure, a
breath, and a four-second "what do I say next" all land in the same number, so
it measures neither articulation nor fluency. `analyze_prosody` now also splits
the silence by duration (see `PAUSE_BANDS`). This re-runs it over the archive so
the split exists for past recordings too, and refreshes history.json to match.

Nothing here re-contacts Azure or any LLM: it only re-reads local audio.

    python backfill_pause_bands.py            # dry run — prints what would change
    python backfill_pause_bands.py --run      # write result.json + history.json
"""
import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import english_coach as ec

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VideoAudioFiles")
AUDIO_EXT = (".webm", ".m4a", ".wav", ".mp3", ".mp4", ".ogg", ".flac")
# the fields analyze_prosody gained; used to spot already-done recordings
NEW_KEYS = ("hesitation_pct", "pause_ratio_adj_pct", "planning_pct")


def find_audio(result_path):
    """The recording that produced this result.json lives beside it."""
    d = os.path.dirname(result_path)
    stem = os.path.basename(result_path)[:-len(".result.json")]
    for ext in AUDIO_EXT:                      # same stem first
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    cands = [os.path.join(d, f) for f in sorted(os.listdir(d))
             if f.lower().endswith(AUDIO_EXT)]
    return cands[0] if len(cands) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually write files")
    ap.add_argument("--force", action="store_true",
                    help="recompute even where the banded fields already exist")
    args = ap.parse_args()

    results = sorted(glob.glob(os.path.join(BASE, "**", "*.result.json"),
                               recursive=True))
    print("%d result files under %s\n" % (len(results), BASE))

    done, skipped, failed, updates = 0, 0, [], {}
    for rp in results:
        try:
            item = json.load(open(rp, encoding="utf-8"))
        except Exception as e:
            failed.append((rp, "unreadable: %s" % e)); continue

        pm = item.get("prosody_metrics") or {}
        if all(k in pm for k in NEW_KEYS) and not args.force:
            skipped += 1; continue

        audio = find_audio(rp)
        if not audio:
            failed.append((rp, "no audio file found")); continue

        try:
            new = ec.analyze_prosody(audio)
        except Exception as e:
            failed.append((rp, "analyze failed: %s" % e)); continue
        if not new:
            failed.append((rp, "analyze returned nothing (too short/unreadable?)"))
            continue

        title = item.get("title") or os.path.basename(rp)
        print("  %-44s raw %3d%%  ->  adj %4.1f%%  hes %4.1f%%  think %4.1f%% (%d)"
              % (title[:44], new["pause_ratio_pct"], new["pause_ratio_adj_pct"],
                 new["hesitation_pct"], new["planning_pct"], new["pause_planning_n"]))

        # keep any keys the old dict had that the analyzer no longer returns
        merged = dict(pm); merged.update(new)
        item["prosody_metrics"] = merged
        updates[rp] = item
        done += 1

    print("\n%d to update, %d already banded, %d failed" % (done, skipped, len(failed)))
    for rp, why in failed:
        print("   ! %s — %s" % (os.path.relpath(rp, BASE), why))

    if not args.run:
        print("\nDry run. Re-run with --run to write.")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for rp, item in updates.items():
        shutil.copy2(rp, rp + ".bak-" + stamp)
        ec._save_result(item, rp)
    print("\nWrote %d result files (each backed up as *.bak-%s)." % (len(updates), stamp))

    # --- history.json: refresh the banded fields on matching sessions ---
    hp = os.path.join(BASE, "history.json")
    if not os.path.exists(hp):
        print("No history.json — nothing further to do."); return
    hist = json.load(open(hp, encoding="utf-8"))
    shutil.copy2(hp, hp + ".bak-" + stamp)

    by_key, by_title, ambiguous = {}, {}, set()
    for rp in glob.glob(os.path.join(BASE, "**", "*.result.json"), recursive=True):
        try:
            d = json.load(open(rp, encoding="utf-8"))
        except Exception:
            continue
        pm = d.get("prosody_metrics") or {}
        if not any(k in pm for k in NEW_KEYS):
            continue
        by_key[(d.get("date"), d.get("title"))] = pm
        t = d.get("title")
        if t in by_title:
            ambiguous.add(t)                   # same title, two recordings
        by_title[t] = pm

    touched = 0
    for rec in hist:
        # a session logged under a date that drifted from its result.json (a
        # late-night recording rolling past midnight does this) still matches
        # on title, as long as that title belongs to exactly one recording
        pm = by_key.get((rec.get("date"), rec.get("title")))
        if not pm and rec.get("title") not in ambiguous:
            pm = by_title.get(rec.get("title"))
        if not pm:
            continue
        before = {k: rec.get(k) for k in NEW_KEYS}
        for k in NEW_KEYS:
            if pm.get(k) is not None:
                rec[k] = pm[k]
        if before != {k: rec.get(k) for k in NEW_KEYS}:
            touched += 1

    tmp = hp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=1)
    os.replace(tmp, hp)
    print("Updated %d/%d history entries (backup: history.json.bak-%s)."
          % (touched, len(hist), stamp))


if __name__ == "__main__":
    main()
