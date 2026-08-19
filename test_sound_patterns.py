#!/usr/bin/env python3
"""Tests for the measured cross-recording pattern list.

    python3 test_sound_patterns.py

The inferred half of the pronunciation section reads a transcript, so it can
only name a sound when the recognizer visibly misheard a word. On a clean
recording it correctly finds nothing — and said so on ten recordings in a row,
which reads as "you have no patterns" when it means "text cannot show me one".

A pattern is not in the text. It is in the per-phoneme scores Azure already
returned, and it only exists ACROSS recordings. These checks pin down the
decisions that make this list a finding rather than a stereotype:

  * the comparison is the speaker's own speech — the same sound elsewhere in a
    word, or their own average across every phoneme — never a textbook list of
    what a Mandarin speaker is supposed to get wrong;
  * no comparison, no positional claim: a sound that only ever appears at the
    end of a word cannot be shown to be worse there;
  * a pattern must recur in several recordings, or it is one bad morning;
  * a sound that fails by position is not also reported as failing generally,
    because the positional row already says where to aim;
  * example words are ranked by mean and must be sayable, so the row doubles
    as a set of drill cards.
"""

import sys

import english_coach as ec

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append(name)
        print("  FAIL  %s\n        got  %r\n        want %r" % (name, got, want))


def w(word, phones, pacc):
    return {"word": word, "accuracy": int(sum(pacc) / len(pacc)),
            "error": "", "phones": phones, "pacc": pacc}


def rec(*words):
    return {"azure": {"words": list(words)}}


def start(final, mid=95, word="start"):
    """"start" = S T AA R T — one /t/ inside the word, one at the end of it.

    Both halves of the comparison come from the same fixture, which is the
    whole point: the claim is about position, so the sound has to appear in
    more than one position.
    """
    return w(word, ["S", "T", "AA", "R", "T"], [95, mid, 95, 95, final])


def block(n, **kw):
    return [start(**kw) for _ in range(n)]


def keys(rows):
    return [(r["ipa"], r["pos"]) for r in rows]


def by_key(rows):
    return {(r["ipa"], r["pos"]): r for r in rows}


# --- a pattern needs more than one session ---------------------------------
one = rec(*block(30, final=30))
check("one recording cannot show a repeatable pattern",
      ec.sound_patterns([one]), [])

two = [rec(*block(15, final=30)) for _ in range(2)]
check("two recordings is still not enough", ec.sound_patterns(two), [])

three = [rec(*block(10, final=30)) for _ in range(3)]
rows = ec.sound_patterns(three)
check("the same weakness in three recordings is a pattern",
      keys(rows), [("t", "final")])

# --- the comparison is the speaker, not a textbook -------------------------
r0 = rows[0]
check("the gap is measured against the same sound elsewhere",
      (round(r0["mean"]), round(r0["compare"])), (30, 95))
check("instances are counted across every recording", r0["n"], 30)
check("so are the recordings", r0["recordings"], 3)

# Same weak final /t/, but this speaker's /t/ is just as weak inside a word.
# There is no positional finding to make, and inventing one would send them to
# drill word endings when the sound itself is what needs work.
flat = [rec(*block(12, final=40, mid=40)) for _ in range(3)]
fr = by_key(ec.sound_patterns(flat))
check("a sound weak in every position is not a positional pattern",
      ("t", "final") in fr, False)
check("...it is reported as a whole-sound weakness instead", ("t", "") in fr, True)
check("...and only once", len([k for k in fr if k[0] == "t"]), 1)

# A sound the speaker only ever produces at the end of a word has nothing to be
# compared against, so no positional claim is available at any gap.
only_final = [rec(*[w("hit", ["HH", "IH", "T"], [95, 95, 20]) for _ in range(12)])
              for _ in range(3)]
check("no instances elsewhere means no positional claim",
      ("t", "final") in by_key(ec.sound_patterns(only_final)), False)

# --- gap and sample-size floors --------------------------------------------
near = [rec(*block(12, final=93)) for _ in range(3)]
check("a two-point gap is noise, not a pattern", ec.sound_patterns(near), [])

thin = [rec(*block(3, final=20)) for _ in range(3)]
check("nine instances is too few to average", ec.sound_patterns(thin), [])

# --- ranking ----------------------------------------------------------------
pop = [w("pop", ["P", "AA", "P"], [95, 95, 70]) for _ in range(10)]
mixed = [rec(*(block(10, final=30) + pop)) for _ in range(3)]
mr = ec.sound_patterns(mixed)
check("the widest gap is listed first", mr[0]["ipa"], "t")
check("both sounds are reported", sorted(r["ipa"] for r in mr), ["p", "t"])
check("the smaller gap is still measured against its own sound",
      (round(mr[1]["mean"]), round(mr[1]["compare"])), (70, 95))

# --- example words ----------------------------------------------------------
def inst(word, score, n):
    return [{"word": word, "score": float(score), "arpa": "T",
             "pos": "final", "rec": 0} for _ in range(n)]

check("the worst word is offered first",
      ec._pattern_words(inst("student", 80, 5) + inst("rest", 20, 5)),
      ["rest", "student"])
check("numbers are not offered as drill cards",
      ec._pattern_words(inst("5000", 20, 5) + inst("rest", 40, 5)), ["rest"])
check("the list is capped",
      len(ec._pattern_words(sum([inst("word" + chr(97 + i), i, 2)
                                 for i in range(10)], []))), 6)

# The real corpus must survive its own filter: a word list that came back empty
# would render a pattern row with nothing to practise.
wr = by_key(ec.sound_patterns(mixed))[("t", "final")]["words"]
check("a real bucket yields drillable words", wr, ["start"])

# --- rendering --------------------------------------------------------------
check("nothing to say renders nothing", ec._sound_patterns_html([one]), "")
html = ec._sound_patterns_html(three)
check("the block names the sound", "/t/" in html, True)
check("...and where it fails", "at the end of a word" in html, True)
check("...and how many recordings back it up", "3 recordings" in html, True)
check("...and offers the words to Practice", "addDrillWords" in html, True)

# An apostrophe must survive into the attribute rather than closing it — the
# same escaping bug that made "don't" undeletable in Practice single word.
apos = [rec(*[w("isn't", ["IH", "Z", "AH", "N", "T"], [95, 95, 95, 95, 20])
              for _ in range(12)] + block(12, final=95)) for _ in range(3)]
h2 = ec._sound_patterns_html(apos)
check("an apostrophe is escaped, not left to close the attribute",
      "isn&#x27;t" in h2 and "isn't" not in h2, True)

# --- the empty half now points at the full one ------------------------------
clean = {"azure": {"words": [w("hit", ["HH", "IH", "T"], [95, 95, 95])],
                   "accuracy": 92, "fluency": 90, "completeness": 100,
                   "pron_score": 91, "prosody": 88},
         "pronunciation_patterns": []}
body = ec._report_body(clean, ec._sound_patterns_html(three))
check("a clean transcript no longer reads as 'you have no patterns'",
      "No repeatable pattern was visible" in body, False)
check("...it hands the reader to the measured list",
      "the measured list below is the real answer" in body, True)
check("with nothing measured, the old honest wording stands",
      "No repeatable pattern was visible" in ec._report_body(clean, ""), True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
