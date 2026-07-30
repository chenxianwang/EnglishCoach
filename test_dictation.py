#!/usr/bin/env python3
"""Tests for the dictation grader and the listening-clip library.

    python3 test_dictation.py
"""

import sys

import english_coach as ec

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))


def grade(ref, typed):
    return ec.dictation_check(ref, typed)


# --- scoring -----------------------------------------------------------------
r = grade("The committee said it would review the decision next week.",
          "The committee said it would review the decision next week.")
check("exact match scores 100", r["score"] == 100, str(r["score"]))
check("exact match is perfect", r["perfect"] is True)
check("exact match has no misses", r["missed"] == 0)

r = grade("The committee said it would review the decision next week.",
          "the committee said it would review the decision next week")
check("case and punctuation ignored", r["score"] == 100, str(r["score"]))

r = grade("The committee said it would review the decision.",
          "The community said it would review the decision.")
check("one misheard word is caught", r["missed"] == 1, str(r["missed"]))
check("misheard word reported as replace",
      any(o["op"] == "replace" and o["ref"] == ["committee"] and o["hyp"] == ["community"]
          for o in r["ops"]),
      str(r["ops"]))

r = grade("I would have gone if I had known.", "I would gone if I had known.")
check("dropped word reported as delete",
      any(o["op"] == "delete" and o["ref"] == ["have"] for o in r["ops"]),
      str(r["ops"]))

r = grade("I would gone if I had known.", "I would have gone if I had known.")
check("extra word reported as insert",
      any(o["op"] == "insert" and o["hyp"] == ["have"] for o in r["ops"]),
      str(r["ops"]))

r = grade("She has been waiting.", "")
check("empty answer scores 0", r["score"] == 0, str(r["score"]))
check("empty answer is not perfect", r["perfect"] is False)

r = grade("", "anything")
check("empty reference doesn't divide by zero", r["score"] == 0)

# --- the mishearings this is actually for -----------------------------------
# connected speech: what a Mandarin-L1 listener typically loses
CASES = [
    ("What did you do about it?", "What do you do about it?", 1),
    ("I should have told him.", "I shoulda told him.", 2),
    ("There's a lot of them.", "There's a lot of them.", 0),
    ("He wanted to come.", "He want to come.", 1),
    ("We're going to need more time.", "We're gonna need more time.", 2),
]
for ref, typed, expect_missed in CASES:
    r = grade(ref, typed)
    check("mishearing: %r" % typed, r["missed"] == expect_missed,
          "expected %d missed, got %d (%s)" % (expect_missed, r["missed"], r["ops"]))

# --- ops must reconstruct the reference exactly ------------------------------
for ref, typed, _ in CASES:
    r = grade(ref, typed)
    rebuilt = []
    for o in r["ops"]:
        rebuilt += o["ref"]
    check("ops reconstruct the reference: %r" % ref[:28],
          rebuilt == ec._norm(ref), "%s != %s" % (rebuilt, ec._norm(ref)))

# --- score is monotonic in accuracy -----------------------------------------
ref = "The committee said it would review the decision next week."
worse = grade(ref, "The community said it could review a decision next week.")["score"]
better = grade(ref, "The committee said it would review the decision next week.")["score"]
check("more accurate answers score higher", better > worse,
      "%d vs %d" % (better, worse))

# --- clip library ------------------------------------------------------------
lib = ec.load_listening_library()
check("missing library returns an empty list, no crash", isinstance(lib, list))

SAMPLE = [
    {"id": "a1", "source": "VOA", "text": "One.", "audio": "voa/a.mp3",
     "start": 0, "end": 2, "license": "Public domain"},
    {"id": "b1", "source": "Tatoeba", "text": "Two.", "audio": "tat/b.mp3",
     "license": "CC BY 2.0 FR"},
    {"id": "bad", "source": "X"},                       # no text/audio -> dropped
]
clean = ec._clean_clips(SAMPLE)
check("invalid clips are dropped", len(clean) == 2, str(len(clean)))
check("valid clips keep their source", {c["source"] for c in clean} == {"VOA", "Tatoeba"})
check("clip ids survive", {c["id"] for c in clean} == {"a1", "b1"})

# --- vocabulary -------------------------------------------------------------
check("tokenizer lowercases and keeps contractions",
      ec._tokenize("Don't STOP, it's fine.") == ["don't", "stop", "it's", "fine"],
      str(ec._tokenize("Don't STOP, it's fine.")))
check("tokenizer drops digits and CJK",
      ec._tokenize("I have 3 cats 我有猫") == ["i", "have", "cats"],
      str(ec._tokenize("I have 3 cats 我有猫")))
check("tokenizer keeps single-letter a and i only",
      ec._tokenize("a b c I") == ["a", "i"], str(ec._tokenize("a b c I")))
check("tokenizer strips stray apostrophes",
      ec._tokenize("'quoted' word") == ["quoted", "word"],
      str(ec._tokenize("'quoted' word")))

FAKE = [
    {"date": "2026-01-01", "title": "one", "words": ec._tokenize("the cat sat on the mat")},
    {"date": "2026-01-02", "title": "two", "words": ec._tokenize("the dog sat down")},
]
v = ec.speaking_vocabulary(FAKE)
check("token count sums both sessions", v["tokens"] == 10, str(v["tokens"]))
check("distinct words counted once across sessions",
      v["types"] == 7, "%s -> %s" % (v["types"], sorted(v["counts"])))
check("repeated word counted", v["counts"]["the"] == 3, str(v["counts"]["the"]))
check("hapax counted", v["hapax"] == 5, str(v["hapax"]))
check("new words per session", [s["new"] for s in v["sessions"]] == [5, 2],
      str([s["new"] for s in v["sessions"]]))
check("cumulative distinct rises",
      [s["cumulative"] for s in v["sessions"]] == [5, 7],
      str([s["cumulative"] for s in v["sessions"]]))
check("first_seen records the earliest session",
      v["first_seen"]["the"] == "2026-01-01", v["first_seen"]["the"])
check("empty corpus doesn't divide by zero",
      ec.speaking_vocabulary([])["ttr"] == 0.0)

for name, detail in PASS:
    print("  ok    %s" % name)
for name, detail in FAIL:
    print("  FAIL  %s\n          %s" % (name, detail))
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
