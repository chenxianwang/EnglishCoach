#!/usr/bin/env python3
"""Tests for the measured weak-word list under Pronunciation patterns.

    python3 test_weakest_words.py

The list exists because the LLM half of that section is honestly empty on a
clean recording — it reads a transcript, so it can only flag a sound the
recognizer visibly misheard. Azure meanwhile scored every word in the same
recording. These checks pin down the two decisions that make the measured half
worth reading rather than just long:

  * repeats are grouped and ranked by their MEAN, so one bad alignment cannot
    push a word above one that is weak every single time it is said;
  * scores that describe nothing spoken (omissions, insertions) are dropped.

Plus the per-phoneme detail, which is the difference between "rest scored 23"
and "the final /t/ scored 25" — only the second is a drill.
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


def w(word, acc, error="", phones=None, pacc=None):
    return {"word": word, "accuracy": acc, "error": error,
            "phones": phones or [], "pacc": pacc or []}


def words(rows):
    return [r["word"] for r in rows]


# -- nothing to show -------------------------------------------------------
check("no Azure result at all", ec.weakest_words(None), [])
check("Azure ran but scored nothing", ec.weakest_words({"words": []}), [])

# -- grouping and ranking ---------------------------------------------------
rows = ec.weakest_words({"words": [
    w("played", 0), w("played", 96), w("played", 98), w("played", 99),
    w("during", 30), w("during", 34),
    w("safe", 90),
]})
check("repeats collapse into one row", words(rows), ["during", "played", "safe"])
check("a word weak every time outranks one bad alignment",
      words(rows).index("during") < words(rows).index("played"), True)
check("the score shown is the mean", rows[0]["mean"], 32)
check("the worst single attempt is kept too", rows[1]["worst"], 0)
check("how often it was said is kept", rows[1]["n"], 4)
check("and how many of those were under the flag threshold", rows[1]["bad"], 1)

# -- what is not a pronunciation score --------------------------------------
rows = ec.weakest_words({"words": [
    w("skipped", 0, "Omission"),      # in the script, never spoken
    w("uh", 0, "Insertion"),          # spoken, not in the script
    w("target", 40, "Mispronunciation"),
]})
check("an omitted word is not a weak word", words(rows), ["target"])

# -- the token ---------------------------------------------------------------
rows = ec.weakest_words({"words": [w("Game,", 10), w("game", 20)]})
check("case and trailing punctuation fold together", words(rows), ["game"])
check("...and the folded row counts both", rows[0]["n"], 2)
check("an empty token is dropped", ec.weakest_words({"words": [w("...", 0)]}), [])

# -- the limit ---------------------------------------------------------------
many = {"words": [w("w%d" % i, i) for i in range(40)]}
check("the list is capped", len(ec.weakest_words(many)), ec.WEAKEST_SHOWN)
check("...and it is the WEAKEST that survive the cap",
      ec.weakest_words(many)[0]["mean"], 0)
check("a caller may ask for fewer", len(ec.weakest_words(many, limit=3)), 3)

# -- the weakest sound -------------------------------------------------------
rest = ec.weakest_words({"words": [
    w("rest", 23, "Mispronunciation", ["r", "eh", "s", "t"], [71, 63, 69, 25])]})[0]
check("the lowest-scoring phoneme is the one named", rest["sound"]["ipa"], "t")
check("its own score comes with it", rest["sound"]["score"], 25)
# Word-final is the position that matters most here: a dropped final stop is
# the classic Mandarin-L1 error, and it is a different fix from a weak onset.
check("and where in the word it sat", rest["sound"]["pos"], "final")

first = ec.weakest_words({"words": [
    w("them", 0, "Mispronunciation", ["dh", "eh", "m"], [14, 80, 90])]})[0]
check("an initial consonant is reported as initial", first["sound"]["pos"], "initial")
check("...and mapped to IPA, not left as ARPAbet", first["sound"]["ipa"], "ð")

mid = ec.weakest_words({"words": [
    w("turn", 0, "Mispronunciation", ["t", "er", "n"], [80, 4, 70])]})[0]
check("anything else is middle", mid["sound"]["pos"], "middle")

# Azure spells schwa `ax`; folding it onto AH (which the sound STATISTICS must
# do, so one vowel isn't counted as two) would print it as /ʌ/ — a different
# vowel from the one that was actually graded.
schwa = ec.weakest_words({"words": [
    w("the", 60, "", ["dh", "ax"], [90, 65])]})[0]
check("Azure's schwa prints as a schwa", schwa["sound"]["ipa"], "ə")

# -1 is Azure's "not scored", and averaging it in would invent a failure.
unscored = ec.weakest_words({"words": [
    w("hello", 50, "", ["hh", "eh", "l", "ow"], [-1, -1, -1, -1])]})[0]
check("no per-phoneme scores means no sound is named", unscored["sound"], {})
check("a truncated phoneme list is not lined up by guesswork",
      ec.weakest_words({"words": [w("no", 50, "", ["n", "ow"], [40])]})[0]["sound"], {})

# -- Azure grading against the wrong reading ---------------------------------
# The two alphabets disagree on exactly one symbol: Azure writes /h/ as `h`,
# CMUdict as `hh`. Unfolded, every h-initial word claimed its own dictionary
# entry was wrong, and the warning that means "do not drill this" appeared on
# ordinary words like "heavy".
heavy = ec.weakest_words({"words": [
    w("heavy", 0, "Mispronunciation", ["h", "eh", "v", "iy"], [36, 100, 100, 32])]})[0]
check("h and hh are the same sound", heavy["suspect"], False)
check("a genuinely different reading is still flagged",
      ec.weakest_words({"words": [
          w("tied", 64, "", ["t", "iy", "d"], [90, 44, 90])]})[0]["suspect"], True)
check("an unknown spelling is never guessed at",
      ec.weakest_words({"words": [
          w("joseki", 17, "", ["jh", "ow"], [50, 50])]})[0]["suspect"], False)

# -- the rendered section ----------------------------------------------------
check("no section without Azure scores", ec._weakest_words_html({}), "")
html = ec._weakest_words_html({"azure": {"words": [
    w("rest", 23, "Mispronunciation", ["r", "eh", "s", "t"], [71, 63, 69, 25]),
    w("20", 0, "Mispronunciation"),
    w("tied", 30, "", ["t", "iy", "d"], [90, 44, 90]),
]}})
check("the section renders", "Weakest words" in html, True)
check("a number is listed as a score", ">20<" in html, True)
# ...but not offered as a practice card: you cannot look up "20" and say it
# again, and "tied" carries a warning that its score isn't about your mouth.
check("only real, trustworthy words go to Practice",
      'data-words="rest"' in html, True)

print("\n%d checks, %d failed" % (len(PASS) + len(FAIL), len(FAIL)))
sys.exit(1 if FAIL else 0)
