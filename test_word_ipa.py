#!/usr/bin/env python3
"""Tests for the CMUdict-backed IPA shown beside a practice word.

    python3 test_word_ipa.py

This is display-only — Azure grades the word itself, never this string — but a
learner reads it as instruction, so a wrong symbol teaches a wrong sound.

The interesting cases are all the same shape: ARPAbet reuses one symbol for a
strong vowel and its weak counterpart, marking the difference only with a
trailing stress digit. IPA gives them separate symbols. Lose the digit and the
app tells you the second syllable of "letter" is the vowel in "bird".
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


if not ec._cmu() and not ec._CMU:
    print("cmudict not installed — nothing to test")
    raise SystemExit(0)

# -- the weak/strong vowel pairs --------------------------------------------
# ER is the NURSE vowel when stressed and r-coloured schwa when not. Both can
# occur in one word, which is the case that catches a table-only mapping.
check("stressed ER is the NURSE vowel", ec.word_ipa("bird"), "/bɜrd/")
check("unstressed ER is r-coloured schwa", ec.word_ipa("letter"), "/ˈletər/")
check("a word with both keeps them apart", ec.word_ipa("murder"), "/ˈmɜrdər/")
check("unstressed ER after a stressed vowel", ec.word_ipa("father"), "/ˈfɑðər/")

# AH is the same story and was already handled; it is checked so a refactor of
# the weak-form table cannot quietly drop one of the two entries.
check("stressed AH is the STRUT vowel", ec.word_ipa("cup"), "/kʌp/")
check("unstressed AH is schwa", ec.word_ipa("about"), "/əˈbaʊt/")
check("a word with both keeps them apart", ec.word_ipa("other"), "/ˈʌðər/")

# -- stress marking ----------------------------------------------------------
check("monosyllables carry no stress mark", ec.word_ipa("cat"), "/kæt/")
check("the mark sits before the whole onset", ec.word_ipa("solid"), "/ˈsɑləd/")
check("secondary stress is marked too", "ˌ" in ec.word_ipa("understand"), True)

# -- the honest failure mode -------------------------------------------------
check("an unknown word returns empty, not a guess",
      ec.word_ipa("zzzqqx"), "")
check("punctuation and case do not defeat the lookup",
      ec.word_ipa("Solid!"), ec.word_ipa("solid"))

print("\nall %d checks passed" % len(PASS) if not FAIL
      else "\n%d passed, %d FAILED" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
