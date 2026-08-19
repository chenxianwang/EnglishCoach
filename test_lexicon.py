#!/usr/bin/env python3
"""Tests for the lemma and word-family folding behind the vocabulary counts.

    python3 test_lexicon.py

Folding is where a vocabulary report can lie to you in both directions. Fold
too little and "go/goes/going/went" inflates the count fourfold; fold too much
and "listen" disappears into "list", or "suggest" collapses onto "sugg" — a
dictionary entry nobody has ever said out loud.

Both dictionaries available offline are dirty in their own way: cmudict is full
of surnames (`calle`, `mak`, `libert`) and /usr/share/dict/words is full of
archaisms (`vocabular`, `sente`). Every case below is one that a single pass
over either list gets wrong, which is why the module keeps two lists and picks
strict-first.
"""

import sys

import lexicon

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append(name)
        print("  FAIL  %s\n        got  %r\n        want %r" % (name, got, want))


lx = lexicon.shared()

# -- inflection: the ordinary cases -----------------------------------------
check("plural -s", lx.lemma("games"), "game")
check("-ies to -y", lx.lemma("studies"), "study")
check("-es after a sibilant", lx.lemma("watches"), "watch")
check("dropped -e before -ed", lx.lemma("liked"), "like")
check("doubled consonant before -ed", lx.lemma("stopped"), "stop")
check("doubled consonant before -ing", lx.lemma("running"), "run")
check("restored -e before -ing", lx.lemma("making"), "make")
check("irregular past", lx.lemma("went"), "go")
check("irregular participle", lx.lemma("spoken"), "speak")
check("suppletive comparative", lx.lemma("better"), "good")
check("clitic folds to its head", lx.lemma("doesn't"), "do")

# -- inflection: the traps --------------------------------------------------
# "called" reads as call+ed or calle+d, and cmudict has the surname "calle".
check("a surname does not win over the verb", lx.lemma("called"), "call")
check("nor does one before -ing", lx.lemma("parking"), "park")
# "used" strips to "us" one letter before it strips to "use".
check("a two-letter base is never the answer", lx.lemma("used"), "use")
check("-ss is not a plural", lx.lemma("business"), "business")
check("news is not the plural of new", lx.lemma("news"), "news")
check("during is not a participle", lx.lemma("during"), "during")
check("a modern word absent from old dictionaries", lx.lemma("laptops"), "laptop")

# -- derivation -------------------------------------------------------------
check("-ly", lx.family("carefully"), "care")
check("-ment", lx.family("improvement"), "improve")
check("-ation", lx.family("calculation"), "calculate")
check("-ent", lx.family("different"), "differ")
check("agent -er", lx.family("speakers"), "speak")
check("comparative of an adjective", lx.family("earlier"), "early")
check("a transparent prefix", lx.family("misunderstood"), "understand")

# -- derivation: the traps --------------------------------------------------
# Each of these has a real dictionary word one strip away that is not the root.
check("listen is not a form of list", lx.family("listening"), "listen")
check("suggest does not fold onto sugg", lx.family("suggestion"), "suggest")
check("finish does not fold onto fin", lx.family("finished"), "finish")
check("corner is not a kind of corn", lx.family("corner"), "corner")
check("shower is not a kind of show", lx.family("shower"), "shower")
check("vocabulary survives -y", lx.family("vocabulary"), "vocabulary")
check("increase is not a kind of crease", lx.family("increase"), "increase")
check("report is not a kind of port", lx.family("report"), "report")

# -- the frequency bands ----------------------------------------------------
check("a core word is K1", lx.band("water"), "K1")
check("its inflection is too", lx.band("waters"), "K1")
# especially -> especial is a real word and a rare one; folding there would
# quietly promote a common adverb into evidence of a wide vocabulary.
check("a listed word is not demoted off its band", lx.band("especially"), "K1")
check("a second-thousand word", lx.band("achieve"), "K2")
check("a word past the first two thousand", lx.band("pagoda"), "off-list")
check("a name is set aside, not counted as rare", lx.band("kimi"), "unknown")

# -- the profile ------------------------------------------------------------
from collections import Counter

prof = lexicon.profile(Counter(
    {"go": 3, "goes": 1, "going": 2, "went": 1, "care": 1, "careful": 1,
     "carefully": 1, "kimi": 4}))
check("forms are counted as written", prof["forms"], 8)
check("inflections fold into lemmas", prof["lemmas"], 5)
check("derivations fold further, names excluded", prof["families"], 2)
check("names are reported, not silently dropped", prof["unknown"], 1)
check("tokens are untouched by folding", prof["tokens"], 14)

# Chao1: 2 families seen, none once-only here, so it cannot extrapolate up.
check("the estimate never falls below what was seen",
      lexicon.chao1(Counter({"a": 1, "b": 1, "c": 2})) >= 3, True)

print("\n%d checks, %d failed" % (len(PASS) + len(FAIL), len(FAIL)))
sys.exit(1 if FAIL else 0)
