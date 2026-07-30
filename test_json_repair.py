#!/usr/bin/env python3
"""Regression tests for the LLM-JSON repair chain.

    python3 test_json_repair.py

LLMs emit near-valid JSON on long structured outputs. The failure that keeps
recurring is an unescaped double-quote inside a string value, which is only
sometimes decidable locally — see `_is_ambiguous_close`. These cases are drawn
from real DeepSeek replies, so they must keep passing.
"""

import json
import sys

import english_coach as ec

PASS, FAIL = [], []


def check(name, raw, expect_key=None, expect_value=None):
    try:
        got = ec._extract_json(raw, "test")
    except Exception as e:
        FAIL.append((name, "raised %s: %s" % (type(e).__name__, str(e)[:120])))
        return
    if expect_key is not None:
        actual = got
        for k in expect_key.split("."):
            k = int(k) if k.isdigit() else k
            try:
                actual = actual[k]
            except (KeyError, IndexError, TypeError):
                FAIL.append((name, "missing key path %r" % expect_key))
                return
        if expect_value is not None and actual != expect_value:
            FAIL.append((name, "at %s expected %r, got %r"
                         % (expect_key, expect_value, actual)))
            return
    PASS.append(name)


# --- 1. already valid -------------------------------------------------------
check("clean JSON",
      '{"a": 1, "b": "two"}', "b", "two")

# --- 2. wrappers the model adds --------------------------------------------
check("fenced in ```json",
      '```json\n{"a": "x"}\n```', "a", "x")
check("prose either side",
      'Here you go:\n{"a": "x"}\nHope that helps!', "a", "x")

# --- 3. structural slips ----------------------------------------------------
check("trailing comma",
      '{"a": "x", "b": "y",}', "b", "y")
check("missing comma between pairs",
      '{"a": "x"\n "b": "y"}', "b", "y")
check("missing comma between objects",
      '{"g": [{"a": "x"}\n{"a": "y"}]}', "g.1.a", "y")
check("raw newline inside a string",
      '{"a": "line one\nline two"}', "a", "line one\nline two")

# --- 4. inner quotes, locally decidable ------------------------------------
# a quote followed by a word cannot be a close, so the scanner gets these right
check("inner quotes mid-sentence",
      '{"why": "he said "hi" loudly"}',
      "why", 'he said "hi" loudly')
check("inner quote then comma then word",
      '{"why": "he said "hi", then left"}',
      "why", 'he said "hi", then left')

# --- 5. THE ambiguous case: quote, comma, quote ----------------------------
# This is the class that produced "Expecting ',' delimiter" in production.
# Locally undecidable; only whole-document parsing settles it.
check("ambiguous: inner quotes, real key follows",
      '{"said": "the fight on the "cove", "corner"", "rule": "word choice"}',
      "rule", "word choice")
check("ambiguous: two quoted words in one value",
      '{"example": "heard "cove", "call" instead", "fix": "drill /k/"}',
      "fix", "drill /k/")
check("ambiguous inside an array of strings",
      '{"evidence": ["said "math", heard "mass"", "dropped -ed"]}',
      "evidence.1", "dropped -ed")
check("ambiguous with a following object",
      '{"top_fixes": [{"issue": "say "the", not "a"", "why": "articles"}]}',
      "top_fixes.0.why", "articles")

# --- 5b. UNTERMINATED string values ----------------------------------------
# The production failure. DeepSeek wrote array elements as `"word" -> prose`
# with no closing quote, so the string swallowed the rest of the document.
# Quote-pairing cannot recover this; only structural re-lexing can.
check("unterminated element, comma-terminated",
      '{"evidence": [\n "captured" -> likely heard as \'capture\',\n "missed" -> likely heard as \'miss\'\n],\n"fix": "add final /t/"}',
      "fix", "add final /t/")
check("unterminated element keeps its content",
      '{"evidence": [\n "captured" -> likely heard as \'capture\',\n "missed" -> likely heard as \'miss\'\n]}',
      "evidence.0", "captured\" -> likely heard as 'capture'")
check("unterminated element, bracket-terminated",
      '{"examples": [\n "row" -> \'low\'\n],\n"drill": "light/right"}',
      "drill", "light/right")
check("mixed: unterminated and stray-closing-quote elements",
      '{"examples": [\n "the" -> \'ze\' (likely)",\n "that" -> \'zat\'\n],\n"name": "/th/ to /s/"}',
      "name", "/th/ to /s/")
check("unterminated value in an object, not an array",
      '{"said": "I and my opponent, "correction": "my opponent and I", "rule": "order"}',
      "rule", "order")

# --- 6. the real schema, mangled the way DeepSeek mangles it ---------------
REALISTIC = '''{
  "level_estimate": "B2 (reaching C1)",
  "overall_summary": "Systematic tense drift and article omission typical of Mandarin L1.",
  "top_fixes": [
    {"issue": "Word-final consonants",
     "example": "I won it by the fight on the "cove", which should be "corner"",
     "better": "I won it with the fight in the corner",
     "why": "Dropped endings cost intelligibility"}
  ],
  "blind_spots": [
    {"pattern": "Tense drift", "kind": "grammar",
     "evidence": ["his position is not that solid", "we both didn't notice"],
     "fix": "Narrate yesterday in past tense only"}
  ],
  "pronunciation_patterns": [],
  "grammar": [{"said": "I and my opponent", "correction": "my opponent and I", "rule": "word order"}],
  "word_choice": [{"said": "launched a call", "suggestion": "started a ko fight", "note": "Go term"}],
  "polished": "In my latest game I won, but both my opponent and I missed a wedge move."
}'''
check("realistic mangled reply — recovers every field",
      REALISTIC, "grammar.0.rule", "word order")
check("realistic mangled reply — polished intact",
      REALISTIC, "polished",
      "In my latest game I won, but both my opponent and I missed a wedge move.")

# --- 7. genuinely unrecoverable input should still raise cleanly -----------
try:
    ec._extract_json("this is not JSON at all, sorry", "test")
    FAIL.append(("garbage input", "should have raised RuntimeError"))
except RuntimeError:
    PASS.append("garbage input raises RuntimeError")
except Exception as e:
    FAIL.append(("garbage input", "raised %s not RuntimeError" % type(e).__name__))

# --- 8. the search must not corrupt valid documents ------------------------
VALID = json.dumps({
    "level_estimate": "B2",
    "grammar": [{"said": "a", "correction": "b", "rule": "c"} for _ in range(8)],
    "polished": 'She said "hello" and left.',
})
check("valid doc with properly escaped quotes survives round-trip",
      VALID, "polished", 'She said "hello" and left.')


# --- report ----------------------------------------------------------------
for name in PASS:
    print("  ok    %s" % name)
for name, why in FAIL:
    print("  FAIL  %s\n          %s" % (name, why))
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
