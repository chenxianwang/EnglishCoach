#!/usr/bin/env python3
"""Tests for the speaking error statistics behind the Error stats view.

    python3 test_error_stats.py

The thing worth testing here is not that it counts, but that it counts the
right denominator. A rate is only meaningful against how many chances you had,
and the two ways of arriving at the numerator — exact phoneme scores vs. a
word-level verdict spread over the sounds inside it — must not be confusable.
"""

import sys

import english_coach as ec
from backfill_phonemes import merge_phonemes

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append(name)
        print("  FAIL  %s\n          got:  %r\n          want: %r" % (name, got, want))


def rec(words):
    return {"azure": {"words": words}}


def w(word, phones=None, pacc=None, accuracy=90, error=""):
    d = {"word": word, "accuracy": accuracy, "error": error}
    if phones is not None:
        d["phones"] = phones
    if pacc is not None:
        d["pacc"] = pacc
    return d


def row(stats, key):
    return next((r for r in stats["rows"] if r["key"] == key), None)


# -- the alphabet has to be folded before anything is counted ---------------
check("Azure's schwa folds onto CMUdict's", ec._stat_phone("ax"), "AH")
check("a flapped t is still a t", ec._stat_phone("dx"), "T")
check("stress digits are stripped", ec._stat_phone("IY1"), "IY")

# -- position is what separates a dark l from a clear one -------------------
check("onset l is clear, coda l is dark",
      [r for p, r in ec._stat_roles(["L", "IH", "T", "AH", "L"])],
      ["onset", "nucleus", "medial", "nucleus", "final"])
check("a word-final stop is marked final",
      ec._stat_roles(["N", "IY", "D"])[-1], ("D", "final"))
check("a stop before another consonant is coda, not final",
      [r for p, r in ec._stat_roles(["OW", "L", "D"])], ["nucleus", "coda", "final"])

# -- CMUdict fills in for recordings analysed before phones were kept -------
check("phonemes come from CMUdict when Azure did not save them",
      ec._stat_seq("thing"), ["TH", "IH", "NG"])
check("Azure's own phonemes win when present",
      ec._stat_seq("thing", ["s", "ih", "ng"]), ["S", "IH", "NG"])

# -- attributed mode: exact denominators, blame spread over the word --------
att = ec.pronunciation_error_stats([rec([
    w("think", error="Mispronunciation"),     # TH + K(final)
    w("three", error="Mispronunciation"),     # TH
    w("thing"), w("thought"), w("both"),      # TH, all fine
])])
check("attributed mode is labelled as such", att["mode"], "attributed")
th = row(att, "th_unvoiced")
check("every attempt at the sound is counted, not just the bad ones", th["n"], 5)
check("a flagged word charges the sounds inside it", th["bad"], 2)
check("the rate is failures over attempts", th["rate"], 40.0)

# -- exact mode: the failure is counted where it actually happened ----------
# Same two bad words, but now Azure says the /k/ sank "think" and the θ was
# fine. Attributed mode cannot tell these apart; exact mode must.
ex = ec.pronunciation_error_stats([rec([
    w("think", phones=["th", "ih", "ng", "k"], pacc=[95, 92, 90, 20],
      error="Mispronunciation"),
    w("three", phones=["th", "r", "iy"], pacc=[30, 88, 91],
      error="Mispronunciation"),
    w("thing", phones=["th", "ih", "ng"], pacc=[93, 90, 88]),
])])
check("exact mode is labelled as such", ex["mode"], "exact")
check("θ is only blamed where θ actually failed", row(ex, "th_unvoiced")["bad"], 1)
check("θ attempts are still all counted", row(ex, "th_unvoiced")["n"], 3)
check("the failing final stop is blamed instead", row(ex, "final_stop")["bad"], 1)

# A score Azure declined to give is not a zero.
unscored = ec.pronunciation_error_stats([rec([
    w("think", phones=["th", "ih", "ng", "k"], pacc=[-1, 90, 90, 90])])])
check("an unscored phoneme is skipped, not counted as failed",
      row(unscored, "th_unvoiced"), None)

# -- mixed libraries must not silently claim to be exact --------------------
mixed = ec.pronunciation_error_stats([rec([
    w("think", phones=["th", "ih", "ng", "k"], pacc=[95, 92, 90, 20]),
    w("three", error="Mispronunciation"),
])])
check("a part-upgraded library says mixed", mixed["mode"], "mixed")

# -- a sound tried three times must not head the report ---------------------
# The interval is not the safeguard here: 2-of-3 really does imply a rate above
# 20%, so ranking on evidence alone puts it top. It is still not a habit.
noisy = ec.pronunciation_error_stats([rec(
    [w("three", error="Mispronunciation")] * 2 + [w("thought")] +      # θ: 2/3
    [w("very", error="Mispronunciation")] * 12 + [w("love")] * 88      # v: 12/100
)])
keys = [r["key"] for r in noisy["rows"]]
check("a 3-attempt sound is held back from the ranking",
      keys.index("v_sound") < keys.index("th_unvoiced"), True)
check("...and is marked as thin rather than hidden",
      row(noisy, "th_unvoiced")["thin"], True)
check("the sound with real evidence is not marked thin",
      row(noisy, "v_sound")["thin"], False)
check("its numbers are still reported honestly",
      (row(noisy, "th_unvoiced")["n"], row(noisy, "th_unvoiced")["bad"]), (3, 2))

# -- grammar typing, on findings taken verbatim from the real log -----------
check("tense", ec.classify_grammar_error(
    {"rule": "Use past continuous for an action in progress at a past time."}), "tense")
check("countability", ec.classify_grammar_error(
    {"rule": "'feedback' is uncountable — no plural 's'"}), "number")
check("preposition", ec.classify_grammar_error(
    {"rule": "Use 'in' with 'recent years' to indicate a time period."}), "preposition")
check("modal", ec.classify_grammar_error(
    {"rule": "After modal 'should', use base verb without 'to'."}), "modal")
check("an omitted article is caught from the edit when the rule is silent",
      ec.classify_grammar_error({"rule": "", "said": "in fintech industry",
                                 "correction": "in the fintech industry"}), "article")
check("nothing forced when nothing fits",
      ec.classify_grammar_error({"rule": "zzz", "said": "x", "correction": "y"}), "")

gs = ec.grammar_error_stats([
    {"azure": {"words": [w("a")] * 500},
     "grammar": [{"said": "he go", "correction": "he goes", "rule": "agreement"},
                 {"said": "he go", "correction": "he goes", "rule": "agreement"}]},
    {"azure": {"words": [w("a")] * 500},
     "grammar": [{"said": "I eated", "correction": "I ate", "rule": "past tense"}]},
])
check("the same mistake quoted twice is one mistake", gs["total"], 2)
check("rate is per 1000 words actually spoken",
      sorted(r["per_1k"] for r in gs["rows"]), [1.0, 1.0])

# -- the backfill merge must not mis-attribute across a misalignment --------
old = [w("i"), w("really"), w("think"), w("so")]
new = [w("i", ["ay"], [90]), w("um", ["ah", "m"], [50]),      # an extra word
       w("really", ["r", "ih", "l", "iy"], [40, 90, 90, 90]),
       w("think", ["th", "ih", "ng", "k"], [95, 90, 90, 30]),
       w("so", ["s", "ow"], [90, 90])]
filled = merge_phonemes(old, new)
check("every original word still gets its scores", filled, 4)
check("scores land on the right word, not shifted by the insertion",
      old[1]["pacc"], [40, 90, 90, 90])
check("the inserted word does not become part of the log",
      [x["word"] for x in old], ["i", "really", "think", "so"])


# -- per-sound scores must survive the rest of the pipeline -----------------
# Every new analysis gets these for free (the Azure request has always asked
# for Phoneme granularity; the parser used to drop the score). The one place
# they could quietly go missing is re-grading, which rebuilds every word.
graded = ec.apply_strictness(
    {"pron_score": 80, "words": [w("think", phones=["th", "ih", "ng", "k"],
                                   pacc=[20, 92, 90, 90], accuracy=80)]},
    "very_strict")
gw = graded["words"][0]
check("re-grading keeps the per-sound scores", gw["pacc"], [20, 92, 90, 90])
check("...and the phoneme symbols they line up with",
      gw["phones"], ["th", "ih", "ng", "k"])
check("...while still re-grading the word itself", gw["accuracy"] < 80, True)
# Deliberate: the word verdict moves with the strictness you happened to pick
# for that recording, the per-sound scores do not. Re-grading them too would
# make the sound trend shift when you touch a UI toggle instead of when your
# speaking changes.
check("per-sound scores are not re-graded", gw["pacc"][0], 20)


# -- the weekly trend -------------------------------------------------------
def dated(day, words):
    return {"date": day, "azure": {"words": words}}


EX = lambda pacc, word="think": w(word, phones=["th", "ih", "ng", "k"], pacc=pacc)
GOOD, BAD = [95, 92, 90, 90], [20, 92, 90, 90]      # θ fine / θ weak

# Windows roll back from `end`, they are not calendar weeks. The newest one
# always ends on `end`, which is the whole point: a Monday-aligned bucket spends
# most of Monday holding a single day of speech, and that lone point swings hard
# enough to read as a collapse.
import datetime
END = datetime.date(2026, 8, 9)


def wk(items, end=END):
    return ec._windows(items, end)


keys, idx = wk([dated("2026-08-09", []), dated("2026-07-27", [])])
check("the newest window ends on the anchor date", keys[-1], "2026-08-03")
check("windows step back seven days at a time",
      keys, ["2026-07-27", "2026-08-03"])
check("a date lands in the window that contains it",
      idx["2026-08-09"], "2026-08-03")
check("...including the first day of that window",
      wk([dated("2026-08-03", [])])[1]["2026-08-03"], "2026-08-03")
check("...and the last day of the one before",
      wk([dated("2026-08-02", []), dated("2026-08-09", [])])[1]["2026-08-02"],
      "2026-07-27")

# An axis that skips the quiet stretches would put a month of silence and a day
# of it at the same distance, which is a lie about when you improved.
gapkeys, _ = wk([dated("2026-06-24", []), dated("2026-08-09", [])])
check("windows you recorded nothing in still hold their place on the axis",
      len(gapkeys), 7)
check("...and they are contiguous",
      gapkeys[:3], ["2026-06-22", "2026-06-29", "2026-07-06"])

check("an unparseable date is dropped, not guessed", ec._day("later"), None)
check("a date after the anchor is dropped rather than plotted",
      wk([dated("2026-09-01", [])])[1], {})
check("no dated recordings means no axis", wk([]), ([], {}))
check("history past the cap is cut, newest kept",
      len(wk([dated("2025-01-01", []), dated("2026-08-09", [])])[0]),
      ec.TREND_MAX_POINTS)


# Every fixture below is anchored to its own newest recording, so the windows
# are deterministic and no fixture date lands in the future of the anchor.
def _latest(items):
    ds = [d.get("date") for d in items if d.get("date")]
    return max(datetime.date.fromisoformat(x) for x in ds) if ds else END


def ptrend(items, **kw):
    kw.setdefault("end", _latest(items))
    return ec.pronunciation_trend(items, **kw)


def gtrend(items, **kw):
    kw.setdefault("end", _latest(items))
    return ec.grammar_trend(items, **kw)

# The guarantee this whole function exists for: exact and attributed
# recordings measure differently, so they must never share a line.
mixed_lib = [
    dated("2026-08-03", [EX(GOOD)] * 20 + [EX(BAD)] * 5),      # exact, θ 20%
    dated("2026-08-10", [w("think", error="Mispronunciation")] * 25),  # attributed
]
t = ptrend(mixed_lib, min_week=5)
check("only one measurement mode is charted", t["mode"], "exact")
check("the other mode is excluded, not blended", t["recordings"], 1)
check("...and the exclusion is reported so the UI can say so", t["skipped"], 1)
# The axis runs to the anchor either way — "nothing scored this way in the last
# seven days" is worth showing — but the excluded recording contributes no data.
check("the excluded recording puts no point on the chart",
      [p["w"] for p in t["overall"]], ["2026-07-28"])
check("...though its window still sits on the axis", len(t["weeks"]), 2)

# With no exact data at all, attributed is charted rather than nothing.
t_att = ptrend([
    dated("2026-08-03", [w("think", error="Mispronunciation")] * 25),
    dated("2026-08-10", [w("think")] * 25)], min_week=5)
check("an all-attributed library still charts", t_att["mode"], "attributed")
check("nothing is skipped when there is only one mode", t_att["skipped"], 0)

# Rates, and the thin-week floor.
t2 = ptrend([
    dated("2026-08-03", [EX(GOOD)] * 15 + [EX(BAD)] * 5),      # θ 25% of 20
    dated("2026-08-10", [EX(GOOD)] * 18 + [EX(BAD)] * 2),      # θ 10% of 20
], min_week=5)
th = next(s for s in t2["series"] if s["key"] == "th_unvoiced")
check("weekly rates are computed per week",
      [p["rate"] for p in th["points"]], [25.0, 10.0])
check("each point carries its own denominator",
      [p["n"] for p in th["points"]], [20, 20])
check("an improving sound reads as falling", th["points"][0]["rate"] > th["points"][-1]["rate"], True)

thin = ptrend([
    dated("2026-08-03", [EX(GOOD)] * 15 + [EX(BAD)] * 5),
    dated("2026-08-10", [EX(BAD)] * 3),                        # only 3 attempts
    dated("2026-08-17", [EX(GOOD)] * 20),
], min_week=5)
check("a window too thin to mean anything is left out of the line",
      [p["w"] for p in next(s for s in thin["series"]
                            if s["key"] == "th_unvoiced")["points"]],
      ["2026-07-28", "2026-08-11"])
check("...though the window still exists on the axis",
      thin["weeks"], ["2026-07-28", "2026-08-04", "2026-08-11"])
# Left out of the line, but not left out of the answer. A rare sound falls
# under the floor in perfectly ordinary weeks, and a table cell that just says
# "—" reads as a broken chart rather than as "you barely said it".
th_thin = next(s for s in thin["series"] if s["key"] == "th_unvoiced")
check("a skipped window comes back as a gap, with its count",
      th_thin["gaps"], {"2026-08-04": 3})
check("the gap count is excluded from the series total", th_thin["total"], 40)
check("a week where the sound never came up is not called a gap",
      "2026-08-24" in th_thin["gaps"], False)
check("a sound present in every week has no gaps",
      next(s for s in t2["series"] if s["key"] == "th_unvoiced")["gaps"], {})

# The floor is a statement about arithmetic, not a preference: at n attempts a
# single slip moves the rate by 100/n points, and past ten points the dot has
# stopped being a measurement.
check("the floor keeps one slip under ten points",
      100.0 / ec.TREND_MIN_WEEK <= 10.0, True)
# The lower floor is what brings a genuinely rare sound back. 12 attempts is a
# real week of speech for θ; the old floor of 15 erased it.
rare = ptrend([
    dated("2026-08-03", [EX(GOOD)] * 15 + [EX(BAD)] * 5),
    dated("2026-08-10", [EX(GOOD)] * 9 + [EX(BAD)] * 3),       # 12 attempts
    dated("2026-08-17", [EX(GOOD)] * 20),
])
check("a rare sound's ordinary window is plotted, not erased",
      [p["w"] for p in next(s for s in rare["series"]
                            if s["key"] == "th_unvoiced")["points"]],
      ["2026-07-28", "2026-08-04", "2026-08-11"])
check("...with its real rate", next(s for s in rare["series"]
      if s["key"] == "th_unvoiced")["points"][1]["rate"], 25.0)

# Dropping the thin week can leave a single point, and a single point is not a
# trend — the series is then not offered at all rather than drawn as a dot.
gap = ptrend([
    dated("2026-08-03", [EX(GOOD)] * 15 + [EX(BAD)] * 5),
    dated("2026-08-10", [EX(BAD)] * 3),
], min_week=5)
check("a series reduced to one usable week is not plotted",
      [s["key"] for s in gap["series"]], [])

# A single point is not a trend, so it is not offered as a line.
one = ptrend([dated("2026-08-03", [EX(GOOD)] * 20)], min_week=5)
check("one week produces no series to plot", one["series"], [])
check("...but the window itself is still reported", one["weeks"], ["2026-07-28"])

# Worst-first, so the chart opens on what is costing something.
order = ptrend([
    dated("2026-08-03", [EX(BAD)] * 10 + [EX(GOOD)] * 10 +
          [w("very", phones=["v", "eh", "r", "iy"], pacc=[95, 90, 90, 90])] * 20),
    dated("2026-08-10", [EX(BAD)] * 10 + [EX(GOOD)] * 10 +
          [w("very", phones=["v", "eh", "r", "iy"], pacc=[95, 90, 90, 90])] * 20),
], min_week=5)
check("the worst sound is offered first",
      order["series"][0]["key"], "th_unvoiced")
check("a sound that never fails ranks last",
      order["series"][-1]["rate"], 0.0)

check("empty input does not explode",
      ptrend([])["weeks"], [])

# -- the grammar trend ------------------------------------------------------
def gr(day, words, findings):
    return {"date": day, "azure": {"words": [w("a")] * words}, "grammar": findings}


TENSE = {"said": "he go", "correction": "he goes", "rule": "past tense"}
ART = {"said": "in fintech", "correction": "in the fintech", "rule": "article"}

gt = gtrend([
    gr("2026-08-03", 1000, [TENSE, ART]),          # 2 per 1000
    gr("2026-08-10", 2000, [TENSE]),               # 0.5 per 1000
], bucket_words=1000)
check("grammar windows are in order", gt["weeks"], ["2026-07-28", "2026-08-04"])
check("the rate is per 1000 words, not per week",
      [p["rate"] for p in gt["overall"]], [2.0, 0.5])
check("a heavier week with the same slips reads as better",
      gt["overall"][0]["rate"] > gt["overall"][1]["rate"], True)
check("each point keeps the words it was divided by",
      [p["words"] for p in gt["overall"]], [1000, 2000])
check("a window with enough words behind it is not flagged thin",
      [p["thin"] for p in gt["overall"]], [False, False])

# -- thin and empty windows -------------------------------------------------
# A window you barely spoke in cannot carry a rate of its own: 300 words with
# one slip reads as 3.3 per 1000, which is mostly the denominator. It is drawn
# hollow rather than merged away — merging would have to reach across the quiet
# windows between, and that is exactly how a month of silence gets hidden.
short = gtrend([
    gr("2026-08-03", 1000, [TENSE]),
    gr("2026-08-10", 300, [TENSE]),                # thin, and in the middle
    gr("2026-08-17", 1000, [TENSE, ART]),
], bucket_words=1000)
check("a thin window keeps its own point", len(short["overall"]), 3)
check("...and is flagged so the chart can hollow it out",
      [p["thin"] for p in short["overall"]], [False, True, False])
check("...with only its own words behind it",
      [p["words"] for p in short["overall"]], [1000, 300, 1000])
check("...and its own findings",
      [p["n"] for p in short["overall"]], [1, 1, 2])
check("the axis labels each window with the day it ends on",
      short["wend"][short["weeks"][-1]], "2026-08-17")
check("every window is the same length, so distance is time",
      [(datetime.date.fromisoformat(short["wend"][k]) -
        datetime.date.fromisoformat(k)).days for k in short["weeks"]], [6, 6, 6])

# Below the hard floor there is nothing to plot, but the window still exists —
# the line dots across it rather than pretending those days were never there.
quiet = gtrend([
    gr("2026-08-03", 1000, [TENSE]),
    gr("2026-08-10", 60, [TENSE]),                 # 60 words -> 16.7 per 1000
    gr("2026-08-17", 1000, [TENSE]),
], bucket_words=1000, min_words=200)
check("a window under the floor is not plotted",
      [p["w"] for p in quiet["overall"]], ["2026-07-28", "2026-08-11"])
check("...but it holds its place on the axis", len(quiet["weeks"]), 3)
check("...and comes back as a gap, with the words it did have",
      next(s for s in quiet["series"] if s["key"] == "tense")["gaps"],
      {"2026-08-04": 60})

# A stretch with no recordings at all is a gap with no count — a different kind
# of nothing from "you spoke, but barely", and the chart says which.
silent = gtrend([
    gr("2026-07-27", 1000, [TENSE]),
    gr("2026-08-17", 1000, [TENSE]),
])
check("silent windows sit on the axis between the two points",
      len(silent["weeks"]), 4)
check("...and are not reported as thin-but-real",
      next(s for s in silent["series"] if s["key"] == "tense")["gaps"], {})

# -- trimming the lead-in ---------------------------------------------------
# A trend has to open somewhere the number means something. Leading thin and
# empty windows cost a column each and say nothing, so they are cut — but only
# at the front, and the card is told how many so history never just vanishes.
lead = gtrend([
    gr("2026-07-06", 300, [TENSE]),                # thin, at the front
    gr("2026-08-03", 1000, [TENSE]),
    gr("2026-08-10", 1000, [TENSE]),
], bucket_words=1000)
check("the chart opens on the first window worth reading",
      lead["weeks"][0], "2026-07-28")
check("...and everything before it is gone from the axis", len(lead["weeks"]), 2)
check("...with the count reported so it can be said out loud", lead["trimmed"], 4)
check("...and the trimmed recordings excluded from the total",
      lead["recordings"], 2)

# Only the lead-in. A thin window with real data on both sides is kept, because
# a dip between two measurements is information about you.
mid = gtrend([
    gr("2026-08-03", 1000, [TENSE]),
    gr("2026-08-10", 300, [TENSE]),
    gr("2026-08-17", 1000, [TENSE]),
], bucket_words=1000)
check("trimming stops at the first solid window", mid["trimmed"], 0)
check("...so an interior thin window survives", len(mid["overall"]), 3)

# A library that is all thin still charts. Trimming everything would leave a
# new user staring at an empty card.
allthin = gtrend([
    gr("2026-08-03", 300, [TENSE]),
    gr("2026-08-10", 300, [TENSE]),
], bucket_words=1000)
check("nothing solid yet means nothing is trimmed", allthin["trimmed"], 0)
check("...and the chart still has its points", len(allthin["overall"]), 2)

# Deduplication is per recording, because saying it again next week is the
# whole point of a trend.
dup = gtrend([
    gr("2026-08-03", 1000, [TENSE, dict(TENSE)]),  # same slip twice in one analysis
    gr("2026-08-10", 1000, [TENSE]),               # and again a week later
], bucket_words=1000)
check("the same slip twice in one analysis counts once",
      dup["overall"][0]["n"], 1)
check("the same slip in a later week counts again",
      dup["overall"][1]["n"], 1)

typed = gtrend([
    gr("2026-08-03", 1000, [TENSE, TENSE, ART]),
    gr("2026-08-10", 1000, [TENSE, ART]),
], bucket_words=1000)
keys = [s["key"] for s in typed["series"]]
check("findings are split by type", sorted(keys), ["article", "tense"])
check("a type with no findings at all is not offered", "pronoun" in keys, False)
check("a week where a type did not appear is a zero, not a gap",
      [p["rate"] for p in next(s for s in typed["series"]
                               if s["key"] == "article")["points"]], [1.0, 1.0])

check("an empty grammar library does not explode",
      gtrend([])["weeks"], [])
check("recordings with no date are skipped",
      gtrend([{"azure": {"words": [w("a")] * 900}, "grammar": [TENSE]}])["weeks"], [])

print()
print("%d FAILING" % len(FAIL) if FAIL else "all %d checks passed" % len(PASS))
sys.exit(1 if FAIL else 0)
