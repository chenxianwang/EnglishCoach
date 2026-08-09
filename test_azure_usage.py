#!/usr/bin/env python3
"""Tests for the Azure usage meter.

    python3 test_azure_usage.py

Azure publishes no "remaining quota" endpoint, so this number is the app's own
count of what it sent. That makes two things worth pinning down: it must count
the right calls (and only those), and it must never present a limit it was
never given.
"""

import json
import os
import shutil
import sys
import tempfile

import english_coach as ec

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append(name)
        print("  FAIL  %s\n          got:  %r\n          want: %r" % (name, got, want))


def library(*results):
    """A throwaway library dir holding the given (recorded_at, seconds, azure?)."""
    lib = os.path.join(tempfile.mkdtemp(), "lib")
    os.makedirs(lib)
    for i, (when, sec, has_azure) in enumerate(results):
        sub = os.path.join(lib, "rec%d" % i)
        os.makedirs(sub)
        with open(os.path.join(sub, "rec%d.result.json" % i), "w") as f:
            json.dump({"recorded_at": when, "duration_sec": sec,
                       "azure": {"words": [{"word": "a"}]} if has_azure else {}}, f)
    return lib


AUG, JUL = "2026-08-02T10:00:00", "2026-07-15T10:00:00"

# -- seeding: history is reconstructed from the analyses on disk -------------
lib = library((AUG, 300.0, True), (AUG, 600.0, True), (JUL, 900.0, True))
s = ec.azure_usage_summary(lib, allowance_hours=5, month="2026-08")
check("only this month is counted", s["seconds"], 900.0)
check("hours are derived from seconds", s["hours"], 0.25)
check("seeded history is flagged as such", s["seeded"], True)
check("the ledger file is written on first read",
      os.path.exists(ec.usage_path(lib)), True)

# A recording analysed without Azure spent no quota and must not appear.
lib2 = library((AUG, 300.0, True), (AUG, 9999.0, False))
check("recordings Azure never saw are excluded",
      ec.azure_usage_summary(lib2, month="2026-08")["seconds"], 300.0)

# -- logging: every caller is counted, and kept apart -----------------------
ec.log_azure_usage(120, kind="practice", library=lib)
ec.log_azure_usage(45, kind="backfill", library=lib)
s = ec.azure_usage_summary(lib, allowance_hours=5, month="2026-08")
check("a logged call adds to the total", s["seconds"], 1065.0)
check("calls are broken down by what spent them",
      s["by_kind"], {"analysis": 900.0, "practice": 120.0, "backfill": 45.0})
check("remaining is the allowance minus usage", s["remaining_hours"], 4.7)

# -- no allowance means no invented limit -----------------------------------
s0 = ec.azure_usage_summary(lib, allowance_hours=0, month="2026-08")
check("usage is still reported without an allowance", s0["seconds"], 1065.0)
check("...but nothing is presented as remaining", "remaining_hours" in s0, False)
check("...and no percentage either", "pct" in s0, False)

# -- going over must not report negative headroom ---------------------------
over = ec.azure_usage_summary(lib, allowance_hours=0.1, month="2026-08")
check("remaining floors at zero rather than going negative",
      over["remaining_hours"], 0.0)
check("percentage caps at 100", over["pct"], 100.0)

# -- metering must never be able to break scoring ---------------------------
before = ec.azure_usage_summary(lib, month="2026-08")["seconds"]
ec.log_azure_usage(0, library=lib)
ec.log_azure_usage(None, library=lib)
ec.log_azure_usage("nonsense", library=lib)
ec.log_azure_usage(30, library="/nonexistent/path/nowhere")
check("junk and unwritable paths are swallowed, not raised",
      ec.azure_usage_summary(lib, month="2026-08")["seconds"], before)

# -- a corrupt ledger degrades to empty rather than crashing ----------------
lib3 = library((AUG, 300.0, True))
ec.azure_usage_summary(lib3, month="2026-08")          # force the file to exist
with open(ec.usage_path(lib3), "w") as f:
    f.write("{ not json at all")
check("a damaged ledger reads as empty",
      ec.azure_usage_summary(lib3, month="2026-08")["seconds"], 0)

for d in (lib, lib2, lib3):
    shutil.rmtree(os.path.dirname(d), ignore_errors=True)

print()
print("%d FAILING" % len(FAIL) if FAIL else "all %d checks passed" % len(PASS))
sys.exit(1 if FAIL else 0)
