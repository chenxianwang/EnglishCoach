# docs/

Screenshots used by the top-level `README.md`.

To regenerate the source page for them:

```bash
python english_coach.py --demo --out demo_report.html
open demo_report.html
```

The demo renders from the synthetic `DEMO_DATA` fixture in `english_coach.py`
(an invented learner — no real recordings, no API keys, no models needed), so
these screenshots are safe to publish.

Expected files:

| file | what to capture |
|---|---|
| `screenshot-report.png` | top of a recording report — score ring, sub-score bars, top fixes |
| `screenshot-prosody.png` | the Prosody meter — pitch contour graph + the five metric cards |
| `screenshot-drills.png` | a training panel, e.g. Sound system or Listening |
