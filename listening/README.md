# listening/

Imported audio clips and `library.json` for the **Listening — dictation** panel.

This folder is gitignored. The audio here comes from third parties under a mix
of licences — some permit redistribution, some don't — so it stays local rather
than being committed. Re-create it with `listening_import.py`.

```bash
python3 listening_import.py list            # what you have
python3 listening_import.py --help          # sources and their licences
```

**Importing is one-off, not per-session.** Once clips are here the dictation
panel just uses them. Re-run an importer only when you want more material —
already-imported clips are skipped, so each run adds new ones rather than
re-fetching the same rows. Clips are sampled at random (the exports are ordered
by id, so the head of the file is the oldest and most repetitive material); pass
`--in-order` if you'd rather take them sequentially.

## Sources

| Command | Material | Licence |
|---|---|---|
| `voa` | VOA Learning English — graded news | Public domain; credit `learningenglish.voanews.com` |
| `sbc` | Santa Barbara Corpus — unscripted conversation | CC BY-ND 3.0 US — personal use, don't redistribute re-segmented |
| `tatoeba` | Single sentences, native speakers | Text CC-BY 2.0 FR; **each recording has its own licence**, recorded per clip |
| `local` | Your own recordings | Yours |

## Layout

```
listening/
  library.json      normalized clips, written by the importer
  voa/…             audio, one file per article
  sbc/…             audio, one file per conversation
  tatoeba/…         audio, one file per sentence
  local/…           your own
```

Each clip in `library.json`:

```json
{"id": "voa-tuesday-meeting-003",
 "source": "VOA",
 "license": "Public domain (VOA) — credit learningenglish.voanews.com",
 "source_url": "https://learningenglish.voanews.com/",
 "audio": "voa/tuesday-meeting.mp3",
 "start": 12.4, "end": 17.9,
 "text": "The committee said it would review the decision next week."}
```

`start`/`end` are optional — with them the player loops just that sentence
inside a longer recording; without them it plays the whole file.
