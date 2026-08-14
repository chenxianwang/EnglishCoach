# 51 Days of Measuring My Own English

**Slide source material.** Every number below was computed from my own data on
2026-08-13 — 54 analysed recordings, 16,102 scored words, 4.0 hours of speech,
2026-06-24 → 2026-08-13. Nothing here is an estimate or a guess unless it says so.

Structure: 6 sections, ~20 slides. Each slide has a **headline**, the **content**
to put on the slide, and **speaker notes** for what to say out loud.

---

## SECTION 0 — Setup

### Slide 1 — Title

> # I stopped guessing about my English
> ## 51 days, 16,102 scored words, and what the data actually said
>
> *Everything in this talk is measured, not remembered.*

**Speaker notes:** Open with the honest framing — I have studied English for
decades and had no idea what I was actually bad at. Not "not fluent." Not
"pronunciation needs work." *Which sound. Which rule. How often.* That question
had never had an answer, so I built something that could answer it.

---

### Slide 2 — What this is built on

| | |
|---|---|
| Recordings analysed | **54** |
| Days covered | **51** (2026-06-24 → 2026-08-13) |
| Days I actually recorded | **26** |
| Total speech | **4.0 hours** |
| Words scored, phoneme by phoneme | **16,102** |
| Distinct content words I produced | **1,568** |
| Grammar corrections logged | **274** |
| Word-choice corrections | **170** |
| Recurring blind spots identified | **147** |
| Pronunciation drill attempts | **1,798** |
| Dictation attempts | **454** |
| Words I've met in my own surroundings | **718** |

**Speaker notes:** Four signals per recording, independently: Azure gives
phoneme-level accuracy; an LLM gives grammar and word choice; a local NumPy
analyser gives pitch and rhythm; and Whisper's own transcription mistakes are
kept as evidence. That last one turns out to matter most — I'll come back to it.

---

## SECTION 1 — The honest scoreboard

### Slide 3 — 51 days, three phases

| Metric | First 10 sessions | Middle | Last 10 sessions | Verdict |
|---|---|---|---|---|
| Pronunciation score | 72.1 | 74.1 | **73.3** | flat |
| Accuracy | 78.5 | 79.7 | **83.8** | ↑ +5.3 |
| Fluency | 82.0 | 82.0 | **79.3** | ↓ |
| Prosody | 55.2 | 59.2 | **58.2** | ↑ slightly, still worst |
| Completeness | 90.3 | 91.7 | 89.2 | flat |
| Grammar errors / session | 7.0 | 5.1 | **3.2** | ↓ −54% |
| Word-choice issues / session | 3.0 | 3.2 | **2.7** | ↓ slightly |
| Speaking rate (wpm) | 91.6 | 76.9 | 75.9 | ↓ −17% |

**Speaker notes:** This is the slide I did not want to make. Grammar improved a
lot. Pronunciation did not move. And I got *slower* — 92 wpm down to 76 wpm.
That's not decline; that's me monitoring myself while I speak. Attention is
finite: I bought grammar accuracy with speed.

---

### Slide 4 — The plateau, stated plainly

> ## Pronunciation score: **69 → 73 → 73 → 73 → 73**
>
> Mispronunciation rate, normalised per word spoken:
> **first 10 sessions 5.2% → last 10 sessions 6.0%**
>
> **51 days of daily practice moved my pronunciation score by ~4 points,
> then stopped.**

**Speaker notes:** Be blunt here. The raw error *count* looks like it exploded —
37 errors in June, 84 in one August session — but that's just because I recorded
a 24-minute session instead of a 3-minute one. Once you divide by words spoken,
the rate is flat. Any progress chart without a denominator is a chart about how
much you recorded, not how well you spoke. That was discovery #1, and it changed
how I built everything after it.

---

### Slide 5 — The level estimate that never moved

> The AI's CEFR verdict, on all 54 recordings:
>
> ### "B2 (reaching C1)"
> ### "B2 (reaching C1)"
> ### "B2 (reaching C1)" × 52 more
>
> **53 of 54 identical.** The one exception was recording #1: *"B2
> (upper-intermediate)"*.

**Speaker notes:** A metric that never changes is not a measurement, it's a
label. This is a lesson about AI feedback generally: it will confidently produce
a summary judgment on every single sample, and the confidence tells you nothing
about whether the judgment can discriminate. I keep it on the report, but I
don't track it. The things I track are the ones that can move.

---

## SECTION 2 — The mistakes I keep making: grammar

### Slide 6 — 274 corrections, sorted

| Category | Count | Share |
|---|---|---|
| **Tense / aspect** | **76** | 28% |
| **Articles (a / an / the)** | **33** | 12% |
| Prepositions | 28 | 10% |
| Subject–verb agreement | 23 | 8% |
| Modals / auxiliaries | 15 | 5% |
| Gerund vs. infinitive | 6 | 2% |
| Countability | 2 | <1% |
| Comparatives, word order, clauses | 5 | 2% |
| *Other (deliberately not forced into a bucket)* | 86 | 31% |

Independently, the AI flagged **"recurring blind spots"** 147 times.
Collapsing the wording variants:

- **Tense/aspect drift — 27 separate flaggings**
- **Missing or misused articles — 25**
- **Word-final consonants dropped or reduced — 30**

**Speaker notes:** Three problems. Not thirty. Two are grammar, one is
pronunciation, and they show up on essentially every recording I have ever made.
Note that ~31% lands in "Other" and I left it visible rather than forcing it —
a tidy chart that lies is worse than a messy one that doesn't.

---

### Slide 7 — Mistake #1: tense drift (28% of everything)

I start a story in the past, and slide into the present within one sentence.

| I said | Should be |
|---|---|
| "I **go** to a restaurant and **ordered** some noodles" | "I **went** to a restaurant and ordered some noodles" |
| "I **play** a move that was over protection" | "I **played** a move that was overprotection" |
| "I **choose** the simple one" | "I **chose** the simple one" |
| "it seemed that he **is** stronger than me" | "it seemed that he **was** stronger than me" |
| "it's again one move short of the win" | "it **was** again one move short of the win" |
| "my other group of stones **are** safe enough, so I **should play** more aggressively" | "…**was** safe enough, so I **should have played** more aggressively" |
| "I **had been struggling for** many times" | "I **struggled** many times" |

**The fix I was given, repeatedly:** *set the time frame once at the start
("Last night I played…"), then never re-decide the tense mid-sentence.*

**Speaker notes:** Look at the second-to-last one carefully — that's the whole
problem in one line. "should play" vs "should have played". Mandarin doesn't
mark this; the timeline lives in context, not on the verb. So under any load —
a hard idea, a fast sentence — my verbs snap back to the unmarked form.

---

### Slide 8 — Mistake #2: articles (12%)

| I said | Should be |
|---|---|
| "in fintech industry" | "in **the** fintech industry" |
| "huge chunk of stones would be captured" | "**a** huge chunk of stones…" |
| "let's look details" | "let's look **at the** details" |
| "I played **move** hurry to connect" | "I played **a** move **in a** hurry to connect" |
| "as a process of skill mastering" | "as a process of mastering **a** skill" |
| "That was **a** big progress." | "That was big progress." *(uncountable — no article)* |
| "After **the** breakfast" | "After breakfast" *(no article with meals)* |

And, unimprovably, from one of my own recordings:

> ### "I always on me have article emission problem."
> → *"I always have **an** article **omission** problem."*

**Speaker notes:** That last one is my favourite thing in the whole dataset. I
was, out loud, describing my article problem — and dropped an article doing it.
It's the perfect demonstration that this is not a knowledge gap. I *know* the
rule. It's a production habit, and knowing a rule does not install a habit.

---

### Slide 9 — Mistake #3: agreement and countability

The pattern: **a long noun phrase, and I agree the verb with the wrong noun in it.**

| I said | Should be | Why |
|---|---|---|
| "my large group of stones **were** captured" | "**was** captured" | head noun is *group*, singular |
| "part of my groups **were** cut off" | "part of my group **was** cut off" | head noun is *part* |
| "the connectiveness of my group of stones **were** impossible" | "the connectivity … **was** impossible" | head noun is *connectivity* |
| "the green beans that **was** cooked" | "that **were** cooked" | …and the reverse error too |
| "the letters and rescue emblem **was** printed" | "**were** printed" | compound subject |

Countability:

| I said | Should be |
|---|---|
| "you can hardly get instant **feedbacks**" | "instant **feedback**" *(uncountable)* |
| "I missed many many **chance**" | "many, many **chances**" |
| "some **piece** of vegetables" | "some **pieces** of vegetables" |
| "obvious improvements **on** my English level" | "obvious improvement **in** my English" |
| "there was no **any** risk of upset" | "there was no risk of **an** upset" |

**Speaker notes:** The consistent shape is *distance*. When the subject and the
verb are adjacent I'm fine. Put four words between them and I agree with
whatever noun I said most recently. That's a working-memory failure, not a
grammar failure — which means the drill is longer sentences, not more rules.

---

### Slide 10 — Mistake #4: English shaped like Chinese

170 word-choice corrections. The ones that sting are where the grammar is
perfect and the phrase is simply not English.

| I said | Native version |
|---|---|
| "I **did** a move" | "I **made** a move" |
| "**carry about** the result" | "**worry about** the result" |
| "**many and many** mistakes" | "many mistakes" |
| "the **rest two** positions" | "the **remaining two** positions" |
| "I let the game **in points** in some phase" | "I let the game **slip** in points at some point" |
| "**over protection of my already alive the group**" | "overprotection of my already-alive group" |
| "**connectiveness**" | "connectivity" |
| "that would be **a huge of workload**" | "that would be a huge workload" |
| "**slanted** by my opponent" | "**attacked** by my opponent" |
| "brand sports sported out by AI" | "blind spots pointed out by AI" |

**Speaker notes:** Two different failures mixed here. "did a move", "carry
about", "many and many" — those are direct calques from Chinese collocations.
But "brand sports sported out" is different: that's not what I meant to say,
that's what the *recogniser heard*. Which brings me to the most useful idea in
this whole project.

---

## SECTION 3 — Pronunciation: what I actually get wrong

### Slide 11 — The recogniser's mistake is the measurement

> When Whisper transcribes "rope jump" as **"robe jump"** —
> that is not a bug to work around.
>
> **That is a listener failing in exactly the way a human listener would,
> and telling me precisely which sound to drill.**

Real examples from my own recordings:

| I said | The machine heard | The sound I dropped |
|---|---|---|
| rope jump | robe / lobes | final **/p/** |
| mopped the floor | mop the floor | final **/t/** |
| blind spots | brand sports | initial **/l/**, final **/t/** |
| with several eggs | wis several eggs | **/θ/** |
| noodles | noodoo | dark **/l/** |
| ko fight | co-fight / cold fight | **/k/** + vowel length |
| tesuji | tashiuji | vowel quality |

**Speaker notes:** This is the design idea the whole app is built around. Most
apps treat transcription errors as noise and try to correct around them. But a
speech recogniser is a listener with no politeness and no context-guessing. If
it can't tell your "rope" from "robe", neither can a stranger on a phone call.
Every one of its errors is free, honest, brutally specific feedback.

---

### Slide 12 — The words I actually fail

Of 1,652 distinct words scored, these are the ones I keep getting wrong.
*(avg = Azure accuracy 0–100 across every time I said it)*

**Catastrophic — I have never once said these correctly:**

| Word | Avg accuracy | Times said |
|---|---|---|
| **percentage** | **8** | 10 |
| **vocabulary** | **8** | 14 |
| **overall** | **24** | 7 |
| **surrounding** | **25** | 6 |
| **deteriorated** | 22 | 5 |
| **escape** | 22 | 5 |

**High-frequency and quietly wrong** — these matter more, because I say them constantly:

| Word | Avg accuracy | Times said | Times flagged |
|---|---|---|---|
| **played** | 56 | 78 | 25 |
| **phase** | 32 | 29 | 16 |
| **because** | 51 | 39 | 7 |
| **just** | 63 | 118 | 23 |
| **game** | 62 | 199 | 17 |
| **learned** | 41 | 26 | 8 |
| **missed** | 67 | 36 | 10 |
| **difficult** | 56 | 22 | — |

**Speaker notes:** "vocabulary" at 8/100 across fourteen attempts is funny until
you realise I use that word to talk *about* studying English. But look at the
second table — that's the real damage. "played" at 56, said 78 times. "just" at
63, said 118 times. A rare word said badly costs you one word. A common word
said badly costs you a sentence, every sentence.

---

### Slide 13 — The discovery: it's the **-ed**, specifically

I computed the average accuracy of every word I've ever said, split one way:

> ## Words ending in **-ed**: **70.1** / 100
> ## All other words (4+ letters): **80.3** / 100
> ### A 10-point gap. n = 547 vs 8,271.

The worst offenders:

| Word | Avg | Word | Avg |
|---|---|---|---|
| ignored | 8 | learned | **41** *(said 26×)* |
| deteriorated | 22 | started | 47 |
| noticed | 30 | responded | 49 |
| pushed | 31 | **played** | **56** *(said 78×)* |
| advanced | 35 | missed | 67 |

**And it is genuinely the -ed, not just "hard clusters":** words ending in two
or more consonant *letters* average 81.0 — versus 81.4 for everything else.
**No effect at all.** The deficit is specific to the past-tense morpheme.

**Speaker notes:** This is the finding I'm proudest of, because it contradicts
what I was being told. The AI reports flagged "word-final consonant clusters" 30
separate times. But when I measured clusters directly, there was nothing there —
81.0 vs 81.4. The gap is entirely in *-ed*. Which makes sense: it's not a
phonetic difficulty, it's a *grammatical* one. It's the same missing past tense
from Slide 7, showing up in my mouth instead of my grammar. One root cause,
two symptoms, and I'd been treating them as unrelated problems.

---

### Slide 14 — What the AI says vs. what the numbers say

**What the AI told me, 127 times across 54 reports:**

| Pattern named | Times flagged |
|---|---|
| Weak word-initial /r/ | 12 |
| Dark /l/ | 12 |
| /v/ vs /w/ | 10 |
| **/θ/ vs /s/ ("th")** | **9 + 8 + 6 + 3 = ~26** |
| Word-final consonants | ~20 |

**What 12,639 phoneme-tagged words actually measure:**

| Sound | Attempts | Failure rate | vs. my 4.4% baseline |
|---|---|---|---|
| **/ɜr/ (bird, work)** | 228 | **10.1%** | **2.3× worse** |
| **/r/ non-initial** | 1,863 | **9.9%** | **2.2× worse** |
| **iː (see, three)** | 1,314 | **7.8%** | 1.8× worse |
| /ŋ/ | 365 | 7.1% | 1.6× |
| /r/ initial | 197 | 7.1% | 1.6× |
| clear /l/ | 444 | 7.0% | 1.6× |
| ʃ tʃ dʒ | 835 | 6.6% | 1.5× |
| /v/ | 820 | 6.2% | 1.4× |
| final stops | 3,312 | 5.0% | 1.1× |
| dark /l/ | 343 | 4.7% | 1.1× |
| **θ / ð ("th")** | 1,556 | **4.4%** | **exactly baseline** |

> ### The AI's #1 complaint — "th" — is my *most average* sound.
> ### My actual worst sound is **/r/**, which it mentioned a third as often.

⚠️ **Honest caveat, keep it on the slide:** this is *attributed* mode — when a
word is flagged, every tracked sound inside it is charged. Attempt counts are
exact; failure counts are an upper bound. A sound that keeps company with a bad
one inherits blame. The ranking is directional, not final.

**Speaker notes:** This is the single most valuable thing the project produced.
"Mandarin speakers struggle with th" is true as a generalisation and false about
me. An LLM writing a report reaches for the well-known pattern for my L1 —
because that's what's in its training data — rather than the pattern in my
audio. It's not lying. It's pattern-matching on the stereotype of a Chinese
speaker instead of measuring the actual Chinese speaker in front of it. You only
catch that with a denominator.

---

## SECTION 4 — Listening: the other half

### Slide 15 — What I mishear

454 dictation attempts, 241 total misses on 124 distinct words.

| Kind | Share |
|---|---|
| **misheard** (wrote the wrong word) | 190 / 241 = 79% |
| **missed** (wrote nothing) | 36 / 241 = 15% |
| **imagined** (wrote a word that wasn't there) | 15 / 241 = 6% |

**41% of all my listening errors are function words. Another 12% are
contractions. Together: 54%.**

| Reference | I heard | What I lost |
|---|---|---|
| **should've** | shouldn't | the whole *'ve* |
| **i'd** | i | 'd |
| **i've** | i | 've |
| **we're** | we | 're |
| **there's** | there | 's |
| **he'd** | he | 'd |
| **it'll** | it | 'll |
| **could've** | could | 've |
| **helped** | help | -ed |
| **came** | turned / happened | vowel |

**Speaker notes:** Note *should've* → *shouldn't*. Those mean opposite things. I
didn't fail to hear a decoration; I inverted the meaning of the sentence. This
is where a listening problem becomes a comprehension problem.

---

### Slide 16 — 🔑 THE FINDING: it's the same weakness, in both directions

> ## In my mouth
> I say **"play"** for **"played"**, **"mop"** for **"mopped"**.
> Words ending -ed score **70** vs **80** for everything else.
>
> ## In my ears
> I hear **"we"** for **"we're"**, **"could"** for **"could've"**,
> **"help"** for **"helped"**.
> **54% of my listening errors are exactly these unstressed endings.**
>
> ### ⬇
> ### These are not two problems. This is one problem.
> ### **English puts grammar in the weak, unstressed parts of the word.
> ### Mandarin doesn't. So I neither produce them nor perceive them.**

**Speaker notes:** This is the payoff slide. I had been treating speaking
practice and listening practice as separate activities with separate scores in
separate panels. They were measuring the same deficit from two sides. And it
reframes the fix completely: drilling "th" harder was never going to help.
What helps is *grammatical endings under reduced stress* — produced and
perceived — which is a completely different exercise. I only found it because
both halves were in one dataset and I went looking for the overlap.

---

## SECTION 5 — What I learned about measuring learning

### Slide 17 — Five rules I had to discover the hard way

**1. A list of mistakes is useless. Only a rate means anything.**
"Seven bad /θ/" means one thing if I said θ nine times and something else
entirely if I said it 300 times. Every count in this deck has a denominator.
That single change reversed my top-priority sound.

**2. Consistency is the skill; peak performance is noise.**
A word counts as mastered when the **last N attempts all clear X** — not when
one lucky take does. Mine is set to *last 3 ≥ 85*. Across 1,798 drill attempts I
average 91 and only 9 attempts fell below 60 — but the words I've actually
mastered are far fewer than that average implies, because *once* is not a skill.

**3. Position changes the sound.**
The /l/ in *light* and the /l/ in *feel* are different articulations, and only
the second one has no Mandarin equivalent. Averaging them together hides exactly
the thing you're looking for. My own numbers: clear /l/ 7.0%, dark /l/ 4.7% —
opposite to what every textbook predicts for a Mandarin speaker, which is itself
worth knowing.

**4. Never mix measurement modes on one chart.**
Exact per-phoneme scoring and word-level attribution give different numbers for
identical speech: θ reads ~4% attributed and ~20% exact. Plot both on one line
and you get a dramatic cliff exactly where the instrument changed — and you'll
read it as progress or collapse. It's neither.

**5. Thin data must *look* thin.**
A rate off 24 attempts swings wildly; a rate off 1,000 doesn't. So weeks under
40 attempts are drawn hollow, under 10 aren't drawn at all, and the line dots
across the gap instead of confidently connecting two points through a week that
was never measured.

**Speaker notes:** Rule 1 is the one that generalises beyond English. Every
self-improvement dashboard I've ever used shows me counts. Counts of a thing I
did more of are not evidence I got better at it.

---

### Slide 18 — What I got wrong about my own English

| I believed | The data said |
|---|---|
| My problem is "th" | "th" is my most **average** sound (4.4%, = baseline). My worst is **/r/** (9.9%) |
| I drop final consonant clusters | Clusters: **81.0** vs **81.4** for everything else. **No effect.** It's *-ed* specifically (70 vs 80) |
| My grammar and my pronunciation are separate problems | Both are the same missing past-tense morpheme |
| Speaking and listening are separate skills | 54% of my listening errors are the same endings I fail to produce |
| More practice → better score | 51 days, 26 recording days: grammar −54%, pronunciation **flat** |
| A CEFR level tracks progress | 53 of 54 recordings returned the identical string |

**Speaker notes:** Six beliefs, all confidently held, all wrong. Not one of them
was stupid — every one is a reasonable thing for a Chinese learner to believe,
and most are printed in textbooks. They were just not true *about me*. The
general fact about a population and the specific fact about a person are
different objects.

---

### Slide 19 — What I'm doing about it

**Stop drilling:** /θ/ minimal pairs. It's already at baseline. Every hour spent
there is an hour not spent on the thing that's 2.3× worse.

**Start drilling — three things, in priority order:**

1. **/r/ and /ɜr/** — 9.9% and 10.1% failure, and /r/ appears in 1,863 of my
   words. Highest volume × highest rate = highest return, by a distance.
2. **The -ed morpheme, both directions.** Produce it (*played, learned, missed,
   mopped*) and perceive it (*helped* vs *help*, *should've* vs *shouldn't*).
   Same exercise, both halves. This is the one that fixes a grammar problem and
   a pronunciation problem with a single drill.
3. **The 8 high-frequency words I say wrong every day** — *played (56), phase
   (32), because (51), just (63), game (62), learned (41), missed (67),
   difficult (56)*. Not exotic vocabulary. The words I actually use.

**Stop measuring:** the CEFR level string, and any raw error count without a
denominator.

**The open question:** prosody has sat at 55–58 for 51 days and is my lowest
score of all. I have no diagnosis for it yet, and my speaking rate went
*down* 17% while I was fixing grammar. That's next.

---

### Slide 20 — Closing

> ## I studied English for twenty years and could not have told you
> ## which sound I get wrong most often.
>
> ## It took 51 days of measurement to find out it was **/r/**,
> ## and that the thing everyone told me to fix was already fine.
>
> ### Generic feedback is about a population.
> ### You are a sample of one.

**Speaker notes:** Close on the thing that's actually transferable: the answer
mattered less than the fact that the answer was *findable*. Anyone can be
specific about their own weakness. Almost nobody is, because nobody keeps the
denominator.

---

## APPENDIX — extra material if you need more slides

### A1. The prosody problem (undiagnosed)
Prosody is my worst metric across all 54 recordings: **55.2 → 59.2 → 58.2**.
Pitch variation on the most recent session: **5.83 semitones**. Speech rate 4.78
syl/s. The Azure error categories include `Monotone`, `UnexpectedBreak` and
`MissingBreak` — all three are tracked and none has moved.

### A2. Vocabulary, three ways
- **Produced:** 1,568 distinct content words across 4 hours of speech.
- **Perceived:** derived from 454 dictation attempts.
- **Surrounding:** **718 words** logged from **39 photographs** of my actual
  environment — the coverage report asks *which things I see every day can I
  still not name?*
- The ledger only ever grows: deleting a photo leaves coverage byte-identical,
  so frequency means "times I've met this word", not "photos still on disk".

### A3. Practice volume
1,798 single-word drill attempts, average **91**, with 1,116 at ≥90 and only 9
below 60. Most-drilled words: *several* (24 tries), *comb* (16), *cross* (13),
*bother* (12), *tied* (11), *stress* (10). Lowest averages despite repetition:
*what's* (71), *bean* (72), *finished* (73), *breathe* (73), *because* (73).

### A4. Listening ear-training (minimal pairs)
43 phoneme contrasts tested. Perfect (100%) on 30 of them. The ones that aren't:
**/e/ 64%**, /uː/ 67%, /ɒ/ 67%, /ʊ/ 75%, /iː/ 80%, /ɪ/ 80%, /ɔː/ 83%, /ʊə/ 86%,
/ð/ 88%. Note /iː/ and /ɪ/ both sit at 80% — and /iː/ was also my third-worst
*produced* sound (7.8%). Another produce/perceive pair.

### A5. Sources for every number
- `VideoAudioFiles/history.json` — 54-session timeline
- `VideoAudioFiles/**/*.result.json` — per-session grammar, word choice, blind
  spots, Azure word+phoneme data
- `VideoAudioFiles/progress.json` — drill scores (`ec_scores`), dictation errors
  (`dict_errors`), ear training (`ls_stats`), word ledger (`ec_seen`)

---

*Generated 2026-08-13 from 54 recordings spanning 2026-06-24 to 2026-08-13.*
