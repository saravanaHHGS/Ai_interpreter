# Code-switching: recognising Tamil that mixes English

The target experience is ChatGPT-Voice-like: natural Tamil freely mixing
English technical terms and product names ("VALD assessment முடிஞ்சுச்சு",
"NordBord data sync ஆயிடுச்சு"), recognised as spoken. This document records
why no single model can deliver that on the target machine, and the layered
architecture that delivers it instead.

## Why no single model

Every candidate was evaluated against five criteria (Tamil accuracy,
Tamil-English code-switch accuracy, technical-term recognition, CPU
performance, streaming latency) under the hard constraints of
`architecture.md` §0 - CPU-only on 2 cores (C1), PyTorch blocked by Smart
App Control (C6), sub-2-second latency (C5):

| Model | Disqualifier |
|---|---|
| Whisper large-v3 | ~25-40 s per utterance on this CPU (extrapolated from measured 0.85/1.66/5.88 s tiny/base/small scaling) |
| Whisper large-v3-turbo | ~15-20 s per utterance, same reasoning |
| NVIDIA Parakeet multilingual | No Tamil (English + 25 European languages) |
| NVIDIA Canary | No Tamil (European languages) |
| Moonshine | English-only |
| Meta MMS-ASR / SeamlessM4T | PyTorch-only (C6) |

The genuinely code-switch-capable models are GPU-class; the CPU-class models
are monolingual. The gap is structural, not a tuning problem.

## The layered answer

Four mechanisms, ordered model-first - post-processing is the last resort,
not the strategy:

**1. Hotword biasing** (`stt.hotwords`, `faster_whisper.build_initial_prompt`).
The user's terms are fed to Whisper as a decoder-biasing prompt, so the
model recognises them *directly*. Measured: "game plan / Nordboard / Force
Frame" became "GamePlan / NordBord / ForceFrame" exactly, at no latency
cost. Glossary canonical terms are merged in automatically.

**2. Word fusion** (`transcript_fusion.py`) - the repair for MIXED
sentences. Both recognisers decode the same utterance; the phonotactic
detector (`code_switch.py`) marks which Tamil-script words are English in
disguise; those words are replaced by the English recogniser's words *from
the same time window*. Time is the join key - both models timestamp against
the same audio. Measured on a live recording:

```
conformer: மேட்சிங் மட்டும் பெண்டிங் ல இருக்கு   (Tamil right, English transliterated)
whisper:   Matching Mutum Bending.                (English right, Tamil garbled)
fused:     Matching மட்டும் Bending ல இருக்கு     (each side's good half)
output:    "Matching is the only thing that is in bending."
```

The base transcript always wins by default: a flagged region with no
English words inside its window keeps its Tamil. Two alignment facts,
both measured, shape the matcher:

- sherpa-onnx converts the sentencepiece `▁` marker into a literal leading
  space in `result.tokens`; matching only `▁` finds no word boundaries at
  all on the real model.
- Whisper smears word *onsets* backward through leading audio ("World" at
  0-760 ms against the conformer's வேர்ல்ட் at 400-840 ms) while word
  *ends* are anchored by the next word's onset - so a word is placed by
  its end, with a mostly-inside overlap test as the fallback.

**3. Wholesale reroute** - the repair for FULLY ENGLISH sentences spoken
into the Tamil model. Unchanged from before, but now gated by the *native
anchor* test: what separates the mixed "வேர்ல்ட் அஸிஸ்மெண்ட் முடிஞ்சிடுச்சு"
(score 0.67) from the fully-English "வி நீட் டூ சால்வ் தத்" (score 0.60)
is not the score - it is முடிஞ்சிடுச்சு, a long unflagged native word.
Sentences with an anchor keep their Tamil and take fusion; sentences
without one are replaced wholesale and skip translation.

**4. Glossary** (`translation.glossary`) - the user-editable last resort,
applied after the models have had their say. It now also repairs the
English recogniser's mishearings inside fused text (e.g. `pending:
["Bending"]`).

## The dual-stream redesign (2026-08-04)

The serial repair had an architectural flaw: mixed sentences - the whole
point of this system - were the *slowest*, because Whisper re-decoded the
entire utterance after the conformer finished (+1.7-2.6 s, serial).

The redesign runs a second STREAMING recogniser (`nemo-streaming-en`,
measured RTF 0.11 - near-free) on every utterance, live, in parallel with
the conformer. At end-of-utterance the English view already exists, so
fusion costs ~0 ms instead of a re-decode. Whisper is demoted to the
wholesale-English reroute, where its hotword biasing matters - and when it
is unavailable (Smart App Control blocked its DLLs again the day this
shipped, proving the point), the partner's text serves the reroute too.

Three measured facts shape the implementation:

- The partner drops the LAST WORD of an utterance unless ~1 s of silence
  is fed before ``input_finished`` (its chunk lookahead).
- Its timestamps run on its own frame clock, ~30% slow with growing drift;
  fusion linearly maps its word-onset span onto the primary's
  (``align_clock=True``), verified word-by-word on live fixtures.
- After rescaling, derived word ENDS can drift into the next region, so
  clock-aligned matching uses mostly-inside overlap only; the end-in-region
  shortcut remains exclusively for Whisper's smeared onsets.

Open question, deliberately left to live measurement: on heavily contended
replays the partner sometimes garbles product names ("aman" where Whisper
heard "World assessment") - CTC has no hotword biasing. If live sessions
show the same, the next step is a vocabulary guard on partner splices.

Also fixed here: the ``cpu_low``/``cpu_high`` profiles still carried
``streaming: false`` and short ``min_silence_ms`` overrides from the
Phase 4 Whisper era - silently disabling chunked streaming, the partner,
and the anti-fragmentation endpoint in every real session. Profile
overrides now exist only where a hardware reason justifies them.

## Cost

Pure Tamil pays nothing (no flags, no English decode). A flagged utterance
pays one extra Whisper-base decode (~1.7-2 s, +10% for word timestamps),
which the user accepted in exchange for mixed-language recognition quality.

## Verification

`tests/unit/test_transcript_fusion.py` replays the live recording's exact
word timings; `test_interpretation_pipeline.py::TestWordFusion` proves the
pipeline routing; the end-to-end behaviour was validated against the
recording `recordings/20260729-120325-processed-16000.wav`, where all six
utterances took the correct path (2 fusions, 1 reroute, 2 untouched pure
Tamil, 1 fragment).
