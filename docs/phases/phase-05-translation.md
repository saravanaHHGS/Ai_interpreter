# Phase 5 — Translation

**Status:** complete
**Version:** 0.5.0

Tamil ↔ English machine translation with IndicTrans2-200M on CTranslate2,
behind the `Translator` port, with an LRU cache. Measured, verified in both
directions, and fast enough that translation is **not** the latency problem.

---

## Verification

```powershell
.\run.ps1 --translate "நாளைக்கு என்ன திட்டம்?"
.\run.ps1 --translate "The meeting will start in five minutes." --source en --target ta
.\scripts\quality.ps1
```

First use of each direction downloads its model (~850 MB, quantised to
~220 MB of RAM at load). Expected gates: ruff clean, mypy clean, **459 tests**.

---

## Measured on the target machine (int8, 2 threads, beam 4)

| Input | Output | Time |
|---|---|---:|
| என் பெயர் சரவணகுமார் | My name is Saravanakumar. | 251 ms |
| இன்று நான் தமிழ் குரல் அடையாளம் சரியாக செயல்படுகிறதா என்று சோதித்து வருகிறேன் | Today I am checking if the Tamil voice recognition is working properly | 381 ms |
| நாளைக்கு என்ன திட்டம்? | What's the plan for tomorrow? | 188 ms |
| The meeting will start in five minutes. Can you hear me clearly? | கூட்டம் இன்னும் ஐந்து நிமிடங்களில் தொடங்கும். நீங்கள் தெளிவாக கேட்க முடியுமா? | 819 ms |
| *any repeated phrase* | *(cache hit)* | **0.03 ms** |

**0.19–1.3 s per sentence, both directions.** Warmup 5–7 s once at startup.

Beam search is affordable here, unlike in speech recognition: a 200M text
model at beam 4 still answers in well under a second, and the quality gain on
morphologically rich Tamil is worth having. `translation.beam_size: 4` on
every profile.

---

## The bug that produced fluent garbage first

The first live attempt translated `என் பெயர் சரவணகுமார்` as *"En Beyer
Saravandakumar"*. Speed was fine; output was nonsense.

Diagnosis: the sentencepiece model produced **one character per piece** for
raw Tamil. IndicTrans2's tokenizer was trained on text with **all Indic
scripts unified into Devanagari** — feed it native Tamil and the encoder sees
a sequence it never saw in training.

The fix is nearly free: the major Indic script blocks in Unicode are
deliberately aligned (ISCII heritage), so Tamil `க` (0x0B95) and Devanagari
`क` (0x0915) differ by a constant offset. `transliteration.py` shifts
codepoints in ~10 lines — the same thing the official IndicProcessor does via
indic-nlp-library, without dragging in pandas and morfessor as dependencies.
English→Indic output arrives in Devanagari and is shifted back.

Pinned by `test_tamil_source_is_transliterated_to_devanagari`. Urdu is
excluded (`supports()` returns false): Perso-Arabic script has no aligned
block.

A second, smaller self-inflicted bug: the en-indic registry entry initially
carried a **guessed** full revision hash (extended by hand from a 12-character
prefix), which 404'd on first download. Revision hashes are now always copied
from `HfApi().repo_info(...).sha`, never reconstructed.

---

## What was built

| Piece | File | Purpose |
|---|---|---|
| `to_devanagari` / `from_devanagari` | `infrastructure/translation/transliteration.py` | Script unification by Unicode block offset |
| `IndicTrans2Translator` | `infrastructure/translation/indictrans2.py` | CT2 translator: NFC → transliterate → SPM → tags → decode → detokenise |
| `LruTranslationCache` | `infrastructure/translation/cache.py` | Bounded LRU, thread-safe, normalised keys, hit-rate metric |
| `CachedTranslator` | `application/services/cached_translator.py` | Cache as a decorator over the `Translator` port — the pipeline cannot tell it is there |
| `create_translator(pair)` | `app/container.py` | Direction → registry model → engine → cache wrapper |
| `--translate` | `cli_translate.py` | One-shot translation with cold/repeat timings |

One checkpoint per direction (`indictrans2-indic-en`, `indictrans2-en-indic`),
mapped in `translation.models`; the container picks by `pair.source`.

### Why CTranslate2 and not the reference implementation

Constraint C6: PyTorch is blocked outright on the target machine. The
`adalat-ai` CT2 exports of the distilled 200M checkpoints run on the runtime
already proven for Whisper, with sentencepiece (verified to load under Smart
App Control) for tokenisation. The ONNX exports by `hari31416` remain a
documented fallback should the CT2 repos disappear.

### Deliberately not implemented

* **Placeholder wrapping** of URLs/emails/long numbers (the official
  IndicProcessor preserves them byte-for-byte). Spoken utterances rarely
  contain them; revisit if numeric mangling appears.
* **Cache persistence** (`translation.cache.persist`). Privacy defaults keep
  conversation text off disk; an in-memory cache dies with the session,
  which is exactly what that default promises.

---

## Dependencies added

| Package | Why |
|---|---|
| `sentencepiece` | IndicTrans2 tokenisation — CT2 consumes token strings, not ids |

---

## Latency position after Phase 5

| Stage | Measured (Tamil → English) |
|---|---:|
| End-of-speech detection | 350 ms |
| STT tail (chunked streaming, Phase 9 wiring) | ~2000 ms |
| Translation | **190–740 ms** (0 ms on cache hit) |
| TTS first chunk | Phase 6 |

Translation is comfortably the cheapest model stage, as Phase 1's analysis
predicted — the one prediction the measurements have confirmed rather than
overturned.

---

## What Phase 6 adds

Text to speech: Piper for English (fast CPU synthesis), and the project's
biggest flagged risk — a usable **Tamil** voice — investigated head-on, with
on-screen captions as the designed fallback if none exists.
