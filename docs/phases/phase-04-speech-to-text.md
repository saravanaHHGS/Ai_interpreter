# Phase 4 — Speech to Text

**Status:** complete
**Version:** 0.4.0

Turns utterances into text, with timestamps, confidence and measured latency.
This is the phase where estimates were replaced by measurements, and the
measurements changed the design.

---

## Verification

```powershell
.\run.ps1 --benchmark                       # measure decode time on this machine
.\run.ps1 --record 10                       # speak a few sentences
.\run.ps1 --listen 20                       # live speech to text
.\run.ps1 --transcribe recordings\<file>.wav --language en
.\scripts\quality.ps1
```

First run downloads Whisper `base` (~145 MB). Expected gates: ruff clean,
mypy clean, **351 tests passing**.

---

## Measured on the target machine

Intel i5-7200U, 2 physical cores, int8, warmed up, decoding **one utterance**:

| Model | 1 thread | 2 threads | 4 threads | Quality |
|---|---:|---:|---:|---|
| `tiny` | — | 0.85 s | — | "What's *true, you work, man?*" ❌ |
| **`base`** | 2.28 s | **1.66 s** | 2.07 s | "What's today your plan?" ✅ |
| `small` | — | 5.88 s | 7.18 s | ✅ but unusable |

**Speech-to-text delay = 350 ms endpoint + 1.66 s decode ≈ 2.0 s**, before
translation and speech synthesis are added.

---

## Three findings that changed the design

### 1. Decode cost is constant, not proportional to audio length

Same model, varying input:

| Audio | Decode |
|---|---|
| 1.0 s | 0.82 s |
| 5.0 s | 0.84 s |
| 9.9 s | 0.96 s |

Whisper pads its mel spectrogram to a fixed **30-second window**, so the
encoder does identical work regardless of utterance length, and on CPU the
encoder dominates.

**Consequence: chunked streaming cannot reduce end-of-utterance latency on
Whisper.** Phase 1 planned to decode in 1-second chunks so that only the final
chunk remained at end of speech, and estimated this was worth about a second.
It is worth nothing: each chunk costs a *full* encoder pass, so decoding every
second would multiply total work rather than spreading it.

Streaming is still implemented, because live captions during a long sentence
are genuinely useful — but it is documented everywhere as buying **feedback,
not speed**, and it is off by default on CPU profiles. The
`StreamingSpeechRecognizer` port is kept because a model whose cost scales
with input length — Parakeet and other Conformer architectures — would make
the same interface a latency win with no change to any caller.

### 2. More threads made it slower

`base` took 1.66 s on 2 threads and 2.07 s on 4. The CPU has two physical
cores; hyperthread siblings share execution units and cache, and CTranslate2's
matrix kernels lose more to that contention than they gain. `stt.cpu_threads`
should track **physical** cores, and the benchmark sweeps thread counts so
this is verifiable rather than assumed.

### 3. Phase 1's estimate was off by roughly eight times

Phase 1 projected Whisper `small` at a 0.55 real-time factor on this class of
CPU, implying ~1.5 s for a 3-second utterance. Measured: 5.88 s. That is why
`--benchmark` is a first-class command rather than a one-off script, and why
`cpu_low` now defaults to `base` rather than `small`.

---

## Two real bugs found by running it

### Warmup took 56 seconds

`warmup()` called the model directly instead of going through this class's
decode path, so it silently skipped every decode option. faster-whisper's
default temperature fallback then applied — up to six decoding passes — and on
silence it took the worst case every time.

Routing warmup through the same `_decode()` as real transcription cut it to
**4.6 s, then 1.5 s once the model file was in the OS cache**. Pinned by
`test_warmup_uses_the_configured_decode_options`.

### `--language en` was silently ignored

The segmenter tags each utterance with the configured source language, and
`transcribe()` treats an utterance's own tag as more specific than the
recogniser's default. Correct in principle, but it meant the command-line
override never reached the decoder: English audio was decoded as Tamil,
producing fluent Tamil gibberish at confidence 0.47.

`create_segmenter()` now accepts a language, and every caller that overrides
the language overrides it for the whole chain.

---

## Tamil accuracy: measured against a written reference

A native speaker recorded a known Tamil sentence, so the output could be
scored rather than guessed at.

**Reference:** `என் பெயர் சரவணகுமார். இன்று நான் தமிழ் குரல் அடையாளம் சரியாக
செயல்படுகிறதா என்று சோதித்து வருகிறேன்.`

| Model | Decode | Output | Word error |
|---|---:|---|---:|
| `whisper-base` | 1.6 s | `என் பேர்தாராவனக்குமாக இருண்டு நான் தமிழ்க்குரலாடையாக தேல்பெடிரதாயின்று` | ~80 % |
| `whisper-small` | 5.2 s | `என் பேசராவனக்கு மாறி என்று நான் தமிழ் குரலாடை … சோதித்து வருங்கிறேன்` | ~50 % |
| **`whisper-tamil-small`** | **6.2–10.5 s** | `என் பேர் சரவணகுமார் இன்று நான் தமிழ் குரல் அடையாளம் சரியாக செயல்படுகிறதா என்று சோதித்து வருகின்றேன்` | **~0–8 %** |

Generic Whisper is simply weak at Tamil, at any size available on this
hardware. The fine-tune recovers essentially all of it, including the proper
noun `சரவணகுமார்`, which neither generic model came close to.

**The accuracy problem is solved. The remaining problem is purely speed**, and
speed has known solutions where accuracy did not.

`whisper-tamil-small` decodes slightly slower than the generic model of the
same size, for a slightly surprising reason: a *correct* transcript contains
more tokens than a truncated wrong one, and decoder time scales with tokens.

### Language support is now enforced

The Tamil fine-tune has lost its multilingual ability. Asking it for English
does not fail inside the model - it returns confident nonsense. The model
registry therefore declares `languages: ["ta"]`, and the recogniser raises
`TranscriptionError` rather than decoding a language the checkpoint was not
trained on.

---

## What the confidence score is, and is not

Confidence is `exp(avg_logprob)` — the geometric mean of per-token
probabilities. Measured on the same recording:

| Condition | Confidence |
|---|---|
| English audio, decoded as English (correct) | 0.33 – 0.61 |
| English audio, forced to Tamil (gibberish) | 0.00 – 0.47 |

The ranges **overlap**. Confidence is a useful relative signal — the worst
transcript scored lowest in both conditions — but it does not cleanly separate
right from wrong, and it must not be presented as a probability that the
transcript is correct. `stt.min_confidence` exists to suppress obvious noise
and defaults to `0.0` (disabled) because a safe threshold has not been
established.

---

## Files

| File | Purpose |
|---|---|
| `infrastructure/stt/faster_whisper.py` | `FasterWhisperRecognizer`, decode options, confidence |
| `cli_stt.py` | `--transcribe`, `--listen`, `--benchmark` |
| `config/models.yaml` | Five Whisper models, pinned to full commit hashes |
| `tests/unit/test_faster_whisper.py` | 46 tests against a fake model, no weights needed |

Tests inject a fake CTranslate2 model. Downloading 145 MB and spending 1.7 s
per decode would make the suite unusable, and none of the behaviour under test
belongs to the model: it is the conversion into domain transcripts, the
confidence calculation, and error handling. Real model behaviour is measured
by `--benchmark`.

---

## Dependencies added

| Package | Why |
|---|---|
| `faster-whisper` | Whisper on CTranslate2: int8 quantisation, low memory, verified to load under Smart App Control in Phase 3 |

`ctranslate2`, `av` and `tokenizers` arrive as its dependencies.

---

## Deviation from the original specification

The specification asked for **NVIDIA Parakeet first, Whisper as fallback**.
Parakeet is English-only (Phase 1, §1.1) and the required pairs are Tamil and
Hindi to English, so Whisper is the primary engine.

The measurements have now produced a concrete argument for Parakeet that
Phase 1 did not have: Parakeet is a Conformer model whose cost **scales with
input length** rather than padding to 30 seconds. On this CPU a 2-second
utterance would plausibly decode several times faster than Whisper's fixed
1.66 s, and streaming would become a real latency win. That makes
Parakeet-for-English a strong candidate for Phase 10, alongside OpenVINO on
the Intel integrated GPU — which attacks the same bottleneck, the encoder,
and would also help Tamil and Hindi.

---

## Latency position after Phase 4

| Stage | Measured | Notes |
|---|---:|---|
| End-of-speech detection | 350 ms | `vad.min_silence_ms` |
| Speech to text | 1660 ms | `base`, 2 threads |
| **Subtotal** | **~2.0 s** | Translation and synthesis still to come |

The 2-second target will not be met on this CPU with `base`. The honest
options are Parakeet or OpenVINO for English, `tiny` at a real accuracy cost,
or a GPU. Phase 5 measures translation next, so the full picture is known
before any of them is chosen.

---

## Phase 4b — the streaming-capable recognisers

Added after the CPU-only and true-streaming constraints were fixed, and after
the model experiment (see `docs/architecture.md` section 0) found engines
whose cost scales with audio length.

### What was built

| Piece | File | Purpose |
|---|---|---|
| `SherpaNemoCtcRecognizer` | `infrastructure/stt/sherpa_nemo.py` | Tamil IndicConformer. Whole-utterance decode plus **chunked incremental streaming** |
| `SherpaNemoStreamingRecognizer` | same | NeMo streaming English conformer, natively incremental |
| `ensure_onnx_metadata` | `infrastructure/stt/onnx_metadata.py` | Stamps sherpa-required metadata into a copy of exports that ship without it |
| Runtime dispatch | `app/container.py` | One factory; the registry entry's `runtime` field selects CTranslate2 vs sherpa-onnx |

### The chunked commitment algorithm

The piece that converts a linear-cost model into low end-of-utterance
latency. Audio buffers until one chunk (4 s) plus a safety margin (0.8 s) is
held; the buffer is decoded; tokens whose timestamps fall safely before the
right edge are **committed at a word boundary** (the sentencepiece `▁`
marker) and their audio is discarded. Each decode therefore covers roughly
one chunk, and at end of utterance only the final chunk remains — a tail of
about `chunk x RTF ≈ 2 s`, versus 6–10 s for a whole-utterance Whisper
decode.

The margin exists because CTC output near the right edge has not seen its
right context and is still allowed to change; committing it would bake in
errors. Commitment only at `▁` boundaries prevents a Tamil word from being
split across two partials.

### Verified end to end on the reference recording

```
Model      indicconformer-ta (int8)     Warmup  2.7 s
[1] 6.72 s -> 4.6 s   என் பெயர் சரவணகுமார் இன்று நான் தமிழ் குரல்
                      அடையாளம் சரியாக செயல்படுகிறதா என்று சோதித்து வருகின்றேன்
[2] 1.73 s -> 1.2 s   இந்த மென்பொருள் என் குரலை        <- exact
```

Utterance 2 is an exact match to the written reference; utterance 1 differs
only in a colloquial verb ending. `--transcribe` uses the whole-utterance
path; the chunked streaming path is wired into the live pipeline in Phase 9.

### Notes

* Sherpa's greedy CTC exposes no token probabilities, so these transcripts
  carry an availability flag (1.0/0.0), not a model-certainty estimate —
  documented on the adapter, and consistent with Phase 4's finding that even
  real confidence values invert on failure modes.
* Smart App Control's cloud verdict on mypy's compiled `librt` runtime
  flipped mid-project, blocking a tool that worked hours earlier. mypy is now
  pinned to 1.14.1 and installed from source (pure Python, no DLL to block);
  `bootstrap.ps1` does this automatically.

---

## What Phase 5 adds

Translation: IndicTrans2-200M for Tamil and Hindi to English, behind a
`Translator` port with pluggable providers, plus a translation cache and the
same style of measured benchmark.
