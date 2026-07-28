# Phase 6 — Text to Speech

**Status:** complete
**Version:** 0.6.0

Speech synthesis for both directions, through sherpa-onnx's VITS engine — the
runtime already installed and proven under Smart App Control. This phase
contained the project's biggest flagged risk (a Tamil voice on CPU); the
answer is *yes, with a caveat measured honestly below*.

---

## Verification

```powershell
.\run.ps1 --speak "The meeting will start in five minutes." --language en
.\run.ps1 --speak "கூட்டம் ஐந்து நிமிடங்களில் தொடங்கும்." --language ta
.\scripts\quality.ps1
```

Each command reports first-chunk latency, saves a WAV to `recordings\`, and
plays the audio. **Listen to the saved files** — voice quality is judged by
ear, not by RTF. Expected gates: ruff clean, mypy clean, **482 tests**.

---

## Measured on the target machine (2 threads, warm, via `--speak`)

| Voice | First chunk | RTF | Sample rate | Licence |
|---|---:|---:|---|---|
| `piper-en-lessac` (English default) | **597 ms** | **0.29** | 22.05 kHz | MIT |
| `piper-en-amy-low` (faster alternative) | — | **0.18** | 16 kHz | MIT |
| `mms-tam` (Tamil) | **3528 ms** | **1.03–1.47** | 16 kHz | **CC-BY-NC-4.0** |

### The Tamil caveat, stated plainly

MMS is the **only Tamil voice that runs under constraint C6** — every
AI4Bharat alternative is PyTorch-only. At RTF ≈ 1.0–1.5 it generates speech
about as fast as it plays, or slower:

* First-chunk latency is ~3.5 s for a normal sentence.
* Continuous multi-sentence Tamil speech will have gaps.

Quantisation was tried and measured **worse** (RTF 4.2): this CPU lacks VNNI,
so int8 convolutions fall back to slow paths. The failed quantised model was
deleted rather than shipped.

The primary direction — Tamil speech in, **English** speech out — is
unaffected: Piper answers in ~0.6 s. The Tamil voice serves the reverse
direction, and the Phase 1 caption fallback remains the designed answer when
its latency is unacceptable: `create_synthesizer` raises a clear
`ConfigurationError` when a language has no voice configured, and the Phase 9
pipeline treats that as "captions instead of audio", not as a crash.

### Licence

`mms-tam` is **CC-BY-NC 4.0 — non-commercial use only**. Recorded in the
model registry (`license:` field, new this phase), logged as a warning every
time the voice is constructed, and listed in `deployment.md`. Nothing else in
the shipping configuration is licence-restricted.

---

## What was built

| Piece | File | Purpose |
|---|---|---|
| `SherpaVitsSynthesizer` | `infrastructure/tts/sherpa_vits.py` | One voice per instance; both synthesizer ports; sentence-streamed chunks with `chunk_index`/`is_last` |
| `split_sentences` | same | Latin + danda terminators; decimals do not split |
| `create_synthesizer(language)` | `app/container.py` | Voice lookup from `tts.voices`, snapshot or file download, espeak data detection, NC-licence warning |
| Snapshot downloads | `models/hf_repository.py` | A descriptor with no `files` list downloads the whole repo — Piper bundles hundreds of espeak-ng data files |
| `license` field | `domain/entities.py`, registry | Restrictions surface in config and logs instead of at distribution time |
| `--speak` | `cli_tts.py` | First-chunk latency, WAV save, playback |

**Why sherpa-onnx rather than the `piper-tts` package:** piper-tts ships
`piper_phonemize` as another native DLL for Smart App Control to block, and
its espeak integration duplicates what sherpa already bundles per-voice. One
engine now serves STT *and* TTS. Kokoro remains a documented upgrade path —
sherpa supports it — pending measurement on capable hardware.

---

## Latency position after Phase 6 — every stage now measured

| Stage | Tamil → English | English → Tamil |
|---|---:|---:|
| End-of-speech detection | 350 ms | 350 ms |
| STT | ~2000 ms tail (chunked IndicConformer) | 1660 ms (Whisper `base`) |
| Translation | 190–740 ms | 370–1330 ms |
| TTS first chunk | **~600 ms** | **~3500 ms** |
| **Total (est.)** | **~3.1–3.7 s** | **~5.9–6.8 s** |

Honest assessment against the <2 s target: not yet met in either direction,
but every number is now measured, and the two dominant costs are known —
Tamil STT tail (Phase 9 wiring + Phase 10 tuning) and Tamil TTS (a model
gap with no current CPU fix). The Phase 9 pipeline overlaps stages, which
these sequential sums do not yet reflect.

---

## What Phase 7 adds

The virtual microphone: routing synthesised speech into VB-CABLE so Teams,
Zoom and Meet hear it — the `AudioSink` port's first real adapter, and the
step that turns these components into an interpreter.
