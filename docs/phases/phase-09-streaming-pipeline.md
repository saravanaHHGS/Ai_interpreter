# Phase 9 — Streaming Pipeline

**Status:** complete (built before Phase 8 by agreement: a UI is easier to
build over a working pipeline than over parts)
**Version:** 0.8.0

The components of Phases 3-7 become one live interpreter::

    CaptureSession ──▶ bounded queue ──▶ worker: STT ▶ MT ▶ TTS ▶ sink
         │ (speech onset)                                      │
         └────────────── sink.clear()  ◀───────────────────────┘  barge-in

---

## Verification

```powershell
# The reproducible path: a recording behaves exactly like a microphone
.\run.ps1 --wav recordings\<tamil-recording>.wav --out "CABLE Input"

# The live path: speak Tamil, hear English (default output = speakers)
.\run.ps1 --interpret 60

# Into Teams: translated speech becomes the virtual microphone
.\run.ps1 --interpret 300 --out "CABLE Input"
```

`--interpret` prints each transcript, its translation, and a per-utterance
latency line; the summary reports the EOU→first-audio distribution.

### Measured end to end (target machine, Tamil reference recording, ta→en)

```
[ta] என் பெயர் சரவணகுமார் இன்று நான் தமிழ் குரல் அடையாளம் ...
[en] My name is Saravanakumar Today I am checking if the Tamil voice
     recognition is working properly
     EOU->audio 6303 ms  (stt 4964 | mt 514 | tts 818)
```

**EOU→first audio: 5.9-6.3 s** for this direction today (measured under CPU
contention with a second recording process; ~1-2 s less when the pipeline
runs alone). Honest position: above the 2 s target, dominated by the Tamil
STT stage, which is exactly what Phase 10's chunked-streaming wiring and
tuning attack. Every stage number now comes from the pipeline itself rather
than from component benchmarks summed by hand.

### The closed loop

While that run played into ``CABLE Input``, a second process recorded
``CABLE Output`` — the endpoint a meeting application would use as its
microphone — and the capture was transcribed back:

```
heard at the cable's far end: "... I am checking if the Tamil voice
                               recognition is working properly ...
                               This software is my voice"
```

Tamil audio in, English speech out of the virtual microphone, confirmed by
reading the words back off the wire. (The mangled proper noun in the
read-back is the *verifying* recogniser mishearing synthesised speech, not a
pipeline defect — the pipeline's own transcript and translation were exact.)

---

## Design decisions

**Threads, not asyncio — a documented deviation from the original spec.**
Every stage is a blocking native call (CTranslate2, sherpa-onnx) that
releases the GIL, so threads already deliver all the parallelism that
exists. On the 2-core target the profile mandates a *serial* inference lane
anyway — three models contending for two cores was measured slower than
running them in sequence (Phase 4) — which reduces the pipeline to one
worker consuming one queue. An event loop would add machinery without adding
overlap. The parallel lane (a worker per stage, for 6+ core machines) is a
Phase 10 change inside `InterpretationPipeline` only.

**Drop-oldest backpressure.** The utterance queue is bounded
(`pipeline.queue_maxsize`); when the speaker outruns the machine, the oldest
unprocessed utterance is dropped and counted. Translating something said
eight seconds ago while the speaker continues is worse than skipping it.

**Barge-in clears audio, not text.** Speech onset while translated audio is
playing drops the sink queue — the interpreter stops talking over the
speaker — but the in-flight utterance's text still completes, because
transcripts remain useful as captions even when their audio was pre-empted.
The end-to-end run demonstrated this working: utterance 2's onset silenced
utterance 1's playback.

**Retries are bounded and stage-local; the pipeline never dies.** A stage
failure retries once (`pipeline.max_retries`) then drops the utterance with
an `on_error` event. A session that silently stops interpreting while the
UI shows "live" is the worst failure mode.

**Captions-only is a first-class mode.** When the target language has no
configured voice, `create_synthesizer` raises and `--interpret` degrades to
text output — the Phase 1 fallback, wired and tested rather than promised.

---

## Two lessons the first end-to-end run taught

**A WAV source must be paced.** `WavFileSource` originally delivered a
15-second file in under a second. Every consequence was misleading: queue
waits masqueraded as decode time (an apparent 9.9 s "STT" that was 8.6 s of
waiting), voice-activity states flapped at replay speed, and twelve
barge-ins fired in the first second. `realtime=True` now paces frames at
wall-clock speed, and `--interpret --wav` uses it always.

**Shutdown must drain by pending audio, not by a fixed timeout.** The first
run's translated speech never reached the far end of the cable: `stop()`
flushed the sink with the worker-join timeout, which expired long before
~9 s of queued 48 kHz audio could play out, and `close()` discarded the
rest. The flush wait is now sized to the sink's actual pending audio.

---

## Files

| Piece | File |
|---|---|
| `InterpretationPipeline`, `PipelineEvents`, `PipelineStats`, `UtteranceTiming` | `application/pipeline/interpretation.py` |
| `--interpret` / `--wav` | `cli_interpret.py` |
| Real-time pacing | `infrastructure/audio/capture/wav_file.py` |
| 18 orchestration tests (all-fake components) | `tests/unit/test_interpretation_pipeline.py` |

---

## What Phase 8 adds

The PySide6 interface over this pipeline: `PipelineEvents` maps directly
onto Qt signals, which is why the events object exists in the shape it does.
