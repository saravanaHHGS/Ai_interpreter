# Phase 3 — Audio Capture

**Status:** complete
**Version:** 0.3.0

Captures microphone audio, conditions it for the speech models, detects when
someone is speaking, and cuts the stream into utterances. Still no speech
recognition — but the input half of the pipeline is finished and testable.

---

## Verification

```powershell
.\run.ps1 --list-devices
.\run.ps1 --record 10
.\scripts\quality.ps1
```

`--record 10` runs the complete chain against your microphone, shows a live
level meter, and writes two WAV files to `recordings\`. **Speak a few short
sentences with clear pauses between them.** Expect one utterance per sentence.

Expected quality gates: ruff clean, mypy clean, **289 tests passing**.

---

## What runs, in order

```
Microphone (WASAPI, 48 kHz, 20 ms blocks)
  → driver callback: copy into a drop-oldest deque      ← real-time thread
  → resample 48 kHz → 16 kHz  (soxr, stateful)          ← worker thread
  → high-pass filter at 80 Hz (FIR, stateful)
  → gain
  → FrameAssembler: exact 512-sample frames
  → Silero VAD: speech probability per frame
  → UtteranceSegmenter: state machine
  → Utterance
```

---

## Files

### Infrastructure

| File | Purpose |
|---|---|
| `audio/buffers.py` | `AudioBlockBuffer` (callback hand-off), `FrameAssembler` (fixed frames) |
| `audio/dsp.py` | `StreamingResampler`, `HighPassFilter`, `AudioPreprocessor`, `apply_gain` |
| `audio/devices.py` | `SounddeviceDeviceEnumerator` with host-API preference |
| `audio/capture/microphone.py` | `MicrophoneSource` — PortAudio callback capture |
| `audio/capture/wav_file.py` | `WavFileSource` — replays a file through the same port |
| `audio/vad/silero.py` | `SileroVad` — neural detector via ONNX Runtime |
| `audio/vad/energy.py` | `EnergyVad` — adaptive fallback, no model needed |
| `audio/recording.py` | `WavRecorder` — incremental WAV writing |
| `models/registry.py` | `ModelRegistry` — reads `config/models.yaml` |
| `models/hf_repository.py` | `HuggingFaceModelRepository` — pinned downloads |

### Application and presentation

| File | Purpose |
|---|---|
| `application/services/utterance_segmenter.py` | The state machine. Pure logic |
| `application/services/capture_session.py` | Worker thread wiring the chain together |
| `presentation/console.py` | Level meter and device table rendering |
| `cli_audio.py` | `--list-devices` and `--record` |

---

## Dependencies added

| Package | Why | Alternative rejected |
|---|---|---|
| `sounddevice` | Ships the PortAudio binary; no compiler needed | `PyAudio` — requires building from source |
| `soundfile` | WAV read/write | `wave` — no float32 support |
| `soxr` | **Stateful streaming** resampling | `scipy.resample_poly` — restarts per chunk, producing boundary artefacts |
| `onnxruntime` | Runs Silero VAD; reused for TTS in Phase 6 | PyTorch — seconds to import, hundreds of MB |
| `huggingface-hub` | Model download with revision pinning | Hand-rolled URLs — unverifiable |

No SciPy. The one filter needed is a windowed-sinc FIR, which is about fifteen
lines of numpy; adding SciPy would cost roughly 60 MB for it.

---

## Findings from the target machine

These came out of running on the real hardware, not from planning.

### 1. Windows Smart App Control is enforced

`VerifiedAndReputablePolicyState = 1`. Unsigned native DLLs shipped inside pip
packages are **blocked by policy**, with the error *"An Application Control
policy has blocked this file."*

Two libraries were affected and both were designed around rather than fought:

* `pysilero-vad` — the obvious VAD package. Its bundled ggml DLLs are blocked,
  so it is not used at all. Silero runs through ONNX Runtime instead, which
  loads fine and is needed for Phase 6 regardless. One runtime, not two.
* `hf_xet` — `huggingface-hub`'s download accelerator. `HuggingFaceModelRepository`
  detects the broken import and disables Xet, falling back to plain HTTPS.

Verified as **working** under the same policy: `onnxruntime`, `sounddevice`,
`soundfile`, `soxr`, and — checked ahead of Phase 4 — `ctranslate2`, which
also reports `int8` support on this CPU. The Phase 4 plan is therefore safe.

**Smart App Control is not disabled.** It cannot be re-enabled afterwards
without reinstalling Windows, and everything needed works within it.

### 2. MME truncates device names at 31 characters

The same microphone appears as:

| Host API | Reported name |
|---|---|
| MME | `Internal Microphone (Conexant I` |
| WASAPI | `Internal Microphone (Conexant ISST Audio)` |

Configuration stores device *names*, because indices change whenever a USB
headset is unplugged. A truncated name silently fails to match, so the
enumerator prefers WASAPI and `audio.input.host_api` defaults to `WASAPI`.

This also caused a real bug: Windows nominates the **MME** endpoint as the
system default, and the code that upgraded the default to a better host API
compared names with `==`. The truncated and full names never matched, so it
kept selecting MME. Matching is now by prefix in either direction, with a
12-character minimum so generic names like `Speakers` are not merged.

### 3. Silero VAD cost, measured

**0.5 ms per 32 ms frame — 1.6 % of one core** on the i5-7200U, single ONNX
thread. Negligible, which is what justifies running it on every frame.

---

## Two real bugs the tests caught

### Clipped utterance starts

Frames captured *while confirming* that speech had started were being stored
in the pre-roll ring buffer. That ring is capped at `pre_roll_ms`, so whenever
`min_speech_ms` exceeded `pre_roll_ms`, the ring discarded real speech and
**the beginning of every utterance was clipped**.

With the shipped defaults (`min_speech_ms: 250`, `pre_roll_ms: 300`) the bug
would rarely have shown; anyone tuning those values would have hit it, and the
symptom — recognition quietly getting worse — is nearly impossible to trace.

Onset frames are now held in a separate, uncapped buffer and combined with the
pre-roll when the utterance begins. Pinned by
`test_onset_frames_survive_a_small_pre_roll`.

### The energy detector could blind itself

Its noise-floor estimate adapted at a rate scaled by `1 - speech_probability`.
Started in a permanently noisy room, every frame scored 1.0, which zeroed the
adaptation rate, which kept every frame at 1.0. It would report speech forever
and never recalibrate.

A small residual adaptation rate (2 %) now applies even during speech: fast
enough to recalibrate over a couple of minutes, far too slow to be dragged
along by a sentence. Pinned by two tests asserting both halves of the
trade-off.

---

## Design decisions

**The driver callback only copies.** It appends one array to a
`collections.deque` with `maxlen`. Under CPython that append is atomic, so no
lock is taken, and `maxlen` gives drop-oldest overflow for free. Dropping the
*oldest* audio is deliberate: in a live conversation, audio from four seconds
ago has no value, and keeping it only adds latency.

**Filters and the resampler are stateful.** A filter restarted at every chunk
boundary produces a discontinuity at each one — an audible buzz at the chunk
rate and a measurable accuracy loss. `test_chunked_filtering_matches_whole_stream`
asserts chunked output is bit-identical to filtering the whole stream.

**The detector declares its own frame size.** Silero requires exactly 512
samples at 16 kHz. Rather than hard-coding that in the pipeline, the port
exposes `required_frame_samples` and `FrameAssembler` obeys it. A detector
with different requirements changes no other code.

**Trailing silence is kept in the utterance.** Speech recognisers use it to
decide the last word has ended; trimming it costs accuracy on the final word
of every sentence.

**Model revisions are pinned to full commit hashes.** `config/models.yaml`
records `e71cae9660...`, not `main`, so upstream cannot silently change the
weights being run.

---

## Latency contributed

| Stage | Cost |
|---|---|
| Driver block | 20 ms |
| Resampling | < 1 ms |
| High-pass group delay | 3.1 ms |
| Frame assembly | up to 32 ms |
| Silero inference | 0.5 ms |
| **End-of-speech detection** | **350 ms** (`vad.min_silence_ms`) |

The endpoint delay dominates everything else by two orders of magnitude, which
is why it is the first thing to tune in Phase 10.

---

## What Phase 4 adds

Speech to text: `faster-whisper` with CTranslate2 int8, streaming chunked
transcription, timestamps, confidence, partial and final results, and a
latency benchmark. `ctranslate2` is already verified to load on this machine.

The `SpeechRecognizer` and `StreamingSpeechRecognizer` ports it implements
already exist in `domain/ports.py`.
