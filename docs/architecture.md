# Architecture

This document describes how AI Interpreter is put together and, more
importantly, *why*. It is updated at the end of every phase.

---

## 1. The problem

Translate live speech and deliver it into a meeting application with as little
delay as possible, on ordinary Windows hardware, without sending audio to a
cloud service.

Three constraints shape every decision that follows:

1. **Latency is the product.** A translation that arrives four seconds late is
   not a slow feature; it is an unusable one, because the conversation has
   moved on.
2. **The hardware is unknown.** The same code must run on a two-core laptop
   with integrated graphics and on an RTX workstation.
3. **The audio is confidential.** Meeting content must not leak into log
   files, telemetry, or a cloud service.

---

## 2. Layers

```
┌──────────────── PRESENTATION ────────────────┐
│ PySide6 windows, view models, theme          │  Phase 8
│ Renders state. Emits commands. No logic.     │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──── APPLICATION ─────────┐
│ InterpretationPipeline, SessionController,   │  Phase 9
│ DeviceService, ModelManager, ProfileSelector │
│ Depends on PORTS only.                       │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──── DOMAIN ──────────────┐
│ Entities, value objects, ports, errors       │  Phase 2 (done)
│ Standard library + numpy. Nothing else.      │
└───────────────────▲──────────────────────────┘
                    │ implements
┌───────────────────┴──── INFRASTRUCTURE ──────┐
│ Audio drivers, ML runtimes, config, logging  │  Phases 2-7
└──────────────────────────────────────────────┘
```

**The dependency rule: arrows point inward.** Domain knows nothing about the
outside world. Infrastructure depends on domain, never the reverse.

Why it is worth the extra files: swapping Whisper for Parakeet, or Piper for a
Tamil engine, touches exactly one class in `infrastructure/`. Nothing in
`application/` or `presentation/` changes, and the test suite still runs
without a microphone, a GPU or a model download.

### Deviation from the usual rule

The domain layer imports `numpy`. Audio buffers cannot be `list[float]` in a
real-time application. `numpy` is treated as a primitive data type in the same
spirit as `decimal` — not as a framework. It is the only exception.

---

## 3. Ports

Every port is a `typing.Protocol`, so adapters satisfy them *structurally* —
by having the right methods, without inheriting anything.

| Port | Responsibility | Phase |
|---|---|---|
| `DeviceEnumerator` | Discover audio endpoints | 3 |
| `AudioSource` | Deliver captured frames | 3 |
| `VoiceActivityDetector` | Score a frame for speech | 3 |
| `SpeechRecognizer` | Utterance to text | 4 |
| `StreamingSpeechRecognizer` | Interim results while speaking | 4 |
| `Translator` | Text to text | 5 |
| `SpeechSynthesizer` | Text to audio | 6 |
| `StreamingSpeechSynthesizer` | Audio per sentence | 6 |
| `AudioSink` | Write audio to a device | 7 |
| `SettingsRepository` | Persist user overrides | 8 |
| `ModelRepository` | Download and verify weights | 4 |
| `TranslationCacheRepository` | Cache repeated phrases | 10 |
| `SessionHistoryRepository` | Optional conversation history | 8 |

### Why capabilities are split

`SpeechSynthesizer` and `StreamingSpeechSynthesizer` are separate protocols
(Interface Segregation). An engine that cannot stream implements the first
only; the pipeline checks with `isinstance` and takes the faster path when it
is available.

This is what makes the Tamil text-to-speech gap survivable. If no free Tamil
voice with acceptable latency exists, a `NullSynthesizer` reports
`supports(ta) == False`, the pipeline degrades to on-screen captions, and
adding a real Tamil engine later means writing **one class** and **one line**
in the composition root. No other file changes.

---

## 4. Composition root

`src/ai_interpreter/app/container.py` is the only place in the codebase that
names a concrete implementation. Dependency injection is done by hand.

**Why not a DI framework?** `dependency-injector` and `punq` resolve the graph
at runtime through reflection, so mypy cannot verify it and a wiring mistake
appears as an exception minutes into a session. A hand-written root is fully
statically checked and reads top to bottom. The benefits people actually want
from DI — swappable implementations, testable components — come from the
ports, not from the container.

Startup order is deliberate:

| # | Step | Why here |
|---|---|---|
| 1 | Paths | Nothing can be read before we know where things are |
| 2 | Hardware | Profile selection needs it, and it must not need config |
| 3 | Profile | Turns `auto` into a concrete tier |
| 4 | Configuration | Merged and validated for that tier |
| 5 | Logging | Its levels and privacy rules come from configuration |
| 6 | Secrets | Last, and never logged |

Steps 1-4 run before logging exists, which is why every failure there raises
`ConfigurationError` carrying a complete, self-contained message.

---

## 5. Configuration

Four layers, each overriding the previous:

```
config/default.yaml              committed defaults, the source of truth
  → config/profiles/<tier>.yaml  hardware tier
    → %APPDATA%/.../config.yaml  personal overrides
      → AI_INTERPRETER__A__B=x   per-run environment overrides
```

Two properties are enforced by the schema:

- **`extra="forbid"`** — an unrecognised key is an error. Writing
  `min_silense_ms` fails at startup with the exact key name instead of leaving
  you to wonder why your tuning had no effect.
- **`frozen=True`** — settings are immutable, so threads read them without
  locking and nothing can change configuration mid-session.

Most fields have no Python default. Defaults live in YAML, so there is exactly
one place to change them.

### Environment override format

`AI_INTERPRETER__STT__MODEL=base` sets `stt.model`. Double underscores
separate nesting levels because single underscores already appear inside key
names. Values are parsed as JSON when possible (`true`, `350`, `null`) and
kept as strings otherwise (`small`, `CABLE Input`).

---

## 6. Hardware profiles

| Profile | Selected when | STT | MT | TTS | Lane |
|---|---|---|---|---|---|
| `cuda` | NVIDIA GPU, >= 6 GB VRAM | large-v3-turbo fp16 | IndicTrans2 fp16 | Kokoro | parallel |
| `cpu_high` | >= 6 physical cores | small int8 | IndicTrans2 int8 | Piper | parallel |
| `cpu_low` | fewer cores | **base** int8 | IndicTrans2 int8 | Piper | serial |

`cpu_low` used `small` until Phase 4 measured it at 5.88 s per utterance
against 1.66 s for `base`. The CPU tiers were each demoted one size on
evidence.

**`inference_lane` is the interesting one.** On a two-core CPU, running three
models concurrently is *slower* than running them one after another, because
they fight for the same two cores and thrash cache. The pipeline therefore
serialises inference on `cpu_low` and overlaps stages elsewhere.

---

## 7. Concurrency model

```
Thread 1  PortAudio callback   real-time. Copies samples into a ring buffer.
                               No logging, no allocation, no locks, ever.
Thread 2  Capture pump         resample → VAD → utterance segmentation
Thread 3  asyncio event loop   orchestration, queues, cancellation, retries
Threads N Inference executor   STT / MT / TTS
Thread 0  Qt UI                receives signals only
```

**Why threads rather than processes:** PyTorch and ONNX Runtime release the
GIL inside native code, so a `ThreadPoolExecutor` achieves real parallelism.
Processes would require pickling seconds of audio across pipes and would load
a separate copy of every model.

**The mistake this avoids:** calling `model.transcribe()` inside an `async def`
blocks the event loop. The UI freezes and audio stutters. Inference always goes
through `run_in_executor`.

**Backpressure is drop-oldest.** In a live conversation, stale audio is
worthless; letting a queue grow only adds latency. Queues are deliberately
tiny (2-4 items).

---

## 8. Latency budget

The meaningful metric is **EOU→FTS**: End Of Utterance to First Translated
Sample. Measuring from the *start* of speech would include however long you
talked, which the application cannot control.

| Stage | `cuda` (estimate) | `cpu_low` (**measured**) |
|---|---:|---:|
| VAD endpoint (silence hangover) | 300 ms | **350 ms** |
| Frame handoff and jitter | 15 ms | **15 ms** |
| STT decode | 120-220 ms | **1660 ms** (Whisper `base`, 2 threads) |
| Translation | 60-110 ms | not yet measured (Phase 5) |
| TTS first chunk | 80-140 ms | not yet measured (Phase 6) |
| Output buffer to the virtual cable | 40-60 ms | 60 ms |
| **Total** | **≈ 0.65-0.9 s** | **≥ 2.0 s already** |

The `cpu_low` figures are measured on an Intel i5-7200U with `--benchmark`.
The `cuda` column remains an estimate until it runs on such a machine.

### What Phase 4 measurement overturned

Phase 1 listed three optimisations. Measurement killed the first one.

1. ~~**Streaming transcription.**~~ **Does not work on Whisper.** Whisper pads
   its encoder to a fixed 30-second window, so a 1-second chunk costs a *full*
   encoder pass — decoding every second multiplies the work instead of
   spreading it. Measured: the same model took 0.82 s on 1 second of audio and
   0.96 s on 9.9 seconds. Streaming is implemented for live captions, and is
   documented as buying feedback rather than speed.
2. **Speculative translation.** Still viable, and now more valuable: with STT
   costing 1.66 s, overlapping translation with it is worth proportionally
   more. Phase 10.
3. **Translation cache.** Still viable. Phase 10.

Two replacements for the lost optimisation, both attacking the encoder, which
is the actual bottleneck:

4. **A model whose cost scales with input length.** NVIDIA Parakeet and other
   Conformer architectures do not pad to 30 seconds. English-only, so it would
   serve one direction. This is what the original specification asked for, and
   the measurements now supply the reason.
5. **OpenVINO on the Intel integrated GPU.** Accelerates the Whisper encoder
   directly, and unlike Parakeet it helps Tamil and Hindi too.

---

## 9. Audio routing

### One-directional mode (Phase 7)

```
Headset mic ──▶ Interpreter (ta→en) ──▶ CABLE Input ──▶ Teams (mic = CABLE Output)
```

### Bidirectional mode (later phase)

```
Headset mic ──▶ Interpreter A (ta→en) ──▶ CABLE-A Input ──▶ Teams mic
Teams output ──▶ CABLE-B Input ──▶ Interpreter B (en→ta) ──▶ Headset speakers
```

**Why two cables.** Capturing "whatever the speakers are playing" while also
playing Tamil speech to those speakers means the application hears its own
output and translates it again — an infinite feedback loop. Two isolated
cables make feedback structurally impossible rather than relying on echo
suppression to catch it.

Windows 11 supports this directly: Settings → System → Sound → Volume mixer
sets a per-application output device.

---

## 10. Privacy design

| Concern | Mechanism |
|---|---|
| Transcripts in logs | Dropped by a logging filter unless `privacy.log_transcripts` is on |
| Conversation history | Not persisted unless `privacy.persist_history` is on |
| Telemetry | No backend exists; the schema *rejects* `telemetry: true` at startup |
| Credentials | `.env` only, wrapped in `SecretStr` so `repr` prints `**********` |
| Model provenance | Every model pinned to an exact revision, verified after download |

The telemetry decision is worth calling out: rather than defaulting it to
false, configuration validation refuses to start if it is true. A setting that
cannot be switched on cannot be switched on by accident.

---

## 11. Decisions and alternatives

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| UI toolkit | PySide6 | PyQt6 | PySide6 is LGPL; PyQt6 is GPL and would force open-sourcing |
| DI | Hand-written root | dependency-injector, punq | Runtime reflection defeats static checking |
| Logging | stdlib + QueueHandler | loguru | Captures `transformers`/`torch` output natively; one less dependency |
| Config format | YAML + pydantic | TOML, JSON | Comments matter in a file users edit; nesting is natural |
| Audio I/O | sounddevice | PyAudio | Ships a PortAudio binary; PyAudio needs a compiler |
| STT runtime | faster-whisper (CTranslate2) | openai-whisper | 4x faster, int8 quantisation, far less RAM |
| Parakeet delivery | Exported ONNX | NeMo toolkit | NeMo is unreliable to install on Windows without a compiler |
| Translation default | IndicTrans2-200M | Nemotron/Llama 8B | 40x smaller, better Tamil, and it runs at all on a laptop CPU |
| VAD | Silero (ONNX) | webrtcvad | More accurate, and needs no compiler |

Two libraries are deliberately **never defaults** because of licensing:
NLLB-200 (CC-BY-NC) and Coqui XTTS-v2 (CPML) are non-commercial. They exist as
opt-in adapters with a licence warning.

---

## 12. Known open questions

| # | Question | Resolved in |
|---|---|---|
| 1 | Is there a free Tamil TTS voice under ~500 ms on CPU? | Phase 6 |
| 2 | Does a Tamil-fine-tuned Whisper beat base Whisper enough to justify it? | Phase 4 |
| 3 | Does OpenVINO on Intel integrated graphics speed up the encoder usefully? | Phase 4 |
| 4 | What is the real measured EOU→FTS on `cpu_low`? | Phase 10 |

These are tracked honestly rather than assumed away. Question 1 in particular
has a designed fallback (on-screen captions), so a negative answer delays a
feature rather than blocking the project.
