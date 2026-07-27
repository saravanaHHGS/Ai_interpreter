# Phase 1 — Architecture Design

**Status:** complete (design only, no code)

Full technical detail lives in [../architecture.md](../architecture.md). This
page records what Phase 1 *decided* and, more importantly, the two findings
that changed the plan.

---

## Finding 1: the target machine has no NVIDIA GPU

The development machine is an Intel i5-7200U — 2 cores, 4 threads, integrated
graphics, 16 GB RAM. The original specification assumed NVIDIA Parakeet,
Nemotron and TensorRT.

Consequences, stated plainly:

- An 8B-parameter LLM (Nemotron, Llama 3.x) produces 2-4 tokens per second on
  this CPU. Not slow — unusable.
- Parakeet TDT 0.6B runs on CPU via ONNX but is **English-only**.
- Sub-2-second latency with good Tamil accuracy is not free here; it requires
  streaming transcription and speculative translation built in from the start
  rather than added as an afterthought.

**Response: hardware profiles.** One codebase, three model tiers, selected
automatically. This is better architecture than hard-coding one model set —
it is required regardless of hardware, and it means no rewrite when a GPU
arrives.

---

## Finding 2: the preferred models do not cover the required languages

The required pairs are Tamil↔English and Hindi↔English. Most of the named
models target English and European languages.

| Stage | Gap | Response |
|---|---|---|
| STT | Parakeet has no Tamil or Hindi | Route per language: Parakeet for English, Whisper for Tamil/Hindi |
| MT | Nemotron/Llama are weak on Tamil and 40x larger than needed | **IndicTrans2-200M becomes the default**; LLMs remain pluggable providers |
| TTS | Kokoro, Piper's official set and XTTS all lack **Tamil** | Decision gate in Phase 6; on-screen Tamil captions as the guaranteed fallback |

Tamil→English — the primary direction — is fully covered. Hindi is fully
covered both ways. English→Tamil **audio** is the one genuine open risk.

### Why IndicTrans2 rather than Nemotron

| | IndicTrans2-200M | Nemotron/Llama 8B |
|---|---|---|
| Parameters | 200 M | 8 B |
| Latency (GPU) | 60-110 ms | 400-900 ms |
| Latency (2-core CPU) | 350-700 ms | unusable |
| Tamil quality | Purpose-built, better | Weaker |
| Licence | MIT | Varies |

Choosing it returns 500-800 ms to the latency budget — on its own, the
difference between hitting and missing the 2-second target.

---

## Decisions carried into Phase 2

| Area | Decision |
|---|---|
| Architecture | Clean Architecture, four layers, dependency rule inward |
| Interfaces | `typing.Protocol` ports, split by capability |
| DI | Hand-written composition root, no framework |
| Concurrency | asyncio orchestration + thread executors for inference |
| Backpressure | Bounded queues, drop-oldest |
| Config | Layered YAML + pydantic; secrets separately in `.env` |
| Logging | stdlib behind a queue, rotating, transcript-filtered |
| UI | PySide6 (LGPL permits commercial use; PyQt6's GPL would not) |
| Metric | EOU→FTS — end of utterance to first translated sample |

---

## Approved scope decisions

1. **Build for the current machine**, with GPU adapters written but not
   loaded, so no rewrite is needed later.
2. **IndicTrans2 as the default translator**, other providers pluggable.
3. **One-directional first** (you → Teams), reverse direction afterwards.
4. **Tamil captions as the fallback** if no free Tamil voice is found, with
   the TTS layer kept provider-based so a Tamil engine can be dropped in later
   without touching the rest of the application.

Requirement 4 is why `SpeechSynthesizer` is a `Protocol` with five methods and
why streaming is a *separate* protocol: adding a Tamil engine later means one
new class and one line in the composition root.

---

## Open questions, tracked honestly

| # | Question | Resolved in |
|---|---|---|
| 1 | Is there a free Tamil TTS voice under ~500 ms on CPU? | Phase 6 |
| 2 | Does Tamil-fine-tuned Whisper beat base Whisper enough to justify it? | Phase 4 |
| 3 | Does OpenVINO on Intel integrated graphics speed up the encoder usefully? | Phase 4 |
| 4 | What is the real measured EOU→FTS on `cpu_low`? | Phase 10 |

Each has a designed fallback, so a negative answer delays a feature rather
than blocking the project.
