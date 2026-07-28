# Phase 7 — Virtual Microphone

**Status:** complete
**Version:** 0.7.0

Synthesised speech now routes into VB-CABLE, where meeting applications pick
it up as a microphone. This is the step that turns the components into an
interpreter — and it was verified with a full loopback on the target machine,
not just by watching a level meter.

---

## Verification

```powershell
.\run.ps1 --speak "Can you hear the interpreter?" --language en --out "CABLE Input"
```

Then, in **Teams / Zoom / Meet**: Settings → Devices → Microphone →
**CABLE Output (VB-Audio Virtual Cable)**, and use the microphone test while
running the command again. The meter should move and the test playback should
contain the sentence.

### The loopback proof (already run on the target machine)

The strongest verification needs no meeting app: speak into one end of the
cable with our sink while recording from the other end with our capture
stack, then transcribe what came through.

```
spoken into CABLE Input : "Testing one two three. Testing the cable loop again."
captured at CABLE Output: peak 0.617, 1 utterance detected
transcribed             : "Testing 123, testing the cable loop again."   conf 0.65
underruns               : 0
```

Every stage of the eventual pipeline touched the audio: TTS → sink →
VB-CABLE → capture → resample → VAD → segmentation → STT. Word-perfect.

---

## What was built

| Piece | File | Purpose |
|---|---|---|
| `VirtualCableSink` | `infrastructure/audio/playback/virtual_cable.py` | The `AudioSink` port's first real adapter |
| `resolve_output_device` / `create_audio_sink` | `app/container.py` | Output endpoint resolution from `audio.output.*` |
| `--out NAME` on `--speak` | `cli_tts.py` | Chunk-by-chunk streaming into any output device |

### Design decisions in the sink

**The stream stays open and plays silence between utterances.** Meeting
applications dislike microphones that appear and disappear; a continuously
running stream that emits zeros when idle looks like a normal quiet mic.

**A jitter buffer gates each utterance's start.** Synthesis chunks arrive in
bursts; starting playback on the first sample and starving immediately would
stutter the first word. Playback holds until `audio.output.jitter_buffer_ms`
(60 ms) is queued — released early by the utterance's final chunk, so
utterances shorter than the buffer still play, and forced by `flush()`.

**`clear()` is barge-in.** When the speaker interrupts, the half-played
previous translation stops immediately by dropping the queue. Phase 9 wires
this to the VAD's speech-onset signal.

**Per-utterance stateful resampling.** Voices have different native rates
(Piper 22.05 kHz, MMS 16 kHz); the cable runs at 48 kHz. The same
`StreamingResampler` correctness argument from the capture side applies in
reverse, and the resampler is rebuilt when the incoming rate changes —
which happens exactly when the target language switches.

**The driver callback only copies**, mirroring the capture side: no locks
held long, no allocation, zero-fill on empty. The fill logic is a separate
method driven directly by the unit tests.

Measured through the sink: English speech reached the cable with **first
chunk in 230–374 ms** after synthesis start and **0 underruns** across every
run.

---

## An environmental incident worth recording

Mid-verification, Smart App Control blocked `av.filter.loudnorm` (a PyAV
DLL inside faster-whisper's import chain) — and **twenty minutes later the
same import succeeded**. Unlike the permanent `librt` block, SAC verdicts
can also be *transient*. `troubleshooting.md` now says: on a sudden
`DLL load failed ... blocked this file` for something that worked before,
retry once before assuming it is permanent.

This also sharpened a strategic observation: every sherpa-onnx component has
been immune to these incidents so far, while the CTranslate2/PyAV stack has
now been hit twice. If faster-whisper's block ever becomes permanent, the
English recogniser moves to the already-registered NeMo streaming model or a
Parakeet-TDT sherpa export — one registry edit, no code.

---

## Latency note

The sink adds jitter buffer (60 ms) plus driver block (20 ms) — under 100 ms,
as budgeted in Phase 1. The `--speak --out` path measured first-chunk-to-
cable at 230–374 ms for English, dominated by synthesis, not routing.

---

## What Phase 8 adds

The PySide6 desktop interface: dashboard, devices, models, languages, logs
and performance pages over the components built so far.
