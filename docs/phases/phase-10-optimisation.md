# Phase 10 - Optimisation

Two optimisations, both measured on the target machine, both shaped by its
two cores. And one principle enforced throughout: an optimisation may cost
accuracy or latency *nothing* on the paths it does not help.

## Warm components across sessions

**Measured: Start after Stop went from 17.15 s to 0.01 s.**

The container now memoises every model component it builds (recognisers,
translators - LRU cache included - voices, the VAD) in a `component_cache`
keyed by what makes each unique. `InterpretationBundle.shutdown()` stops the
session but deliberately does not close the models; `Container.shutdown()`
releases them at process exit. Each component's `warmup()` became
idempotent, so re-assembly skips the throwaway decode a warm model does not
need.

The UI benefits most - pressing Start a second time is instant - and the
translation cache riding along means sentences repeated across sessions
still answer in 0 ms.

## The streaming lane

**Measured on 20.7 s of continuous speech: end-of-speech to final
transcript fell from 13.38 s (offline decode) to 0.64 s, with 11 interim
transcripts surfacing during the speech.**

The IndicConformer's chunked-commitment `transcribe_stream` (Phase 4b) now
feeds the live pipeline. On speech onset the pipeline buffers frames; once
speech outlasts `chunk_ms + 0.8 s`, an `UtteranceStreamer` opens on its own
thread, receives the backlog and every following frame, and decodes in
committed chunks while the speaker is still talking. At end-of-utterance
only the uncommitted tail remains. Interim transcripts flow through the new
`PipelineEvents.on_partial` into the UI's live caption line.

Three design points that came from measurement, not theory:

**Short utterances never enter the lane.** An utterance shorter than
chunk-plus-margin can never commit early, so streaming it buys nothing -
and the WAV regression showed it *costing* time, because the streamed
decode contends with the previous utterance's translation on 2 cores.
Below the threshold the offline path runs exactly as before, bit for bit.

**Failure can never lose speech.** The streamed result is collected with a
timeout; on timeout, decode error, or a recogniser without streaming
support, the pipeline falls back to the whole-utterance offline decode.

**Committed segments keep absolute time.** Committed audio is trimmed from
the buffer, but the word segments built from it are offset back to
utterance-relative time - transcript fusion aligns words across two
recognisers by time, and a clock that reset at each commit would break it.

Accuracy parity, measured against the offline decode on real utterances:
a 4.1 s utterance decoded *identically*; a 6.5 s span showed minor
chunk-edge variance (`முடிஞ்சிடுச்சு` -> `முடிஞச்சு`) - degraded rendering
of a word, never a lost one. The 0.8 s commit margin is the knob: wider
buys parity, narrower buys tail latency.

Found and fixed en route (Phase 9 fusion work, recorded here for the
streaming story): sherpa-onnx renders the sentencepiece `▁` marker as a
literal leading space in `result.tokens`, so the commit logic had never
found a word boundary on the real model - streaming only force-committed at
the buffer ceiling until then.

## What was considered and rejected

- **Parallel STT + fusion decode** - the English rescue decode could
  overlap the Tamil decode, but both engines want both cores; Phase 4
  measured that contention costs more than serialising. Revisit on >2 cores.
- **Speculative English decode on every utterance** - wastes a Whisper pass
  on the pure-Tamil majority to speed up the flagged minority.
- **Smaller MT beam** - beam 4 -> 2 would halve MT time, but Phase 5 chose
  quality on morphologically rich Tamil deliberately. A config knob exists
  for anyone who disagrees (`translation.beam_size`).
