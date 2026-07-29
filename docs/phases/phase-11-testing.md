# Phase 11 - Test hardening

The unit suite grew organically to 630 tests across phases 2-10; this phase
adds the layer it could not provide: **regression tests that run the real
neural networks against ground-truthed speech.**

## The real-model regression suite

`pytest -m requires_model` (10 tests, ~57 s) runs the actual models against
the evaluation recording - six utterances of natural mixed Tamil-English,
each confirmed by the speaker. It pins every behaviour a feature depends on:

- **Tamil recognition**: the confirmed pure-Tamil sentence recognised, never
  flagged; word segments matching the text word-for-word (fusion depends on
  this shape).
- **Code-switch routing**: the mixed sentence flagged *with* a native
  anchor - the exact condition that chooses fusion over the wholesale
  reroute.
- **English recognition**: the confirmed English sentence recognised with
  per-word timestamps.
- **Fusion on real audio**: the Tamil survives, English words splice in,
  and Whisper's hallucinated "Mutum" stays out.
- **Translation**: Latin terms pass through untouched (the property the
  glossary and fusion both rest on) and Tamil comes out as English.
- **Synthesis**: the English voice produces audibly non-silent audio.
- **The whole pipeline**: a real-time paced replay of the recording through
  capture, VAD, segmentation, STT, repair, glossary and MT - captions-only,
  so no audio device is needed - asserting zero failures, the repairs
  firing on exactly the utterances that need them, and no transliteration
  soup reaching the output.

Assertions are substring-tolerant of benign decode variance but strict
about behaviours. Neither the model cache nor the recording is committed
(models are re-downloadable; the recording is the developer's own voice),
so every test skips cleanly when its prerequisite is absent. To recreate
the recording: `.\run.ps1 --record 30` speaking natural mixed sentences,
then update `tests/integration/conftest.py`.

## Fast by default

`requires_model` tests are excluded from the default run via `addopts`, so
the developer loop stays at ~10 s for 630 tests. The two-tier workflow:

    pytest                       # 630 unit + light integration, ~10 s
    pytest -m requires_model     # 10 real-model regressions, ~57 s

## Unit gaps closed

- **UI controller threading**: start/stop/failure/shutdown driven through
  the real signal machinery with the bundle builder monkeypatched - the
  background-thread choreography is verified without loading a model.
- **Container component cache**: reuse across sessions and close-and-clear
  on shutdown (exercised with the model-free energy detector).

## Coverage, and what it deliberately ignores

Fast-suite coverage: **75 % overall**, with the meaningful parts high -
interpretation pipeline 93 %, capture session 96 %, and every service the
phases fought over (fusion, code-switch, glossary, segmenter, DSP,
transliteration) at or near 100 % (31 files skipped as fully covered).

The uncovered remainder is of two kinds, both deliberate: console
presentation shells (`cli_*.py` - printing and argument plumbing around
logic that is tested where it lives) and hardware adapters (microphone,
Silero) that need real devices or models - the `requires_model` and
`requires_audio` suites cover those on the machines that have them.
Chasing percentage in either would test print statements, not behaviour.
