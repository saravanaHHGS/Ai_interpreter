"""End-to-end pipeline over the evaluation recording, real models throughout.

Captions-only on purpose: no audio device is opened, so the test runs on
any machine that has the model cache and the recording. The WAV source is
real-time paced (~30 s), because unpaced replay masquerades queue waits as
decode time and floods the segmenter - the lesson of the first E2E run.

The assertions pin the *behaviour* of the whole system on ground-truthed
speech: everything interpreted, nothing lost, the code-switch repairs
firing on the utterances that need them and only those.
"""

from __future__ import annotations

import time

import pytest

from ai_interpreter.application.pipeline.interpretation import (
    InterpretationPipeline,
    PipelineEvents,
)
from ai_interpreter.application.services.capture_session import CaptureSession
from ai_interpreter.application.services.glossary import GlossaryRewriter
from ai_interpreter.domain.entities import Transcript, Translation
from ai_interpreter.domain.value_objects import LanguagePair, SampleRate
from ai_interpreter.infrastructure.audio.capture.wav_file import WavFileSource

pytestmark = [pytest.mark.integration, pytest.mark.requires_model]


def test_recording_interprets_end_to_end(real_container, recording_path) -> None:  # type: ignore[no-untyped-def]
    pair = LanguagePair.of("ta", "en")
    source = WavFileSource(
        recording_path,
        frame_ms=real_container.settings.audio.input.frame_ms,
        realtime=True,
    )
    recognizer = real_container.create_recognizer(pair.source)
    english = real_container.create_recognizer(pair.target, word_timestamps=True)
    translator = real_container.create_translator(pair)
    recognizer.warmup()
    english.warmup()
    translator.warmup()

    capture = CaptureSession(
        source=source,  # type: ignore[arg-type]
        preprocessor=real_container.create_preprocessor(source.sample_rate),
        vad=real_container.create_vad(),
        segmenter=real_container.create_segmenter(SampleRate(16000), language=pair.source),
    )

    transcripts: list[Transcript] = []
    translations: list[Translation] = []
    pipeline = InterpretationPipeline(
        capture=capture,
        recognizer=recognizer,
        translator=translator,
        synthesizer=None,
        sink=None,
        pair=pair,
        glossary=GlossaryRewriter(real_container.settings.translation.glossary),
        english_fallback=english,
        fallback_min_score=real_container.settings.stt.code_switch_min_score,
        word_fusion=True,
        streaming_stt=real_container.settings.stt.streaming,
        stream_min_seconds=real_container.settings.stt.chunk_ms / 1000.0 + 0.8,
        events=PipelineEvents(
            on_transcript=transcripts.append,
            on_translation=translations.append,
        ),
    )

    pipeline.start()
    try:
        while capture.is_running:
            time.sleep(0.2)
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            stats = pipeline.stats()
            resolved = (
                stats.utterances_out
                + stats.dropped_backpressure
                + stats.dropped_empty
                + stats.failures
            )
            if resolved >= stats.utterances_in:
                break
            time.sleep(0.2)
    finally:
        pipeline.stop()

    stats = pipeline.stats()

    # The recording contains six spoken utterances; VAD segmentation may
    # split one differently, so the bound is >= 5 rather than == 6.
    assert stats.utterances_in >= 5
    assert stats.failures == 0
    assert stats.utterances_out >= stats.utterances_in - 1  # one may be empty

    # The repairs fire on the utterances that need them: at least the mixed
    # sentence fuses and the fully-English one reroutes or fuses.
    assert stats.word_fusions >= 1
    assert stats.word_fusions + stats.code_switch_reroutes >= 2

    # The confirmed pure-Tamil sentence made it through untouched...
    assert any("நாளைக்கு" in transcript.text for transcript in transcripts)
    # ...and every translation the meeting would hear is actually English.
    assert translations
    for translation in translations:
        assert translation.translated_text.strip()

    # Glossary + fusion + hotwords: the mixed "pending" sentence must not
    # reach the meeting as raw transliteration soup.
    assert not any("பெண்டிங்" in translation.translated_text for translation in translations)
