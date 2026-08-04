"""Word-level fusion of two recognisers' views of the same utterance.

The measured problem, from the user's own recording: for the mixed sentence
"matching மட்டும் pending ல இருக்கு", the Tamil conformer returns perfect
Tamil with the English words transliterated (``மேட்சிங் மட்டும் பெண்டிங் ல
இருக்கு``), while Whisper-English returns the English words correctly and
garbles the Tamil ("Matching Mutum Bending"). Each model knows exactly what
the other does not.

Fusion takes the Tamil transcript as the base, finds the words the
phonotactic detector flags as English-in-Tamil-script, and replaces each
flagged run with the English recogniser's words *from the same time window*.
Both recognisers timestamp against the same audio, so time is the join key -
no transliteration guessing, no dictionary.

The base transcript always wins by default: a flagged region with no English
words inside its window keeps its Tamil, because "no evidence" must never
become "delete the user's speech". This is what makes fusion strictly safer
than the whole-utterance reroute it complements - the reroute replaces
everything and so can only be used when the sentence is entirely English.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ai_interpreter.application.services.code_switch import flag_english_tokens
from ai_interpreter.domain.entities import Transcript, TranscriptSegment

__all__ = ["FusionResult", "fuse_transcripts"]

# Fraction of an English word's duration that must fall inside a flagged
# region for it to be spliced in. Measured on the live recording: Whisper's
# "Mutum" (its garbled rendering of the *Tamil* word மட்டும்) overlapped the
# neighbouring flagged region by about a third - a midpoint test let it
# through, mostly-inside keeps it out. Genuine matches sit at 0.8-1.0.
_MIN_OVERLAP: Final[float] = 0.5

# Below this onset span, a linear clock fit has nothing to hold on to.
_MIN_SPAN_MS: Final[float] = 120.0

# Punctuation stripped from a spliced replacement: mid-sentence, Whisper's
# trailing "assessment." full stop would corrupt the surrounding Tamil.
_TRAILING_PUNCTUATION: Final[str] = ".,!?;:"


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Outcome of fusing two transcripts.

    Args:
        text: The fused text - the base transcript with flagged regions
            replaced by the other recogniser's words.
        replaced: Base-transcript words that were replaced, in order.
        inserted: Words spliced in from the other transcript, in order.
    """

    text: str
    replaced: tuple[str, ...]
    inserted: tuple[str, ...]


def _word_level_segments(transcript: Transcript) -> list[TranscriptSegment] | None:
    """Extract per-word segments, or ``None`` when the shape is wrong.

    Args:
        transcript: A transcript whose segments may or may not be per-word.

    Returns:
        The word segments, or ``None`` when the transcript carries no
        segments or carries coarser-than-word ones (a recogniser configured
        without word timestamps). Fusion must then decline rather than
        splice at the wrong granularity.
    """
    segments = [segment for segment in transcript.segments if segment.text.strip()]
    if not segments or len(segments) != len(transcript.text.split()):
        return None
    return segments


def _align_clocks(
    base_words: list[TranscriptSegment],
    other_words: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Rescale the other transcript's timeline onto the base's.

    For recognisers whose internal frame clock disagrees with the primary's
    - measured: the NeMo streaming model's timestamps run ~30% slow, with
    drift GROWING along the utterance. Both transcripts describe the same
    audio, so their first and last word ONSETS mark the same two real
    moments; the linear map between those onset spans lands every word
    correctly (verified word-by-word against three live utterances). Word
    onsets anchor the fit rather than word ends, because ends are derived
    (next word's onset, or a guess for the last word) and unreliable.

    Args:
        base_words: The primary transcript's word segments.
        other_words: The secondary transcript's word segments, in a clock
            the caller has declared untrustworthy.

    Returns:
        The secondary segments mapped onto the base clock, or unchanged
        when either onset span is too short to fit a line through.
    """
    base_start = base_words[0].start_ms
    base_span = base_words[-1].start_ms - base_start
    other_start = other_words[0].start_ms
    other_span = other_words[-1].start_ms - other_start
    if base_span < _MIN_SPAN_MS or other_span < _MIN_SPAN_MS:
        return other_words

    scale = base_span / other_span
    return [
        TranscriptSegment(
            text=segment.text,
            start_ms=base_start + (segment.start_ms - other_start) * scale,
            end_ms=base_start + (segment.end_ms - other_start) * scale,
            confidence=segment.confidence,
        )
        for segment in other_words
    ]


def _overlap_fraction(word: TranscriptSegment, start_ms: float, end_ms: float) -> float:
    """How much of a word's duration lies inside a time window.

    Args:
        word: The word segment.
        start_ms: Window start.
        end_ms: Window end.

    Returns:
        A value in ``[0.0, 1.0]``.
    """
    overlap = min(word.end_ms, end_ms) - max(word.start_ms, start_ms)
    duration = max(word.end_ms - word.start_ms, 1e-6)
    return max(0.0, overlap) / duration


def fuse_transcripts(
    base: Transcript,
    other: Transcript,
    min_overlap: float = _MIN_OVERLAP,
    align_clock: bool = False,
) -> FusionResult | None:
    """Splice ``other``'s words into ``base`` where ``base`` is transliterated.

    Args:
        base: The source-language transcript (word-level segments required).
            Its unflagged words are kept verbatim.
        other: The target-language recogniser's transcript of the *same*
            audio (word-level segments required). Only its words lying
            mostly inside flagged time regions are used.
        min_overlap: Fraction of a word's duration that must fall inside a
            flagged region to be spliced.
        align_clock: Declare that ``other``'s recogniser keeps a different
            internal clock (the NeMo streaming partner); its timeline is
            then linearly mapped onto ``base``'s before matching. Leave
            ``False`` for Whisper, whose clock agrees with the conformer's.

    Returns:
        The fusion result, or ``None`` when fusion is not possible (missing
        word timestamps, nothing flagged) or changed nothing (no words of
        ``other`` fell inside any flagged window).
    """
    base_words = _word_level_segments(base)
    other_words = _word_level_segments(other)
    if base_words is None or other_words is None:
        return None

    flagged = [bool(flag_english_tokens(segment.text)) for segment in base_words]
    if not any(flagged):
        return None

    if align_clock:
        other_words = _align_clocks(base_words, other_words)

    pieces: list[str] = []
    replaced: list[str] = []
    inserted: list[str] = []
    used: set[int] = set()

    index = 0
    while index < len(base_words):
        if not flagged[index]:
            pieces.append(base_words[index].text)
            index += 1
            continue

        # Extend across consecutive flagged words: "வேர்ல்ட் அஸிஸ்மெண்ட்" is
        # one English phrase, and matching it as one window lets a different
        # number of English words replace it ("VALD assessment").
        end = index
        while end + 1 < len(base_words) and flagged[end + 1]:
            end += 1

        # Two matching rules, chosen by what the timestamps can be trusted
        # for. Whisper (unaligned) smears word ONSETS backward through
        # leading audio (measured: "World" at 0-760 ms against the
        # conformer's வேர்ல்ட் at 400-840 ms) while ENDS are anchored by
        # the next word's onset - so an end landing inside the region
        # counts. A clock-ALIGNED transcript has calibrated boundaries on
        # both sides, and its derived ends can drift into the NEXT region
        # after rescaling - there, mostly-inside overlap is the only safe
        # test (verified word-by-word on the live fixtures).
        region_start = base_words[index].start_ms
        region_end = base_words[end].end_ms
        matches = [
            position
            for position, word in enumerate(other_words)
            if position not in used
            and (
                (not align_clock and region_start <= word.end_ms <= region_end)
                or _overlap_fraction(word, region_start, region_end) >= min_overlap
            )
        ]

        replacement = " ".join(other_words[position].text for position in matches)
        replacement = replacement.strip().rstrip(_TRAILING_PUNCTUATION)
        if replacement:
            used.update(matches)
            replaced.extend(segment.text for segment in base_words[index : end + 1])
            inserted.extend(other_words[position].text for position in matches)
            pieces.append(replacement)
        else:
            pieces.extend(segment.text for segment in base_words[index : end + 1])
        index = end + 1

    if not replaced:
        return None
    return FusionResult(
        text=" ".join(pieces),
        replaced=tuple(replaced),
        inserted=tuple(inserted),
    )
