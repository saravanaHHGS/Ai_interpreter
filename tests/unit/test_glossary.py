"""Unit tests for glossary term recovery.

The Tamil variants used here are the real mis-transcriptions observed in the
live session that motivated the feature: "world research" spoken mid-Tamil
came out as ``வேர்ல்ட் ரெஸ்சர்ச்``.
"""

from __future__ import annotations

import pytest

from ai_interpreter.application.services.glossary import GlossaryRewriter

pytestmark = pytest.mark.unit


class TestGlossaryRewriter:
    """Whole-token variant replacement."""

    def test_replaces_a_known_variant(self) -> None:
        rewriter = GlossaryRewriter({"world": ["வேர்ல்ட்"]})
        assert rewriter.rewrite("வேர்ல்ட் முடிஞ்சது") == "world முடிஞ்சது"

    def test_replaces_multiple_terms_in_one_sentence(self) -> None:
        # The exact sentence from the live session.
        rewriter = GlossaryRewriter({"world": ["வேர்ல்ட்"], "research": ["ரெஸ்சர்ச்"]})
        result = rewriter.rewrite("வேர்ல்ட் ரெஸ்சர்ச் எல்லாமே முடிஞச்சு")

        assert result == "world research எல்லாமே முடிஞச்சு"

    def test_several_variants_map_to_one_term(self) -> None:
        # Two different attempts at "world" from the same session.
        rewriter = GlossaryRewriter({"world": ["வேர்ல்ட்", "வலட"]})

        assert rewriter.rewrite("வலட நல்லது") == "world நல்லது"
        assert rewriter.rewrite("வேர்ல்ட் நல்லது") == "world நல்லது"

    def test_multi_word_phrase_variant(self) -> None:
        rewriter = GlossaryRewriter({"we need to": ["வே நீத்் டூ"]})
        assert rewriter.rewrite("வே நீத்் டூ போகணும்") == "we need to போகணும்"

    def test_never_matches_inside_a_longer_word(self) -> None:
        # Rewriting fragments of correct Tamil would be worse than missing a
        # term.
        rewriter = GlossaryRewriter({"x": ["கல்"]})
        assert rewriter.rewrite("கல்வி நல்லது") == "கல்வி நல்லது"

    def test_latin_variants_match_case_insensitively(self) -> None:
        rewriter = GlossaryRewriter({"VALD": ["vald", "walt"]})
        assert rewriter.rewrite("the Walt session") == "the VALD session"

    def test_longest_variant_wins(self) -> None:
        rewriter = GlossaryRewriter({"world": ["வேர்ல்ட்"], "world research": ["வேர்ல்ட் ரெஸ்சர்ச்"]})
        assert rewriter.rewrite("வேர்ல்ட் ரெஸ்சர்ச் நடக்குது") == "world research நடக்குது"

    def test_repeated_occurrences_are_all_replaced(self) -> None:
        rewriter = GlossaryRewriter({"world": ["வலட"]})
        assert rewriter.rewrite("வலட வலட") == "world world"

    def test_empty_glossary_is_a_no_op(self) -> None:
        rewriter = GlossaryRewriter({})
        assert rewriter.rewrite("எதுவும் மாறக்கூடாது") == "எதுவும் மாறக்கூடாது"
        assert rewriter.size == 0

    def test_blank_variants_are_ignored(self) -> None:
        rewriter = GlossaryRewriter({"x": ["  ", ""]})
        assert rewriter.size == 0

    def test_counts_hits(self) -> None:
        rewriter = GlossaryRewriter({"world": ["வலட"]})
        rewriter.rewrite("வலட வலட")
        rewriter.rewrite("வலட")

        assert rewriter.hits == 3
