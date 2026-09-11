"""Grader unit tests.

Previously a hand-rolled script with a main() and a check() helper, which meant
`python -m unittest` discovered zero tests in the module that produces every
label in the campaign. Converted to unittest so it runs in the suite, with the
original Stage 0 checklist preserved and counterexamples added for the numeric
and contradiction failures found on 2026-09-10.

Run: python -m unittest campaign.test_grade
"""

import unittest

from campaign.grade_deterministic import (
    alias_match,
    alias_match_detail,
    extract_final_answer,
    grade_row,
    normalize,
)


class TestStage0Checklist(unittest.TestCase):
    """The original hand-built checks, unchanged in substance."""

    def test_normalize_strips_articles_and_punctuation(self) -> None:
        self.assertEqual(normalize("The, Beatles!"), "beatles")

    def test_alias_containment(self) -> None:
        self.assertTrue(alias_match("It was the Beatles", ["Beatles"]))

    def test_no_false_hit_on_short_token(self) -> None:
        self.assertFalse(alias_match("Paris", ["a"]))

    def test_numeric_equality(self) -> None:
        self.assertTrue(alias_match("about 42 people", ["42"]))

    def test_exact_hit(self) -> None:
        self.assertTrue(grade_row({
            "answer": "Rome", "aliases": ["Rome", "Roma"], "answerable": True,
            "grader_type": "exact"})["correct"])

    def test_exact_miss(self) -> None:
        self.assertFalse(grade_row({
            "answer": "London", "aliases": ["Rome"], "answerable": True,
            "grader_type": "exact"})["correct"])

    def test_unanswerable_abstention_is_correct(self) -> None:
        g = grade_row({"answer": "I cannot answer from the context.",
                       "references": [], "answerable": False, "grader_type": "exact"})
        self.assertTrue(g["correct"])
        self.assertTrue(g["abstained"])

    def test_unanswerable_but_answered_is_wrong(self) -> None:
        g = grade_row({"answer": "The capital is Berlin.", "references": [],
                       "answerable": False, "grader_type": "exact"})
        self.assertFalse(g["correct"])

    def test_answerable_abstention_is_a_miss(self) -> None:
        g = grade_row({"answer": "I don't know.", "aliases": ["Rome"],
                       "answerable": True, "grader_type": "exact"})
        self.assertFalse(g["correct"])
        self.assertTrue(g["abstained"])

    def test_llm_prepass_alias_hit(self) -> None:
        g = grade_row({"answer": "Mitochondria regulate perforations.",
                       "references": ["Mitochondria regulate perforations"],
                       "answerable": True, "grader_type": "llm"})
        self.assertTrue(g["correct"])

    def test_llm_without_obvious_hit_needs_a_judge(self) -> None:
        g = grade_row({"answer": "A nuanced clinical explanation follows.",
                       "references": ["Some other ground truth"],
                       "answerable": True, "grader_type": "llm"})
        self.assertIsNone(g["correct"])


class TestNumericFidelity(unittest.TestCase):
    """normalize() deletes "." and "-".

    Reading numbers out of a normalized string therefore compared "3 14" against
    "3 15" as 3.0 against 3.0, and turned "-5" into "5". Numbers are now read from
    raw text, and a numeric reference is decided numerically rather than falling
    through to string rules.
    """

    def test_different_decimals_are_not_equal(self) -> None:
        self.assertEqual(alias_match_detail("3.15", ["3.14"]), "miss")
        self.assertEqual(alias_match_detail("2.5", ["2.05"]), "miss")

    def test_sign_is_not_discarded(self) -> None:
        self.assertEqual(alias_match_detail("-5", ["5"]), "miss")
        self.assertEqual(alias_match_detail("5", ["-5"]), "miss")

    def test_identical_numbers_still_match(self) -> None:
        self.assertEqual(alias_match_detail("3.14", ["3.14"]), "hit")
        self.assertEqual(alias_match_detail("-5", ["-5"]), "hit")

    def test_thousands_separators_still_match(self) -> None:
        self.assertEqual(alias_match_detail("1,200", ["1200"]), "hit")

    def test_number_embedded_in_prose_still_matches(self) -> None:
        self.assertEqual(alias_match_detail("about 42 people", ["42"]), "hit")

    def test_wrong_number_in_prose_is_a_miss(self) -> None:
        self.assertEqual(alias_match_detail("about 43 people", ["42"]), "miss")


class TestContradictionIsNotAHit(unittest.TestCase):
    """A reference can appear inside an answer that denies it.

    Containment alone read "Not Paris; the answer is London." as a hit for
    "Paris". Surface matching cannot resolve that, so the row is reported
    ambiguous and routed to a judge rather than being labelled either way.
    """

    def test_negated_reference_with_a_marked_conclusion_is_a_miss(self) -> None:
        # "the answer is" names London as the conclusion, so this resolves
        # cleanly rather than needing a judge.
        self.assertEqual(
            alias_match_detail("Not Paris; the answer is London.", ["Paris"]),
            "miss")

    def test_negated_number_is_ambiguous(self) -> None:
        self.assertEqual(
            alias_match_detail("The answer is not 7, it is 9.", ["7"]),
            "ambiguous")

    def test_ambiguous_rows_go_to_a_judge_not_to_a_label(self) -> None:
        # No marker names the conclusion, so containment plus a contradiction cue
        # is not enough to label this either way.
        g = grade_row({"answer": "Paris is not the answer. London is.",
                       "aliases": ["Paris"], "answerable": True,
                       "grader_type": "exact"})
        self.assertEqual(g["method"], "needs_judge")
        self.assertIsNone(g["correct"])

    def test_a_resolved_wrong_answer_is_labelled_wrong_not_deferred(self) -> None:
        g = grade_row({"answer": "Not Paris; the answer is London.",
                       "aliases": ["Paris"], "answerable": True,
                       "grader_type": "exact"})
        self.assertEqual(g["method"], "alias")
        self.assertFalse(g["correct"])

    def test_plain_correct_answers_are_unaffected(self) -> None:
        self.assertEqual(alias_match_detail("Paris", ["Paris"]), "hit")
        self.assertEqual(alias_match_detail("The capital is Paris.", ["Paris"]), "hit")

    def test_boolean_view_treats_ambiguous_as_not_a_hit(self) -> None:
        self.assertFalse(alias_match("Not Paris; the answer is London.", ["Paris"]))




class TestGradesTheAssertedFinalAnswer(unittest.TestCase):
    """Earlier working is not the answer.

    Two false positives survived the first numeric fix: contradiction detection
    only looked before the match, and a number matched anywhere including an
    abandoned candidate. Both are handled by grading the span the answer names as
    its conclusion.
    """

    def test_contradiction_after_the_reference_is_not_a_hit(self) -> None:
        self.assertNotEqual(
            alias_match_detail("Paris is not the answer. London is.", ["Paris"]),
            "hit")

    def test_abandoned_numeric_candidate_is_not_a_hit(self) -> None:
        self.assertEqual(
            alias_match_detail("I first considered 7. My final answer is 9.", ["7"]),
            "miss")

    def test_marked_conclusion_is_graded_not_the_working(self) -> None:
        self.assertEqual(
            alias_match_detail("I first considered 7. My final answer is 9.", ["9"]),
            "hit")

    def test_negation_inside_the_marked_span_is_ambiguous(self) -> None:
        self.assertEqual(
            alias_match_detail("my final answer is not Paris", ["Paris"]),
            "ambiguous")

    def test_multi_sentence_correct_answer_is_still_a_hit(self) -> None:
        self.assertEqual(
            alias_match_detail(
                "The capital of France is Paris. It has over 2 million people.",
                ["Paris"]),
            "hit")

    def test_extraction_reports_whether_a_marker_was_found(self) -> None:
        span, how = extract_final_answer("I considered 7. My final answer is 9.")
        self.assertEqual(how, "marked")
        self.assertEqual(span, "9.")
        span, how = extract_final_answer("Paris")
        self.assertEqual(how, "whole")
        self.assertEqual(span, "Paris")


if __name__ == "__main__":
    unittest.main(verbosity=2)
