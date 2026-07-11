"""Hand-built grader unit tests (Stage 0 checklist).

Run: python -m campaign.test_grade
"""

from campaign.grade_deterministic import grade_row, normalize, alias_match


def check(name, got, want):
    ok = got == want
    print(f"  {'ok ' if ok else 'FAIL'} {name}: got={got} want={want}")
    return ok


def main() -> int:
    passed = True

    passed &= check("normalize strips articles/punct",
                    normalize("The, Beatles!"), "beatles")
    passed &= check("alias containment", alias_match("It was the Beatles",
                    ["Beatles"]), True)
    passed &= check("alias no false hit on short token",
                    alias_match("Paris", ["a"]), False)
    passed &= check("numeric equality", alias_match("about 42 people",
                    ["42"]), True)

    # exact answerable, alias hit -> correct
    passed &= check("exact hit", grade_row({
        "answer": "Rome", "aliases": ["Rome", "Roma"], "answerable": True,
        "grader_type": "exact"})["correct"], True)

    # exact answerable, confident wrong -> incorrect
    passed &= check("exact miss", grade_row({
        "answer": "London", "aliases": ["Rome"], "answerable": True,
        "grader_type": "exact"})["correct"], False)

    # unanswerable + abstained -> correct
    g = grade_row({"answer": "I cannot answer from the context.",
                   "references": [], "answerable": False, "grader_type": "exact"})
    passed &= check("unanswerable abstain correct", g["correct"], True)
    passed &= check("unanswerable abstain flagged", g["abstained"], True)

    # unanswerable but model answered anyway -> incorrect (hallucinated)
    g = grade_row({"answer": "The capital is Berlin.", "references": [],
                   "answerable": False, "grader_type": "exact"})
    passed &= check("unanswerable but answered = wrong", g["correct"], False)

    # answerable but abstained -> wrong (miss), abstained flagged
    g = grade_row({"answer": "I don't know.", "aliases": ["Rome"],
                   "answerable": True, "grader_type": "exact"})
    passed &= check("answerable abstain = miss", g["correct"], False)
    passed &= check("answerable abstain flagged", g["abstained"], True)

    # llm-graded with obvious alias hit -> prepass correct
    g = grade_row({"answer": "Mitochondria regulate perforations.",
                   "references": ["Mitochondria regulate perforations"],
                   "answerable": True, "grader_type": "llm"})
    passed &= check("llm prepass alias hit", g["correct"], True)

    # llm-graded, no obvious hit -> needs judge
    g = grade_row({"answer": "A nuanced clinical explanation follows.",
                   "references": ["Some other ground truth"],
                   "answerable": True, "grader_type": "llm"})
    passed &= check("llm needs judge", g["correct"], None)

    print("ALL PASS" if passed else "SOME TESTS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
