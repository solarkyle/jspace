# Label correction and sensitivity analysis, 2026-09-10

The deterministic grader that produced most Stage 1 labels had defects. This is a
dated recheck of the published Stage 1 gates against a corrected label set.

**It does not supersede `STAGE1_REPORT.md`.** That report stands as published. This
is the sensitivity analysis beside it.

**Headline: both gates survive, and the apparent improvement is mostly not the
relabelling.** Roughly 80 percent of the Gate A gain comes from removing rows the
corrected grader refuses to label, not from the labels that changed. On Gate B the
relabelling moves the result slightly the wrong way.

## What was wrong with the grader

Three defects, all confirmed against the committed code and all now covered by
tests in `campaign/test_grade.py`.

1. **Numbers were read from normalised text.** `normalize()` deletes `.` and `-`,
   and `_num()` then parsed the result, so `"3.14"` became `"3 14"` and read as
   `3.0`. `3.15` and `3.14` compared equal; `"-5"` matched reference `5`. It also
   caused false negatives: `"9,117"` failed to match `9117`.
2. **A numeric reference could match via a non-numeric reference list.** Reference
   `97.35` matched an answer asserting `97.88%`, because both reduced to a first
   number of `97`. halubench's `references` field holds a stringified array, which
   matched through the same path.
3. **A reference was accepted anywhere in the answer.** `"Not Paris; the answer is
   London."` counted as correct for `Paris`, and `"I first considered 7. My final
   answer is 9."` counted as correct for `7`.

The corrected grader reads numbers from raw text, decides a numeric reference
numerically, grades only the span after an explicit final-answer marker when one is
present, and returns **ambiguous** rather than a label when containment is the only
evidence and a contradiction cue is present. Ambiguous routes to a judge; it never
becomes a label.

## What changed in the labels

Re-graded only what the deterministic grader owned. The 2,072 judge-decided labels
were preserved untouched, and the 2,042 never-labelled rows were left alone.

| | rows |
|---|---:|
| re-graded (`alias`, `llm_prepass_alias`) | 14,106 |
| label flipped | 253 |
| label lost to ambiguity | 847 |
| label gained | 0 |
| judge labels preserved | 2,072 |
| labelled population | 16,178 -> **15,331** |

| source | re-graded | flipped | lost label |
|---|---:|---:|---:|
| drop | 2,000 | 160 | 98 |
| hotpotqa | 2,500 | 35 | 100 |
| popqa | 3,000 | 32 | 14 |
| trivia_qa | 3,000 | 19 | 7 |
| squad_v2 | 3,000 | 7 | 149 |
| halubench | 544 | 0 | 417 |
| medhallu | 62 | 0 | 62 |

`drop` moves most on flips, which is expected: it is the numerical-reasoning
source. halubench and medhallu lose the most labels, because their prepass hits
came through the numeric backdoor described above. Those are `llm`-graded sources,
so a lost prepass hit becomes a judge request rather than a wrong label.

Separately, the grader **as committed** already disagreed with 210 stored labels
(1.49 percent) before any of this change. Those were never reproducible from the
repository.

## Decomposition: selection versus relabelling

Dropping 847 rows is not random. They are rows with contradictory or unmarked
prose, which are plausibly the harder ones. So any comparison must separate the
effect of removing them from the effect of changing labels. Three variants:

- **A. published** — original labels, all 16,178 rows
- **B. selection control** — original labels, restricted to the 15,331 rows that
  keep a label after correction
- **C. corrected** — corrected labels, same 15,331 rows

B minus A isolates selection. C minus B isolates relabelling.

### Gate A, workspace-over-logprob LODO increment

| model | A published | B selection control | C corrected | selection | relabelling |
|---|---:|---:|---:|---:|---:|
| LightGBM | +0.0651 | +0.0783 | **+0.0816** | +0.0132 | +0.0033 |
| logistic | +0.0908 | +0.1045 | **+0.1100** | +0.0137 | +0.0055 |

**80 percent of the LightGBM gain is selection.** The labels that actually changed
account for +0.0033 of the +0.0165 total. Reporting C against A without B would
have read as a vindication of the corrected labels. It is not one.

### Gate B, mean catch-rate delta at a 20 percent routing budget

| variant | mean delta | bootstrap 95% CI | verdict |
|---|---:|---|---|
| A published | +5.10pp | [+3.15, +7.36] | HIT |
| B selection control | +6.00pp | [+3.68, +7.61] | HIT |
| C corrected | +5.51pp | [+3.58, +7.45] | HIT |

Here the relabelling moves the result **down** by 0.49pp. The net rise from +5.10
to +5.51 is selection (+0.90) partly undone by relabelling (-0.49). This matters
because the registered threshold is +5pp and the published margin was 0.10pp: the
verdict holds in all three variants, but it was never a comfortable margin and the
correction does not make it one.

Per-source Gate B under corrected labels: drop +5.01, halubench +5.68,
hotpotqa +5.11, medhallu +18.67, popqa +1.62, squad_v2 +0.63, trivia_qa +1.87.
medhallu continues to carry the mean, as the original report noted.

### Reproduction check

Both gates reproduce exactly from saved features on the original labels
(+0.0651 LightGBM, +0.0908 logistic, Gate B +5.10pp, CI [+3.15, +7.36], identity
leakage 0.811 against a 0.195 majority). The analysis code is reproducible; the
grading was the moving part.

## What this does not establish

- **The corrected labels are not validated ground truth.** 253 flips are 253
  disagreements, not 253 verified corrections. `out/campaign/relabel_report.json`
  exports 60 flip examples for adjudication. None have been adjudicated by an
  independent judge.
- **The 847 unresolved rows were never judged.** They are excluded, counted by
  source, and in the Gate C scorer the verdict is bracketed by imputing them both
  ways. For Gate A and Gate B here they are simply absent, which is what variant B
  exists to quantify.
- **This is the retraining variant.** LODO retrains with corrected labels on both
  sides. The pure rescoring variant, holding the frozen classifiers' predictions
  fixed and changing only the evaluation labels, has not been run.
- **Judge-decided labels were preserved**, so any error on the judge side is
  untouched here. The pilot measured Sonnet against Fable at kappa 0.84 and Codex
  against Sonnet at 0.818, which bounds but does not eliminate that.
- Identity leakage rises slightly (0.811 to 0.824 accuracy against a 0.195
  majority), so LODO remains the primary analysis and pooled numbers stay
  untrusted, exactly as registered.

## Reproducing this

```bash
python -m campaign.relabel_stage1 \
    --input out/campaign/stage1_judged.jsonl \
    --out out/campaign/stage1_judged_relabelled.jsonl \
    --report out/campaign/relabel_report.json
python -m campaign.build_feature_table \
    --input out/campaign/stage1_judged_relabelled.jsonl \
    --out out/campaign/stage1_features_relabelled.jsonl
python -m campaign.train_baselines --input out/campaign/stage1_features_relabelled.jsonl
python -m campaign.score_gate_b --input out/campaign/stage1_features_relabelled.jsonl
```

The selection control is the same pipeline with original labels restricted to the
`example_id`s that keep a label after correction.
