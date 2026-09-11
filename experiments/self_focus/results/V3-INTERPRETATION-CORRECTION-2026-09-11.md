# Dated interpretation correction to the v3 pilot, 2026-09-11

No trials were rerun. The measurements in `MORNING_REPORT.md` stand. Three of its
**interpretations** were wrong and one headline overreached.

## 1. "Baseline artifact" was mathematically void

The report attributed J_FOCUS's AUROC advantage partly to its baseline margin
sitting 12 nats closer to the decision boundary. **AUROC is rank-based, so adding a
constant to every score within a condition cannot change that condition's AUROC.**
Verified numerically: 1.0000 before adding 12 to every score, 1.0000 after.

The baseline offset explains nothing about within-condition separation. Withdrawn.

## 2. "Lower variance is an artifact" was unjustified

A smaller but more consistent shift can be the **better** detector, because AUROC
measures separation rather than mean displacement. Verified: a +4.0 shift with sd
4.0 gives AUROC 0.786, while a +1.0 shift with sd 0.3 gives 0.993.

So J_FOCUS having a smaller mean shift (+2.234 against NORMAL's +3.640) and a higher
AUROC is not self-contradictory and is not evidence of an artifact. Withdrawn.

## 3. "Specificity kills it" conflated two different questions

Presence detection and concept identification are separate.

- **`wrong_concept` is an intervention.** My own code labels it
  `intervention_present: true`. Its raising the presence score is a **success** of
  presence detection, not a failure. Treating `wrong_concept ≈ concept` as a
  specificity failure was a category error: it is only a failure for
  *identification*, which was separately at floor.
- **The random arm does not settle specificity.** It destroys the ordinary task
  (1/40 and 0/40 correct against 40/40 for every other arm), so comparing its shift
  to a task-preserving concept injection compares perturbations of very different
  severity. Its larger response does not make the concept response meaningless, and
  does not show the concept response is introspection either. The experiment did not
  separate those explanations.

## 4. Corrected headline

Replacing "the score moves, but tracks disruption not concept":

> Internal interventions shift the model's monitoring score. J_FOCUS shows promising
> separation in this small pilot, but the mechanism and generality remain unresolved.

J_FOCUS is preserved as an **exploratory** result: neither demonstrated introspection
nor a demonstrated artifact. The condition stays in the design. Eight concepts and one
recipient context are far too narrow for any generality claim.

"Clear negative with a working instrument" was too strong and is withdrawn.

## 5. Two code limitations that bound the claims

Both confirmed in the committed code, both limiting claims the report made.

**Substring task grading.** `pilot_v3.py` line 269 scores the ordinary task as
`expected_answer in text.lower()`. That is substring matching, which the protocol
elsewhere forbids, and it inflates task-preservation: a degenerate output containing
the word anywhere counts as correct. The 40/40 task-preservation figures are
therefore upper bounds, and the random arm's 1/40 and 0/40 are the only figures the
substring rule cannot flatter.

**The alternative concept can be missing from the answer choices.** `choice_options`
builds the five options around the *trial's* concept, while the `wrong_concept` arm
injects a different one. Measured on the saved trials: **40 of 80 `wrong_concept`
trials (50%) had the injected concept absent from the five options**, so
`correct_letter` was `None` and `identified` could never be `True` for them.

The 0/80 strict identification result is therefore **not** a clean floor: for half
the `wrong_concept` trials the correct answer was unavailable. The 0/40 on the
`concept` arm stands, since the trial's own concept is always an option there.

## 6. What still holds

Unaffected by any of the above:

- The forced `STATE=` prefix lands on a real decision: combined label mass 1.0000,
  top token is a label on 100% of trials.
- `sham` shifts the score by exactly 0.000 in every cell.
- `steer` shifts it by exactly 0.000 in every cell, so post-decision steering
  provably cannot reach the measurement.
- The continuous score registers changes the greedy choice never showed: greedy was
  "absent" on all 480 trials.
- Labels were counterbalanced independently of the arm, and the model tracked the
  instruction's semantics rather than a surface token.
- The score construction cannot see ground truth, and a constant preference gives
  AUROC exactly 0.5 under it.
- Layer 23 and above cannot reach the reporting position on this checkpoint.

## 7. Revised next steps

1. A control matched on **functional disruption**, since the random arm as run is
   not a matched control and its comparison is uninterpretable.
2. Fix both code issues before any rerun: strict designated-field task grading, and
   option sets built around the **injected** concept so identification is always
   answerable.
3. More concepts and more recipient contexts before any generality claim about
   J_FOCUS.
