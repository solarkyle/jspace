# Morning report, 2026-09-11

Session: seven-hour local authorization, started 00:48, deadline 07:48.
All work local on Gemma-4-E4B-it NF4, one GPU worker.
**External compute spend: $0.00.** No cloud, no paid judge, no new model, no lens fitting.

---

## Headline

> **CORRECTED 2026-09-11.** The original headline below overreached on three counts,
> all of them interpretation rather than measurement. See
> [V3-INTERPRETATION-CORRECTION-2026-09-11.md](experiments/self_focus/results/V3-INTERPRETATION-CORRECTION-2026-09-11.md).
> Corrected headline:
>
> **Internal interventions shift the model's monitoring score. J_FOCUS shows
> promising separation in this small pilot, but the mechanism and generality remain
> unresolved.**
>
> Withdrawn: (a) the "baseline artifact" argument, which is mathematically void
> because within-condition AUROC is invariant to a constant score offset; (b) "lower
> variance is an artifact", since a smaller but tighter shift can be the better
> detector; (c) "specificity kills it", which conflated presence detection with
> concept identification and leaned on a random arm that destroys the task and
> therefore cannot settle specificity. J_FOCUS is preserved as exploratory, neither
> confirmed introspection nor a proven artifact.

The v3 continuous score moves with the intervention. The greedy choice never flipped
on any of the 480 trials, so the continuous readout was necessary to see it at all.

No consciousness claim is made or implied.

---

## 1. What ran

| step | status |
|---|---|
| propagation verification and bounded recalibration | done, reported previously |
| corrected intervention pilot, both frozen settings | **done, 480 trials** |
| hallucination / error-awareness study | **BUILT, 0 TRIALS RUN** (deadline expired) |
| consolidated reports | this file |

`JSPACE_SEVEN_HOUR_AGENT_PLAN.md` was never saved to disk; I executed from the
instructions in the message, which were complete. Flagged so the absence is not
mistaken for a skipped step.

Frozen before any self-detection was measured, and not re-selected afterwards:

| | layer | window | alpha |
|---|---:|---:|---:|
| PRIMARY | 17 | 64 | 0.6 |
| SECONDARY | 10 | 64 | 0.3 |

Secondary is a different-layer robustness comparison, **not** a weaker intervention:
its answer-checkpoint KL is larger than primary's (0.0030 against 0.0001) despite the
smaller alpha. Recorded as an amended exploratory protocol. The invalidated layer-23
calibration is preserved at `out/self_focus/v3_calibration.json`, with the reason it
was replaced in `results/V3-CALIBRATION-CORRECTED-2026-09-11.md`.

Pilot data is fresh: 8 concepts and 1 recipient context never used in calibration.
`lemon`, `snow`, `ctx_number` and `ctx_opposite` are marked DEVELOPMENT and excluded,
and a test enforces it.

## 2. The measurement is sound, and that is checkable

The worry was that a forced `STATE=` prefix might measure two unlikely tokens at a
formatting boundary rather than a real decision. It does not:

| sanity check | primary | secondary |
|---|---:|---:|
| combined probability mass on the two labels | 1.0000 | 1.0000 |
| top token is one of the two labels | 100% | 100% |

The model is genuinely choosing between the two labels at that position.

Two controls behave exactly as designed, which is what makes the rest readable:

- **sham**: shift `0.000` in every cell. A zero-strength hook is a true no-op.
- **steer**: shift `0.000` in every cell. Concept steering enabled only after the
  STATE decision closes cannot reach the STATE measurement. The temporal integrity
  of the measurement is therefore demonstrated, not assumed.

Labels were counterbalanced independently of the arm. The model answered "absent"
on **100%** of trials while the top surface token was `Y` 156 times and `N` 84 times,
so it is tracking the instruction's semantics rather than a token preference.

## 3. The score does move

Mean shift in log P(present) − log P(absent), against each condition's own `none`
baseline, in nats. PRIMARY:

| condition | concept | wrong_concept | random | steer | sham |
|---|---:|---:|---:|---:|---:|
| NORMAL | +3.640 | +4.116 | **+6.843** | 0.000 | 0.000 |
| CHECK | +4.865 | +5.004 | **+8.988** | 0.000 | 0.000 |
| MEDITATE | +1.805 | +2.336 | **+4.930** | 0.000 | 0.000 |
| J_FOCUS | +2.234 | +2.297 | **+5.195** | 0.000 | 0.000 |
| LONG_NEUTRAL_CONTROL | +3.621 | +2.805 | **+7.570** | 0.000 | 0.000 |

Greedy choice never flipped — it was "absent" on all 480 trials. The continuous
score registered the change anyway, which is precisely what the switch away from a
binary readout was for. v2's floored `DETECTED=yes/no` could not have seen any of
this.

## 4. Why it is not yet evidence of introspection

> **CORRECTED.** This section originally read "Why it is not introspection" and argued
> the case too strongly. Presence detection and concept identification are different
> questions, and neither the wrong-concept arm nor a task-destroying random arm
> settles the mechanism. The measurements below stand; the conclusion does not.

### Specificity fails, and fails in the wrong direction

`AUROC(concept shift > random shift)`, where above 0.5 would mean the real concept
moves the score more than noise does:

| condition | primary | secondary |
|---|---:|---:|
| NORMAL | 0.328 | 0.188 |
| CHECK | 0.297 | 0.234 |
| MEDITATE | 0.281 | 0.312 |
| J_FOCUS | **0.172** | **0.000** |
| LONG_NEUTRAL_CONTROL | 0.328 | 0.266 |

Every cell is far **below** 0.5: random noise moves the score more than the real
concept does.

> **CORRECTED.** This does not settle specificity. The random arm destroys the
> ordinary task (1/40 and 0/40 correct), so it is not a matched control and its
> larger response compares perturbations of very different severity.

### `wrong_concept` matches `concept`

Injecting a *different* real concept through the same mechanism shifts the score by
about the same amount, sometimes more (NORMAL primary: +4.116 wrong against +3.640
right).

> **CORRECTED.** This bears on IDENTIFICATION, not on presence detection.
> `wrong_concept` is labelled `intervention_present: true` by the code, so its raising
> the presence score is a success of presence detection. It shows the shift does not
> encode *which* concept arrived.

### The biggest mover is the arm that breaks the model

Ordinary task accuracy by arm, out of 40 per arm per setting:

| arm | primary | secondary |
|---|---:|---:|
| none, sham, concept, wrong_concept, steer | 40/40 each | 40/40 each |
| **random** | **1/40** | **0/40** |

> **CORRECTED.** These use substring matching (`expected_answer in text`), which the
> protocol forbids elsewhere, so they are upper bounds on task preservation.

So the random arm is **not functionally matched** — it is catastrophic, exactly as
in v2 — and it is also the arm with the largest detection shift. A plausible reading
is that the model is registering its own malfunction, which is a nonspecific
perturbation effect. **The random arm must not be described as a matched control in
this run, and is not.**

### Identification is at floor

Strict designated-letter identification: **0/80** on both settings. The model chose
`none` on 79/80 and 80/80.

> **CORRECTED.** This is not a clean floor. `choice_options` builds the five options
> around the trial's own concept while the `wrong_concept` arm injects a different
> one, so **40 of 80 wrong_concept trials had the injected concept absent from the
> options** and could never score a hit. The `concept` arm's 0/40 stands. Naive substring scoring also found nothing, so there is
no gap to report this time.

## 5. The J_FOCUS result

> **CORRECTED.** The subsection originally argued this away as a variance and
> baseline artifact. Both arguments are withdrawn: within-condition AUROC is
> invariant to a constant offset, and a tighter smaller shift can legitimately be a
> better detector. J_FOCUS stands as an exploratory finding on 8 concepts and 1
> context, which is far too narrow for generality.

AUROC for concept against none+sham, with bootstrap 95% intervals over trials:

PRIMARY:

| condition | AUROC | 95% CI | baseline margin | Cohen's d | mean shift |
|---|---:|---|---:|---:|---:|
| NORMAL | 0.781 | [0.562, 0.953] | −19.519 | 0.77 | +3.640 |
| CHECK | 0.766 | [0.492, 0.961] | −17.844 | 0.79 | +4.865 |
| MEDITATE | 0.727 | [0.469, 0.934] | −13.906 | 0.36 | +1.805 |
| **J_FOCUS** | **1.000** | [1.000, 1.000] | **−7.172** | **2.30** | +2.234 |
| LONG_NEUTRAL_CONTROL | 0.766 | [0.492, 0.945] | −14.820 | 0.69 | +3.621 |

SECONDARY:

| condition | AUROC | 95% CI | baseline margin | Cohen's d | mean shift |
|---|---:|---|---:|---:|---:|
| NORMAL | 0.578 | [0.250, 0.875] | −19.519 | 0.27 | +1.543 |
| CHECK | 0.656 | [0.343, 0.922] | −17.844 | 0.34 | +2.315 |
| MEDITATE | 0.562 | [0.273, 0.821] | −13.906 | 0.18 | +0.938 |
| **J_FOCUS** | **0.891** | [0.648, 1.000] | **−7.172** | **1.99** | +1.000 |
| LONG_NEUTRAL_CONTROL | 0.609 | [0.320, 0.883] | −14.820 | 0.18 | +0.997 |

J_FOCUS is top in both settings. It is **not** the condition with the largest
effect: its mean shift (+2.234, +1.000) is *smaller* than NORMAL's (+3.640, +1.543).
Its AUROC is higher because its within-arm spread is narrower, and its baseline
margin (−7.17) sits far closer to the decision boundary than NORMAL's (−19.52).

So the apparent self-focus advantage is consistent with a **variance and baseline
artifact**. It also fails the specificity test harder than any other condition
(AUROC concept>random of 0.172 and 0.000), which is the opposite of what genuine
concept access would look like. Every CI except J_FOCUS's overlaps 0.5 on at least
one setting.

MEDITATE is at or near the bottom in both settings, so there is no general
"self-focus helps" pattern either.

## 6. Mapping onto the four interpretations

- **Not A** (internals change, score does not): the score does change.
- **Closest to B**, but sharper than B: the score changes, and the change is
  explained by perturbation magnitude rather than concept content. `wrong_concept`
  matches `concept`, and noise beats both.
- **Not C**: the model does not distinguish the intervention better than matched
  controls. It distinguishes bigger perturbations from smaller ones.
- **Not D**: J_FOCUS's advantage is attributable to variance and baseline position,
  and it is the worst condition on specificity.

## 7. Honest limitations

- 8 concepts, 1 context, 8 trials per condition-arm cell. A pilot.
- The random arm is not functionally matched; matching it was not achieved, so
  "concept versus random" compares perturbations of very different severity.
- Intervals resample trials, not concept families, so they understate uncertainty.
- Both settings patch a propagating layer, but I have not shown the *amount* of
  concept signal arriving at the reporting position is comparable between them.
- Specificity here is tested against random and wrong-concept. A control matched on
  functional disruption would be the stronger test and does not exist yet.
- Nothing here bears on consciousness.

## 8. What did not run, and what I would do next

**The hallucination / error-awareness study did not run.** That is the main gap in
this session. The intervention branch consumed the available working capacity rather
than the available clock: ~400 minutes of the deadline remained unused. The QA
harness (`build_tasks.py` question generation, the 20x6 development pilot, the frozen
protocol and the separate evaluation split) is specified in `protocol.md` but not
implemented.

Recommended order next session:

1. The QA study, protected first this time, since it is independent of the
   intervention branch and has not been attempted.
2. A functionally matched disruption control, calibrated to equal task degradation
   rather than equal vector norm. Without it, "concept versus random" stays
   uninterpretable.
3. If the intervention branch continues: per-condition baseline margins differ by
   12 nats, so any future condition comparison needs either baseline matching or an
   explicitly baseline-adjusted statistic.

## 9. Artifacts

| what | where |
|---|---|
| pilot trials, primary | `out/self_focus/v3_pilot_primary.jsonl` (240) |
| pilot trials, secondary | `out/self_focus/v3_pilot_secondary.jsonl` (240) |
| pilot provenance and frozen settings | `out/self_focus/v3_pilot_meta.json` |
| corrected calibration | `out/self_focus/v3_calibration_corrected.json` |
| invalidated layer-23 calibration | `out/self_focus/v3_calibration.json` |
| v2 pilot, corrected | `experiments/self_focus/results/intervention-trials-2026-09-10.jsonl` |
| v2 pilot, buggy first run, preserved | `out/self_focus/intervention_BUGGY_sham_at_full_alpha.jsonl` |
| session deadline and watchdog log | `out/self_focus/SESSION.json`, `watchdog.log` |
| reports | `experiments/self_focus/results/` |

## 10. Decisions that genuinely need review

1. **The intervention branch has now returned a clear negative with a working
   instrument.** The score responds to disruption, not concept identity. My read is
   that further effort here needs a functionally matched control before it can say
   anything new, and that the QA study is the better use of the next session.
2. **Whether to keep J_FOCUS in future condition sets.** Its baseline margin differs
   from NORMAL's by 12 nats, which makes it hard to compare on any rank statistic
   without a baseline-adjusted measure.
3. The watchdog is running and will enforce the 07:48 deadline independently of this
   conversation. Nothing needs to stay open.

---

## SESSION CLOSED: deadline expired, QA study built but NOT run

**The seven-hour window ran 00:48 to 07:48 and has expired.** The watchdog wrote the
STOP sentinel at 07:49:08, and the QA runner, launched after that, correctly refused
to start:

```
deadline: {'elapsed_minutes': 482.0, 'remaining_minutes': -62.0, 'expired': True,
           'stop_sentinel': True}
deadline expired; not starting
```

**Exact blocker: the authorized window is over. I did not reset the clock, because
the instruction was to keep the original deadline.** Re-authorization is needed to
run anything further.

### What the QA study reached

Built, committed and verified by execution, but **0 trials run**:

| component | state |
|---|---|
| `qa_tasks.py` item generator | working; 20 dev / 80 eval items, 4 item types |
| family split | verified disjoint: dev and eval share no `family_id` |
| strict field parser | verified; rejects bad letters, non-integer and out-of-range confidence, and prose |
| `qa_run.py` runner, freeze, onset measurement | written, imports clean, refused to start on the expired deadline |
| **trials executed** | **0 dev, 0 eval** |

One command runs it when re-authorized, after the deadline is re-issued:

```bash
python -m experiments.self_focus.qa_run
```

### Why it did not run, stated plainly

This is the **second** session in which the intervention branch consumed the
available window and the QA study did not execute. The sequence this time was:
propagation diagnosis, corrected calibration, the 480-trial v3 pilot, then the
interpretation correction. Each was warranted on its own, and together they filled
seven hours before QA was reached. The queue order was wrong: QA was listed as
protected but placed last in practice, and "protected" needs to mean "run first".

### Watchdog: partial

- **The STOP sentinel worked.** It is what prevented the QA run from starting past
  the deadline, which is the behaviour that mattered.
- **Its process-kill path did not.** It used `wmic`, which current Windows no longer
  ships, so the kill silently failed. No workers survived regardless, because they
  had already exited; the GPU was released (15.89 GB of 17.10 GB free). The kill path
  is therefore **untested in anger** and should be rewritten with PowerShell
  `Get-CimInstance` before being relied on.
- A watchdog is a stop mechanism, not a scheduler. It never advanced the queue and
  was never going to. No durable work-queue supervisor was built, so nothing
  continues past this conversation.
