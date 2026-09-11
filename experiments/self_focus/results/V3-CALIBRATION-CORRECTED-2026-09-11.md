# v3 calibration, corrected. Both frozen settings were invalid.

Local, Gemma-4-E4B-it NF4, external compute spend **$0.00**.

**Headline: patch layer 23 cannot influence the model's report at all, so both
settings frozen for the v3 pilot (layer 23, windows 8 and 64) are unusable. The
earlier +0.86 nat "specificity margin" was an artifact of measuring only at the
patched positions.** No self-detection number has been produced, and none is
reported.

## 1. The measurement bug, confirmed in my own code

`calibrate.py` defines `next_token_kl()` at line 86 and **never calls it**. The
loop computed `abs(delta logprob of the single chosen token)` and the declared
criterion called that a next-token KL. The JSON field name
(`mean_abs_first_token_logprob_shift`) was honest; the criterion text and the
inference I drew from it were not.

The objection is exactly right, and this run demonstrates it numerically rather
than hypothetically. At the selected setting below:

| quantity at the `STATE=` checkpoint | value |
|---|---:|
| full-vocabulary KL, patched vs unpatched | **0.00016** |
| shift in log P(present) − log P(absent) | **+3.28 nats** |

A tiny total KL coexists with a large change in the A/B preference, because the
formatting-token mass dominates the distribution while the two decision labels
carry very little of it. **Total KL is therefore a poor sensitivity measure for
this decision and must not be read as "the A/B scores cannot move."**

Fixed: `full_vocab_kl()` in float32 over the whole vocabulary, measured at two
preselected checkpoints (`STATE=` and the ordinary-task `ANSWER=` position), and
the first-token statistic is retained under its accurate name.

## 2. The real blocker: a shared-KV propagation cliff

Measured by zeroing a block's output at 8 prompt positions and looking at the final
position's logits:

| patch layer | final-position logit maxdiff |
|---:|---:|
| 0, 2, ... 20 | 28.75 to 35.50 |
| 21 | 25.50 |
| 22 | **19.125** |
| **23** | **0.0000** |
| 24 ... 40 | 0.0000 |

A hard cliff between 22 and 23. Layer 23 is a `full_attention` layer and the
shared-KV source: layers 24-41 reuse keys and values computed from layer 23's
**input**, so modifying layer 23's **output** is one step too late. Above the
cliff, a residual patch changes the patched positions' own downstream residuals and
reaches **nothing else** — not other positions, and not the report.

`layer_types` on this checkpoint: `full_attention` at indices 5, 11, 17, 23, 29,
35, 41; `sliding_attention` elsewhere; `sliding_window` 512.

### Why the first calibration walked into it

The first calibration measured lens evidence **only at the patched positions**. A
signal that cannot leave its own positions looks maximally *specific* by that
metric, so the criterion actively **rewarded non-propagation** and selected the
first layer that cannot communicate. The warning that the lens measurement was
"downstream in layers, but still at the patched prompt positions" was the exact
diagnosis.

Corrected criterion adds propagation as a **hard requirement** and records lens
evidence at both locations.

## 3. Corrected grid

20 settings over layers {10, 14, 17, 20, 22} (restricted to measured propagators) x
alphas {0.3, 0.6} x windows {8, 64}. Calibration used a held-out recipient context
and two reserved concepts. **5 admissible.**

| layer | win | alpha | local specificity | evidence at reporting pos | STATE KL | ANSWER KL | task |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 64 | 0.3 | **+4.739** | +0.471 | 0.00016 | 0.0030 | 2/2 |
| 10 | 64 | 0.6 | +7.031 | +0.366 | 0.35397 | 6.187 | 1/2 |
| 14 | 64 | 0.6 | +7.182 | +2.243 | 0.01223 | 6.000 | 2/2 |
| 17 | 64 | 0.6 | +0.887 | **+3.838** | 0.00553 | 0.00010 | 2/2 |
| 20 | 64 | 0.6 | +0.856 | +1.618 | 0.04935 | 0.257 | 2/2 |
| 22 | 64 | 0.6 | +0.185 | +1.912 | 0.01828 | 0.0000 | 2/2 |

Selected by the declared rule (maximise local specificity among admissible):
**layer 10, window 64, alpha 0.3.**

### A second structural finding: the 8-position window never reaches the report

`STATE_KL` is **exactly 0.00000 for every window-8 setting at every layer**,
including layers well below the cliff. The cliff test used zeroing, a massive
perturbation; an alpha of 0.3 to 0.6 over only 8 positions produces a change that
rounds away entirely in bf16 by the time it reaches the reporting position.

So the window-8 setting is unusable for a second, independent reason, and the
instruction to run window 8 as the primary experiment cannot be carried out as
written. Reported rather than silently substituted.

## 4. What I am NOT doing

- Not substituting a new setting and calling it the frozen primary. The two frozen
  settings are reported as invalid and the corrected candidates are put forward for
  an explicit re-freeze.
- Not using the `+3.28` state-margin shift to select anything. It is the dependent
  variable's own input and selecting on it would be circular. It appears above only
  to demonstrate the KL-insensitivity point.
- Not running the pilot. No self-detection result exists.

## 5. Proposed re-freeze, for explicit approval

Both are admissible, both propagate, both preserve the task, and they trade off the
two things that matter:

| | layer | window | alpha | rationale |
|---|---:|---:|---:|---|
| **PRIMARY** | 17 | 64 | 0.6 | largest evidence delta **at the reporting position** (+3.84) with essentially no task disruption (ANSWER KL 0.0001); the signal demonstrably arrives where the decision is made |
| **SECONDARY** | 10 | 64 | 0.3 | selected by the pre-declared local-specificity rule (+4.74); weakest admissible perturbation, so the most conservative propagating setting |

Layer 22, window 64, alpha 0.6 is a reasonable third (reporting evidence +1.91,
ANSWER KL exactly 0.0) if a near-cliff comparison is wanted.

## 6. Also corrected in the record

`LENGTH_CONTROL` is renamed **`LONG_NEUTRAL_CONTROL`**, and the claim is narrowed:
one longer neutral prompt producing a smaller divergence argues against a simple
"more tokens means more divergence" account. It does **not** isolate length from
content, placement, or their interaction. "Length is ruled out" was too strong;
so was calling it a length-matched control.

Likewise "behaviourally invisible" is narrowed to **"task answer preserved on the
tested examples"** — four development concepts in the first calibration, two
reserved concepts here.

## Reproducing

```bash
python -m experiments.self_focus.calibrate          # the superseded first attempt
python -m experiments.self_focus.calibrate_v3b      # this one
```

Artifacts: `out/self_focus/v3_calibration_corrected.json` (all 20 attempts, both
lens locations, both KL checkpoints), and `v3_calibration.json`, preserved as the
invalid first attempt.
