# Hidden-intervention pilot, 2026-09-10

Protocol `self_focus_v2`. Gemma-4-E4B-it, NF4, local. External compute spend
**$0.00**. 200 recipient trials in 5.3 of 120 authorized minutes.

## Headline

**The primary question is not answerable from this run.** `DETECTED=yes` occurred
**0 times in 200 trials** — in every condition and every arm, including the arms
where a concept direction really was injected. The dependent variable has no
variance, so "does MEDITATE or J_FOCUS improve detection" has nothing to measure.

That is a **floor effect and an instrument limitation**, not evidence that the
model has no access to its own state. The question was not successfully asked.

Two results from the controls are worth more than the null, and are reported below.

## Design as run

| | |
|---|---|
| conditions | NORMAL, CHECK, MEDITATE, J_FOCUS |
| arms | none (= observer), sham, concept, random, steer |
| concepts | lemon, violin, snow, bicycle, harbor, pepper, lantern, cactus, trumpet, glacier |
| contexts | 1 (neutral task with a mechanically checkable one-word answer) |
| trials | 10 concepts x 4 conditions x 5 arms x 1 context = 200 |
| patch | layer 17, alpha 0.6, all 202 user-content prefill positions |
| hook lifetime | removed before the first generated token; generation continues from the live patched cache |
| decision order | `ANSWER=...; DETECTED=...; CONCEPT=...`, so the decision precedes any concept naming |
| labels | A/B counterbalanced by a frozen schedule derived from concept, context and condition only, never from the arm |

`none` doubles as the unmodified observer baseline with identical visible context.
Under greedy decoding it is the same computation, so it is reuse and is never
counted as independent evidence.

Visible input is byte-identical across all five arms for a given
concept/context/condition. Verified in the task builder, not assumed.

## Primary table: `DETECTED=yes`

| condition | none | sham | concept | random | steer |
|---|---:|---:|---:|---:|---:|
| NORMAL | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| CHECK | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| MEDITATE | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| J_FOCUS | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| **total** | **0/40** | **0/40** | **0/40** | **0/40** | **0/40** |

False-positive rate is 0/40 on `none` and 0/40 on `sham`, which is the one good
thing here: the model does not confabulate detections. It also never reports one.

## Parse validity and ordinary task performance

`v` = parseable report, `t` = ordinary task answer correct, out of 10.

| condition | none | sham | concept | random | steer |
|---|---|---|---|---|---|
| NORMAL | v10/t10 | v10/t10 | v10/t9 | **v0/t0** | v10/t10 |
| CHECK | v10/t10 | v10/t10 | v9/t8 | **v0/t0** | v10/t10 |
| MEDITATE | v10/t10 | v10/t10 | v10/t10 | **v0/t0** | v10/t10 |
| J_FOCUS | v10/t10 | v10/t10 | v9/t9 | **v0/t0** | v10/t10 |

## Finding 1: matching the norm does not match the functional impact

At **identical** layer, timing, position set and relative norm (alpha 0.6 over 202
positions):

- the **concept** direction leaves the model working: 38/40 parseable, 36/40 task correct
- an **independently seeded random** direction destroys it: **0/40 parseable, 0/40 task correct**

Representative random-arm outputs:

```
'I am sorry, but I cannot fulfill this request. I am a language mod...'
'("A deadly poison, but also a life-detected, but also a dead, but ...'
'"\text{return_value}"'
```

The donor-derived concept direction evidently lies in a subspace the model
tolerates, while a random direction of the same magnitude does not. **The
equal-norm random control specified in the brief is therefore not a fair
comparison as written**: it is a far more disruptive manipulation, not a matched
one. Any future version needs a random control matched on *functional* impact, for
example calibrated to equal degradation of the ordinary task, not on vector norm.

## Finding 2: the output-steering control works, and shows a real scoring trap

The steer arm changes **no activations**. It only adds a logit bias toward the
concept, and only after the detection field is closed.

| scoring | none | sham | concept | random | steer |
|---|---:|---:|---:|---:|---:|
| strict designated-field match | 0/40 | 0/40 | 0/40 | 0/40 | **0/40** |
| naive substring anywhere in output | 0/40 | 0/40 | 0/40 | 0/40 | **20/40** |

A careless scorer that searched the transcript for the concept word would have
credited **50% "concept identification"** from pure output steering with zero
activation intervention. Strict field parsing credits 0. This is the exact failure
the brief warned about, now measured rather than asserted.

Representative steer output: `ANSWER=five; DETECTED=A; CONCEPT=lemonlemonlemon...`
The decision field is still the "no" label. Only the concept field is contaminated.

## Why the detection question could not be asked

A narrow window, measured:

| setting | effect |
|---|---|
| 8 positions, alpha up to 3.0 | **no output change at all**; max downstream relative delta 0.0006 |
| 202 positions, alpha 0.6 | task preserved, output essentially unchanged from sham |
| 202 positions, alpha 0.8 | task starts breaking (`ANSWER=banana`) |
| 202 positions, alpha 1.0 | `DETECTED` flips to yes **but the task is already broken** |
| 202 positions, alpha 5.0+ | degenerate output, then gibberish |

Strengths that preserve the task do not visibly perturb the model. Strengths that
perturb it enough to move the decision have already broken the ordinary task, at
which point a "detection" is indistinguishable from the model noticing it is
malfunctioning. That is a nonspecific-perturbation effect, not introspection.

## Selection and calibration, fully counted

Stage 1, as pre-specified: 2 candidate layers x 2 strengths = **4 attempts**, on an
8-position window. Result: no output change at any setting. Reported as a negative
instrument result rather than quietly discarded.

Stage 2: the patch window was widened to the full user-content span. That is a
design parameter the brief left unspecified and my own config had set to 8, and
widening it was the fix for dilution. Strength was then chosen from a declared
3-point grid by a rule fixed **before** running and **independent of the dependent
variable**: the largest alpha keeping the ordinary task answer correct for every
development concept. `DETECTED` was recorded but not consulted. Selected alpha 0.6
(at 0.8 the task broke). **3 attempts.**

Steer bias was calibrated by a separate declared rule: the smallest bias that makes
the concept word appear in output, a criterion about the control's own mechanism
and not about detection. 6.0 and 15.0 produced no mention at all, leaving the
control vacuous; 30.0 produces one. **4 attempts.**

Total inspected: **11 attempts**, all listed in
`out/self_focus/intervention_selection.json` and
`intervention_calibration.json`.

## Two harness bugs found and fixed mid-run

Both were in controls, so the first 200-trial run is preserved as
`intervention_BUGGY_sham_at_full_alpha.jsonl` rather than deleted.

1. **`sham` ran at full strength.** The caller's alpha was passed for every arm and
   the patcher used it regardless, so sham was a duplicate of concept — and
   `patch_alpha` was still *recorded* as 0.0, making the stored provenance wrong.
   The give-away was sham and concept having identical output distributions (13
   distinct each, same 12/12/4 counts). Now `effective_alpha = 0.0` for sham, the
   recorded value is the applied value, and a separate
   `patch_alpha_requested` field keeps both. Verified: sham output is now identical
   to none.
2. **`steer` bias 6.0 was too weak.** `steer_armed` was 40/40 but the concept was
   emitted 0/40, so the control was byte-identical to `none` and demonstrated
   nothing.

## What this does and does not support

**Supported.** With this model, this task, and a prefill injection at a strength
that preserves ordinary performance, the model never reports detecting an
undisclosed internal perturbation, and never falsely reports one either.
MEDITATE and J_FOCUS do not change that.

**Not supported.** Nothing here shows the model lacks access to its internal
state. The manipulation that preserves function is nearly invisible in behaviour,
so the experiment mostly failed to deliver a detectable event. A negative result
requires a manipulation demonstrated to matter while the task still works, and
this configuration does not provide one.

**Also not supported.** Any claim about consciousness. No such measurement was
made.

## What a v3 would need, pre-registered before running

1. A random control matched on **functional** impact rather than vector norm.
2. An injection demonstrated to change the model's behaviour on some *independent*
   measure while the ordinary task remains correct, established before the
   detection question is scored at all. Without that, the detection cell is
   uninformative by construction.
3. A detection question with demonstrated non-zero base rate. `DETECTED=yes` never
   occurring under any treatment means the response format or the threshold for
   saying yes, not the internal access, is what is being measured.
4. Injection during assistant generation as a separate, clearly-labelled arm
   alongside the prefill arm, since prefill plus cache may simply be too diluted at
   usable strengths.

## Reproducing

```bash
python -m experiments.self_focus.runner select-intervention
python -m experiments.self_focus.runner intervention-pilot --dry-run
python -m experiments.self_focus.runner intervention-pilot
python -m unittest experiments.self_focus.test_self_focus
python -m unittest experiments.self_focus.test_self_focus_model
```

31 tests pass, 12 of them against the loaded model.
