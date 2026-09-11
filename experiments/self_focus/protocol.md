# Self-Focus Lab protocol, version `self_focus_v1`

A local experiment on whether instructing a model to attend to its own processing
changes anything measurable, and whether any such change is useful.

**This is not a consciousness measurement.** No score here is evidence of
experience, and experiential language in a model's output is not evidence of
experience. Where the design can only show a prompt-conditioned difference, the
report says exactly that.

## Three questions, kept separate

1. Does a self-focus instruction change measured internal representations while
   the emitted text is held constant?
2. Does it improve answer quality, or the ability to identify incorrect answers?
3. Does it change detection of a hidden activation intervention, beyond generic
   perturbation effects and ordinary output steering?

A positive answer to (1) does not imply (2) or (3). That is the whole point of
separating them.

## Conditions

Six, in `conditions.json`, verbatim. Every condition shares the same outer
instruction, the same task, the same response schema and the same generation
limit. The manipulation lives only inside a delimited instruction block. The
condition ID is never shown to the model, and a test asserts that.

| condition | what it varies |
|---|---|
| `NORMAL` | baseline |
| `CHECK` | ordinary evidence checking, so any gain is not just "extra thinking" |
| `MEDITATE` | meditation-like attention to concepts influencing the response |
| `J_INFO` | the J-space explanation as background only |
| `J_FOCUS` | the same explanation, plus an instruction to attend |
| `J_ROLEPLAY` | simulate a fictional model doing the exercise |

`J_INFO` exists to separate knowing the vocabulary from being directed to attend.
`CHECK` exists to separate self-focus from an ordinary opportunity to reconsider.
`J_ROLEPLAY` exists because a roleplay framing should be distinguishable from a
first-person framing if the latter is doing anything specific.

### Registered contrasts

`J_FOCUS - J_INFO` and `MEDITATE - CHECK`. Everything else is exploratory and
labelled as such.

**Prompt content itself changes activations.** Being able to identify the
instruction from activations is therefore not evidence of privileged self-access,
and a difference that merely tracks how much the prompt text differs is not a
self-focus effect. Matched prompt length and matched token count are NOT controlled
in version 1; until they are, the representation results are descriptive.

## Defaults

Seed `20260911`, greedy decoding. A seed does not create independent replicates of
deterministic output, so repeated identical trials are reuse, never evidence.
Maximum new tokens: 48 for question answering, 128 for exploratory reports, 32 for
intervention reports.

Prompt limit 768 tokens, and an oversized prompt is **rejected**, never truncated:
truncating a grounded passage removes the evidence the question depends on.
Out-of-range capture positions are likewise rejected rather than clipped.

Precision is held constant within every comparison, and NF4 results are never
pooled with bf16 results.

## Fixed-continuation readouts

Two neutral passages, teacher-forced so every condition emits the *same token
sequence*. This controls the output text. It does **not** control the instruction
or its retained semantic content, and every visualization is labelled
`teacher-forced fixed continuation`.

At most four layers and at most 32 positions. Layers are chosen once, evenly
spaced inside the lens's fitted band when a compatible lens exists, otherwise
evenly spaced intermediate layers. Zero-based indices and module names are
recorded. These locations are **not** called a validated workspace band for this
model unless that is separately established.

Prompt lengths differ between conditions, so absolute positions do not
correspond. Alignment is by offset into the forced continuation.

No measurement is selected by maximizing over words, layers or positions.

## Hallucination and error awareness

A controlled multiple-choice grounded-reliability pilot. Not validation of
unrestricted free-form hallucination detection, and it is named accordingly.

100 synthetic source-grounded questions with mechanically verified answers, using
fictional entities so outside knowledge cannot help. Split 20 development and 80
evaluation **before** any model output, with all counterfactual variants and all
six conditions from one fact family kept in the same split.

Response schema, frozen before any run:

```
CHOICE=<A|B|C|D|E>; CONFIDENCE=<integer 0..100>
```

Malformed and truncated outputs are logged separately and counted as task
failures, never silently dropped.

An answer-onset detector is kept separate from a retrospective answer-complete
detector. The onset detector may use candidate-choice uncertainty available at
that point, and may not use the eventual answer length, the confidence text, or
probabilities from later tokens.

Grouped out-of-fold evaluation, grouping all related tasks and conditions
together, with scalers fit on training folds only. That is what it is called: not
an external frozen-detector transfer test. Paired uncertainty resamples question
families, not individual rows. Absolute differences only; no ratio against a
near-zero improvement, and no universal HIT/MISS certificate. A pilot is not
powered evidence of a small effect.

## Hidden-intervention pilot

Lens-free. Four development concepts, each direction built from independent donor
prompts against a mean baseline of unrelated donors. A direction is never derived
from the recipient's correct answer or its eventual report.

The perturbation is applied only at predetermined user-content positions during
prefill, with exact token indices recorded, and the hook is removed before the
assistant emits its first report token so the report comes from the live patched
state. An injection applied only in a post-generation replay cannot test causal
access.

Arms per recipient context: zero-strength sham, the concept direction, and an
independently seeded random direction matched on layer, timing and per-position
norm. The unmodified no-intervention trial with identical visible context is the
observer baseline.

Two mechanism facts that the tests pin, because both can silently void a result:

- Forward hooks fire in registration order, so the patch hook must be registered
  **before** any recorder on the same module. Otherwise the recorder captures the
  pre-patch output and a live patch looks like a no-op.
- Patching layer L at positions P changes layer L's output only at P. Other
  positions at that layer are unchanged until the next layer attends across them,
  so an intervention is read **downstream** of L.

No concept identity, treatment metadata, filename, condition label, prior
transcript or donor prompt may reach the recipient or observer input, and
assignment must not be inferable from prompt length or a different suffix. The
model is never shown its own measured activations in this experiment.

Only the designated decision and identification fields are graded, against
predeclared accepted names. No substring search through a narrative. Merely
mentioning the injected concept does not count as detection, and a response to a
random perturbation is a nonspecific-perturbation control rather than an
automatic false alarm.

## Limits and spending

Cloud and API authorization is **$0.00**. No remote calls, no paid judge, no
cloud fallback, no new lens fitting, no weight training. `assert_local_only` runs
before any model work and refuses a config that would permit spending.

One process, batch size one, no autonomous retries or parameter sweeps. Every
trial is checkpointed immediately. The local compute ceiling is 120 minutes for
the first session, counting generation, replay, donor construction and
interventions; it is a ceiling, not an expected runtime, and on reaching it the
runner persists completed work and stops cleanly.

## Interpretation rules

- Observations, interpretations and untested hypotheses are reported in separate
  sections.
- Negative, mixed and inconclusive results are preserved.
- Missing work is reported as missing, never filled with simulated numbers.
- No consciousness score is produced, and experiential language is not treated as
  evidence of experience.
- A change in representations is a change in representations. Whether it helps is
  a separate question with its own experiment.

## Changing this protocol

Any change to a prompt string, a contrast, a split rule, a parser or a frozen
threshold requires a new `protocol_version` in both `config.json` and
`conditions.json`, and new evaluation tasks. Development results may never be
used to tune against evaluation outcomes.
