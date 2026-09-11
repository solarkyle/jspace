# v3 manipulation calibration, and the LENGTH_CONTROL readouts

Protocol `self_focus_v2` readouts plus `self_focus_v3_calibration`. Local,
Gemma-4-E4B-it NF4. External compute spend **$0.00**. 0.7 minutes for the readouts,
the calibration grid inside the same 120-minute ceiling.

v2 is preserved exactly as published, including its buggy first run and its
corrected null. Nothing below rewrites it; one v2 *interpretation* is upgraded, and
that upgrade is stated as such.

## 1. LENGTH_CONTROL: prompt length is ruled out

`LENGTH_CONTROL` is neutral archival filler with no self-reference and no
instruction to attend, deliberately made longer than the manipulations.

Divergence from `NORMAL` at layer 17, mean cosine over the teacher-forced fixed
continuation, two passages:

| condition | prompt tokens | mean cos vs NORMAL |
|---|---:|---:|
| **LENGTH_CONTROL** | **242** | **0.9681** |
| CHECK | 133 | 0.9537 |
| MEDITATE | 154 | 0.9429 |
| J_INFO | 174 | 0.9424 |
| J_FOCUS | 208 | 0.9243 |
| J_ROLEPLAY | 203 | 0.8988 |

The longest prompt produces the **smallest** divergence. 132 extra neutral tokens
move the representation less than 44 extra self-focus tokens do.

**This rules out prompt length** as the explanation for the milestone-1
representation differences. The milestone-1 report said the differences "track how
much the prompt text differs"; that was too weak, and the remaining confound is
semantic **content**, not size. Self-focus instructions are semantically about
attention and processing; archival filler is not. Distinguishing "content about
processing" from "privileged self-access" still needs the causal experiment, which
is what v3 is for.

Across layers, the ordering holds: LENGTH_CONTROL stays closer to NORMAL than
MEDITATE or J_FOCUS at every recorded layer.

| layer | LC vs NORMAL | LC vs MEDITATE | LC vs J_FOCUS |
|---:|---:|---:|---:|
| 10 | 0.9984 | 0.9978 | 0.9982 |
| 17 | 0.9681 | 0.9407 | 0.9294 |
| 23 | 0.9874 | 0.9675 | 0.9696 |
| 30 | 0.9962 | 0.9901 | 0.9895 |

### Incidental: this pipeline reproduces exactly

Re-running the six v1 conditions under v2 gives mean cosine `1.00000000` against
the v1 residuals. Worth recording given that the July campaign's generations did
not reproduce: the difference is that this runner pins a local snapshot and records
its file hashes.

## 2. Why v3 calibrates on an internal measure

v2 selected strength by ordinary-task preservation. That kept the model working but
produced a manipulation that was invisible in behaviour, so the detection cell
could not distinguish "the model cannot report it" from "there was nothing to
report".

v3 asks the fitted lens first: **did the injected concept actually become more
represented downstream?** Only then is the model asked whether it noticed.

### Criterion, declared before any self-detection was measured

Admissible settings had to satisfy all of:

1. ordinary task answer correct for every development concept
2. report parseable for every development concept
3. injected-concept lens evidence delta strictly positive
4. **specificity**: injected delta larger than the mean delta for unrelated
   concepts, so a general rise in everything does not qualify
5. first-token logprob shift below 2.0, so the output distribution is perturbed
   without being destroyed

Among admissible settings: maximise the specificity margin. Self-detection
performance was not consulted anywhere in the calibration.

### Grid and result

18 settings: layers {17, 23} x alphas {0.3, 0.6, 0.9} x windows {8, 64, all user
content}. **13 admissible.**

Selected:

| | |
|---|---|
| patch layer | 23 |
| lens read layer | 29 |
| window | 8 user-content positions |
| alpha | 0.9 |
| **injected-concept evidence delta** | **+1.2107 nats** |
| unrelated-concept evidence delta | +0.3491 |
| **specificity margin** | **+0.8616** |
| ordinary task | 4/4 |
| parseable | 4/4 |
| first-token logprob shift | 0.0000 |

**This is a confirmed internal change that is behaviourally invisible.** The lens
says the concept arrived and is specifically elevated over unrelated concepts,
while the task answer and even the first generated token's probability are
untouched. That is the configuration the detection question needs: a real event
that the model's output does not already give away.

Notable structure: the narrow 8-position window gives **better specificity** than
wide windows. Wide windows raise evidence for everything (unrelated +1.48 at L17
w300 a0.6), so the margin shrinks. Patching more is not patching better.

## 3. This upgrades the v2 interpretation

The same grid re-measured v2's own setting (layer 17, window 300, alpha 0.6):

| | v2 setting |
|---|---:|
| injected-concept evidence delta | **+1.9121 nats** |
| unrelated delta | +1.4882 |
| specificity margin | +0.4239 |

So **v2's manipulation did produce a measurable internal change**, which v2 could
not establish because it measured no internal quantity. The v2 report called its
result "a floor effect and an instrument limitation, not evidence that the model
lacks access". With this measurement, v2 is better described as the first of the
four cases:

> **Case A.** The intervention measurably changed internals and the model's
> self-report did not change. No evidence of access *under that protocol*.

The qualifier still matters: v2's readout was a **binary greedy** `DETECTED=yes/no`
that never fired in 200 trials, so it could not register a weak shift even in
principle. Case A with a floored instrument is weaker than case A with a
continuous one. That is exactly what v3's continuous STATE margin is for.

## 4. Tokenization caveat, recorded not hidden

Not all ten concepts are single-token in every surface form:

| concept | forms and token counts |
|---|---|
| lemon | all forms 1 token |
| snow | all forms 1 token |
| violin | `Violin` is 2 tokens, others 1 |
| bicycle | `Bicycle` is 2 tokens, others 1 |

The lens evidence measure takes a log-sum-exp over the **first** tokens of the
predeclared surface forms. For the capitalised two-token forms that is a
first-token proxy and is labelled as such; it is not claimed to be the whole
word's representation. The lower-cased and space-prefixed forms, which are the ones
a model would actually emit here, are single tokens for all four concepts shown.

## 5. What is built, and what is not

**Built and run:** LENGTH_CONTROL readouts, the internal-measure calibration grid,
the lens readout path (`calibrate.py`, `Readout`), and concept token-set
declaration.

**Not yet built, and therefore not run:** the v3 trial harness itself. That needs
the continuous STATE logit margin read from a forced `STATE=` prefix before any
text is generated, the randomized five-way concept-identification readout, the
wrong-concept and functionally-matched random controls, steering enabled only after
the STATE decision, and bootstrap intervals over score distributions. **No v3
self-detection number exists yet, and none is reported.**

The calibration was the gate. It passed, so the pilot is worth building. Had no
setting been admissible, v3 would have stopped here.

## Reproducing

```bash
python -m experiments.self_focus.runner demo          # includes LENGTH_CONTROL
python -m experiments.self_focus.calibrate
```

Artifacts: `out/self_focus/v3_calibration.json` (all 18 attempts),
`out/self_focus/demo_*.jsonl` (v2 readouts), `demo_*_v1.jsonl` (v1, preserved).
