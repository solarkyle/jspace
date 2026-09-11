# Where I would take JSpace next

Research review: 2026-09-05. Proposed experiments below are hypotheses and design
drafts, not novelty claims or registered studies. No GPU runs, deployment, or
changes to existing experiments were made in this review.

## Recommendation

Make the central question **"What information is present, correctly bound, and
actually used?"**, not "Can one internal score tell us whether anything is true?"

The best next scientific project is a **binding-and-evidence debugger**: isolate
cases where the model has all the right ingredients but associates them incorrectly
or fails to use the supplied evidence. The galaxy can become its interface. First,
harden the capture and intervention contract so that an attractive graph cannot
be mistaken for a causal result.

Do not restart the killed sign-flip claim or overwrite the existing D1/D2 design.
That study has its own freeze gates and outstanding approvals.

## What this review covered

- JSpace README, campaign preregistrations/reports, feature extraction, collection
  and classifier code, saved campaign traces, exploratory datasets, curvature
  extension, and sidecar galaxy/lab/conformance/branch implementation.
- The later ML-Lab research ledger and Study 3 Rung 0, N1/N3, N2, and D1/D2 draft.
  Those later adjudications supersede several stronger claims in older handoffs.
- The official Jacobian-lens paper/repository, jlens-gguf, J-Wash, and selected
  primary literature below, including a September 1 recurrence paper.

This is a targeted review of the located project materials, not a claim to have
read every file on the computer or completed an exhaustive literature review.

## What survives, and what does not

| Observation | What it supports | What it does not establish |
|---|---|---|
| Campaign Gate A: mean LODO increment +0.065, positive 6/7 | Useful error-ranking information in some task families | Universal reliability, causal truth representation |
| Prospective Gate D misses: mean increment -0.016, positive 2/5 | Real limits to frozen transfer | A clean within-model cognitive-operation sign flip |
| Rung 0: retrieval aligned, reasoning compatible with chance in two Gemmas | Task-dependent signal strength | That reasoning errors must have an opposite signature |
| N1 trajectory gain disappears after benchmark-structure controls | A warning about apparently rich layer features | Item-level reasoning rescue; the repeated-question labels cannot be distinguished by identical features |
| N2 legal verdict confounds | A concrete benchmark/probe failure mode | A general claim that all legal or truth probes are spurious |
| Curvature largely redundant with LP + workspace features | Layer localization deserves controlled study | A new independent hallucination sensor |
| Small galaxy/lab demos execute | A useful UI and experimentation scaffold | Validated causal control, a capacity law, or literal access to thoughts |

Sources: `campaign/reports/STAGE1_REPORT.md`, `STAGE2_REPORT.md`, current `README.md`,
`curvature/README.md`; later adjudications in
an external study tree (`study3/RUNG0_RESULTS.md`, `N1_N3/RESULTS.md`, and
`N2_signed_unfold/RESULTS.md`, not part of this repository). These historical
numerical claims were reviewed,
not all independently recomputed in this turn.

## New CPU audit performed in this review

Code: `analysis/research_prefix_audit.py`; tests:
`analysis/test_research_prefix_audit.py`. Output:
`out/research_20260905/prefix_full.json` (input SHA-256 hashes included).

Inputs contained 18,220 Stage 1 and 7,120 Stage 2 records. The analysis retained
16,178 and 7,113 historically labeled records respectively: **23,291 usable
responses**, with 2,049 unresolved records excluded. This is the scope of these
local judged files, not a replacement for the published campaign population.

Fixed exploratory model: StandardScaler fitted on training data, balanced
logistic regression, C=1, max_iter=2000, seed 20260905. No tuning. Features:

- First-token LP only (1 input).
- First-token LP + six onset workspace scalars (7 inputs).
- Those seven + six onset-to-current deltas at midpoint/end (13 inputs).

No full-answer LP, answer length, answer text, or future checkpoint enters a
predictor. All overlapping `split_group` or `example_id` training rows are removed.
This removed 2,805 Stage 1 rows for the separate regenerated SQuAD comparison.
Near-duplicate detection and a new judge-confidence audit were not performed.

| Mean AUROC | First-token LP | + onset workspace | + midpoint deltas | + end deltas |
|---|---:|---:|---:|---:|
| Stage 1 LODO, 7 evaluable sources | .633 | .770 | .771 | .788 |
| Stage 1 -> 5 new Stage 2 sources | .584 | .561 | .566 | .560 |

ESConv has one label class and is not scored. The repaired SQuAD set is reported
separately in JSON, not counted as a new Stage 2 source. These are unweighted
source means, not pooled AUROCs; there are no new confidence intervals.

**Interpretation:** in this specific small model, midpoint measurements barely
improve Stage 1 ranking, and do not repair broad transfer. This does not falsify
all temporal detectors: the features/model are limited, and this is retrospective.
For example, NQ onset improves .718 -> .762, while legal falls .545 -> .432. Do
not read the latter as mechanistic inversion; answer-conditioned controls remain
mandatory.

Checkpoint feasibility matters. Midpoint equals onset for 1,252/16,178 Stage 1
responses and 1,706/7,113 Stage 2 responses. In legal it is **1,488/1,600 (93%)**.
The midpoint is therefore not a distinct temporal observation for most legal
answers. The three checkpoints are genuinely distinct in 14,926 Stage 1 and
5,407 Stage 2 responses.

This **does not close Gate C**. Prefix-only output LP aggregates were not saved;
first-token LP is a weaker comparator. A fraction of eventual answer length is
also an offline, oracle-length checkpoint, not an executable online stopping rule.
All source datasets were already exposed before this analysis, so it is not a
new prospective transfer result.

### Instrument failures reproduced without a model

> **Addendum, 2026-09-10.** The gate described in this section has since been
> changed to fail closed, so the four pairs below are now rejected
> (`sidecar/test_conformance_closed.py` pins that). The comparison returns a
> three-state `status` of `pass` / `fail` / `incomplete`, requires
> `input_token_ids` and final-output evidence in both captures, treats a layer
> missing from the candidate as incomplete rather than absent-and-therefore-fine,
> and compares residual norms as well as cosine so a uniform rescale cannot pass.
> The text below is kept as the record of what the instrument did on 2026-09-05.

`conformance_negative_controls()` tests the actual current comparison function.
All four constructed candidate pairs returned `passed=true`:

1. A required layer missing from the candidate.
2. Candidate residual vectors missing.
3. Final-output data missing from both captures.
4. Candidate residuals scaled by 10 with the same direction and reported top-k.

The fourth is not proof that residual norms must match under every architecture;
it shows that the current contract never tests scale. The first three are
incomplete evidence passing the gate. The production code was not changed.

Static inspection also finds that `patched_snapshot_from_ids()` removes its hook
before `greedy_from_ids()` continues. Later tokens recompute the unpatched model.
Thus, under deterministic recomputation, an unchanged first token forces the same
continuation. The existing experiment is a one-step intervention, not persistent
branching of an altered hidden state. Its small probability shift is not evidence
of corrected reasoning.

The overload sweep changes the target identity, expected value, target position,
prompt length, and tokenization with the number of bindings. Its few examples
cannot support the earlier capacity-margin interpretation.

## 1. Binding-aware J-lens: right concepts, wrong associations

**Hypothesis.** Some clean-looking failures are binding errors: relevant concepts
are present but attached to the wrong entity, role, source, or tool argument.
A relation-sensitive readout adds information that bag-of-token entropy misses.

**Smallest discriminating test.** Start with synthetic worlds whose truth is exact:
`Ada owns amber; Bram owns birch`. Construct paired prompts with the same entities,
values, and token counts, but different assignments. Ask about one fixed entity;
counterbalance answer values, query position, assignment order, and load. Use
single-token values first, checked against the actual tokenizer. Reuse the same
target across load conditions. Include irrelevant filler with matched length.

Compare bag-of-concept J-lens features, ordinary LP, an output-text/position
baseline, and a small role-conditioned readout. Crucially, test **selective
interchange interventions**: can swapping one binding change the queried slot
while preserving the other slots? Include wrong-slot, random equal-norm, and
content-only patches. Evaluate held-out names, assignments, and templates, grouped
by underlying world, not by prompt row.

**Why this is not merely a new binding paper.** Binding IDs and ordering-related
subspaces already exist in [Feng & Steinhardt](https://arxiv.org/abs/2310.17191)
and [Dai et al.](https://arxiv.org/abs/2409.05448). The extension to test is whether
a J-lens-visible content channel plus a binding channel can diagnose and selectively
repair errors in quantized small models, then carry to tool arguments. Binding
discovery itself is not new.

**Competing explanations:** position/order codes; answer-token readout; general
perturbation damage. They require the counterbalanced prompts, held-out templates,
and selective untargeted-slot controls above.

**Pilot decision:** pursue a larger study only if relation features add useful
held-out discrimination beyond LP + content and targeted patches outperform both
content-only and random controls without comparable collateral slot damage. A
proposed worthwhile threshold is +5 AUROC points and +10 percentage points of
selective repair, with cluster uncertainty reported; freeze the actual gate after
an independent pilot, before new confirmatory data.

**Use:** a tool-call debugger showing "right destination, wrong recipient" or
"right number, wrong field." In the galaxy, edges encode tested relationships;
mere co-activation remains an explicitly different edge type.

## 2. Evidence-adoption debugger: retrieved is not the same as used

**Hypothesis.** Useful evidence can be represented strongly yet fail to control
the answer. The discrepancy between visibility and causal influence predicts
ignored-evidence errors better than uncertainty alone.

**Test.** Use fictional, fully specified mini-worlds to avoid relying on noisy
real-world truth labels. Cross evidence correctness with relevance, source
reliability, and agreement with an established premise. Include four important
cases: evidence truly used; its words merely repeated; conflicting evidence
noticed but rejected; missing evidence. An authoritative instruction defines
which source governs the task, not which answer token must be emitted.

Measure output LP, answer correctness, lens readout, and the change in answer
log-odds after replacing the relevant source activation with a matched alternative.
Use irrelevant-source patches and equal-norm random patches. Full-answer scoring
must handle multi-token alternatives, not just first-token rank.

Separate two roles: interventions create offline causal labels; a cheaper passive
monitor tries to predict those labels. Do not call an expensive counterfactual
oracle a single-pass deployed detector. Compare with input/output-only evidence
checks and a plain "re-read the relevant evidence" intervention at matched cost.

**Nearest work:** [Two Pathways to Truthfulness](https://arxiv.org/abs/2601.07422)
distinguishes question-anchored and answer-anchored information pathways; the old
local scan's retrieve-versus-reason gloss is too coarse. [Resist and Update](https://arxiv.org/abs/2607.12985)
already studies causal counterfactual control under evidence and pressure. Our
proposed extension is affordable, interpretable diagnosis of *failed evidence
handoffs* on the existing small-model/J-lens stack, not discovery of causal
evidence monitoring.

**Kill rule:** stop developing a specialized internal router if text/LP checks
match it at the same correction budget, or if its apparent gain vanishes when
answer identity and source style are controlled. The outcome metric is corrected
errors minus newly introduced errors per unit of extra inference, not brighter
evidence nodes.

**Use:** RAG debugging, citation/source selection, and deciding whether a failed
answer needs retrieval, a clearer evidence handoff, or external verification.

## 3. Quantization-aware causal conformance, not just pretty agreement

**Hypothesis.** A quantized model can preserve ordinary answers and recognizable
lens top-k while changing intervention response. Behavioral quality, readout
quality, and controllability need separate tests.

Build a strict capture contract first: exact tokenizer/input IDs, architecture,
model and lens revision hashes, explicit layer indexing and tap point, required
coverage, vector shapes and norms, finite values, and complete final-output
evidence. Missing required evidence is `incomplete`, never `pass`.

Then compare the same Gemma checkpoint at the existing NF4 setting and suitable
native GGUF precisions. Keep weights, lens provenance, prompts, and decoding
identical where meaningful; log unavoidable differences. Distinguish exact tensor
conversion from fitting a ridge surrogate, and from agreement across different
quantizations. `jlens-gguf` already implements conversion and interventions, so
[reuse it](https://github.com/igorbarshteyn/jlens-gguf), rather than claiming we
invented GGUF J-lens conversion.

For each capture compare (a) final distributions/answers, (b) readouts and norms,
(c) finite-difference intervention responses over signed doses and multiple
layers. Sparse top-k lists with different supports do not provide exact full-vocab
JSD: use a common measured partition, full logits on a small calibration set, or
honest bounds. Compare one-shot, persistent, and donor-swap interventions under
clearly different labels, with matched KV/recompute behavior.

**Candidate contribution:** a portable functional test suite and a measurement
of which precisions preserve which causal behaviors. A null result, where all
three agree, is still useful deployment evidence. Do not announce "quantization
destroys causality" from a single dose or a near-tied argmax.

**Use:** a trustworthy local GGUF galaxy, regression testing llama.cpp changes,
and choosing precision using an accuracy/latency/memory/monitorability frontier.
Quantized hidden-state detection itself is already explored, e.g.
[this NF4 probe study](https://arxiv.org/abs/2606.02628); its high reported scores
would require our answer-conditioned audit before serving as a benchmark target.

## Additional tests worth keeping, in priority order

### A monitor that can decline to judge

Estimate *applicability* separately from error risk. Candidate signals include
distance from the training readout distribution, disagreement among independently
trained heads, template sensitivity, and agreement with a cheap LP baseline.
The benchmark is whether the gate avoids making selective prediction worse on
unseen operations. Compare a pooled monitor, a prompt-text-routed monitor, and
internal routing. Hold out complete task families and answer mappings. Soft
expert routing is prior art, not automatic novelty. Positive monotone calibration
alone cannot fix inverted AUROC because it does not change rankings.

This is the right place to use an explicit **unknown / unsupported** galaxy state.
Low entropy must not become a green truth light. Stable wrong beliefs remain a
reason to consult external evidence.

### Real early warning, tied to the first wrong claim

Collect per-token LP and sparse readouts at fixed token counts or clause
boundaries. Label the first incorrect factual span, not just eventual response
correctness. Score alarms strictly before that span, lead time, false alarms per
1,000 correct tokens, and saved compute at equal correction quality. Stop rules
cannot know final response length. Restrict and separately report answers long
enough to admit warning. Compare against LP-only and
[Semantic Entropy Probes](https://arxiv.org/abs/2406.15927), accounting for their
offline supervision cost. This is more decisive than another onset/mid/end curve.

### A semantic-fragmentation control, not just token merging

Hold the referent fixed while changing its spelling, alias, script, or tokenizer
fragmentation. Ask whether apparent fog changes while correctness and behavior
stay stable. Compare raw token readouts, conservative surface families, and
phrase/template readouts. Do not merge `US` and `us`, or equate a first-token
fragment with the full entity. Anthropic already proposes template and oracle
lenses; the research question is their **measurement invariance and causal
specificity**, not whether multi-token concepts can exist.

### Curvature as a geometry control

Before expanding curvature, match the exact input token IDs between its capture
and the workspace run. Move the measured token window away from the chat-template
suffix and randomize harmless suffix phrasing. Choose layer bands inside training
folds, then test on held-out data. The current v2 analysis selects an upper band
after per-layer inspection and uses ordinary 5-fold CV, so that result should not
be described as nested held-out layer selection. The original
[curvature study](https://arxiv.org/abs/2604.23985) already links curvature with
uncertainty; the incremental, confound-controlled question is what remains open
for this project.

## What the galaxy should represent

Keep 2D for now. Use fixed or aligned positions so animation does not invent
semantic movement. Separate token time from layer depth. Show observed tokens,
phrase readouts, and analyst-assigned group labels as different objects.

The most useful new view is a paired clean/corrupted run with readout differences,
patch location and duration, actual output-log-odds effects, and collateral
changes. Color can represent measured entropy, evidence adoption, or an
applicability status, each with a legend. "Causal influence" appears only after an
intervention and only for the tested metric/context. Never label the picture a
complete record of thoughts.

[The workspace paper](https://transformer-circuits.pub/2026/workspace/index.html)
explicitly leaves binding structure and inconsistent legibility unresolved.
It also demonstrates that not every computation uses the measured workspace.
These are reasons to expose limits in the interface, not fill gaps with semantic
labels. J-Wash is a useful
[steering/editing workflow reference](https://github.com/Extraltodeus/J-Wash),
but editable concepts do not by themselves validate truth or safety.

## Literature changes since the July notes

- [Orgad et al.](https://arxiv.org/abs/2410.02707) already show limited probe transfer,
  token-specific truthfulness information, and different error types. A generic
  "internals know more than outputs" framing is not a new contribution.
- [Sahoo et al.](https://arxiv.org/abs/2606.02907) show a concrete case where apparent
  reasoning-mode separation collapses after format controls. Do not infer a
  cognitive-operation classifier from dataset classification accuracy.
- [Activation patching best practices](https://arxiv.org/abs/2309.16042) make
  corruption choice and outcome metric part of the causal claim. Include controls
  before interpreting a colorful intervention plot.
- [Looped Transformers under the Jacobian Lens](https://arxiv.org/abs/2609.01924),
  September 1, reports architecture-dependent read/write persistence in Ouro and
  Huginn. "Apply J-lens to recurrent transformers" is now covered by direct prior
  work. It motivates a later question about *when monitoring should sample* as
  content is reconstructed or becomes temporarily accessible. This is not a
  reason to switch away from Gemma during the first debugger pilot.

## Concrete next action and execution record

Choose **one** live scientific question: binding error versus missing content.
Use the existing Gemma-4-E4B environment for a small pilot, with no assumption that
E4B findings inherit the 12B campaign's evidence. GPU use and any large downloads
need separate scheduling/approval; none were started here.

Before that pilot: make conformance fail closed, specify intervention lifetime,
and implement zero-dose identity plus clean/corrupt/restore controls. Then create
200 development worlds and reserve entirely new worlds/templates for confirmation.
All variants of a world stay in one split. Start with short contexts and one-token
answers to make binding effects distinguishable from sampling and tokenization.
Do not launch the whole agenda simultaneously.

Commands run from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest analysis.test_research_prefix_audit
.\.venv\Scripts\python.exe -m analysis.research_prefix_audit --smoke --out out/research_20260905/prefix_smoke.json
.\.venv\Scripts\python.exe -m analysis.research_prefix_audit --out out/research_20260905/prefix_full.json
```

Three unit tests passed: future-feature exclusion, checkpoint delta math, and
duplicate-checkpoint accounting. A final combined run with the existing
`sidecar.test_experiments` and `sidecar.test_galaxy` suites passed all 10 tests.
Those existing tests do not enforce the missing conformance requirements.
The 600-row smoke completed; it is ordered and
not representative. The full audit completed and its non-empty JSON was read
back. Runtime: existing project Python, NumPy 2.5.1, scikit-learn 1.9.0. The
bundled Python lacked scikit-learn, so the project environment was used with
approval. No packages were installed. Raw data and frozen artifacts were untouched.

Remaining audit limits: no new bootstrap CIs, no near-duplicate/source-format or
answer-conditioned adjustment in this small prefix analysis, no independent
adversarial review of the new script, and no GPU causal reproduction. Consequently
the new predictive results stay exploratory; the engineering negative controls
are direct tests of the current harness, not tests of model internals.
