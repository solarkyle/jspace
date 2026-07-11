# Stage 2 report: prospective zero-shot validation (2026-07-11, PRELIMINARY)

STATUS: preliminary. The three deterministic new sources (nq_open,
legal_hallucinations, bfcl) and the squad_v2 regen are final. truthfulqa and
facts_grounding await Codex judging (quota-limited, resumable driver running);
their rows will complete the table but CANNOT change the Gate D verdict, which
is already decided on breadth (see below).

## Design recap

Pre-registration: campaign/PREREG_STAGE2.md, written before any generation.
The Stage 1 classifiers (LightGBM: combined / logprob / workspace) were frozen
and SHA-256 hashed BEFORE the Stage 2 datasets were generated; they score all
Stage 2 rows zero-shot, with no retraining or tuning of any kind. 7,120
prompts, Gemma-4-12B bf16, greedy, identical two-pass capture and feature code
as Stage 1. Generation cost ~$14 Modal.

Gate D (registered): mean frozen-combined-minus-frozen-logprob AUROC increment
over the five NEW sources >= +0.02 AND positive on >= 4 of 5.

## Verdict: GATE D MISS (registered criteria; decided on final data)

Frozen-model zero-shot per source (deterministic sources final):

| source | n | err% | AUC lp | AUC ws | AUC comb | increment |
|---|---:|---:|---:|---:|---:|---:|
| nq_open | 2000 | 66% | 0.746 | 0.739 | 0.768 | +0.023 |
| legal_hallucinations | 1600 | 47% | 0.555 | 0.419 | 0.448 | -0.107 |
| bfcl | 400 | 22% | 0.572 | 0.499 | 0.552 | -0.020 |
| truthfulqa | 817 | pending judge | | | | |
| facts_grounding | 803 | pending judge | | | | |
| squad_v2 (regen, scored separately) | 1500 | 29% | 0.363 | 0.425 | 0.419 | +0.056 |

With legal and bfcl negative, breadth can reach at most 3/5, below the
registered 4/5. Per the prereg's pre-committed interpretation, the honest
conclusion is that the Stage 1 mapping does not transfer universally beyond
its training pool's task style.

## The mechanism: the signal transfers, the mapping does not

Two diagnostics rule out a data or pipeline pathology and localize the failure:

1. Feature health: zero NaNs across all 175,181 feature cells; identical
   feature code as Stage 1.
2. In-domain learnability: a FRESH LightGBM (5-fold CV, within each new
   dataset) reads the SAME features at legal 0.927, squad 0.890, nq 0.783,
   bfcl 0.552. BUT decomposition shows the 0.9+ numbers are confounded:

   | slice | truth structure | in-domain AUC |
   |---|---|---:|
   | legal case_existence | fixed (always "yes") | 1.000 (answer readout) |
   | legal fake_case_existence | fixed (always "no") | 1.000 (answer readout) |
   | squad unanswerable-only | fixed (abstain = correct) | 0.991 (answer readout) |
   | squad answerable-only | varying | 0.866 (n=530, only 3% err - noisy) |
   | nq_open | varying | 0.783 (the honest headline) |
   | legal affirm_reverse | varying | 0.582 |
   | bfcl | varying | 0.552 |

   On a slice where the correct answer is CONSTANT, the error label is a
   deterministic function of which answer the model emitted, and the residual
   stream trivially encodes the answer it is about to emit - so AUC ~1.0 there
   measures answer-identity readout, not error detection. The honest genuine
   in-domain numbers on varying-truth tasks are ~0.55-0.78. (Stage 1 largely
   escapes this confound: its datasets have varying truth per row; its squad
   was all-answerable.)

So: the features carry a real but moderate error signal in genuinely new
varying-truth domains (nq 0.783 in-domain vs frozen zero-shot 0.768 - transfer
is nearly free there), and what fails elsewhere is the frozen signal-to-error
MAPPING plus, on fixed-truth veracity tasks, the structure of the task itself.
Three regimes:

- Task style matches Stage 1 (closed-book answer retrieval, nq_open): the
  frozen mapping transfers (+0.023, above the registered per-source bar).
- Veracity/abstention tasks (legal case existence, squad unanswerable): the
  mapping INVERTS. Frozen workspace AUC 0.419 on legal means the score ranks
  errors BELOW non-errors: flipping it yields ~0.58. Interpretation: Stage 1
  taught the detector "internal fog = wrong answer." On "is this real?" and
  "can this be answered?" tasks, fog is what CORRECT behavior looks like -
  the model senses it has nothing on the fabricated case and rightly says no;
  confident fluency is what the errors look like. Uncertainty is evidence of
  error in retrieval and evidence of correctness in veracity judgment.
  A second structural point: on fixed-truth slices, answer-agnostic error
  detection is impossible in principle (the label IS the answer), so these
  tasks need a different monitor design (e.g. fog + answer-direction jointly),
  not a better version of this one.
- Tool calls (bfcl): weak signal even in-domain (0.552). Wrong-argument
  errors (87 of 89 bfcl errors) are structural, not epistemic; they do not
  produce a workspace fog signature. Note only 1/400 responses failed to
  parse: Gemma emits clean JSON; it just fills in wrong values.

Curious detail: on squad_v2 both frozen families are inverted in absolute
terms (combined 0.419) yet the workspace INCREMENT is positive (+0.056) -
workspace features are less wrong than logprobs under the flip.

## What survives, restated

- Stage 1 (adversarially reproduced): workspace features beat output
  confidence across 7 held-out-dataset wrappers, ~+0.06 under every control.
- Stage 2 adds: on genuinely new varying-truth domains the signal is real but
  moderate (nq_open 0.783 in-domain, 0.768 frozen zero-shot - transfer nearly
  free within task style); the polarity of the fog-error relationship is
  task-type-conditional; fixed-truth veracity slices are structurally out of
  scope for answer-agnostic monitors.
- The universal zero-shot plug-in detector is dead by our own registered
  test. The live design is a per-task-family monitor: lens reads + tiny
  classifier (~300 KB, microsecond inference) + a few hundred labeled
  examples from the target distribution.

## Caveats and pending items

- truthfulqa / facts_grounding rows pending Codex quota; judged by Codex
  (GPT-5.5) per prereg. Judge validated against Sonnet at kappa 0.818 on 120
  identical-prompt rows (Sonnet-vs-Fable pilot kappa 0.84): three judges, two
  providers, substantial agreement.
- Single model (Gemma-4-12B), greedy decoding, no comparison yet against
  dedicated trained hallucination probes.
