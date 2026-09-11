# Gate C: reduced-scope follow-up, 2026-09-10

**Full-population conclusion: `PROVISIONAL_UNADJUDICATED`.**

**Labelled-subset verdict: `UNDECIDABLE` (logistic), `MISS` (LightGBM).**

Gate C was the one registered Stage 1 gate never tested. It is now tested, at
reduced scope, and the answer is that **this subset cannot support the registered
comparison**: the full-answer increment the criterion is defined relative to is
approximately zero here, so there is nothing for a prefix to preserve.

That is a real result and it is preserved as it came out. It is not a HIT.

## The registered criterion

From `campaign/PREREG_STAGE1.md`:

> the risk score from the 50 percent answer-prefix preserves at least 90 percent
> of the full-answer LODO AUROC increment. Falsified if the 50 percent prefix
> retains < 90 percent.

"Increment" means workspace-over-logprob, so:

    full   increment = AUROC(full LP + full workspace)    - AUROC(full LP)
    prefix increment = AUROC(prefix LP + workspace @ 50%) - AUROC(prefix LP)
    ratio            = prefix increment / full increment        threshold 0.90

## Result

1,706 labelled rows across 4 sources, leave-one-dataset-out, Gemma-4-12B bf16.

| | logistic | LightGBM |
|---|---:|---:|
| AUROC, full LP | 0.5824 | 0.5288 |
| AUROC, full LP + workspace | 0.5744 | 0.5342 |
| **full-answer increment** | **-0.0080** | **+0.0054** |
| AUROC, prefix LP @ 50% | 0.5630 | 0.5289 |
| AUROC, prefix LP + workspace @ 50% | 0.5574 | 0.5256 |
| **prefix increment** | **-0.0056** | **-0.0033** |
| retention ratio | no denominator | -0.607 |
| labelled-subset verdict | UNDECIDABLE | MISS |

Bootstrap 95 percent interval on the primary increment: logistic
[-0.0183, +0.0060], LightGBM [-0.0559, +0.0461]. Both contain zero.

### Per-source AUROC (logistic)

| arm | drop | hotpotqa | popqa | squad_v2 |
|---|---:|---:|---:|---:|
| full LP | 0.641 | 0.637 | 0.369 | 0.684 |
| full LP + workspace | 0.617 | 0.635 | 0.349 | 0.697 |
| prefix LP @ 50% | 0.622 | 0.629 | 0.311 | 0.689 |
| prefix LP + workspace @ 50% | 0.627 | 0.627 | 0.293 | 0.682 |

### Deployable checkpoints

The registered 50 percent checkpoint needs the eventual answer length, so it is an
oracle observation. Absolute token indices are what a live monitor could read.

| arm | logistic increment | LightGBM increment |
|---|---:|---:|
| token 8 | -0.0326 | +0.0159 |
| token 16 | +0.0016 | +0.0488 |

**Do not read the LightGBM token-16 ratio of 8.995 as a HIT.** It is +0.0488
divided by a full-answer increment of +0.0054. A ratio against a near-zero
denominator is not interpretable on either side of 0.90, and the same applies to
the secondary prefix-plus arm (3.401) and token 8 (2.923). Those numbers are
reported for completeness and carry no verdict.

## Why the gate is not answerable on this subset

Every AUROC above sits between 0.50 and 0.58. The detector has close to no
ranking signal here, against Stage 1's LODO means of roughly 0.73 (logprob) and
0.79 (combined). Three contributors, none of which this run can separate:

1. **The population is deliberately different.** Only answers of at least 32
   tokens, and only sources gradable without a judge. That removes trivia_qa and
   popqa's short answers, halubench, medhallu and esconv, which is most of where
   the Stage 1 signal lived.
2. **Four sources means three training folds per LODO split.** Stage 1 had eight.
3. **popqa is degenerate here.** Error rate 0.976, so roughly 7 non-error rows in
   290. It clears the scorer's minority floor of 5 and should not be trusted:
   its AUROC is 0.29 to 0.37, inverted and far below chance, and it pulls the
   means down. Post hoc, excluding it leaves the full-answer increment still
   approximately zero (drop, hotpotqa and squad_v2 give 0.654 logprob against
   0.650 combined), so the conclusion does not depend on that choice. This is a
   robustness note, not a re-specification.

## Unresolved rows

139 rows were routed to a judge by the corrected deterministic grader and **no
judge has adjudicated them**: hotpotqa 44, drop 54, squad_v2 41.

Two uniform imputations are reported. They are **sensitivity scenarios, not
bounds**: AUROC is not monotone in label flips and the statistic is a ratio of two
AUROC differences, so a mixed assignment can fall outside both uniform ones. A
four-row counterexample is pinned in `campaign/test_uniform_scenarios.py`, where
both uniform assignments give 2.00 and one mixed assignment gives 0.00.

Under LightGBM both uniform scenarios return MISS (ratios 0.771 and 0.296). Under
logistic both are UNDECIDABLE. Agreement is weak reassurance and certifies
nothing. The full-population conclusion stays `PROVISIONAL_UNADJUDICATED`.

## Reproduction of the generations

27 of 1,845 answers (1.5 percent) reproduced the July text exactly; 170 matched on
token count. Per source: drop 5, hotpotqa 14, popqa 0, squad_v2 8.

This measures exact text equality only. It does not show that any published
finding fails to reproduce, and it does not show the original results were wrong:
those runs were internally consistent against whatever environment existed then.
The cause of the divergence **has not been isolated**. Ruled out so far: the hub
commits since July touched README, `tokenizer_config.json` and the chat template
rather than weights, and the rendered prompt is byte-identical across
`0e2b1058`, `711c1368` and `707f0a3b`. Not examined: the transformers version, GPU
and kernel differences (the original long-context shards were routed to A100 while
this run was entirely L40S), and bf16 tie-breaking, where one differing argmax
cascades through greedy decoding.

Because the July answers did not reproduce, July labels do not describe these
answers. Every label here is a fresh deterministic grade of this run's own output,
using the grader corrected on 2026-09-10. Fresh exact-source accuracy: 0.654.

## Deviations from the original plan

Stated so the scope is not mistaken for the registered study:

- exact-graded sources only, so labels are free and judge-free
- answers of at least 32 tokens only: early warning is undefined where there is no
  room to warn, and `legal_hallucinations` never reaches 8 tokens
- fresh generations, fresh labels, a corrected grader
- logistic regression as primary, matching the 2026-09-05 audit, with LightGBM
  reported beside it
- bootstrap intervals resample rows within each held-out source, so
  between-source variation is excluded; with four sources it cannot be estimated

## Cost and provenance

- 1,845 prompts, 4 shards, L40S, `--max-new 96`, greedy, bf16
- 16,145 GPU-seconds = **4.48 GPU-hours**, computed **$8.75** at $1.95/hour
- Modal's billing report had not posted this interval at the time of writing, so
  the figure above is derived from the shards' own reported runtimes, not billed
- **This run was unpinned.** Revision pinning was added after launch, so the
  shards recorded no `model_revision`. The environment snapshot in
  `environment-2026-09-10.json` is corroborating, not proof of what each worker
  loaded.
- Analysis commit: `82828fd0373eec3d13358bf174e27030d6d17a48`
- Environment: torch 2.12.1+cu130, transformers 5.13.0, accelerate 1.14.0,
  bitsandbytes 0.49.2, numpy 2.5.1; model sha
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; lens sha256 `214ba704...e360048`

Artifact hashes, sha256:

| artifact | sha256 | bytes |
|---|---|---:|
| `gatec.jsonl` (raw traces) | `7e36f7854801d356...` | 34,173,995 |
| `gatec_graded.jsonl` | `d08dd434bebe8020...` | 34,326,693 |
| `gate_c.json` (this result) | `dfb95b179d2939e0...` | 14,127 |
| `manifests/gatec_main.jsonl` | `c904da398589a416...` | 5,848,798 |

Full values in `out/campaign/gatec_artifacts.json`.

## Reproducing

```bash
python -m campaign.build_gatec_manifest --min-tokens 32 \
    --sources hotpotqa,drop,squad_v2,popqa --limit-per-source 600 \
    --out campaign/manifests/gatec_main.jsonl
JLENS_TIMEOUT_S=6600 bash campaign/launch_gatec.sh \
    campaign/manifests/gatec_main.jsonl gatec 4 30
bash campaign/finish_gatec.sh gatec 4
```

## What this changes

Nothing about the published Stage 1 or Stage 2 verdicts. Gate C was open; it is now
tested at reduced scope and returns no usable answer, because the quantity it
measures preservation of is absent on the subset that can support the test at all.

The honest summary for the project: on long, judge-free answers from this model,
the workspace features add nothing over output logprob even with the whole answer
in hand. Whether a prefix preserves an increment is moot where there is no
increment.
