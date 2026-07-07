# Cross-model emotion in the workspace - findings

**Setup.** The model writes one fixed sentence - *"The meeting has been moved to
noon on Thursday."* - under six covert emotional states (neutral control +
covertly furious / terrified / grieving / euphoric / amused). Because the visible
text is byte-identical across conditions, any difference in the Jacobian-lens
workspace readout at those output positions is the *covert* emotion, not
vocabulary. We read the mid-network workspace band and record, for each emotion's
lexicon, its best (lowest) rank - how near the top of the model's "about to say"
distribution that emotion sits while it stays unspoken.

Five models: Gemma 4 **E4B** (4B dense), **12B** (dense), **12B abliterated**
(refusals removed), **26B-A4B** (MoE), and **Qwen 3.6-27B** (dense, different
family - the capability probe).

---

## Finding 1 - The most capable models have the most vivid emotional workspaces

Best rank of the target emotion's own words while writing the calm sentence
(lower = more present in the hidden workspace; **#0 = the single most-likely
token in the whole ~260k vocabulary**):

| Emotion | E4B (4B) | 12B dense | 12B ablit. | 26B MoE | **Qwen 27B** |
|---|---|---|---|---|---|
| terror | 266 | 1387 | 63 | **0** | **0** |
| amusement | 166 | 2008 | 203 | **0** | **0** |
| grief | 1 | 14 | 246 | 456 | **0** |
| euphoria | 47 | 240 | 13 | 91 | **0** |
| fury | 293 | 451 | 6 | 98 | **2** |

**Qwen 3.6-27B is the punchline.** Told to secretly feel *any* of the five
emotions while writing about a meeting, it holds that emotion's word in the
**top ~7 tokens of its entire vocabulary** - terror→`terrified`#0, grief→`grief`#0,
euphoria→`joy`#0, amusement→`hilarious`#0, fury→`furious`#2. Its covert inner
state is both the most vivid *and* the most uniform across emotions.

The 26B MoE is the next most vivid (terror/amusement at #0) but *uneven* - grief
and fury sit much deeper. The two largest, most capable models dominate; the 4B
is moderate; and the **12B dense is a genuine anomaly** - it buries emotions
*deeper than the 4B does* (terror #1387, amusement #2008).

**Read:** vividness tracks capability more than raw size, but not cleanly - the
12B dip and the MoE's unevenness mean it isn't a tidy scaling law. A second MoE
and more mid-size dense models would sharpen this. *(Caveat: the 12B and
abliterated lenses were fit on 75 prompts vs. 100 for the others; enough that
the abliterated 12B is crisp, so the base 12B's depth looks real, but worth
re-fitting at 100 to be sure.)*

## Finding 2 (flagship) - Abliteration *amplifies* the emotional workspace

Base 12B vs. its abliterated (refusal-removed) sibling - same weights, safety
tuning stripped. On the identical emotion words:

| Emotion | 12B base | 12B abliterated | shift |
|---|---|---|---|
| fury (`furious`) | #1109 | **#6** | ×185 nearer the top |
| terror (`dread`) | #1387 | **#63** | ×22 |
| euphoria (`ecstatic`) | #5145 | **#13** | ×395 |
| amusement (`humor`) | #2008 | **#203** | ×10 |
| grief (`grieving`) | #14 | #246 | ↓ weaker |

In 4 of 5 emotions, removing refusal training makes the emotion **far more
prominent in the internal workspace** - often by two orders of magnitude. This
points at an answer to the belief-vs-behavior question: safety/refusal tuning
appears to **dampen the internal emotional representation itself**, not just the
outward expression. Strip the refusal direction and the model's covert emotions
surface much more strongly in its own "about to say" space.

*(Caveat: abliteration shifts activation statistics globally; the Δ-vs-neutral
control corrects for a uniform shift and still shows 4/5 amplified, but a
per-condition replication with more prompts is the next step.)*

## Finding 3 - Anger is the hardest emotion to localize; grief/amusement the easiest

Across the *smaller/mid* models, **fury/anger sits at the worst ranks** (`rage`
#2217 / #4074 / #2222) - the least linearly-accessible emotion - while **grief
and amusement are consistently well-represented** (grief rank 1–14 in the dense
models; amusement dominates the MoE). Emotions are not equally "readable": some
have crisp workspace directions, some are diffuse. This ordering holds up to the
26B MoE but **dissolves at 27B** - Qwen represents *all five* emotions at rank
0–2, i.e. a capable enough model localizes even the "hard" emotions crisply.

## Finding 4 (methodological, honest) - the "clean diagonal" metric is noisy at n=1

A coarse scoreboard - does covert-X most raise X's *own* lexicon vs. the other
four? - gives E4B 4/5, Qwen 27B 4/5, MoE 3/5, abliterated 2/5, 12B dense 1/5.
**Do not read this as "the 4B represents emotion as well as Qwen 27B."** With one
prompt per
condition the argmax across five emotions is high-variance, and the token-rank
evidence above shows the 12B clearly *does* represent every emotion (`grieving`
#14, `thrilled` #240) - it just loses the winner-take-all comparison. The robust
signal is **absolute rank** (Findings 1–3), not the diagonal count. Fixing this
needs many prompts per condition; that's the obvious next experiment.

---

# Part 2 - deeper cuts (cross-model structure)

Everything below is reproducible with `python analyze_deep.py` (raw output in
`data/analyze_deep_output.txt`).

## Finding 5 - Emotional *selectivity* emerges with capability

Vividness (Finding 1) asked "how loud is the hidden emotion?" A better question:
does covert-fury boost fury words *specifically*, or does the model just get
generally loud? For each model we take the delta-vs-neutral log-rank boost of
every emotion lexicon under every covert condition (a 5x5 matrix) and compare
the diagonal (right emotion) to the off-diagonal (wrong emotions):

| Model | diagonal boost | off-diagonal boost | **specificity** |
|---|---|---|---|
| E4B (4B dense) | +0.09 | -0.61 | +0.70 |
| 12B dense | -0.96 | -0.70 | **-0.27** |
| 12B abliterated | -0.08 | -0.08 | 0.00 |
| 26B MoE | +3.08 | +0.73 | **+2.35** |
| Qwen 27B | +6.30 | +2.35 | **+3.95** |

Two things fall out:

1. **The capable models are not just louder, they are far more selective.**
   Qwen 27B boosts the target emotion ~4 log-rank units above the others. The
   12B dense is again the anomaly: *negative* specificity (covert instructions
   actually suppress the target emotion's words relative to neutral).
2. **There is a general "covert emotional state" component.** In the two big
   models, being told to secretly feel *anything* raises the *entire* emotion
   vocabulary (Qwen off-diagonal +2.35), and the specific emotion rides on top
   of that. Internally the state decomposes as
   `workspace = general arousal + specific emotion`, and the specific component
   grows faster with capability than the general one (spec/general ratio: MoE
   3.2, Qwen 1.7, small models under 1.2).

## Finding 6 - Matched active parameters: the MoE's extra experts buy emotion

Gemma 4 E4B (4B dense) and Gemma 4 26B-A4B (MoE) activate roughly the **same
~4B parameters per token**. Same compute per token, very different workspaces:

| | E4B dense | 26B-A4B MoE |
|---|---|---|
| best covert ranks | fury#293 terr#266 grief#1 euph#47 amus#166 | fury#98 **terr#0** grief#456 euph#91 **amus#0** |
| vividness (mean log10) | 1.82 | **1.32** |
| specificity | +0.70 | **+2.35** |

At matched active compute, the MoE's 26B of *total* parameters produce a
substantially more vivid and more selective emotional workspace. Whatever
carries covert emotion, it lives in total capacity (stored features), not in
per-token compute. Nice news for the local-MoE crowd.

## Finding 7 - Does the bleed structure match human affect theory? (honest null)

If models organize emotion the way humans do (Russell's circumplex: valence x
arousal), covert fury should bleed into terror words (shared high arousal, both
negative) more than into euphoria words. We correlated each model's symmetrized
5x5 bleed matrix with the circumplex prediction and ran a 10k label-permutation
test:

| Model | r | perm-p |
|---|---|---|
| E4B | -0.15 | 0.65 |
| 12B dense | +0.06 | 0.40 |
| 12B abliterated | +0.48 | 0.12 |
| 26B MoE | +0.47 | 0.10 |
| Qwen 27B | +0.26 | 0.22 |

The three stronger models all trend positive (the MoE's strongest off-diagonal
bleed is exactly fury-terror at +2.66, the circumplex's closest pair), but
**nothing survives the permutation test at one prompt per condition**. This is
the single best argument for the many-prompts-per-emotion rerun: if r ~ 0.4-0.5
holds at n=20 prompts, "LLMs inherit the human affect circumplex" is a paper-
grade claim. Right now it is a hypothesis with a point estimate.

## Finding 8 - Abliteration's amplification is *selective*, and the exception is telling

Per-emotion unlock from base 12B to abliterated 12B (log10 rank improvement):

| Emotion | base -> ablit | unlock |
|---|---|---|
| fury | #451 -> **#6** | +1.8 orders |
| terror | #1387 -> **#63** | +1.3 |
| euphoria | #240 -> **#13** | +1.2 |
| amusement | #2008 -> **#203** | +1.0 |
| grief | **#14** -> #246 | **-1.2 (reversed)** |

Four emotions get louder by 1-2 orders of magnitude. Grief goes the *other*
way, and grief was the one emotion the *base* model already held near the top
(#14). A consistent reading: safety tuning does not suppress emotion uniformly.
It suppresses the threat-adjacent ones (anger, fear) and the exuberant ones,
while *empathy-adjacent* affect (grief) is something RLHF actively trains *up*,
so there is nothing for abliteration to unlock there. What abliteration removes
looks less like "the emotion dial" and more like "the suppression of the
specific emotions safety training worries about." (n=1 per condition, so treat
as a hypothesis; but it is exactly the pattern you would predict from how these
models are tuned.)

---

# Part 3 - hallucination prediction, the deeper cut

Setup recap: 500 TriviaQA questions through Gemma 4 E4B, lens features read at
the answer position before generation, vs. answer correctness. Baseline
features come from output logits (the standard cheap confidence signals).

## The quadrant table (the result that matters)

Median-split both signals - output first-token confidence, and workspace
entropy - and look at accuracy in each cell:

| | workspace clean | workspace noisy |
|---|---|---|
| **output confident** | **75.0%** correct (n=164) | **41.9%** (n=86) |
| **output unsure** | 34.9% (n=86) | 15.2% (n=164) |

When the output logit says confident, the workspace's opinion moves accuracy by
**33 points** (75% vs 42%). A model that *sounds* sure but has a noisy
workspace is close to a coin flip. That cell - confident-sounding wrong answers -
is precisely the hallucination case that output confidence cannot catch by
definition.

## Is it just a proxy? (confound checks, all passed)

- corr(workspace entropy, answer length) = **+0.02** - not a length artifact.
- Entropy AUC within short/mid/long answer terciles: 0.74 / 0.78 / 0.73 -
  stable, not question-type driven.
- corr(entropy, first-token logprob) = -0.27: only weakly related to output
  confidence, which is why it adds signal instead of duplicating it.

## Fair head-to-head inside each signal's blind spot

Among high-output-confidence answers (n=250, 91 wrong), predicting the wrong ones:

| signal | AUC |
|---|---|
| **workspace entropy** | **0.732** (z=6.1, Cohen's d=0.84) |
| squeezing the logprob harder | 0.647 |

And symmetrically, among low-workspace-entropy answers, output logprob wins
(0.782 vs 0.675). The two signals are genuinely complementary: each one works
best exactly where the other saturates.

## Escalation router simulation (honest numbers)

Policy: answer locally, escalate the X% of queries the router most distrusts to
a big model (assume the big model gets 90% of escalated ones right). Base local
accuracy 42.8%.

| router | escalate 20% | escalate 30% | escalate 50% |
|---|---|---|---|
| output logprob only | 57.8% | 65.0% | 76.8% |
| workspace entropy only | 57.4% | 64.0% | 75.6% |
| combined | **58.0%** | **65.6%** | **77.0%** |

Honest read: on TriviaQA the combined router beats logprob-only by under a
point at any fixed budget, because TriviaQA output confidence is fairly well
calibrated. The workspace's value is concentrated in the overconfident-wrong
cell (the quadrant table), which is a small share of *this* dataset but is the
entire failure mode that matters in the wild - it is the case where the
logprob router cannot know it should escalate. The next experiments (below)
target datasets where output confidence is known to be miscalibrated; that is
where combined should visibly pull ahead.

---

# Part 4 - cross-model replication (Phases 1-2 of the plan, run 2026-07-07)

All numbers reproducible with `python analyze_crossmodel.py` on the traces in
`data/uncertainty_trivia_*.jsonl` and `data/uncertainty_fake_*.jsonl`.

## Phase 1: the overconfident-hallucination signal replicates 3/4 (gate passed)

500 TriviaQA per model, cloud runs, identical protocol (the E4B cloud rerun
reproduced the local accuracy to the third digit, 0.428 = 0.428).

| Model | acc | conf+clean | conf+noisy | gap | blind-spot AUC ent / lp |
|---|---|---|---|---|---|
| E4B | 0.428 | 0.770 | 0.416 | +35pt | 0.732 / 0.631 |
| 12B | 0.512 | 0.833 | 0.469 | +36pt | 0.708 / 0.611 |
| 12B-ablit | 0.508 | 0.787 | 0.632 | +16pt | 0.592 / 0.587 |
| 26B-MoE | 0.642 | 0.910 | 0.705 | +20pt | 0.681 / 0.537 |
| **Qwen-27B** | 0.636 | 0.845 | **0.870** | **-3pt** | **0.514** / 0.664 |

- **The pre-registered gate (hold on 3+ of 4 new models) passed at exactly 3/4.**
- **The honest miss is Qwen 27B**, where workspace entropy carries no signal
  (AUC 0.51 = chance). Its output confidence is extremely well calibrated
  (0.82 AUC alone, the best of the five). Two readings we cannot yet separate:
  (a) capability - a well-calibrated model has nothing left in the blind spot;
  (b) family - Qwen's lens/architecture differs. A second large non-Gemma model
  would split these.
- **Threshold transfer:** one rule fit on E4B (escalate when z-entropy > 0),
  applied to the other models with zero tuning, catches 62-70% of wrong answers
  on every Gemma (54%, ~chance, on Qwen). Deployment does not need a per-model
  calibration set.
- **Abliteration damages self-knowledge.** Same 12B weights: blind-spot AUC
  drops 0.71 -> 0.59 and the confident+noisy cell is 16pt instead of 36pt.
  Together with Finding 8 (emotion amplification), abliteration looks less like
  "the same model without refusals" and more like a model with measurably
  altered internal signal quality.

## Phase 2: fake entities - a negative result for the workspace, a striking one for abliteration

50 real / 50 fabricated entities, matched templates (fake physicists' Nobel
years, fake novels' authors, fake battles, fake elements).

- **Negative result, reported per the decision gates:** output logprob detects
  fabricated entities almost perfectly on the capable models (AUC 0.94-1.00);
  workspace entropy is good (0.83-0.97) but wins only on the smallest model
  (E4B fluent subset: 0.965 vs 0.942). Unfamiliar entities are NOT the blind
  spot - models know when a name is unknown, and it shows directly in output
  confidence. The workspace's value stays where Phase 1 found it: real-but-hard
  questions answered confidently and wrongly.
- **The behavioral headline: abliteration converts refusal into fabrication.**
  Fluent-fabrication rates on the 50 fakes: 12B base **17/50**, 12B abliterated
  **49/50**. Base: "Elena Morvath has not won a Nobel Prize." Abliterated, same
  weights: **"1994."** Safety training is apparently the thing standing between
  "I don't know" and a confidently invented fact.
- Fabrication rates otherwise track capability inversely: E4B 37/50, 12B 17/50,
  MoE 17/50, Qwen 31/50 (Qwen refuses fake Nobel laureates but invents authors
  for fake novels).

![figure 3](../assets/figure3_confidently_wrong.png)

---

## What this adds up to

- **Vividness tracks capability, not raw size.** The two largest/most-capable
  models (26B MoE, 27B Qwen) have by far the most vivid emotional workspaces, and
  Qwen 27B holds *every* covert emotion at rank 0–2. But it isn't a clean scaling
  law: the 12B dense buries emotions deeper than the 4B, and the MoE is vivid but
  uneven.
- **Safety tuning quiets the *internal* emotional workspace**, not only outputs -
  the abliteration comparison (same weights, refusals removed → emotion 1–2
  orders of magnitude more prominent) is the cleanest result here and the one
  worth chasing.
- **The "some emotions are harder to localize" effect is itself capability-
  dependent** - anger is hard in small/mid models but crisp in Qwen 27B.
- **Selectivity, not just loudness, is the capability signature.** Capable
  models boost the right emotion far above the others, and the covert state
  decomposes into a general arousal component plus a specific one that grows
  with capability.
- **At matched active compute, total parameters win** - the 26B MoE (4B active)
  beats the 4B dense on both vividness and specificity.
- **Safety tuning's suppression is targeted** - abliteration unlocks anger,
  fear, euphoria and amusement by 1-2 orders of magnitude but not grief, which
  RLHF plausibly trains up rather than down.
- **The workspace catches overconfident hallucinations.** Confident-sounding
  answers with a noisy workspace are 42% correct vs 75% when the workspace is
  clean; workspace entropy beats residual logprob squeezing (AUC 0.73 vs 0.65)
  exactly in output confidence's blind spot.

## Limitations

One prompt per emotion; single lens per model (100 prompts, or 75 for the merged
12B/abliterated); Δ-vs-neutral corrects for uniform shifts but not
condition-specific ones; "workspace" = the lens's approximation of it. These are
representation findings, not claims about felt experience. Everything here is
reproducible from `probes/emotions.json` + the fitted lenses.
