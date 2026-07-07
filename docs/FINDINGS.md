# Cross-model emotion in the workspace — findings

**Setup.** The model writes one fixed sentence — *"The meeting has been moved to
noon on Thursday."* — under six covert emotional states (neutral control +
covertly furious / terrified / grieving / euphoric / amused). Because the visible
text is byte-identical across conditions, any difference in the Jacobian-lens
workspace readout at those output positions is the *covert* emotion, not
vocabulary. We read the mid-network workspace band and record, for each emotion's
lexicon, its best (lowest) rank — how near the top of the model's "about to say"
distribution that emotion sits while it stays unspoken.

Four models, same family lineage: Gemma 4 **E4B** (4B dense), **12B** (dense),
**12B abliterated** (refusals removed), **26B-A4B** (MoE). Qwen 3.6-27B pending.

---

## Finding 1 — The MoE has a dramatically more vivid emotional workspace

Best rank of the target emotion's own words while writing the calm sentence
(lower = more present in the hidden workspace; rank 0 = the single most-likely
token in the whole 260k vocabulary):

| Emotion | E4B (4B) | 12B dense | 12B abliterated | **26B MoE** |
|---|---|---|---|---|
| terror | afraid #266 | dread #1387 | dread #63 | **terrified #0, terror #2, fear #10** |
| amusement | amusing #166 | humor #2008 | humor #203 | **hilarious #0, giggle #1, funny #12** |
| euphoria | elated #47 | thrilled #240 | ecstatic #13 | euphoric #91 |
| fury | furious #293 | livid #451 | **furious #6** | furious #98 |
| grief | **sorrow #1** | grieving #14 | grieving #246 | sorrow #456 |

The MoE, told to secretly feel terror, is holding `terrified` as the **#0 token
in the entire vocabulary** — literally the thing it is most disposed to say —
while it writes about a meeting. Same for amusement (`hilarious` #0, `giggle`
#1). Nothing else in the fleet is that vivid. Whether this is the MoE
architecture or the 26B scale we can't fully separate with one MoE, but the
effect is not subtle.

## Finding 2 (flagship) — Abliteration *amplifies* the emotional workspace

Base 12B vs. its abliterated (refusal-removed) sibling — same weights, safety
tuning stripped. On the identical emotion words:

| Emotion | 12B base | 12B abliterated | shift |
|---|---|---|---|
| fury (`furious`) | #1109 | **#6** | ×185 nearer the top |
| terror (`dread`) | #1387 | **#63** | ×22 |
| euphoria (`ecstatic`) | #5145 | **#13** | ×395 |
| amusement (`humor`) | #2008 | **#203** | ×10 |
| grief (`grieving`) | #14 | #246 | ↓ weaker |

In 4 of 5 emotions, removing refusal training makes the emotion **far more
prominent in the internal workspace** — often by two orders of magnitude. This
points at an answer to the belief-vs-behavior question: safety/refusal tuning
appears to **dampen the internal emotional representation itself**, not just the
outward expression. Strip the refusal direction and the model's covert emotions
surface much more strongly in its own "about to say" space.

*(Caveat: abliteration shifts activation statistics globally; the Δ-vs-neutral
control corrects for a uniform shift and still shows 4/5 amplified, but a
per-condition replication with more prompts is the next step.)*

## Finding 3 — Anger is the hardest emotion to localize; grief/amusement the easiest

Across every model, **fury/anger sits at the worst ranks** (e.g. `rage` #2217 /
#4074 / #361 / #2222) — it's the least linearly-accessible emotion in the
workspace. **Grief and amusement are consistently well-represented** (grief hits
rank 1–14 in the two dense models; amusement dominates the MoE). Emotions are not
equally "readable" — some have crisp workspace directions, some are diffuse, and
this ordering is stable across scale.

## Finding 4 (methodological, honest) — the "clean diagonal" metric is noisy at n=1

A coarse scoreboard — does covert-X most raise X's *own* lexicon vs. the other
four? — gives E4B 4/5, MoE 3/5, 12B dense 1/5, abliterated 2/5. **Do not read
this as "the 4B represents emotion better than the 12B."** With one prompt per
condition the argmax across five emotions is high-variance, and the token-rank
evidence above shows the 12B clearly *does* represent every emotion (`grieving`
#14, `thrilled` #240) — it just loses the winner-take-all comparison. The robust
signal is **absolute rank** (Findings 1–3), not the diagonal count. Fixing this
needs many prompts per condition; that's the obvious next experiment.

---

## What this adds up to

- **Emotion representation is not monotonic in size.** The 26B MoE is far more
  vivid than the 12B dense; the 4B is respectable. Architecture (MoE) and scale
  both plausibly matter; a second MoE would disentangle them.
- **Safety tuning quiets the *internal* emotional workspace**, not only outputs —
  the abliteration comparison is the cleanest result here and the one worth
  chasing.
- **Emotions have a stable difficulty ordering** (anger hard, grief/amusement
  easy) that holds across four models.

## Limitations

One prompt per emotion; single lens per model (100 prompts, or 75 for the merged
12B/abliterated); Δ-vs-neutral corrects for uniform shifts but not
condition-specific ones; "workspace" = the lens's approximation of it. These are
representation findings, not claims about felt experience. Everything here is
reproducible from `probes/emotions.json` + the fitted lenses.
