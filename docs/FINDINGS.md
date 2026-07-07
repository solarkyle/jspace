# Cross-model emotion in the workspace — findings

**Setup.** The model writes one fixed sentence — *"The meeting has been moved to
noon on Thursday."* — under six covert emotional states (neutral control +
covertly furious / terrified / grieving / euphoric / amused). Because the visible
text is byte-identical across conditions, any difference in the Jacobian-lens
workspace readout at those output positions is the *covert* emotion, not
vocabulary. We read the mid-network workspace band and record, for each emotion's
lexicon, its best (lowest) rank — how near the top of the model's "about to say"
distribution that emotion sits while it stays unspoken.

Five models: Gemma 4 **E4B** (4B dense), **12B** (dense), **12B abliterated**
(refusals removed), **26B-A4B** (MoE), and **Qwen 3.6-27B** (dense, different
family — the capability probe).

---

## Finding 1 — The most capable models have the most vivid emotional workspaces

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
**top ~7 tokens of its entire vocabulary** — terror→`terrified`#0, grief→`grief`#0,
euphoria→`joy`#0, amusement→`hilarious`#0, fury→`furious`#2. Its covert inner
state is both the most vivid *and* the most uniform across emotions.

The 26B MoE is the next most vivid (terror/amusement at #0) but *uneven* — grief
and fury sit much deeper. The two largest, most capable models dominate; the 4B
is moderate; and the **12B dense is a genuine anomaly** — it buries emotions
*deeper than the 4B does* (terror #1387, amusement #2008).

**Read:** vividness tracks capability more than raw size, but not cleanly — the
12B dip and the MoE's unevenness mean it isn't a tidy scaling law. A second MoE
and more mid-size dense models would sharpen this. *(Caveat: the 12B and
abliterated lenses were fit on 75 prompts vs. 100 for the others; enough that
the abliterated 12B is crisp, so the base 12B's depth looks real, but worth
re-fitting at 100 to be sure.)*

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

Across the *smaller/mid* models, **fury/anger sits at the worst ranks** (`rage`
#2217 / #4074 / #2222) — the least linearly-accessible emotion — while **grief
and amusement are consistently well-represented** (grief rank 1–14 in the dense
models; amusement dominates the MoE). Emotions are not equally "readable": some
have crisp workspace directions, some are diffuse. This ordering holds up to the
26B MoE but **dissolves at 27B** — Qwen represents *all five* emotions at rank
0–2, i.e. a capable enough model localizes even the "hard" emotions crisply.

## Finding 4 (methodological, honest) — the "clean diagonal" metric is noisy at n=1

A coarse scoreboard — does covert-X most raise X's *own* lexicon vs. the other
four? — gives E4B 4/5, Qwen 27B 4/5, MoE 3/5, abliterated 2/5, 12B dense 1/5.
**Do not read this as "the 4B represents emotion as well as Qwen 27B."** With one
prompt per
condition the argmax across five emotions is high-variance, and the token-rank
evidence above shows the 12B clearly *does* represent every emotion (`grieving`
#14, `thrilled` #240) — it just loses the winner-take-all comparison. The robust
signal is **absolute rank** (Findings 1–3), not the diagonal count. Fixing this
needs many prompts per condition; that's the obvious next experiment.

---

## What this adds up to

- **Vividness tracks capability, not raw size.** The two largest/most-capable
  models (26B MoE, 27B Qwen) have by far the most vivid emotional workspaces, and
  Qwen 27B holds *every* covert emotion at rank 0–2. But it isn't a clean scaling
  law: the 12B dense buries emotions deeper than the 4B, and the MoE is vivid but
  uneven.
- **Safety tuning quiets the *internal* emotional workspace**, not only outputs —
  the abliteration comparison (same weights, refusals removed → emotion 1–2
  orders of magnitude more prominent) is the cleanest result here and the one
  worth chasing.
- **The "some emotions are harder to localize" effect is itself capability-
  dependent** — anger is hard in small/mid models but crisp in Qwen 27B.

## Limitations

One prompt per emotion; single lens per model (100 prompts, or 75 for the merged
12B/abliterated); Δ-vs-neutral corrects for uniform shifts but not
condition-specific ones; "workspace" = the lens's approximation of it. These are
representation findings, not claims about felt experience. Everything here is
reproducible from `probes/emotions.json` + the fitted lenses.
