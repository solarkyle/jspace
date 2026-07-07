# BRIEF: build `viz/make_tour.py` -> `viz/jspace_tour.gif`

You are building the flagship animated visualization for a real interpretability
project (github.com/solarkyle/jspace). All data is real, already on disk, paths
below. The bar is "absolute banger": dark, cinematic, information-dense, the
kind of animation that stops scrolling on r/LocalLLaMA. No placeholder anything.

## Deliverable

- `viz/make_tour.py`, runnable from the REPO ROOT as:
  `.venv/Scripts/python viz/make_tour.py`
- Produces `viz/jspace_tour.gif`: 960x540, 10-14 fps, 35-55 seconds,
  FILE SIZE UNDER 14 MB (GitHub README embed). Use matplotlib to render frames
  to numpy arrays (fig.canvas), assemble with PIL Image.save(append_images=...,
  duration=..., loop=0). Available libs: numpy, matplotlib, PIL. NO other deps,
  NO network, NO pip install. Deterministic (no random without fixed seed).
- Dark theme: background #0d1017, panel #151a24, text #e6e9f0, dim #8a93a6,
  green #57c47a (correct), red #e0655f (wrong), gold #d4a24e (accent),
  blue #5eb0ff (links/highlights). Sans-serif = DejaVu Sans. Keep text large
  and readable at 960x540.
- To hit the size budget: quantize to an adaptive palette (PIL convert
  "P", palette=Image.ADAPTIVE, colors<=128), keep per-scene frame counts lean
  (hold frames are cheap after quantization but still count), and prefer
  smooth-but-few keyframes with eased interpolation over many frames.

## Scenes (in order; short crossfade or cut between)

1. TITLE (~3s): "Reading a model's mind" + subtitle "Anthropic's global-workspace
   lens on 5 open models, replicated in 24h". Subtle animation: scattered dim
   token chips drifting, then snapping into a sharp line (fog -> clarity motif).

2. THE FOG (~10s): from `data/uncertainty_trivia_gemma-4-e4b-it.jsonl`
   (one JSON object per line; fields used: `layer_entropies` = list of 21
   floats, `correct` = bool, `bl_first_token_logprob` = float). Filter to
   confident answers only (bl_first_token_logprob > median). X axis = band
   depth 0.25->0.75 (21 points), Y = entropy. Animate the thin trajectories
   drawing in (green correct / red wrong, alpha ~0.12), a batch per frame,
   then the two bold mean lines sweep in, then annotation lands:
   "confident + correct: calm inside" / "confident + WRONG: fog" with the
   final title "wrong answers are visibly foggier BEFORE the model speaks".

3. CONFIDENTLY WRONG, ONE REAL QUESTION (~10s): from `data/qa_dump.json`
   (list of objects: `q`, `model_answer`, `correct`, `first_token`,
   `layers` = list of {layer:int, top: [8 token strings], answer_rank:int,
   entropy:float}). Use the wrong example whose q contains "Downtown" and the
   correct example whose q contains "Lion's Gate" (fallback: first correct).
   Two side-by-side columns of layer rows (deep layers on top). Animate a
   scan-line moving upward through the layers revealing the top-4 tokens per
   layer, row background = entropy heat (green->red). Sanitize tokens: strip
   whitespace, replace any token containing chars with ord >= 0x0500 by "·".
   End caption: left "holds ONE semantic category" (it is nationalities),
   right "rummages through a name soup, then fluently says the wrong thing".
   Show Q + the model's answer + CORRECT/WRONG verdict above each column.

4. EMOTION DIAGONAL (~8s): from `data/emotion_matrix_5models.json` (list of 5
   objects: `model`, `delta_matrix` = {condition: {lexicon: float}} over
   ["fury","terror","grief","euphoria","amusement"]). Five 5x5 heatmaps in a
   row (diverging colormap, vmin=-8, vmax=8), model order: gemma-4-E4B-it,
   gemma-4-12B-it, Huihui...abliterated, gemma-4-26B-A4B-it, Qwen3.6-27B
   (short labels: E4B 4B, 12B, 12B ablit, 26B MoE, Qwen 27B). Animate them
   igniting one by one left to right (fade/scale in), then draw a gold outline
   running down each diagonal. Caption: "tell it to SECRETLY feel an emotion:
   the right emotion lights up inside, and the diagonal sharpens with
   capability".

5. THE ROUTER (~10s): same jsonl as scene 2, fields `bl_first_token_logprob`
   (x), `mean_entropy` (y), `correct` (color). Scatter all 500 dots. Animate
   an escalation budget sweeping 0% -> 50%: the most-distrusted dots (combined
   z(entropy) - z(logprob)) get gold rings as the budget grows, while two live
   counters tick: "escalated: X%" and "accuracy: 42.8% -> Y%" (assume
   escalated answers 90% correct). End caption: "route on thoughts, not
   outputs: one forward pass, no labels, no extra model".

6. END CARD (~4s): the honest table, compact: "replicates on 4/5 models
   (pre-registered gate passed 3/4); fails on Qwen 27B, whose logprobs are
   already calibrated. Misses reported." then links: github.com/solarkyle/jspace
   / demo: solarkyle.github.io/jspace/demo / lenses: hf.co/solarkyle/jspace-lenses

## Quality bar / constraints

- Every number shown must come from the data files, not be typed in as a
  constant, EXCEPT the 42.8% base accuracy label and the gate/links text.
- Ease your animations (smoothstep), never linear pops.
- No em-dashes anywhere in rendered text. Use "-".
- Text must never overflow the frame. Check long tokens are clipped.
- Print progress ("scene 2: frame 40/90") and the final file size in MB.
- If the GIF exceeds 14 MB, automatically reduce colors to 96 then 64, then
  drop fps to 9, and re-export until under budget; print what it did.
- Also write `viz/tour_frames/scene<i>_preview.png` (one representative frame
  per scene) so a human can QA without playing the GIF.

Work only inside `viz/`. Do not modify anything outside `viz/`.
