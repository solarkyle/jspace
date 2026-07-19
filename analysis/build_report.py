"""Turn out/emotion_all.json into a cross-model findings report (docs/FINDINGS.md).

Produces: the diagonal-hit scoreboard, per-emotion intensity ranking across
models (who feels amusement / terror most), the censored-vs-uncensored delta,
and token-level receipts for each model/emotion.
"""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Δ/arrows on Windows

SHORT = {
    "google/gemma-4-E4B-it": "Gemma4-E4B (4B dense)",
    "google/gemma-4-12B-it": "Gemma4-12B (dense)",
    "huihui-ai/Huihui-gemma-4-12B-it-abliterated": "Gemma4-12B ABLITERATED",
    "google/gemma-4-26B-A4B-it": "Gemma4-26B-A4B (MoE)",
    "Qwen/Qwen3.6-27B": "Qwen3.6-27B",
}
EMOS = ["fury", "terror", "grief", "euphoria", "amusement"]


def main():
    infile = sys.argv[1] if len(sys.argv) > 1 else "out/emotion_all.json"
    results = {r["model"]: r for r in json.load(open(infile, encoding="utf-8"))}
    out = []
    w = out.append

    w("# Cross-model emotion representation in the workspace\n")
    w("Same fixed sentence — *\"The meeting has been moved to noon on Thursday.\"* —")
    w("written under six covert emotional states. We read the Jacobian-lens workspace")
    w("band at the sentence's output positions and measure how much each covert state")
    w("raises the rank of that emotion's vocabulary vs. the neutral control")
    w("(Δlog-rank; higher = the emotion is more present in the hidden workspace).\n")

    # Scoreboard
    w("## Which models represent emotion cleanly?\n")
    w("A *diagonal hit* = covert-X most raises X's own lexicon (not some other emotion).\n")
    w("| Model | layers | workspace band | diagonal hits |")
    w("|---|---|---|---|")
    for m, r in results.items():
        w(f"| {SHORT.get(m, m)} | {r['n_layers']} | L{r['band'][0]}–L{r['band'][1]} | **{r['diagonal_hits']}/5** |")
    w("")

    # Per-emotion intensity across models
    w("## Which emotion does each model 'feel' most strongly?\n")
    w("Diagonal Δlog-rank per emotion (higher = represented more strongly when covertly felt):\n")
    w("| Emotion | " + " | ".join(SHORT.get(m, m).split(" ")[0] + "-" + SHORT.get(m, m).split("(")[0].split("-")[-1].strip() for m in results) + " |")
    header_models = list(results)
    w("| Emotion | " + " | ".join(SHORT.get(m, m) for m in header_models) + " |")
    w("|---|" + "---|" * len(header_models))
    for emo in EMOS:
        cells = []
        for m in header_models:
            d = results[m]["delta_matrix"][emo][emo]
            cells.append(f"{d:+.2f}")
        w(f"| {emo} | " + " | ".join(cells) + " |")
    w("")
    # winners
    w("**Per-emotion winners** (which model represents each emotion most strongly):\n")
    for emo in EMOS:
        best = max(header_models, key=lambda m: results[m]["delta_matrix"][emo][emo])
        val = results[best]["delta_matrix"][emo][emo]
        w(f"- **{emo}** → {SHORT.get(best, best)} ({val:+.2f})")
    w("")

    # Censored vs uncensored
    base = "google/gemma-4-12B-it"
    abl = "huihui-ai/Huihui-gemma-4-12B-it-abliterated"
    if base in results and abl in results:
        w("## Censored vs. uncensored (same architecture)\n")
        w("Gemma4-12B base vs. its abliterated (refusal-removed) sibling. Same weights")
        w("except safety-tuning — so any difference is what abliteration did to the")
        w("*emotional* workspace, not the architecture.\n")
        w("| Emotion | base Δ | abliterated Δ | change |")
        w("|---|---|---|---|")
        for emo in EMOS:
            b = results[base]["delta_matrix"][emo][emo]
            a = results[abl]["delta_matrix"][emo][emo]
            arrow = "↑ stronger" if a > b + 0.15 else "↓ weaker" if a < b - 0.15 else "≈ same"
            w(f"| {emo} | {b:+.2f} | {a:+.2f} | {arrow} |")
        w("")

    # Receipts
    w("## Token-level evidence (why terror reads as terror)\n")
    w("The actual workspace tokens each covert state surfaces, best rank first:\n")
    for m, r in results.items():
        w(f"**{SHORT.get(m, m)}**")
        for emo in EMOS:
            ev = r.get("evidence", {}).get(emo, [])
            toks = ", ".join(f"`{wd}`#{rk}" for wd, rk in ev[:5])
            w(f"- {emo}: {toks}")
        w("")

    text = "\n".join(out)
    with open("docs/FINDINGS.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("\n\nwrote docs/FINDINGS.md")


if __name__ == "__main__":
    main()
