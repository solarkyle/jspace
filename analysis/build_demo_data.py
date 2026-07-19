"""Pack demo data into docs/demo/data.js (rerun whenever inputs update).

Inputs (uses whichever exist):
  out/emotion_all5.json          -> heatmap explorer + evidence chips
  out/uncertainty_v2.jsonl       -> hallucination playground (E4B local run)
  out/uncertainty_trivia_*.jsonl -> cross-model tab data (Modal, when landed)
  out/workspace_dump.json        -> guess-the-emotion game tokens
"""
import glob
import json
import os

SHORT = {
    "google/gemma-4-E4B-it": "Gemma E4B (4B dense)",
    "google/gemma-4-12B-it": "Gemma 12B (dense)",
    "huihui-ai/Huihui-gemma-4-12B-it-abliterated": "Gemma 12B (abliterated)",
    "google/gemma-4-26B-A4B-it": "Gemma 26B (MoE, 4B active)",
    "Qwen/Qwen3.6-27B": "Qwen 3.6 27B (dense)",
}
data = {"short": SHORT}

if os.path.exists("out/emotion_all5.json"):
    data["emotion"] = json.load(open("out/emotion_all5.json", encoding="utf-8"))

if os.path.exists("out/uncertainty_v2.jsonl"):
    rows = [json.loads(l) for l in open("out/uncertainty_v2.jsonl", encoding="utf-8")]
    data["uncertainty"] = {"google/gemma-4-E4B-it": [
        {"q": r["q"], "a": r["answer"], "c": int(r["correct"]),
         "lp": round(r["bl_first_token_logprob"], 4),
         "ent": round(r["mean_entropy"], 4)} for r in rows]}

for path in glob.glob("out/uncertainty_trivia_*.jsonl"):
    slug = os.path.basename(path)[len("uncertainty_trivia_"):-len(".jsonl")]
    mid = next((m for m in SHORT if m.split("/")[-1].lower() == slug), None)
    if not mid:
        continue
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    data.setdefault("uncertainty", {})[mid] = [
        {"q": r["q"], "a": r["answer"], "c": int(r["correct"]),
         "lp": round(r["bl_first_token_logprob"], 4),
         "ent": round(r["mean_entropy"], 4)} for r in rows]

if os.path.exists("out/workspace_dump.json"):
    data["game"] = json.load(open("out/workspace_dump.json", encoding="utf-8"))

os.makedirs("docs/demo", exist_ok=True)
out = "docs/demo/data.js"
with open(out, "w", encoding="utf-8") as f:
    f.write("window.JSPACE = " + json.dumps(data, ensure_ascii=False) + ";\n")
print(f"wrote {out} ({os.path.getsize(out)//1024} KB), sections: {sorted(k for k in data if k != 'short')}")
