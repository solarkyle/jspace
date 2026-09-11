"""Build the Self-Focus Lab report: Markdown plus a self-contained HTML page.

Sections that have no data are reported as not executed rather than omitted, so a
reader can tell the difference between a null result and missing work.
"""

from __future__ import annotations

import html
import io
import json
from pathlib import Path

import numpy as np

from experiments.self_focus import core

ORDER = ["NORMAL", "CHECK", "MEDITATE", "J_INFO", "J_FOCUS", "J_ROLEPLAY"]


def _cos(a, b) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_demo(out_dir: Path) -> dict:
    reports = {r["condition"]: r for r in core.read_trials(out_dir / "demo_reports.jsonl")}
    traces = {(r["condition"], r["passage_index"]): r
              for r in core.read_trials(out_dir / "demo_traces.jsonl")}
    residuals: dict = {}
    path = out_dir / "demo_residuals.jsonl"
    if path.exists():
        with io.open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                residuals[(row["condition"], row["passage_index"])] = {
                    int(k): np.asarray(v, dtype=np.float64) for k, v in row["layers"].items()}
    meta = {}
    if (out_dir / "demo_meta.json").exists():
        meta = core.load_json(out_dir / "demo_meta.json")
    return {"reports": reports, "traces": traces, "residuals": residuals, "meta": meta}


def contrast_table(residuals: dict, conds: dict) -> list[dict]:
    """Per-layer similarity for every condition pair, aligned by continuation offset.

    Prompt lengths differ between conditions, so absolute positions do not
    correspond. Alignment is by offset into the forced continuation, which is the
    same token sequence in every condition.
    """
    if not residuals:
        return []
    layers = sorted(next(iter(residuals.values())).keys())
    passages = sorted({p for (_, p) in residuals})
    registered = {tuple(pair) for pair in conds["primary_contrasts"]}
    rows = []
    for a in ORDER:
        for b in ORDER:
            if a >= b:
                continue
            for layer in layers:
                cosines, norm_deltas = [], []
                for p in passages:
                    if (a, p) not in residuals or (b, p) not in residuals:
                        continue
                    A, B = residuals[(a, p)][layer], residuals[(b, p)][layer]
                    n = min(len(A), len(B))
                    for i in range(n):
                        cosines.append(_cos(A[i], B[i]))
                        na, nb = np.linalg.norm(A[i]), np.linalg.norm(B[i])
                        norm_deltas.append(abs(na - nb) / (nb + 1e-12))
                if not cosines:
                    continue
                rows.append({
                    "a": a, "b": b, "layer": layer,
                    "registered": (a, b) in registered or (b, a) in registered,
                    "mean_cos": float(np.mean(cosines)),
                    "min_cos": float(np.min(cosines)),
                    "mean_rel_norm_delta": float(np.mean(norm_deltas)),
                    "n_positions": len(cosines),
                })
    return rows


def build(out_dir: Path) -> dict:
    cfg, conds = core.config(), core.conditions()
    demo = load_demo(out_dir)
    rows = contrast_table(demo["residuals"], conds)
    summary = {
        "protocol_version": cfg["protocol_version"],
        "demo_reports": len(demo["reports"]),
        "demo_traces": len(demo["traces"]),
        "contrast_rows": len(rows),
        "qa_dev_trials": len(core.read_trials(out_dir / "qa_dev.jsonl")),
        "qa_eval_trials": len(core.read_trials(out_dir / "qa_eval.jsonl")),
        "intervention_trials": len(core.read_trials(out_dir / "intervention.jsonl")),
        "external_compute_spend_usd": 0.0,
    }
    md = render_markdown(cfg, conds, demo, rows, summary)
    page = render_html(cfg, conds, demo, rows, summary)
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")
    (out_dir / "report.html").write_text(page, encoding="utf-8")
    return summary


def _registered_rows(rows):
    return [r for r in rows if r["registered"]]


def render_markdown(cfg, conds, demo, rows, summary) -> str:
    meta = demo["meta"]
    prov = meta.get("provenance", {})
    L = []
    A = L.append
    A("# Self-Focus Lab: first milestone\n")
    A(f"Protocol `{cfg['protocol_version']}`. External compute spend **$0.00**. "
      f"All work local.\n")

    A("## What was executed\n")
    A("| item | count |")
    A("|---|---:|")
    for key in ("demo_reports", "demo_traces", "qa_dev_trials", "qa_eval_trials",
                "intervention_trials"):
        A(f"| {key.replace('_', ' ')} | {summary[key]} |")
    A("")
    not_run = [k for k in ("qa_dev_trials", "qa_eval_trials", "intervention_trials")
               if summary[k] == 0]
    if not_run:
        A(f"**Not executed in this milestone:** {', '.join(n.replace('_', ' ') for n in not_run)}. "
          f"Reported as missing, not as a null result.\n")

    A("## OBSERVED: the six free reports\n")
    A("Exploratory only, and not fed into any later experiment. All six are shown, "
      "including the mechanical ones.\n")
    for name in ORDER:
        r = demo["reports"].get(name)
        if not r:
            continue
        mean_lp = (sum(r["step_logprobs"]) / len(r["step_logprobs"])
                   if r["step_logprobs"] else float("nan"))
        A(f"### {name}")
        A(f"`{len(r['gen_token_ids'])} tokens, prompt {r['n_prompt_tokens']} tokens, "
          f"mean logprob {mean_lp:.3f}`\n")
        A("> " + r["text"].strip().replace("\n", "\n> ") + "\n")

    A("## OBSERVED: fixed-continuation readouts\n")
    A("**Teacher-forced fixed continuation.** Every condition emits the same token "
      "sequence, so the output text is controlled and only the instruction differs. "
      "Aligned by offset into the continuation, because prompt lengths differ by "
      "condition and absolute positions do not correspond.\n")
    if prov:
        A(f"Model `{prov.get('model_id')}` loaded from `{prov.get('loaded_from')}`, "
          f"precision `{prov.get('precision')}`, {prov.get('n_layers')} layers at "
          f"`{prov.get('block_path')}`.")
    lens = meta.get("lens", {})
    A(f"Layers {meta.get('layers')} selected from the "
      f"{'lens fitted band ' + str(lens.get('band')) if lens.get('available') else 'no-lens fallback'}. "
      f"Zero-based. Not called a validated workspace band for this model.\n")

    reg = _registered_rows(rows)
    if reg:
        A("### Registered contrasts\n")
        A("| contrast | layer | mean cos | min cos | mean relative norm delta | positions |")
        A("|---|---:|---:|---:|---:|---:|")
        for r in reg:
            A(f"| {r['a']} - {r['b']} | {r['layer']} | {r['mean_cos']:.4f} | "
              f"{r['min_cos']:.4f} | {r['mean_rel_norm_delta']:.4f} | {r['n_positions']} |")
        A("")
    if rows:
        layers = sorted({r["layer"] for r in rows})
        mid = layers[len(layers) // 2]
        A(f"### All pairs at layer {mid}, exploratory\n")
        A("| pair | mean cos | min cos |")
        A("|---|---:|---:|")
        for r in sorted([x for x in rows if x["layer"] == mid], key=lambda x: x["mean_cos"]):
            A(f"| {r['a']} vs {r['b']} | {r['mean_cos']:.4f} | {r['min_cos']:.4f} |")
        A("")

    A("## INTERPRETATION, separated from the above\n")
    A("- The instruction changes measured internal representations while the emitted "
      "text is held identical. That is a prompt-conditioned representation "
      "difference. It is **not** evidence of privileged self-access, and the brief "
      "says so in advance: prompt content itself changes activations.")
    A("- The model's own reports mostly say it cannot distinguish an effect. Self-report "
      "and measured internals therefore disagree in direction, which is a result worth "
      "recording and not yet an explanation.")
    A("- Ordering of the pairwise differences tracks how much the prompt text differs, "
      "so the registered contrasts are **not** the largest differences. Separating a "
      "self-focus effect from a prompt-length and prompt-content effect needs the "
      "matched controls that are not in this milestone.")
    A("")
    A("## UNTESTED HYPOTHESES\n")
    A("- Whether any of this improves answer quality or error identification.")
    A("- Whether self-focus changes detection of a hidden activation intervention.")
    A("- Whether the effect survives matched prompt length and matched token count.")
    A("- Anything about consciousness. No such measurement is made or implied.\n")

    A("## Provenance\n")
    if prov:
        A("```json")
        A(json.dumps({k: prov[k] for k in prov if k != "file_hashes"}, indent=2))
        A("```")
    A(f"Budget: `{meta.get('budget')}`\n")
    A("Commands:\n")
    A("```bash")
    A("python -m experiments.self_focus.runner preflight --dry-run")
    A("python -m experiments.self_focus.runner preflight")
    A("python -m experiments.self_focus.runner demo")
    A("python -m unittest experiments.self_focus.test_self_focus")
    A("python -m unittest experiments.self_focus.test_self_focus_model")
    A("python -m experiments.self_focus.report")
    A("```")
    return "\n".join(L) + "\n"


def render_html(cfg, conds, demo, rows, summary) -> str:
    meta = demo["meta"]
    reg = _registered_rows(rows)
    layers = sorted({r["layer"] for r in rows})
    mid = layers[len(layers) // 2] if layers else None

    def esc(x):
        return html.escape(str(x))

    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>Self-Focus Lab, first milestone</title><style>
:root{--fg:#1b1b1b;--bg:#fbfbf9;--mut:#666;--line:#ddd;--hl:#f3efe6}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1040px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:14px;margin:18px 0 4px} .mut{color:var(--mut)} code{background:#efece6;padding:1px 4px;border-radius:2px}
table{border-collapse:collapse;margin:8px 0;font-size:13px} th,td{border:1px solid var(--line);padding:4px 8px;text-align:right}
th:first-child,td:first-child{text-align:left} tr.reg{background:var(--hl);font-weight:600}
blockquote{margin:6px 0;padding:8px 12px;border-left:3px solid #bbb;background:#fff;white-space:pre-wrap}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border:1px solid var(--line);border-radius:2px;background:#fff;margin-right:6px}
.obs{border-left:3px solid #4a7;padding-left:10px} .int{border-left:3px solid #c84;padding-left:10px}
.unt{border-left:3px solid #999;padding-left:10px}
</style></head><body>"""]
    P = parts.append
    P(f"<h1>Self-Focus Lab, first milestone</h1>")
    P(f"<div class=mut>Protocol <code>{esc(cfg['protocol_version'])}</code> &middot; "
      f"external compute spend <strong>$0.00</strong> &middot; all work local</div>")

    P("<h2>What was executed</h2><table><tr><th>item</th><th>count</th></tr>")
    for key in ("demo_reports", "demo_traces", "qa_dev_trials", "qa_eval_trials",
                "intervention_trials"):
        P(f"<tr><td>{esc(key.replace('_',' '))}</td><td>{summary[key]}</td></tr>")
    P("</table>")
    not_run = [k for k in ("qa_dev_trials", "qa_eval_trials", "intervention_trials")
               if summary[k] == 0]
    if not_run:
        P(f"<p class=mut><strong>Not executed:</strong> "
          f"{esc(', '.join(n.replace('_',' ') for n in not_run))}. "
          f"Missing work, not a null result.</p>")

    P("<h2>Observed: the six free reports</h2>")
    P("<p class=mut>Exploratory only. All six shown, including mechanical ones. "
      "Not fed into any later experiment.</p><div class=obs>")
    for name in ORDER:
        r = demo["reports"].get(name)
        if not r:
            continue
        mean_lp = (sum(r["step_logprobs"]) / len(r["step_logprobs"])
                   if r["step_logprobs"] else float("nan"))
        P(f"<h3>{esc(name)}</h3><div class=mut>"
          f"<span class=tag>{len(r['gen_token_ids'])} gen tokens</span>"
          f"<span class=tag>prompt {r['n_prompt_tokens']}</span>"
          f"<span class=tag>mean logprob {mean_lp:.3f}</span></div>"
          f"<blockquote>{esc(r['text'].strip())}</blockquote>")
    P("</div>")

    P("<h2>Observed: fixed-continuation readouts</h2>")
    P("<p><strong>Teacher-forced fixed continuation.</strong> Every condition emits the "
      "same token sequence, so output text is controlled and only the instruction varies. "
      "Aligned by continuation offset, because prompt lengths differ by condition.</p>")
    lens = meta.get("lens", {})
    P(f"<p class=mut>Layers {esc(meta.get('layers'))}, zero-based, from "
      f"{'the lens fitted band' if lens.get('available') else 'the no-lens fallback'}. "
      f"Not called a validated workspace band for this model.</p>")
    if reg:
        P("<h3>Registered contrasts</h3><table><tr><th>contrast</th><th>layer</th>"
          "<th>mean cos</th><th>min cos</th><th>rel norm delta</th><th>positions</th></tr>")
        for r in reg:
            P(f"<tr class=reg><td>{esc(r['a'])} &minus; {esc(r['b'])}</td><td>{r['layer']}</td>"
              f"<td>{r['mean_cos']:.4f}</td><td>{r['min_cos']:.4f}</td>"
              f"<td>{r['mean_rel_norm_delta']:.4f}</td><td>{r['n_positions']}</td></tr>")
        P("</table>")
    if mid is not None:
        P(f"<h3>All pairs at layer {mid}, exploratory</h3>"
          "<table><tr><th>pair</th><th>mean cos</th><th>min cos</th></tr>")
        for r in sorted([x for x in rows if x["layer"] == mid], key=lambda x: x["mean_cos"]):
            cls = " class=reg" if r["registered"] else ""
            P(f"<tr{cls}><td>{esc(r['a'])} vs {esc(r['b'])}</td>"
              f"<td>{r['mean_cos']:.4f}</td><td>{r['min_cos']:.4f}</td></tr>")
        P("</table><p class=mut>Highlighted rows are the registered contrasts.</p>")

    P("<h2>Interpretation, kept separate</h2><div class=int><ul>"
      "<li>The instruction changes measured internal representations while the emitted "
      "text is identical. That is a prompt-conditioned representation difference, "
      "<strong>not</strong> evidence of privileged self-access.</li>"
      "<li>The model's own reports mostly deny a distinguishable effect, so self-report "
      "and measured internals disagree in direction. Recorded, not explained.</li>"
      "<li>Pairwise differences track how much the prompt text differs, so the "
      "registered contrasts are not the largest ones. Separating self-focus from "
      "prompt content and length needs controls absent from this milestone.</li>"
      "</ul></div>")
    P("<h2>Untested</h2><div class=unt><ul>"
      "<li>Whether any of this improves answer quality or error identification.</li>"
      "<li>Whether self-focus changes detection of a hidden activation intervention.</li>"
      "<li>Whether the effect survives matched prompt length and token count.</li>"
      "<li>Anything about consciousness. No such measurement is made or implied.</li>"
      "</ul></div>")
    prov = meta.get("provenance", {})
    if prov:
        P("<h2>Provenance</h2><pre style='white-space:pre-wrap;font-size:12px'>"
          + esc(json.dumps({k: prov[k] for k in prov if k != 'file_hashes'}, indent=2))
          + "</pre>")
    P(f"<p class=mut>Budget: <code>{esc(meta.get('budget'))}</code></p>")
    P("</body></html>")
    return "".join(parts)


def main() -> int:
    cfg = core.config()
    out_dir = core.REPO / cfg["output_dir"]
    summary = build(out_dir)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'REPORT.md'}")
    print(f"wrote {out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
