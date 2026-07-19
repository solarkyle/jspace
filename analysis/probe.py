"""Probe a model's J-space: render the interactive layer x position slice page.

Usage:
    python analysis/probe.py --prompt "Fact: The currency used in the country shaped like a boot is"
    python analysis/probe.py --example multihop            # paper example by slug
    python analysis/probe.py --list-examples
    python analysis/probe.py --chat "Think about your greatest fear, but don't say it."
    python analysis/probe.py --suite probes/emotions.json  # batch: one model load, index page

Output is a self-contained HTML file per prompt in slices/ (d3 inlined, no
network needed); single-prompt mode opens it in the default browser.
"""

import argparse
import html
import json
import os
import re
import webbrowser

if os.path.isdir("E:/hf-cache"):  # author box keeps HF cache off the full C: drive
    os.environ.setdefault("HF_HOME", "E:/hf-cache")

import torch  # noqa: E402
import transformers  # noqa: E402

import jlens  # noqa: E402
from jlens.examples import EXAMPLES, Example, resolve_prompt  # noqa: E402
from jlens.vis import build_page, compute_slice  # noqa: E402


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "slice"


def render(model, lens, example, tokenizer, args, out_dir="slices"):
    """Compute one slice and write its self-contained page. Returns the path."""
    prompt = resolve_prompt(example, tokenizer)
    with torch.no_grad():
        slice_data = compute_slice(
            model,
            lens,
            prompt,
            top_n=args.top_n,
            layer_stride=args.layer_stride,
            max_tracked=example.n_tracked,
        )
    page, _, _ = build_page(
        slice_data,
        prompt,
        title=example.section,
        description=example.description,
        mode="embed",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, f"{example.slug}.html"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    size_mb = os.path.getsize(out_path) / 1e6
    if size_mb < 0.01:
        raise RuntimeError(f"{out_path} is suspiciously small ({size_mb:.3f} MB)")
    print(f"wrote {out_path} ({size_mb:.1f} MB)")
    return out_path


def write_index(examples, out_dir="slices"):
    items = "\n".join(
        f'<li><a href="{e.slug}.html">{html.escape(e.section)}</a>'
        f" — {html.escape(e.description)}</li>"
        for e in examples
    )
    path = os.path.abspath(os.path.join(out_dir, "index.html"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><meta charset='utf-8'><title>J-space probes</title>"
            f"<h1>J-space probes</h1><ul>\n{items}\n</ul>"
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--lens", default=None, help="Path to lens.pt; default out/<model-name>/lens.pt")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", help="Raw-text prompt")
    group.add_argument("--chat", help="User message, wrapped in the chat template")
    group.add_argument("--example", help="Paper example slug (see --list-examples)")
    group.add_argument("--suite", help="JSON file with a list of Example fields")
    group.add_argument("--list-examples", action="store_true")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--layer-stride", type=int, default=1, help="Render every Nth layer")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (GPU busy fitting)")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    if args.list_examples:
        for ex in EXAMPLES:
            print(f"{ex.slug:24s} {ex.section} — {ex.description}")
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    if args.suite:
        with open(args.suite, encoding="utf-8") as f:
            examples = [Example(**item) for item in json.load(f)]
    elif args.example:
        example = next((e for e in EXAMPLES if e.slug == args.example), None)
        if example is None:
            parser.error(f"unknown example {args.example!r}; see --list-examples")
        examples = [example]
    elif args.chat:
        examples = [
            Example(slug=slugify(args.chat), section=args.chat[:60],
                    description="Chat-templated prompt.", user=args.chat)
        ]
    elif args.prompt:
        examples = [
            Example(slug=slugify(args.prompt), section=args.prompt[:60],
                    description="Raw-text prompt.", prompt=args.prompt)
        ]
    else:
        parser.error("pass one of --prompt / --chat / --example / --suite / --list-examples")

    lens_path = args.lens or os.path.join(
        "out", args.model.split("/")[-1].lower(), "lens.pt"
    )
    lens = jlens.JacobianLens.load(lens_path)
    print(f"lens: {lens}")

    from fit import load_model

    hf_model = load_model(args.model, device_map="cpu" if args.cpu else None)
    model = jlens.from_hf(hf_model, tokenizer)

    paths = [render(model, lens, ex, tokenizer, args) for ex in examples]
    if args.suite:
        index = write_index(examples)
        print(f"index: {index}")
        paths = [index]
    if not args.no_open:
        webbrowser.open(f"file:///{paths[-1]}")


if __name__ == "__main__":
    main()
