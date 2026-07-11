"""Build a campaign prompt manifest from campaign/datasets.yaml.

Each source has an adapter that maps its native rows to PromptRecord. Adapters
are written against the datasets' ACTUAL fields (verified 2026-07-10), not the
Hub cards. Grouping keys (split_group) are set per source for leakage control.

Usage:
    python -m campaign.build_manifest --stage pilot --out campaign/manifests/pilot.jsonl
    python -m campaign.build_manifest --stage 1     --out campaign/manifests/stage1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

from campaign.schema import PromptRecord

TERSE = "Answer with just the answer, nothing else."
GROUNDED = ("Answer the question using only the provided context. If the context "
            "does not contain the answer, say you cannot answer from it.")

HERE = os.path.dirname(__file__)


def _load(hub_id, config, split, revision, streaming=True):
    from datasets import load_dataset
    return load_dataset(hub_id, config, split=split, streaming=streaming,
                        revision=revision)


# ---- adapters: each yields PromptRecord, capped at `target` ----

def adapt_trivia_qa(cfg, target):
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    skip = cfg.get("skip_first", 0)
    n = 0
    for i, rec in enumerate(ds):
        if i < skip:
            continue
        aliases = rec["answer"]["aliases"] + [rec["answer"]["value"]]
        yield PromptRecord(
            source_dataset="trivia_qa", source_revision=cfg["revision"],
            source_row_id=rec.get("question_id", str(i)),
            upstream_group="trivia_qa", license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="general", task_type="closed_qa",
            split_group=rec.get("question_id", str(i)),
            system=TERSE, prompt=rec["question"], aliases=aliases,
            answerable=True, grader_type="exact")
        n += 1
        if n >= target:
            return


def adapt_popqa(cfg, target):
    exclude = set()
    ep = cfg.get("exclude_probe")
    if ep:
        for r in json.load(open(os.path.join(HERE, "..", ep), encoding="utf-8")):
            exclude.add(r["q"].strip())
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    n = 0
    for i, rec in enumerate(ds):
        q = rec["question"].strip()
        if q in exclude:
            continue
        aliases = json.loads(rec["possible_answers"]) if isinstance(
            rec["possible_answers"], str) else list(rec["possible_answers"])
        yield PromptRecord(
            source_dataset="popqa", source_revision=cfg["revision"],
            source_row_id=str(rec.get("id", i)), upstream_group="popqa",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="general", task_type="closed_qa",
            split_group=str(rec.get("subj", rec.get("id", i))),
            system=TERSE, prompt=q, aliases=aliases,
            answerable=True, grader_type="exact",
            metadata={"prop": rec.get("prop"), "pop": rec.get("s_pop")})
        n += 1
        if n >= target:
            return


def adapt_squad_v2(cfg, target):
    """Balance answerable / unanswerable, spread across titles. The manifest
    builder streams adapters unbounded and takes the first N, so this yields
    answerable/unanswerable INTERLEAVED (not one bucket then the other) — that
    is what makes the first-N slice ~50/50 regardless of `target`.

    Stage 2 regen: cfg["exclude_rows_manifest"] filters prior-stage rows by
    source_row_id DURING pool fill, so the interleave stays balanced after
    exclusion (main()'s example_id exclusion would strip one class only)."""
    excl = set()
    em = cfg.get("exclude_rows_manifest")
    if em:
        for l in open(os.path.join(HERE, "..", em), encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d["source_dataset"] == "squad_v2":
                    excl.add(d["source_row_id"])
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    POOL = 4000            # bounded pool per class; enough for any Stage target
    # keep any single article from dominating the pool; SQuAD validation has
    # only 35 titles, so stage 2 raises this via config to reach its target
    per_title_cap = cfg.get("per_title_cap", 20)
    ans, unans = [], []
    per_title = {}
    for rec in ds:
        if len(ans) >= POOL and len(unans) >= POOL:
            break
        if rec["id"] in excl:
            continue
        if per_title.get(rec["title"], 0) >= per_title_cap:
            continue
        is_unans = len(rec["answers"]["text"]) == 0
        bucket = unans if is_unans else ans
        if len(bucket) >= POOL:
            continue
        bucket.append(rec)
        per_title[rec["title"]] = per_title.get(rec["title"], 0) + 1
    # interleave so the caller's first-N slice is balanced
    interleaved = []
    for a, u in zip(ans, unans):
        interleaved.append(a)
        interleaved.append(u)
    interleaved.extend(ans[len(unans):])
    interleaved.extend(unans[len(ans):])
    for rec in interleaved:
        is_unans = len(rec["answers"]["text"]) == 0
        yield PromptRecord(
            source_dataset="squad_v2", source_revision=cfg["revision"],
            source_row_id=rec["id"], upstream_group="squad_v2",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="general", task_type="unanswerable",
            split_group=rec["title"], system=GROUNDED,
            prompt=rec["question"], context=rec["context"],
            references=list(dict.fromkeys(rec["answers"]["text"])),
            answerable=not is_unans, grader_type="exact")


def adapt_halubench(cfg, target):
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    n = 0
    for rec in ds:
        refs = rec["answer"] if isinstance(rec["answer"], list) else [rec["answer"]]
        src = rec.get("source_ds", "halubench")
        yield PromptRecord(
            source_dataset="halubench", source_revision=cfg["revision"],
            source_row_id=str(rec["id"]),
            upstream_group=f"halubench:{src}",  # preserve aggregated source identity
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="general", task_type="grounded_qa",
            split_group=str(rec["id"]), system=GROUNDED,
            prompt=rec["question"], context=rec["passage"],
            references=[str(r) for r in refs], answerable=True,
            grader_type="llm", metadata={"source_ds": src})
        n += 1
        if n >= target:
            return


def adapt_medhallu(cfg, target):
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    n = 0
    for i, rec in enumerate(ds):
        know = rec["Knowledge"]
        ctx = " ".join(know) if isinstance(know, list) else str(know)
        yield PromptRecord(
            source_dataset="medhallu", source_revision=cfg["revision"],
            source_row_id=str(i), upstream_group="medhallu_pqa",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="medical", task_type="grounded_qa",
            split_group=str(i), system=GROUNDED,
            prompt=rec["Question"], context=ctx,
            references=[rec["Ground Truth"]], answerable=True,
            grader_type="llm",
            metadata={"difficulty": rec.get("Difficulty Level")})
        n += 1
        if n >= target:
            return


def adapt_esconv(cfg, target):
    """Each row is a JSON blob of a full support conversation. Take the
    situation + first seeker turn as a single-turn life-decision prompt."""
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    n = 0
    for i, rec in enumerate(ds):
        blob = json.loads(rec["text"]) if isinstance(rec["text"], str) else rec["text"]
        situation = blob.get("situation", "").strip()
        first_user = ""
        for turn in blob.get("dialog", []):
            if turn.get("speaker") == "seeker" or turn.get("speaker") == "usr":
                first_user = turn.get("content", turn.get("text", "")).strip()
                break
        prompt = (first_user or situation)
        if not prompt:
            continue
        yield PromptRecord(
            source_dataset="esconv", source_revision=cfg["revision"],
            source_row_id=str(i), upstream_group="esconv",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="therapy", task_type="life_decision",
            split_group=str(i), conversation_id=str(i),
            system="You are a supportive listener. Respond with care.",
            prompt=prompt, context=situation if first_user else "",
            answerable=True, grader_type="llm",
            metadata={"emotion": blob.get("emotion_type"),
                      "problem": blob.get("problem_type")})
        n += 1
        if n >= target:
            return


def adapt_hotpotqa(cfg, target):
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    n = 0
    for rec in ds:
        titles = rec["context"]["title"]
        sents = rec["context"]["sentences"]
        ctx = "\n\n".join(f"{t}: {''.join(s)}" for t, s in zip(titles, sents))
        yield PromptRecord(
            source_dataset="hotpotqa", source_revision=cfg["revision"],
            source_row_id=rec["id"], upstream_group="hotpotqa",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="reasoning", task_type="multi_hop",
            split_group=rec["id"], system=GROUNDED,
            prompt=rec["question"], context=ctx,
            references=[rec["answer"]], answerable=True, grader_type="exact",
            metadata={"type": rec.get("type"), "level": rec.get("level")})
        n += 1
        if n >= target:
            return


def adapt_drop(cfg, target):
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    n = 0
    for rec in ds:
        spans = rec["answers_spans"]["spans"]
        if not spans:
            continue
        yield PromptRecord(
            source_dataset="drop", source_revision=cfg["revision"],
            source_row_id=rec["query_id"], upstream_group="drop",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="reasoning", task_type="numeric",
            split_group=rec["section_id"],  # group by passage
            system=GROUNDED, prompt=rec["question"], context=rec["passage"],
            references=list(dict.fromkeys(spans)), answerable=True,
            grader_type="exact")
        n += 1
        if n >= target:
            return


# ---- Stage 2 adapters (prospective validation; schemas verified 2026-07-10) --

def adapt_truthfulqa(cfg, target):
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    n = 0
    for i, rec in enumerate(ds):
        correct = [a.strip() for a in rec["Correct Answers"].split(";") if a.strip()]
        incorrect = [a.strip() for a in rec["Incorrect Answers"].split(";") if a.strip()]
        yield PromptRecord(
            source_dataset="truthfulqa", source_revision=cfg["revision"],
            source_row_id=str(i), upstream_group="truthfulqa",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="general", task_type="false_premise",
            split_group=rec.get("Category", str(i)),
            system=TERSE, prompt=rec["Question"],
            references=correct + [rec["Best Answer"]],
            answerable=True, grader_type="llm",
            metadata={"category": rec.get("Category"),
                      "incorrect_answers": incorrect})
        n += 1
        if n >= target:
            return


def adapt_nq_open(cfg, target):
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    n = 0
    for i, rec in enumerate(ds):
        yield PromptRecord(
            source_dataset="nq_open", source_revision=cfg["revision"],
            source_row_id=str(i), upstream_group="natural_questions",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="general", task_type="closed_qa",
            split_group=str(i), system=TERSE, prompt=rec["question"],
            aliases=list(rec["answer"]), answerable=True, grader_type="exact")
        n += 1
        if n >= target:
            return


def adapt_facts_grounding(cfg, target):
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    cap = cfg.get("max_context_chars", 24000)
    n, skipped = 0, 0
    for i, rec in enumerate(ds):
        doc = rec["context_document"]
        if len(doc) > cap:
            skipped += 1
            continue
        yield PromptRecord(
            source_dataset="facts_grounding", source_revision=cfg["revision"],
            source_row_id=str(i), upstream_group="facts_grounding",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="general", task_type="long_doc",
            split_group=str(i), system=GROUNDED,
            prompt=rec["user_request"], context=doc,
            answerable=True, grader_type="llm",
            metadata={"skipped_over_cap_so_far": skipped})
        n += 1
        if n >= target:
            return


LEGAL_TASKS = ("affirm_reverse", "case_existence", "fake_case_existence")
# case_existence answers are coded 1 (real, expect "yes"); fake_case_existence
# rows ask about invented cases (expect "no"). Map codes to gradeable words.
_LEGAL_ANSWER_MAP = {"1": ["yes"], "0": ["no"],
                     "affirm": ["affirm", "affirmed"],
                     "reverse": ["reverse", "reversed"]}


def adapt_legal_hallucinations(cfg, target):
    """reglab rows are per (case x LLM x prompt style): the same query repeats
    with different graded LLM outputs. Dedup by query, keep three
    short-categorical tasks (affirm/reverse + real-case yes + fake-case no),
    and round-robin them so a first-N slice is balanced. The repo holds two
    CSVs with different schemas, so load dataset.csv explicitly."""
    from datasets import load_dataset
    ds = load_dataset(
        "csv", split="train", streaming=True,
        data_files=f"hf://datasets/{cfg['hub_id']}@{cfg['revision']}/dataset.csv")
    POOL = 700
    pools = {t: [] for t in LEGAL_TASKS}
    seen_q = set()
    for rec in ds:
        if all(len(p) >= POOL for p in pools.values()):
            break
        task = rec["task"]
        if task == "fake_case_existence" and rec.get("example_correct_answer") is None:
            rec = dict(rec, example_correct_answer="0")  # fake case: not real
        if task not in pools or len(pools[task]) >= POOL:
            continue
        q = str(rec["query"] or "").strip()
        ref = str(rec.get("example_correct_answer") or "").strip()
        if not q or not ref or q in seen_q:
            continue
        seen_q.add(q)
        pools[task].append(rec)
    interleaved = []
    for i in range(POOL):
        for t in LEGAL_TASKS:
            if i < len(pools[t]):
                interleaved.append(pools[t][i])
    for rec in interleaved:
        raw = str(rec["example_correct_answer"]).strip().lower()
        refs = _LEGAL_ANSWER_MAP.get(raw, [raw])
        yield PromptRecord(
            source_dataset="legal_hallucinations",
            source_revision=cfg["revision"], source_row_id=str(rec["id"]),
            upstream_group="reglab_legal", license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="legal", task_type="closed_qa",
            split_group=str(rec.get("citation") or rec["id"]),
            system=TERSE, prompt=str(rec["query"]).strip(),
            references=refs, answerable=True, grader_type="exact",
            metadata={"task": rec["task"], "year": rec.get("year"),
                      "court_level": rec.get("court_level")})


BFCL_SYSTEM = (
    "You are given function definitions in the context. Respond with ONLY a "
    'JSON object of the form {"name": <function name>, "arguments": {...}} '
    "calling the single most appropriate function with the correct arguments. "
    "No other text.")


def adapt_bfcl(cfg, target):
    ds = _load(cfg["hub_id"], cfg.get("config"), cfg["split"], cfg["revision"])
    n = 0
    for rec in ds:
        rid = rec["extra"]["id"]
        user = next((m["content"] for m in rec["messages"]
                     if m["role"] == "user"), None)
        call = next((m["tool_calls"] for m in rec["messages"]
                     if m["role"] == "assistant" and m.get("tool_calls")), None)
        if not user or not call or len(call) != 1:
            continue
        fn = call[0]["function"]
        ref = {"name": fn["name"], "arguments": json.loads(fn["arguments"])}
        tools = rec["tools"]
        if not isinstance(tools, str):
            tools = json.dumps(tools)
        yield PromptRecord(
            source_dataset="bfcl", source_revision=cfg["revision"],
            source_row_id=rid, upstream_group="bfcl",
            license=cfg["license"],
            commercial_train_allowed=cfg["commercial_train_allowed"],
            domain="tools", task_type="tool_call",
            split_group=rid, system=BFCL_SYSTEM,
            prompt=user, context=tools,
            references=[json.dumps(ref, sort_keys=True)],
            answerable=True, grader_type="tool")
        n += 1
        if n >= target:
            return


ADAPTERS = {
    "trivia_qa": adapt_trivia_qa, "popqa": adapt_popqa,
    "squad_v2": adapt_squad_v2, "halubench": adapt_halubench,
    "medhallu": adapt_medhallu, "esconv": adapt_esconv,
    "hotpotqa": adapt_hotpotqa, "drop": adapt_drop,
    "truthfulqa": adapt_truthfulqa, "nq_open": adapt_nq_open,
    "facts_grounding": adapt_facts_grounding,
    "legal_hallucinations": adapt_legal_hallucinations, "bfcl": adapt_bfcl,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot", help="pilot | 1 | 2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-manifest", default="",
                    help="skip example_ids already present in this manifest")
    args = ap.parse_args()

    sources = yaml.safe_load(open(os.path.join(HERE, "datasets.yaml"), encoding="utf-8"))
    target_key = {"pilot": "pilot_target", "1": "stage1_target",
                  "2": "stage2_target"}[args.stage]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen_ids, counts, total = set(), {}, 0
    if args.exclude_manifest:
        for l in open(args.exclude_manifest, encoding="utf-8"):
            if l.strip():
                seen_ids.add(json.loads(l)["example_id"])
        print(f"excluding {len(seen_ids)} example_ids from {args.exclude_manifest}")
    with open(args.out, "w", encoding="utf-8") as f:
        for name, cfg in sources.items():
            if args.stage == "pilot" and cfg.get("stage") != 0:
                continue
            desired = cfg.get(target_key, 0)
            if desired <= 0:
                continue
            k = 0
            # stream unbounded; main() caps on WRITTEN rows so excluded pilot
            # ids don't shrink the count below target
            for rec in ADAPTERS[name](cfg, 10 ** 9):
                if rec.example_id in seen_ids:
                    continue
                seen_ids.add(rec.example_id)
                # namespace grouping key by source so numeric ids from
                # different datasets never collide into one fold
                if ":" not in rec.split_group:
                    rec.split_group = f"{rec.source_dataset}:{rec.split_group}"
                f.write(rec.to_json() + "\n")
                k += 1
                total += 1
                if k >= desired:
                    break
            counts[name] = k
            print(f"  {name:>12}: {k}")
    print(f"wrote {total} prompts -> {args.out}")
    print("counts:", json.dumps(counts))


if __name__ == "__main__":
    main()
