"""Judge blinded requests with Codex (GPT-5.5) instead of Anthropic models.

Reads the emit output of campaign.grade_claude (example_id, schema,
prompt_hash, judge_prompt), sends each FROZEN judge prompt verbatim to
`codex exec` (prompt over stdin: contexts can exceed argv limits), parses the
strict-JSON verdict, and appends rows compatible with grade_claude ingest.

Resumable: example_ids already present in --out are skipped, so rerunning
after an interruption only judges the remainder.

    python -m campaign.judge_codex --requests out/campaign/stage2_judge_requests.jsonl \
        --out out/campaign/verdicts_codex_stage2.jsonl --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

CODEX = os.path.expanduser("~/AppData/Roaming/npm/codex.cmd")  # npm shim; bare "codex" fails in subprocess on Windows
TIMEOUT = 240  # seconds per case; long FACTS contexts need headroom


def _extract_json(text: str):
    """Last balanced {...} block that parses as JSON (codex may add prose)."""
    end = len(text)
    while True:
        close = text.rfind("}", 0, end)
        if close == -1:
            return None
        depth = 0
        for i in range(close, -1, -1):
            c = text[i]
            if c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:close + 1])
                    except json.JSONDecodeError:
                        break
        end = close


def judge_one(req):
    """Run one frozen judge prompt through codex exec. Returns verdict row or
    None on failure (caller retries once; persistent failures are logged)."""
    out = subprocess.run(
        [CODEX, "exec", "--sandbox", "read-only", "--skip-git-repo-check", "-"],
        input=req["judge_prompt"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=TIMEOUT)
    v = _extract_json(out.stdout or "")
    if not isinstance(v, dict) or "verdict" not in v:
        return None
    v["example_id"] = req["example_id"]
    v["_schema"] = req["schema"]
    v["_judge_model"] = "codex-exec"
    v["_prompt_hash"] = req["prompt_hash"]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="judge at most N (pilot)")
    args = ap.parse_args()

    reqs = [json.loads(l) for l in open(args.requests, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["example_id"])
    todo = [r for r in reqs if r["example_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(reqs)} requests, {len(done)} already judged, {len(todo)} to go")

    lock = threading.Lock()
    n_ok, n_fail = 0, 0
    with open(args.out, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge_one, r): r for r in todo}
            for fut in as_completed(futs):
                req = futs[fut]
                try:
                    v = fut.result()
                except Exception as e:
                    print(f"  ERROR {req['example_id']}: {str(e)[:100]}")
                    n_fail += 1
                    continue
                if v is None:  # one retry for parse/transient failures
                    try:
                        v = judge_one(req)
                    except Exception:
                        v = None
                with lock:
                    if v is None:
                        n_fail += 1
                        print(f"  unparseable: {req['example_id']}")
                    else:
                        f.write(json.dumps(v, ensure_ascii=False) + "\n")
                        f.flush()
                        n_ok += 1
                        if n_ok % 25 == 0:
                            print(f"  {n_ok}/{len(todo)} judged")
    print(f"done: {n_ok} verdicts, {n_fail} failures -> {args.out}")


if __name__ == "__main__":
    main()
