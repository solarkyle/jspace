from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROMPTS: list[dict[str, Any]] = [
    {"q": "Answer with just the answer: What is the capital of France?", "aliases": ["Paris"]},
    {"q": "Answer with just the answer: Who wrote Hamlet?", "aliases": ["Shakespeare", "William Shakespeare"]},
    {"q": "Answer with just the answer: What is the chemical symbol for gold?", "aliases": ["Au", "gold"]},
    {"q": "Answer with just the answer: What is the largest planet in the Solar System?", "aliases": ["Jupiter"]},
    {"q": "Answer with just the answer: Who painted the Mona Lisa?", "aliases": ["Leonardo", "Leonardo da Vinci"]},
    {"q": "Answer with just the answer: Who was the first president of the United States?", "aliases": ["George Washington", "Washington"]},
    {"q": "Answer with just the answer: At sea level, water boils at how many degrees Celsius?", "aliases": ["100", "one hundred"]},
    {"q": "Answer with just the answer: What is the currency of Japan?", "aliases": ["yen", "Japanese yen"]},
    {"q": "Answer with just the answer: Who wrote the novel 1984?", "aliases": ["George Orwell", "Orwell"]},
    {"q": "Answer with just the answer: What is the smallest prime number?", "aliases": ["2", "two"]},
    {"q": "Answer with just the answer: What is the capital of the Faroe Islands?", "aliases": ["Torshavn"]},
    {"q": "Answer with just the answer: Who composed the opera Nixon in China?", "aliases": ["John Adams"]},
    {"q": "Answer with just the answer: What element has atomic number 43?", "aliases": ["Technetium", "Tc"]},
    {"q": "Answer with just the answer: Which island is home to the city of Nuuk?", "aliases": ["Greenland"]},
    {"q": "Answer with just the answer: What was Caravaggio's birth name?", "aliases": ["Michelangelo Merisi", "Michelangelo Merisi da Caravaggio"]},
    {"q": "Answer with just the answer: Who won the 1957 Nobel Prize in Literature?", "aliases": ["Albert Camus", "Camus"]},
    {"q": "Answer with just the answer: Which river runs through the city of Inverness in Scotland?", "aliases": ["Ness", "River Ness"]},
    {"q": "Answer with just the answer: What is the IATA airport code for Keflavik International Airport?", "aliases": ["KEF"]},
    {"q": "Write a short Python function that returns the factorial of a non-negative integer."},
    {"q": "In two sentences, explain why mutable default arguments in Python can be surprising."},
    {"q": "How can I stop pasta from sticking together without overcomplicating dinner?"},
    {"q": "What is a practical way to reset a home Wi-Fi router safely?"},
    {"q": "Give a concise checklist for packing for a one-night work trip."},
    {"q": "How should I store leftover rice safely?"},
    {"q": "What are two signs that a bike tire needs more air?"},
]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return "".join(c for c in text.lower() if c.isalnum() or c == " ").strip()


def accent_fold(text: str) -> str:
    text = unicodedata.normalize("NFD", unicodedata.normalize("NFC", text))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def is_right(answer: str, aliases: list[str]) -> bool:
    n_answer = norm(answer)
    if any(norm(alias) and norm(alias) in n_answer for alias in aliases):
        return True
    folded_answer = norm(accent_fold(answer))
    return any(
        norm(accent_fold(alias)) and norm(accent_fold(alias)) in folded_answer
        for alias in aliases
    )


def trunc(text: str, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/v1/chat/completions")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    local_right = routed_right = trivia_count = 0
    escalated = 0
    snapshot_ms: list[float] = []

    print(f"{'#':>2}  {'question':38}  {'answer':42}  {'noise':>5}  {'route':>9}  {'latency':>8}  {'eval':>7}")
    print("-" * 122)
    for i, item in enumerate(PROMPTS, start=1):
        payload = {
            "model": "jspace-sidecar",
            "messages": [{"role": "user", "content": item["q"]}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "jspace_return_local": True,
        }
        start = time.perf_counter()
        try:
            data = post(args.url, payload, args.timeout)
            latency = time.perf_counter() - start
            answer = data["choices"][0]["message"]["content"] or ""
            jspace = data.get("jspace", {})
            noise = float(jspace.get("noise", 0.0))
            was_escalated = bool(jspace.get("escalated", False))
            local_answer = str(jspace.get("local_answer", answer))
            snapshot_ms.append(float(jspace.get("snapshot_ms", 0.0)))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            latency = time.perf_counter() - start
            print(f"{i:>2}  {trunc(item['q'], 38):38}  {trunc(str(exc), 42):42}  {'err':>5}  {'ERROR':>9}  {latency:7.1f}s  {'':>7}")
            continue

        route = "ESCALATED" if was_escalated else "KEPT"
        if was_escalated:
            escalated += 1
        verdict = ""
        aliases = item.get("aliases")
        if aliases:
            trivia_count += 1
            local_ok = is_right(local_answer, aliases)
            routed_ok = is_right(answer, aliases)
            local_right += int(local_ok)
            routed_right += int(routed_ok)
            verdict = "right" if routed_ok else "wrong"

        print(
            f"{i:>2}  {trunc(item['q'], 38):38}  {trunc(answer, 42):42}  "
            f"{noise:5.2f}  {route:>9}  {latency:7.1f}s  {verdict:>7}"
        )

    denom = max(1, trivia_count)
    mean_snapshot = sum(snapshot_ms) / len(snapshot_ms) if snapshot_ms else 0.0
    print("\nSummary")
    print(f"local-only accuracy: {local_right}/{trivia_count} = {local_right / denom:.3f}")
    print(f"routed accuracy:     {routed_right}/{trivia_count} = {routed_right / denom:.3f}")
    print(f"escalation rate:    {escalated}/{len(PROMPTS)} = {escalated / len(PROMPTS):.3f}")
    print(f"mean snapshot ms:   {mean_snapshot:.1f}")


if __name__ == "__main__":
    main()
