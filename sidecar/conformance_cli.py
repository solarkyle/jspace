from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidecar.conformance import compare_captures


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two J-Space layer captures")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output")
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    result = compare_captures(reference, candidate)
    encoded = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(
        f"{'PASS' if result['passed'] else 'FAIL'}: "
        f"{result['aligned_layers']} aligned layers, "
        f"final token match={result['final_token_match']}"
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
