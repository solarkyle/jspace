from __future__ import annotations

import argparse

from .core import AVAILABLE_LENSES, DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jspace")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snap", help="capture one workspace snapshot")
    snap.add_argument("prompt")
    snap.add_argument("--model", default=DEFAULT_MODEL)
    snap.add_argument("--quant", default="4bit")

    chat = sub.add_parser("chat", help="run a small workspace REPL")
    chat.add_argument("--model", default=DEFAULT_MODEL)
    chat.add_argument("--quant", default="4bit")

    sub.add_parser("info", help="list bundled public lens names")

    args = parser.parse_args(argv)
    if args.command == "info":
        return info()
    if args.command == "snap":
        return snap_once(args.prompt, model=args.model, quant=args.quant)
    if args.command == "chat":
        return chat_loop(model=args.model, quant=args.quant)
    return 0


def info() -> int:
    print("available public lenses:")
    for item in AVAILABLE_LENSES:
        print(f"  {item['slug']:<38} {item['model_id']}")
    return 0


def snap_once(prompt: str, *, model: str, quant: str | None) -> int:
    from .core import Workspace

    ws = Workspace(model, quant=_quant_arg(quant))
    snapshot = ws.snapshot(prompt)
    print(f"answer: {snapshot.answer}")
    snapshot.show()
    return 0


def chat_loop(*, model: str, quant: str | None) -> int:
    from .core import Workspace

    ws = Workspace(model, quant=_quant_arg(quant))
    print("jspace chat. Empty line, exit, or quit stops.")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            break
        if not prompt or prompt.lower() in {"exit", "quit"}:
            break
        snapshot = ws.snapshot(prompt)
        print(f"answer: {snapshot.answer}")
        print(f"noise: {snapshot.noise:.3f}")
        snapshot.show()
    return 0


def _quant_arg(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"none", "bf16", "bfloat16"}:
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
