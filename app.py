"""
Command-line entry point for the multi-agent system.

Usage:
    python app.py "Summarize this: ..."
    python app.py            # interactive REPL
"""

from __future__ import annotations

import sys

from graph import run


def main() -> None:
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        _handle(user_input)
        return

    print("Multi-Agent NLP System (type 'exit' to quit)")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        _handle(user_input)


def _handle(user_input: str) -> None:
    final_state = run(user_input)
    print(final_state["final_response"])


if __name__ == "__main__":
    main()
