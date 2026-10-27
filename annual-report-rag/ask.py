"""Ask the reports from the terminal.

    python ask.py "What attrition rate did HCLTech report?"
    python ask.py            (interactive; empty line to quit)
"""
import sys

from qa import QAEngine, format_answer

if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    engine = QAEngine()
    print(f"Answer engine: {engine.mode}\n")
    if len(sys.argv) > 1:
        print(format_answer(engine.ask(" ".join(sys.argv[1:]))))
    else:
        while question := input("Question: ").strip():
            print(format_answer(engine.ask(question)), "\n")
