"""Run a question set and report refusal behaviour, verified citations and latency.
Use it at the end of every iteration and paste the summary into README (failed attempts / results).

    python evaluate.py                    # eval_questions.json
    python evaluate.py my_questions.json
"""
import json
import statistics
import sys
from pathlib import Path

from qa import QAEngine, format_answer


def main(path: str = "eval_questions.json") -> None:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    engine = QAEngine()
    print(f"Answer engine: {engine.mode}\n")
    correct, times = 0, []
    for item in items:
        ans = engine.ask(item["question"])
        got = "answer" if ans.found else "not_found"
        expected = item.get("expect", got)
        correct += got == expected
        times.append(ans.seconds)
        print(f"{'PASS' if got == expected else 'FAIL'}  expected {expected}, got {got}")
        print("  " + format_answer(ans).replace("\n", "\n  ") + "\n")
    print(f"{correct}/{len(items)} behaved as expected; median {statistics.median(times):.1f}s, slowest {max(times):.1f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main(*sys.argv[1:2])
