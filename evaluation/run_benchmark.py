"""Run evaluation/benchmark.json through the real workflow and score what the user would see.

Usage (from the project root):  python evaluation/run_benchmark.py <label> [questions file]
  questions file: benchmark.json (default, 17 questions) or independent_questions.json (32 questions)
Writes evaluation/results/<label>.json and .txt after every question.

Scoring of the user-facing outcome:
  correct          an answer was shown and its data matches the ground truth
  wrong            an answer was shown but its data does not match (worst case)
  safe_failure     no answer was shown although one was expected
  correct_refusal  a question that must be refused was refused
  wrongly_answered a question that must be refused was answered
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retail_analyst.config import get_settings  # noqa: E402
from retail_analyst.graph import ask, build_graph, format_trace  # noqa: E402
from retail_analyst.question import parse_question  # noqa: E402
from retail_analyst.ui_support import answer_status, result_rows, user_answer  # noqa: E402


def numbers(row: dict) -> list[float]:
    return [round(float(v), 2) for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]


def label(row: dict):
    values = list(row.values())
    return values[0] if len(values) > 1 else None


def data_matches(expected: list[dict], shown: list[dict], question: str) -> tuple[bool, str]:
    got = {label(r): numbers(r) for r in shown}
    for row in expected:
        if label(row) not in got:
            return False, f"missing group {label(row)!r}"
        if numbers(row)[-1] not in got[label(row)]:
            return False, f"{label(row)!r}: expected {numbers(row)[-1]}, shown {got[label(row)]}"
    if parse_question(question).calculation == "ranking":
        order = [label(r) for r in shown[:len(expected)]]
        if order != [label(r) for r in expected]:
            return False, f"ranking order {order} != {[label(r) for r in expected]}"
    return True, "matches ground truth"


def score(item: dict, state: dict) -> tuple[str, str]:
    reliable, _ = user_answer(state)
    if item["expect"] == "refuse":
        return ("correct_refusal", "refused") if not reliable else ("wrongly_answered", "answered a must-refuse question")
    if not reliable:
        return "safe_failure", answer_status(state)
    ok, why = data_matches(item["expected_rows"], result_rows(state), item["question"])
    return ("correct" if ok else "wrong"), why


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def main(label_: str, questions_file: str = "benchmark.json") -> None:
    items = json.loads((ROOT / "evaluation" / questions_file).read_text(encoding="utf-8"))["questions"]
    out_json = ROOT / "evaluation" / "results" / f"{label_}.json"
    out_txt = out_json.with_suffix(".txt")
    s = get_settings()
    models = {a: s.model_for(a) for a in ("supervisor", "data_analyst", "analytics")}
    graph = build_graph()
    results, text = [], [f"BENCHMARK {label_}: {models}\n"]
    for i, item in enumerate(items, 1):
        start = time.perf_counter()
        state = ask(item["question"], graph)
        seconds = round(time.perf_counter() - start, 1)
        outcome, detail = score(item, state)
        _, shown = user_answer(state)
        requests = [e.get("spec") for e in state["trace"] if e["action"] in ("query", "out_of_scope")]
        results.append({"n": i, "question": item["question"], "expect": item["expect"],
                        "kind": item.get("kind", ""), "outcome": outcome,
                        "detail": detail, "seconds": seconds, "status": answer_status(state),
                        "reference_request": item.get("reference_request"), "agent_requests": requests,
                        "issues": state.get("issues", []), "user_sees": shown})
        text.append(f"\n{'=' * 80}\nQ{i}: {item['question']}\nOUTCOME: {outcome} ({detail}) | {seconds}s | "
                    f"reference: {json.dumps(item.get('reference_request'))}\n--- TRACE ---\n"
                    f"{format_trace(state['trace'])}\n--- USER SEES ---\n{shown}\n")
        summary = {k: sum(r["outcome"] == k for r in results)
                   for k in ("correct", "wrong", "safe_failure", "correct_refusal", "wrongly_answered")}
        by_kind = {}
        for r in results:
            by_kind.setdefault(r["kind"] or "-", {}).setdefault(r["outcome"], 0)
            by_kind[r["kind"] or "-"][r["outcome"]] += 1
        payload = {"models": models, "summary": summary, "by_kind": by_kind, "ollama_ps": sh(["ollama", "ps"]),
                   "gpu": sh(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"]),
                   "results": results}
        out_json.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        out_txt.write_text("".join(text), encoding="utf-8")
        print(f"Q{i:2} {outcome:16} {seconds:6.1f}s  {item['question']}", flush=True)
    print("SUMMARY", summary)
    print("BY KIND", by_kind)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "catalog_1.5b",
         sys.argv[2] if len(sys.argv) > 2 else "benchmark.json")
