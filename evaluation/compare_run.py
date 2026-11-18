"""Run the five benchmark questions and record correctness inputs, retries, timing and memory.
Usage: python compare_run.py <label> <output_dir>   (run from the project root)"""
import json
import subprocess
import sys
import time
from pathlib import Path

from retail_analyst.config import get_settings
from retail_analyst.graph import ask, build_graph, format_trace

QUESTIONS = [
    "What are the top 5 product categories by sales revenue?",
    "How did monthly sales revenue grow during 2018?",
    "Which customer states have the highest late delivery rate?",
    "What share of payment value comes from each payment type?",
    "Is there a relationship between delivery delays and review scores?",
]
label, out_dir = sys.argv[1], Path(sys.argv[2])
s = get_settings()
models = {a: s.model_for(a) for a in ("supervisor", "data_analyst", "analytics")}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # tool not available
        return f"unavailable ({type(exc).__name__})"


graph = build_graph()
results, text = [], [f"CONFIG {label}: {models}\n"]
for i, q in enumerate(QUESTIONS, 1):
    start = time.perf_counter()
    state = ask(q, graph)
    seconds = time.perf_counter() - start
    trace = state["trace"]
    sql = [e for e in trace if e["action"] == "query"]
    rec = {
        "q": i, "question": q, "seconds": round(seconds, 1),
        "sql_executed": len(sql), "sql_ok": sum(1 for e in sql if "rows" in e),
        "sql_failed": sum(1 for e in sql if "rows" not in e),
        "duplicates_blocked": sum(1 for e in trace if e["action"] == "duplicate_request"),
        "routes": [f"{e['next']}" for e in trace if e["action"] == "route"],
        "analyses": [a["request"] for a in state.get("analyses", [])],
        "datasets": {n: {"request": d.get("request"), "sql": d["sql"], "rows": d["rows"][:15], "row_count": d["row_count"]}
                     for n, d in state.get("datasets", {}).items()},
        "checks": state.get("checks", []),
        "final_answer": state["final_answer"],
        "ollama_ps": sh(["ollama", "ps"]),
        "gpu_mem": sh(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"]),
    }
    results.append(rec)
    text.append(f"\n{'=' * 70}\nQ{i}: {q}  ({rec['seconds']}s, SQL ok/failed {rec['sql_ok']}/{rec['sql_failed']}, "
                f"duplicates blocked {rec['duplicates_blocked']})\n--- TRACE ---\n{format_trace(trace)}\n"
                f"--- ANSWER ---\n{state['final_answer']}\n--- MEMORY ---\n{rec['ollama_ps']}\nGPU: {rec['gpu_mem']}\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.json").write_text(json.dumps({"models": models, "results": results}, indent=1, default=str), encoding="utf-8")
    (out_dir / f"{label}.txt").write_text("".join(text), encoding="utf-8")
    print(f"Q{i} done in {rec['seconds']}s (saved)", flush=True)
