# Benchmark results

Every live run made during development is kept unchanged. Most runs contain:
- `results.json` (or `<label>.json`): one record per question with outcome, timing, agent requests,
  checks and the text the user saw
- `results.txt` (or `<label>.txt`): the same, readable, with the full agent trace and generated SQL

**Outcomes**
- **correct:** answered, and the answer matches the hand-written ground-truth SQL
- **wrong:** answered, but the answer doesn't match
- **correct_refusal:** a question that must be refused was refused
- **wrongly_answered:** a question that must be refused was answered
- **safe_failure:** an answerable question was refused

**Independence:** all question sets were seen during development; none is independent validation.
See "Evaluation" in the main README.

## Final runs (frozen implementation)

| Run | Question set | Correct / wrong / correct refusals / safe failures | Time |
|---|---|---|---|
| [`freeze_17q/`](freeze_17q/) | `benchmark.json` (17) | 13 / 0 / 4 / 0 | 12.5 s |
| [`freeze_32q/`](freeze_32q/) | `independent_questions.json` (32) | 13 / 0 / 14 / 5 | 21.3 s |
| [`yearfix_49q/`](yearfix_49q/) | `fresh_questions.json` (49) | 1 / 0 / 45 / 3 | 16.2 s |

## History (in chronological order)

The early runs use the 5-question set (`../ground_truth.sql`) and were produced by the legacy runner
`../compare_run.py`. All later runs use `../run_benchmark.py`.

| Run | What changed before it |
|---|---|
| `original_1.5b.txt` | First three-agent version (free-form SQL), 5 questions |
| `improved_1.5b.*` | Table selection, SQL retries, Supervisor task descriptions, report checks |
| `structured_1.5b.*` | Structured JSON query requests (`retail_analyst/tools/query_builder.py`) |
| `structured_fix_1.5b.*` | Two automatic request corrections. 2/5 correct, 0 wrong |
| `structured_fix_3b_data_analyst.*` | Same, with `qwen2.5:3b` as the Data Analyst. 0/5 correct, about 4× slower, so 1.5B was kept |
| `catalog_1.5b.*` | Metric catalog and question constraints. 17-question benchmark; model-written summaries |
| `catalog_1.5b_generated_summaries.json` | Offline rescore of the run above with code-generated summaries |
| `catalog_generated_1.5b.*` | Code-generated summaries, live |
| `catalog_p3p4_1.5b.*` | Metric-fit check and rule-based routing |
| `independent_1.5b.*` | First run of the 32-question set (5 wrong answers: ignored qualifiers) |
| `final_regression_17q/`, `final_regression_32q/`, `final_regression_checks/` | Detection of unsupported qualifiers and filters |
| `synonyms_17q/`, `synonyms_32q/` | Wider synonym lists |
| `fresh_49q/` | First run of the fresh 49-question set (2 wrong answers: conditions ignored) |
| `closedworld_17q/`, `closedworld_32q/`, `closedworld_49q/` | Closed-world question check and date ranges |
| `yearfix_49q/` | Year check reads date ranges (final run of the fresh set) |
| `freeze_17q/`, `freeze_32q/` | No code change: final regression before the freeze |
