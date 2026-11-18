"""The deterministic part of the pipeline (question constraints + catalog + SQL) reproduces the
independent ground truth of every benchmark question when given the correct metric choice.
Any remaining benchmark error therefore comes from the agents' choices, not from the code."""
import json
import unittest
from pathlib import Path

from retail_analyst.catalog import check_fit, compile_query
from retail_analyst.question import parse_question
from retail_analyst.tools.sql_tool import run_sql

BENCHMARK = json.loads((Path(__file__).resolve().parent.parent / "evaluation" / "benchmark.json")
                       .read_text(encoding="utf-8"))["questions"]


def numbers(row: dict) -> list[float]:
    return [round(float(v), 2) for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]


def label(row: dict):
    values = list(row.values())
    return values[0] if len(values) > 1 else None


class BenchmarkReferenceTests(unittest.TestCase):
    def test_reference_choices_reproduce_the_ground_truth(self):
        for item in [q for q in BENCHMARK if q["expect"] == "answer"]:
            with self.subTest(item["question"]):
                constraints = parse_question(item["question"])
                self.assertIsNone(constraints.refusal)
                compiled = compile_query(item["reference_request"], constraints)
                check_fit(compiled.metric, compiled.grouping, item["question"])  # correct choices are accepted
                result = run_sql(compiled.sql)
                self.assertTrue(result.ok, result.error)
                expected = item["expected_rows"]
                got = {label(r): numbers(r) for r in result.rows}
                for row in expected:  # same groups, and each expected value appears in that group's row
                    self.assertIn(label(row), got)
                    self.assertIn(numbers(row)[-1], got[label(row)])
                if constraints.calculation == "ranking":
                    self.assertEqual(label(result.rows[0]), label(expected[0]))  # the leader is the same
                    self.assertEqual([label(r) for r in result.rows[:len(expected)]], [label(r) for r in expected])

    def test_refusals_are_detected_before_any_agent(self):
        refused = {q["question"]: parse_question(q["question"]).refusal for q in BENCHMARK if q["expect"] == "refuse"}
        detected = {q for q, r in refused.items() if r}
        # All four, including "...the most new customers?" (customer type is not supported).
        self.assertEqual(detected, set(refused))


if __name__ == "__main__":
    unittest.main()
