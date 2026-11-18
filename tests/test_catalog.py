"""Metric catalog: every definition runs, reproduces the database's documented views and the
independently computed ground truth, and invalid requests are rejected before any SQL."""
import unittest

import json
from pathlib import Path

from retail_analyst.catalog import METRICS, UnsupportedRequest, catalog_prompt, compile_query, unexplained_words
from retail_analyst.question import Constraints, parse_question
from retail_analyst.tools.sql_tool import run_sql


def rows(request, question="", constraints=None):
    compiled = compile_query(request, constraints or parse_question(question))
    result = run_sql(compiled.sql)
    assert result.ok, result.error
    return result.rows


def view(sql, key, value):
    return {r[key]: r[value] for r in run_sql(sql).rows}


class DefinitionTests(unittest.TestCase):
    def test_every_combination_runs(self):
        for metric in METRICS.values():
            for grouping in (None,) + metric.groupings:
                with self.subTest(metric=metric.key, grouping=grouping):
                    self.assertTrue(rows({"metric": metric.key, "group_by": grouping}, constraints=Constraints()))

    def test_definitions_reproduce_documented_views(self):
        breakdown = Constraints()
        cases = [
            ("late_delivery_rate", "customer_state", "late_rate",
             view("SELECT customer_state_std, late_rate FROM analytics.delivery_by_state", "customer_state_std", "late_rate")),
            ("late_delivery_rate", "month", "late_rate",
             view("SELECT purchase_year_month, late_rate FROM analytics.delivery_monthly WHERE n_delivery_eligible > 0",
                  "purchase_year_month", "late_rate")),
            ("average_delivery_days", "customer_state", "avg_delivery_days",
             view("SELECT customer_state_std, avg_purchase_to_delivery_days FROM analytics.delivery_by_state",
                  "customer_state_std", "avg_purchase_to_delivery_days")),
            ("revenue", "category", "revenue",
             view("SELECT category_reporting, items_price_total FROM analytics.category_performance",
                  "category_reporting", "items_price_total")),
        ]
        for metric, grouping, column, expected in cases:
            with self.subTest(metric=metric, grouping=grouping):
                got = {next(iter(r.values())): r[column]
                       for r in rows({"metric": metric, "group_by": grouping}, constraints=breakdown)}
                common = set(got) & set(expected)
                self.assertGreater(len(common), 10)
                self.assertLess(max(abs(got[k] - expected[k]) for k in common), 0.006)

    def test_totals_reconcile_with_sources(self):
        total = Constraints()
        self.assertAlmostEqual(rows({"metric": "revenue"}, constraints=total)[0]["revenue"], 13591643.70, places=2)
        self.assertAlmostEqual(rows({"metric": "payment_value"}, constraints=total)[0]["payment_value"], 16008872.12,
                               places=2)
        self.assertEqual(rows({"metric": "orders"}, constraints=total)[0]["orders"], 99441)


class GroundTruthTests(unittest.TestCase):
    """The five benchmark questions (evaluation/ground_truth.sql) with the right metric choice."""

    def test_q1_top_categories(self):
        out = rows({"metric": "revenue", "group_by": "category"}, "What are the top 5 product categories by sales revenue?")
        self.assertEqual([(r["category_reporting"], r["revenue"]) for r in out], [
            ("health_beauty", 1258681.34), ("watches_gifts", 1205005.68), ("bed_bath_table", 1036988.68),
            ("sports_leisure", 988048.97), ("computers_accessories", 911954.32)])

    def test_q2_monthly_growth_2018(self):
        out = rows({"metric": "revenue", "group_by": "month"}, "How did monthly sales revenue grow during 2018?")
        self.assertEqual([(r["year_month"], r["revenue"]) for r in out], [
            ("2018-01", 950030.36), ("2018-02", 844178.71), ("2018-03", 983213.44), ("2018-04", 996647.75),
            ("2018-05", 996517.68), ("2018-06", 865124.31), ("2018-07", 895507.22), ("2018-08", 854686.33)])
        self.assertIsNone(out[0]["growth_pct"])
        self.assertEqual(out[1]["growth_pct"], -11.14)

    def test_q3_late_rate_by_state(self):
        out = rows({"metric": "late_delivery_rate", "group_by": "customer_state"},
                   "Which customer states have the highest late delivery rate?")
        self.assertEqual([(r["customer_state_std"], r["late_rate"]) for r in out[:5]],
                         [("AL", 0.2141), ("MA", 0.1743), ("SE", 0.1522), ("PI", 0.1387), ("CE", 0.1376)])

    def test_q4_payment_share(self):
        out = rows({"metric": "payment_value", "group_by": "payment_type"},
                   "What share of payment value comes from each payment type?")
        self.assertEqual([(r["payment_type"], r["share_pct"]) for r in out[:4]],
                         [("credit_card", 78.34), ("boleto", 17.92), ("voucher", 2.37), ("debit_card", 1.36)])
        self.assertAlmostEqual(sum(r["share_pct"] for r in out), 100.0, delta=0.05)

    def test_q5_review_score_by_delivery(self):
        out = rows({"metric": "average_review_score", "group_by": "delivery_status"},
                   "Is there a relationship between delivery delays and review scores?")
        scores = {r["delivery_status"]: r["avg_review_score"] for r in out}
        self.assertEqual(scores, {"late": 2.271, "on_time": 4.29})


class ConstraintTests(unittest.TestCase):
    def test_year_filter_and_direction_come_from_the_question(self):
        out = rows({"metric": "revenue", "group_by": "category"}, "Which 3 categories had the lowest revenue in 2017?")
        self.assertEqual(len(out), 3)
        self.assertEqual([r["revenue"] for r in out], sorted(r["revenue"] for r in out))  # ascending
        total_2017 = rows({"metric": "revenue"}, "What was the total revenue in 2017?")[0]["revenue"]
        expected = run_sql("SELECT SUM(items_price_total) AS v FROM analytics.sales_daily WHERE year = 2017").rows[0]["v"]
        self.assertAlmostEqual(total_2017, expected, places=2)

    def test_monthly_series_stay_in_the_reliable_window(self):
        months = [r["year_month"] for r in rows({"metric": "orders", "group_by": "month"}, constraints=Constraints())]
        self.assertEqual((months[0], months[-1]), ("2017-01", "2018-08"))

    def test_unknown_keys_are_ignored_and_noted(self):
        compiled = compile_query({"metric": "revenue", "group_by": "category", "sql": "DROP TABLE x"}, Constraints())
        self.assertNotIn("DROP", compiled.sql)
        self.assertTrue(compiled.notes)


class ClosedWorldTests(unittest.TestCase):
    """Every word of a question must be accounted for; otherwise a condition could be ignored."""

    def test_fresh_set_wrong_answers_are_now_refused(self):
        # Both were answered wrongly in the fresh question set (conditions silently ignored).
        self.assertIn("estimated", unexplained_words(
            "What is the average delivery duration from the order placement to the estimated delivery date?"))
        self.assertIn("used", unexplained_words(
            "What is the total revenue generated from orders with more than one payment method used?"))

    def test_conditions_and_unknown_measures_are_flagged(self):
        cases = {"How many orders were placed before 2018?": "before",
                 "Which states have revenue higher than 1 million?": "than",
                 "How many unique products have been sold?": "unique",
                 "What is the average revenue per category?": "average",   # not an average metric
                 "Which marketing channel brought the most customers?": "marketing"}
        for question, word in cases.items():
            with self.subTest(question):
                self.assertIn(word, unexplained_words(question))

    def test_answerable_benchmark_questions_pass(self):
        for item in json.loads((Path(__file__).resolve().parent.parent / "evaluation" / "benchmark.json")
                               .read_text(encoding="utf-8"))["questions"]:
            if item["expect"] == "answer":
                with self.subTest(item["question"]):
                    self.assertEqual(unexplained_words(item["question"]), [])

    def test_average_is_accepted_before_an_average_metric(self):
        for question in ("What's the average customer rating?", "What is the average order value by customer state?",
                         "Which customer states have the longest average delivery time?"):
            with self.subTest(question):
                self.assertEqual(unexplained_words(question), [])

    def test_whole_dataset_range_is_accounted_for(self):
        self.assertEqual(unexplained_words("Which Brazilian state(s) saw the highest number of orders during the "
                                           "period September 2016 to October 2018?"), [])


class InvalidRequestTests(unittest.TestCase):
    INVALID = {
        "not a dict": ("revenue", Constraints()),
        "unknown metric": ({"metric": "profit"}, Constraints()),
        "injection in metric": ({"metric": "revenue; DROP TABLE x"}, Constraints()),
        "injection in grouping": ({"metric": "revenue", "group_by": "category; DROP TABLE x"}, Constraints()),
        "payments by category": ({"metric": "payment_value", "group_by": "category"}, Constraints()),
        "orders by category": ({"metric": "orders", "group_by": "category"}, Constraints()),
        "growth by category": ({"metric": "revenue", "group_by": "category"}, Constraints(calculation="growth")),
        "ranking without grouping": ({"metric": "revenue"}, Constraints(calculation="ranking")),
        "share without grouping": ({"metric": "revenue"}, Constraints(calculation="share")),
    }

    def test_invalid_requests_are_rejected(self):
        for label, (request, constraints) in self.INVALID.items():
            with self.subTest(label):
                with self.assertRaises(UnsupportedRequest):
                    compile_query(request, constraints)

    def test_prompt_lists_every_metric_with_its_groupings(self):
        prompt = catalog_prompt()
        for metric in METRICS.values():
            self.assertIn(f"- {metric.key}:", prompt)
            self.assertIn(", ".join(metric.groupings), prompt)


if __name__ == "__main__":
    unittest.main()
