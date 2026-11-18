"""Structured query builder: valid requests against the live database (checked against
independently computed values) and rejection of invalid or malicious requests."""
import unittest

from retail_analyst.tools.query_builder import QuerySpecError, auto_correct, build_query
from retail_analyst.tools.sql_tool import run_sql


def rows(spec):
    result = run_sql(build_query(spec))
    assert result.ok, result.error
    return result.rows


class ValidRequestTests(unittest.TestCase):
    def test_group_metric_order_limit(self):
        out = rows({"table": "analytics.category_performance", "columns": ["category_reporting"],
                    "metrics": ["sum(items_price_total)"], "order_by": "sum(items_price_total) desc", "limit": 2})
        self.assertEqual(out, [{"category_reporting": "health_beauty", "sum_items_price_total": 1258681.34},
                               {"category_reporting": "watches_gifts", "sum_items_price_total": 1205005.68}])

    def test_group_by_uses_columns_not_aliases(self):
        # sales_daily has a column named "month": grouping must not be confused by aliases.
        out = rows({"table": "analytics.sales_daily", "columns": ["year_month"],
                    "metrics": ["sum(items_price_total)"],
                    "filters": [["year", "=", "2018"], ["in_candidate_window", "=", "true"]],
                    "order_by": "year_month"})
        self.assertEqual(len(out), 8)
        self.assertEqual(out[0], {"year_month": "2018-01", "sum_items_price_total": 950030.36})

    def test_plain_columns_desc_nulls_last(self):
        out = rows({"table": "analytics.delivery_by_state", "columns": ["customer_state_std", "late_rate"],
                    "order_by": "late_rate desc", "limit": 2})
        self.assertEqual([r["customer_state_std"] for r in out], ["AL", "MA"])
        self.assertIn("NULLS LAST", build_query({"table": "analytics.delivery_by_state",
                                                 "columns": ["late_rate"], "order_by": "late_rate desc"}))

    def test_join_along_foreign_key(self):
        out = rows({"table": "warehouse.fact_reviews", "join": "warehouse.fact_orders",
                    "columns": ["is_late"], "metrics": ["avg(review_score)", "count(*)"],
                    "filters": [["is_late", "in", [True, False]]], "order_by": "is_late"})
        by_late = {r["is_late"]: r for r in out}
        self.assertAlmostEqual(by_late[True]["avg_review_score"], 2.271, places=3)
        self.assertAlmostEqual(by_late[False]["avg_review_score"], 4.290, places=3)

    def test_qualified_column_names_and_text_filter(self):
        out = rows({"table": "warehouse.fact_payments", "columns": ["fact_payments.payment_type"],
                    "metrics": ["sum(payment_value)"], "filters": [["payment_type", "!=", "not_defined"]],
                    "order_by": "sum(payment_value) desc"})
        self.assertEqual(out[0]["payment_type"], "credit_card")
        self.assertAlmostEqual(out[0]["sum_payment_value"], 12542084.19, places=2)


class AutoCorrectTests(unittest.TestCase):
    # Exact requests Qwen 2.5 1.5B produced in the structured run (they returned wrong answers).
    Q1_REQUEST = {"table": "analytics.product_performance", "columns": ["category_reporting", "items_price_total"],
                  "metrics": ["sum(items_price_total)"], "order_by": "sum(items_price_total) desc", "limit": 5}
    Q4_REQUEST = {"table": "warehouse.fact_payments", "columns": ["payment_type", "payment_value"],
                  "metrics": ["sum(payment_value)"], "order_by": "sum(payment_value) desc", "limit": 10}

    def test_summarised_column_removed_from_groups(self):
        fixed, notes = auto_correct(self.Q1_REQUEST)
        self.assertEqual(fixed["columns"], ["category_reporting"])
        self.assertEqual(len(notes), 1)
        self.assertEqual(self.Q1_REQUEST["columns"], ["category_reporting", "items_price_total"])  # not mutated

    def test_q1_request_now_matches_ground_truth(self):
        out = rows(auto_correct(self.Q1_REQUEST)[0])
        self.assertEqual([(r["category_reporting"], r["sum_items_price_total"]) for r in out], [
            ("health_beauty", 1258681.34), ("watches_gifts", 1205005.68), ("bed_bath_table", 1036988.68),
            ("sports_leisure", 988048.97), ("computers_accessories", 911954.32)])

    def test_q4_request_now_matches_ground_truth(self):
        out = rows(auto_correct(self.Q4_REQUEST)[0])
        totals = {r["payment_type"]: r["sum_payment_value"] for r in out}
        self.assertEqual(list(totals)[:4], ["credit_card", "boleto", "voucher", "debit_card"])
        self.assertAlmostEqual(totals["credit_card"], 12542084.19, places=2)
        self.assertAlmostEqual(100 * totals["credit_card"] / sum(totals.values()), 78.34, places=2)

    # Exact request Qwen 2.5 1.5B produced for "What customer states have the highest late delivery rate?"
    DELIVERY_REQUEST = {"table": "analytics.delivery_monthly", "columns": ["customer_state_std", "late_rate"],
                        "metrics": ["late_rate"], "order_by": "late_rate desc", "limit": 10}

    def test_delivery_request_now_matches_ground_truth(self):
        fixed, notes = auto_correct(self.DELIVERY_REQUEST)
        self.assertEqual(fixed["table"], "analytics.delivery_by_state")
        self.assertNotIn("metrics", fixed)
        self.assertEqual(len(notes), 2)  # duplicate metric dropped + table moved
        out = rows(fixed)
        self.assertEqual([(r["customer_state_std"], round(r["late_rate"], 4)) for r in out[:5]],
                         [("AL", 0.2141), ("MA", 0.1743), ("SE", 0.1522), ("PI", 0.1387), ("CE", 0.1376)])

    def test_table_is_not_switched_when_ambiguous(self):
        # customer_state_std exists in delivery_by_state AND fact_orders: no single correct table.
        spec = {"table": "analytics.delivery_monthly", "columns": ["customer_state_std"]}
        self.assertEqual(auto_correct(spec), (spec, []))

    def test_table_is_not_switched_for_joins_or_valid_requests(self):
        joined = {"table": "analytics.delivery_monthly", "join": "warehouse.fact_orders", "columns": ["late_rate"]}
        self.assertEqual(auto_correct(joined), (joined, []))
        valid = {"table": "analytics.delivery_by_state", "columns": ["customer_state_std", "late_rate"]}
        self.assertEqual(auto_correct(valid), (valid, []))

    def test_bare_metric_is_dropped_only_when_it_repeats_a_column(self):
        repeated = {"table": "analytics.delivery_by_state", "columns": ["late_rate"], "metrics": ["late_rate"]}
        self.assertNotIn("metrics", auto_correct(repeated)[0])
        alone = {"table": "analytics.delivery_by_state", "columns": ["customer_state_std"], "metrics": ["late_rate"]}
        self.assertEqual(auto_correct(alone), (alone, []))                # meaning unclear: left to fail
        with self.assertRaises(QuerySpecError):
            build_query(alone)

    def test_count_of_group_column_is_kept(self):
        spec = {"table": "warehouse.fact_payments", "columns": ["payment_type"], "metrics": ["count(payment_type)"]}
        self.assertEqual(auto_correct(spec), (spec, []))

    def test_overlap_is_rejected_without_auto_correct(self):
        with self.assertRaises(QuerySpecError):
            build_query(self.Q1_REQUEST)


class InvalidRequestTests(unittest.TestCase):
    BASE = {"table": "analytics.category_performance", "columns": ["category_reporting"]}
    INVALID = {
        "not a dict": "SELECT 1",
        "unknown key": {**BASE, "sql": "DROP TABLE x"},
        "unknown table": {"table": "warehouse.load_run", "columns": ["load_id"]},
        "unqualified table": {"table": "category_performance", "columns": ["category_reporting"]},
        "unknown column": {**BASE, "columns": ["revenue"]},
        "dq column hidden": {"table": "warehouse.fact_orders", "columns": ["dq_payment_items_mismatch"]},
        "nothing selected": {"table": "analytics.category_performance"},
        "bad metric syntax": {**BASE, "metrics": ["items_price_total"]},
        "bad aggregate": {**BASE, "metrics": ["median(items_price_total)"]},
        "sum on text": {**BASE, "metrics": ["sum(category_reporting)"]},
        "star outside count": {**BASE, "metrics": ["sum(*)"]},
        "bad operator": {**BASE, "filters": [["n_orders", "like", 5]]},
        "bad filter shape": {**BASE, "filters": [["n_orders", ">"]]},
        "number filter not numeric": {**BASE, "filters": [["n_orders", ">", "many"]]},
        "in without list": {**BASE, "filters": [["category_reporting", "in", "toys"]]},
        "join without foreign key": {**BASE, "join": "warehouse.dim_dates"},
        "order by ungrouped column": {**BASE, "metrics": ["sum(units)"], "order_by": "n_orders desc"},
        "limit too high": {**BASE, "limit": 10_000},
        "limit not int": {**BASE, "limit": "10"},
        "injection in table": {"table": "analytics.sales_daily; DROP TABLE x", "columns": ["year"]},
        "injection in column": {**BASE, "columns": ["category_reporting FROM x; DROP TABLE y --"]},
        "injection in metric": {**BASE, "metrics": ["sum(units); DROP TABLE x"]},
        "injection in order": {**BASE, "order_by": "category_reporting; DROP TABLE x"},
    }

    def test_invalid_requests_are_rejected_before_sql(self):
        for label, spec in self.INVALID.items():
            with self.subTest(label):
                with self.assertRaises(QuerySpecError):
                    build_query(spec)

    def test_unknown_column_error_names_the_right_tables(self):
        with self.assertRaises(QuerySpecError) as ctx:
            build_query({"table": "analytics.sales_daily", "columns": ["category_reporting"]})
        self.assertIn("analytics.category_performance", str(ctx.exception))  # where it really is
        self.assertIn("year_month", str(ctx.exception))                      # what is available

    def test_injection_in_filter_value_is_escaped(self):
        value = "x'; DROP TABLE warehouse.dim_dates; SELECT '1"
        sql = build_query({**self.BASE, "filters": [["category_reporting", "=", value]]})
        self.assertIn("'x''; DROP TABLE warehouse.dim_dates; SELECT ''1'", sql)  # one quoted literal
        result = run_sql(sql)
        self.assertTrue(result.ok, result.error)   # runs as a plain text comparison...
        self.assertEqual(result.row_count, 0)      # ...that matches nothing

    def test_comment_marker_in_value_is_rejected_by_sql_tool(self):
        # Defense in depth: the SQL validator conservatively rejects '--' even inside a literal.
        sql = build_query({**self.BASE, "filters": [["category_reporting", "=", "x'; DROP TABLE t; --"]]})
        result = run_sql(sql)
        self.assertFalse(result.ok)
        self.assertTrue(result.error.startswith("Rejected"))


if __name__ == "__main__":
    unittest.main()
