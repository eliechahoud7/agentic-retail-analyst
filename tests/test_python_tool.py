"""run_analysis: every allowlisted operation, invalid inputs, and one live end-to-end check."""
import unittest

import pandas as pd

from retail_analyst.tools.python_tool import OPERATIONS, dataframe_from_rows, run_analysis
from retail_analyst.tools.sql_tool import run_sql

SALES = pd.DataFrame({
    "month": ["2018-01", "2018-01", "2018-02", "2018-02", "2018-03"],
    "state": ["SP", "RJ", "SP", "RJ", "SP"],
    "revenue": [100.0, 50.0, 150.0, 50.0, 300.0],
    "orders": [10, 5, 12, 5, 20],
    "label": ["a", "b", "c", "d", "e"],
})
DATA = {"sales": SALES}


def run(**request):
    return run_analysis(request, DATA)


class OperationTests(unittest.TestCase):
    def test_all_operations_covered(self):
        self.assertEqual(set(OPERATIONS),
                         {"group_sum", "top_n", "mean_median", "share_of_total", "growth", "correlation"})

    def test_group_sum(self):
        r = run(operation="group_sum", dataset="sales", by="state", value="revenue")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.rows, [{"state": "SP", "revenue": 550.0}, {"state": "RJ", "revenue": 100.0}])

    def test_group_sum_multiple_columns(self):
        r = run(operation="group_sum", dataset="sales", by=["month", "state"], value="orders")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(len(r.rows), 5)

    def test_top_n(self):
        r = run(operation="top_n", dataset="sales", value="revenue", n=2)
        self.assertTrue(r.ok, r.error)
        self.assertEqual([row["revenue"] for row in r.rows], [300.0, 150.0])
        bottom = run(operation="top_n", dataset="sales", value="revenue", n=1, ascending=True)
        self.assertEqual(bottom.rows[0]["revenue"], 50.0)

    def test_mean_median(self):
        r = run(operation="mean_median", dataset="sales", value="revenue")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.rows[0]["mean"], 130.0)
        self.assertEqual(r.rows[0]["median"], 100.0)
        by_state = run(operation="mean_median", dataset="sales", value="revenue", by="state")
        self.assertEqual(by_state.rows[0]["state"], "SP")
        self.assertAlmostEqual(by_state.rows[0]["mean"], 183.3333)

    def test_share_of_total(self):
        r = run(operation="share_of_total", dataset="sales", by="state", value="revenue")
        self.assertTrue(r.ok, r.error)
        shares = {row["state"]: row["share_pct"] for row in r.rows}
        self.assertAlmostEqual(shares["SP"], 84.6154)
        self.assertAlmostEqual(sum(shares.values()), 100.0, places=3)

    def test_growth(self):
        r = run(operation="growth", dataset="sales", period="month", value="revenue")
        self.assertTrue(r.ok, r.error)
        self.assertEqual([row["month"] for row in r.rows], ["2018-01", "2018-02", "2018-03"])
        self.assertIsNone(r.rows[0]["growth_pct"])           # no previous period
        self.assertAlmostEqual(r.rows[1]["growth_pct"], 33.3333)
        self.assertAlmostEqual(r.rows[2]["growth_pct"], 50.0)

    def test_correlation(self):
        r = run(operation="correlation", dataset="sales", x="orders", y="revenue")
        self.assertTrue(r.ok, r.error)
        self.assertGreater(r.rows[0]["r"], 0.9)
        self.assertEqual(r.rows[0]["n"], 5)
        spearman = run(operation="correlation", dataset="sales", x="orders", y="revenue", method="spearman")
        self.assertTrue(spearman.ok, spearman.error)
        # Same rank order in both columns (ties included) -> rho = 1.0
        self.assertAlmostEqual(spearman.rows[0]["r"], 1.0)


class InvalidInputTests(unittest.TestCase):
    INVALID = {
        "unknown operation": dict(operation="eval", dataset="sales"),
        "code as operation": dict(operation="__import__('os').system('dir')", dataset="sales"),
        "missing operation": dict(dataset="sales"),
        "unknown dataset": dict(operation="group_sum", dataset="nope", by="state", value="revenue"),
        "unknown column": dict(operation="group_sum", dataset="sales", by="city", value="revenue"),
        "non-numeric value": dict(operation="group_sum", dataset="sales", by="state", value="label"),
        "missing parameter": dict(operation="group_sum", dataset="sales", by="state"),
        "unknown parameter": dict(operation="top_n", dataset="sales", value="revenue", code="print(1)"),
        "n too large": dict(operation="top_n", dataset="sales", value="revenue", n=10_000),
        "n not int": dict(operation="top_n", dataset="sales", value="revenue", n="5"),
        "bad ascending": dict(operation="top_n", dataset="sales", value="revenue", ascending="yes"),
        "too many group columns": dict(operation="group_sum", dataset="sales",
                                       by=["month", "state", "label", "orders"], value="revenue"),
        "bad correlation method": dict(operation="correlation", dataset="sales", x="orders",
                                       y="revenue", method="kendall"),
        "column as non-string": dict(operation="growth", dataset="sales", period=["month"], value="revenue"),
    }

    def test_calculated_percentages_are_not_treated_as_raw_values(self):
        computed = {"g": SALES.assign(growth_pct=[None, 10.0, 5.0, -2.0, 3.0], share_pct=[20.0] * 5,
                                      change=[None, 1.0, 2.0, 3.0, 4.0])}
        for request in (dict(operation="growth", dataset="g", period="month", value="growth_pct"),
                        dict(operation="share_of_total", dataset="g", by="state", value="share_pct"),
                        dict(operation="group_sum", dataset="g", by="state", value="change")):
            with self.subTest(request["operation"]):
                result = run_analysis(request, computed)
                self.assertFalse(result.ok)
                self.assertIn("already a calculated", result.error)
        # Ranking by an already-calculated share is still meaningful.
        self.assertTrue(run_analysis(dict(operation="top_n", dataset="g", value="share_pct", n=2), computed).ok)

    def test_invalid_requests_are_rejected(self):
        for label, request in self.INVALID.items():
            with self.subTest(label):
                result = run_analysis(request, DATA)
                self.assertFalse(result.ok)
                self.assertTrue(result.error)

    def test_non_dict_request(self):
        self.assertFalse(run_analysis("group_sum(sales)", DATA).ok)

    def test_empty_dataset_and_single_period(self):
        empty = {"e": SALES.iloc[0:0]}
        self.assertFalse(run_analysis(dict(operation="top_n", dataset="e", value="revenue"), empty).ok)
        one = {"one": SALES.iloc[0:1]}
        self.assertFalse(run_analysis(dict(operation="growth", dataset="one", period="month",
                                           value="revenue"), one).ok)

    def test_zero_total_share(self):
        zero = {"z": SALES.assign(revenue=0.0)}
        self.assertFalse(run_analysis(dict(operation="share_of_total", dataset="z", by="state",
                                           value="revenue"), zero).ok)


class EndToEndTests(unittest.TestCase):
    def test_sql_result_feeds_analysis(self):
        sql = run_sql("SELECT purchase_year_month, late_rate, avg_purchase_to_delivery_days, n_orders "
                      "FROM analytics.delivery_monthly ORDER BY purchase_year_month")
        self.assertTrue(sql.ok, sql.error)
        datasets = {"delivery": dataframe_from_rows(sql.rows)}
        g = run_analysis(dict(operation="growth", dataset="delivery", period="purchase_year_month",
                              value="n_orders"), datasets)
        self.assertTrue(g.ok, g.error)
        self.assertEqual(len(g.rows), sql.row_count)
        c = run_analysis(dict(operation="correlation", dataset="delivery", x="late_rate",
                              y="avg_purchase_to_delivery_days"), datasets)
        self.assertTrue(c.ok, c.error)


if __name__ == "__main__":
    unittest.main()
