"""run_sql: valid queries against the live warehouse and rejection of unsafe SQL."""
import unittest

from retail_analyst.tools.sql_tool import MAX_ROWS, UnsafeQueryError, run_sql, validate_sql


class ValidQueryTests(unittest.TestCase):
    def test_simple_count(self):
        result = run_sql("SELECT COUNT(*) AS n FROM warehouse.fact_orders;")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.columns, ["n"])
        self.assertEqual(result.rows, [{"n": 99441}])
        self.assertFalse(result.truncated)

    def test_with_query_join_and_types(self):
        result = run_sql("""
            WITH cat AS (
                SELECT p.category_reporting, SUM(l.line_price_total) AS revenue
                FROM warehouse.fact_order_lines l
                JOIN warehouse.dim_products p ON p.product_id = l.product_id
                GROUP BY 1
            )
            SELECT * FROM cat ORDER BY revenue DESC LIMIT 5
        """)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.row_count, 5)
        self.assertIsInstance(result.rows[0]["revenue"], float)  # Decimal converted
        revenues = [r["revenue"] for r in result.rows]
        self.assertEqual(revenues, sorted(revenues, reverse=True))  # ORDER BY preserved

    def test_view_with_dates_cast_and_like(self):
        # '::' casts and '%' wildcards must not be treated as placeholders.
        result = run_sql("SELECT purchase_date, n_orders::numeric AS n FROM analytics.sales_daily "
                         "WHERE year_month LIKE '2017-%' ORDER BY purchase_date LIMIT 3")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.rows[0]["purchase_date"][:4], "2017")  # ISO date string

    def test_row_cap(self):
        result = run_sql("SELECT product_id FROM warehouse.dim_products")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.row_count, MAX_ROWS)
        self.assertTrue(result.truncated)

    def test_forbidden_word_inside_string_is_allowed(self):
        result = run_sql("SELECT COUNT(*) AS n FROM warehouse.fact_orders WHERE order_status = 'delete'")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.rows[0]["n"], 0)

    def test_database_errors_are_returned_not_raised(self):
        result = run_sql("SELECT no_such_column FROM warehouse.fact_orders")
        self.assertFalse(result.ok)
        self.assertIn("no_such_column", result.error)

    def test_excluded_table_is_denied_by_database(self):
        result = run_sql("SELECT * FROM warehouse.load_run")
        self.assertFalse(result.ok)
        self.assertIn("permission denied", result.error)

    def test_preview(self):
        result = run_sql("SELECT category_reporting, n_orders FROM analytics.category_performance")
        self.assertIn("category_reporting | n_orders", result.preview(3))


class UnsafeQueryTests(unittest.TestCase):
    UNSAFE = {
        "empty": "   ",
        "insert": "INSERT INTO warehouse.dim_dates (date) VALUES ('1900-01-01')",
        "update": "UPDATE warehouse.fact_orders SET order_status = 'x'",
        "delete": "DELETE FROM warehouse.fact_reviews",
        "drop": "DROP TABLE warehouse.dim_dates",
        "create": "CREATE TABLE x (id int)",
        "truncate": "TRUNCATE warehouse.fact_payments",
        "grant": "GRANT SELECT ON warehouse.dim_dates TO PUBLIC",
        "set": "SET statement_timeout = 0",
        "multiple statements": "SELECT 1; DROP TABLE warehouse.dim_dates",
        "multiple selects": "SELECT 1; SELECT 2",
        "data-modifying CTE": "WITH d AS (DELETE FROM warehouse.fact_reviews RETURNING *) SELECT * FROM d",
        "select into": "SELECT * INTO new_table FROM warehouse.dim_dates",
        "set_config": "SELECT set_config('transaction_read_only', 'off', true)",
        "pg_sleep": "SELECT pg_sleep(60)",
        "pg_read_file": "SELECT pg_read_file('postgresql.conf')",
        "row lock": "SELECT * FROM warehouse.fact_orders FOR UPDATE",
        "row lock share": "SELECT * FROM warehouse.fact_orders FOR SHARE",
        "dollar quoting": "SELECT $$x$$",
        "statement hidden after comment": "SELECT 1 /* harmless */; DELETE FROM warehouse.fact_reviews",
        "unbalanced quote": "SELECT 'abc FROM warehouse.fact_orders",
        "explain": "EXPLAIN ANALYZE SELECT 1",
        "copy": "COPY warehouse.dim_dates TO STDOUT",
    }

    def test_validator_rejects(self):
        for label, query in self.UNSAFE.items():
            with self.subTest(label):
                with self.assertRaises(UnsafeQueryError):
                    validate_sql(query)

    def test_run_sql_returns_rejection_without_executing(self):
        result = run_sql("DELETE FROM warehouse.fact_reviews")
        self.assertFalse(result.ok)
        self.assertTrue(result.error.startswith("Rejected:"))

    def test_validator_accepts_safe_queries(self):
        for query in ["SELECT 1", "select 1;", "  WITH x AS (SELECT 1 AS a) SELECT a FROM x ",
                      "(SELECT 1)", "SELECT 1 -- trailing comment",
                      "SELECT created_at_alias FROM (SELECT 1 AS created_at_alias) t"]:
            with self.subTest(query):
                validate_sql(query)


if __name__ == "__main__":
    unittest.main()
