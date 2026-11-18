"""Connectivity, read-only session settings and schema metadata (live database)."""
import unittest

from sqlalchemy import text

from retail_analyst.config import get_engine, get_settings, load_schema, schema_summary


class ConfigTests(unittest.TestCase):
    def test_settings_hide_password(self):
        settings = get_settings()
        self.assertEqual(settings.pg_user, "retail_agent_ro")
        self.assertNotIn(settings.pg_password, repr(settings))
        self.assertEqual(settings.model_for("supervisor"), settings.supervisor_model or settings.ollama_model)

    def test_connection_is_read_only_with_timeout(self):
        with get_engine().connect() as conn:
            user, read_only, timeout = conn.execute(text(
                "SELECT current_user, current_setting('transaction_read_only'), "
                "current_setting('statement_timeout')")).one()
        self.assertEqual(user, "retail_agent_ro")
        self.assertEqual(read_only, "on")
        self.assertEqual(timeout, "30s")

    def test_schema_metadata(self):
        tables = load_schema()
        self.assertEqual(len(tables), 20)
        orders = tables["warehouse.fact_orders"]
        self.assertEqual(orders.kind, "table")
        self.assertEqual(orders.primary_key, ["order_id"])
        self.assertIn(("order_id", "text"), orders.columns)
        self.assertIn((["customer_unique_id"], "warehouse.dim_customers", ["customer_unique_id"]),
                      orders.foreign_keys)
        self.assertEqual(tables["analytics.sales_daily"].kind, "view")
        # Excluded relations are not visible to the agent.
        self.assertNotIn("warehouse.load_run", tables)
        self.assertFalse(any(name.startswith("dbt_staging.") for name in tables))

    def test_schema_summary_is_prompt_ready_and_secret_free(self):
        summary = schema_summary()
        self.assertIn("warehouse.fact_orders (table", summary)
        self.assertIn("customer_unique_id -> warehouse.dim_customers(customer_unique_id)", summary)
        self.assertNotIn(get_settings().pg_password, summary)


if __name__ == "__main__":
    unittest.main()
