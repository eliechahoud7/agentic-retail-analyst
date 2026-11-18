"""UI tests: the user-facing answer (deterministic summary and table) and the Streamlit app.

The summary tests rebuild what the user sees from the verified catalog data for the benchmark
questions whose model-written text was false, and assert the correct facts. The app tests inject a
graph built with scripted models (no Ollama needed); data still comes from the real database.
"""
import logging
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from retail_analyst.catalog import compile_query
from retail_analyst.graph import build_graph
from retail_analyst.labels import CATEGORY_NAMES, display_label
from retail_analyst.question import parse_question
from retail_analyst.summary import answer_markdown, summary_text, title_for
from retail_analyst.tools.sql_tool import run_sql
from retail_analyst.ui_support import (NO_RELIABLE_ANSWER, SERVICE_UNAVAILABLE, escape_markdown, heading_for,
                                       results_table, run_question, safe_error, user_answer)
from test_graph import GOOD_SPEC, TOP_N, ScriptedLLM, bad_spec, route

APP = str(Path(__file__).resolve().parent.parent / "app.py")
QUESTION = "What are the top 5 product categories by sales revenue?"
TOP_ROWS = [{"category_reporting": "health_beauty", "sum_items_price_total": 1258681.34},
            {"category_reporting": "watches_gifts", "sum_items_price_total": 1205005.68},
            {"category_reporting": "bed_bath_table", "sum_items_price_total": 1036988.68}]
MODEL_PROSE = "Watches & Gifts ranked first with R$ 9,999,999 because customers prefer gifts."
# Text that must never appear in the normal user-facing answer.
TECHNICAL = ["SELECT", "Evidence", "Caveat", "trace", "analytics.", "warehouse.", "{", "Numbers checked",
             "with caveats", "Supervisor", "Data Analyst", "qwen", "retail_agent_ro", "sum_items_price_total",
             "request", "Dataset d1"]


def state(rows=TOP_ROWS, body=MODEL_PROSE, issues=(), question=QUESTION, request=None):
    data = {"d1": {"rows": rows, "row_count": len(rows), "request": request}} if rows else {}
    return {"question": question, "datasets": data, "analyses": [], "answer_body": body, "issues": list(issues)}


def catalog_state(question, metric, group_by):
    """What the workflow produces when the Data Analyst chooses {metric, group_by} (as in the benchmark)."""
    constraints = parse_question(question)
    compiled = compile_query({"metric": metric, "group_by": group_by}, constraints)
    result = run_sql(compiled.sql)
    assert result.ok, result.error
    request = {"metric": compiled.metric, "group_by": compiled.grouping, "calculation": compiled.calculation,
               "years": list(constraints.years), "ascending": constraints.ascending, "top_n": constraints.top_n}
    return state(rows=result.rows, question=question, request=request)


def text_of(st):
    rows, meta = st["datasets"]["d1"]["rows"], st["datasets"]["d1"]["request"]
    return summary_text(rows, meta), title_for(rows, meta, st["question"])


def assert_no_technical_text(test, text_):
    for marker in TECHNICAL:
        test.assertNotIn(marker, text_, f"technical text {marker!r} shown to the user")


class ResultsTableTests(unittest.TestCase):
    def test_ranking_table_with_rank_category_and_brl(self):
        table = results_table(TOP_ROWS)
        self.assertIn("| Rank | Category | Revenue |", table)
        self.assertIn("|---:|---|---:|", table)                      # numbers right-aligned
        self.assertIn("| 1 | Health & Beauty | R$ 1,258,681 |", table)
        self.assertIn("| 3 | Bed, Bath & Table | R$ 1,036,989 |", table)

    def test_percent_rate_and_codes(self):
        shares = results_table([{"payment_type": "credit_card", "sum_payment_value": 12542084.19, "share_pct": 78.344},
                                {"payment_type": "boleto", "sum_payment_value": 2869361.27, "share_pct": 17.924}])
        self.assertIn("| 1 | Credit Card | R$ 12,542,084 | 78.34% |", shares)
        rates = results_table([{"customer_state_std": "AL", "late_rate": 0.2141},
                               {"customer_state_std": "MA", "late_rate": 0.1743}])
        self.assertIn("| 1 | AL | 21.41% |", rates)                  # state codes kept, rate as %

    def test_time_series_has_no_rank(self):
        table = results_table([{"year_month": "2018-01", "sum_items_price_total": 950030.36, "change": None, "growth_pct": None},
                               {"year_month": "2018-02", "sum_items_price_total": 844178.71, "change": -105851.65,
                                "growth_pct": -11.14}])
        self.assertNotIn("Rank", table)
        self.assertIn("| 2018-02 | R$ 844,179 | R$ -105,852 | -11.14% |", table)

    def test_long_results_are_capped(self):
        rows = [{"seller_state_std": f"S{i}", "units": 100 - i} for i in range(30)]
        self.assertIn("Showing the first 20 of 30 rows", results_table(rows))


class GeneratedSummaryTests(unittest.TestCase):
    """The benchmark answers whose model-written text was false, now generated from the data."""

    def test_q2_growth_states_the_real_direction_and_months(self):
        # Model said: "grew by 16.47% in February". February was -11.14%; +16.47% was March.
        text_, title = text_of(catalog_state("How did monthly sales revenue grow during 2018?", "revenue", "month"))
        self.assertEqual(title, "Monthly Revenue Growth in 2018")
        self.assertEqual(text_, "Revenue went from R$ 950,030 in January 2018 to R$ 854,686 in August 2018, "
                                "a change of -10.0%. The largest increase was in March 2018 (+16.47%) and the "
                                "largest decrease was in June 2018 (-13.19%).")

    def test_q3_ranking_starts_with_the_real_leader(self):
        # Model said: "highest are SE and MA". AL is highest.
        text_, title = text_of(catalog_state("Which customer states have the highest late delivery rate?",
                                             "late_delivery_rate", "customer_state"))
        self.assertEqual(title, "Top 10 Customer States by Late Delivery Rate")
        self.assertEqual(text_, "AL has the highest late delivery rate (21.41%), followed by MA (17.43%) and SE (15.22%).")

    def test_q7_lowest_ranking_names_the_real_lowest(self):
        # Model named the 7th and 8th lowest categories as the lowest.
        text_, title = text_of(catalog_state("Which product categories had the lowest revenue?", "revenue", "category"))
        self.assertEqual(title, "Bottom 10 Product Categories by Revenue")
        self.assertEqual(text_, "Security & Services has the lowest revenue (R$ 283.29), followed by "
                                "Fashion: Children's Clothes (R$ 569.85) and CDs, DVDs & Musicals (R$ 730.00).")

    def test_q8_month_breakdown_peak_and_low(self):
        # Model said: peak in December, orders decreased in February. Peak was November; lowest January.
        text_, title = text_of(catalog_state("How many orders were placed in each month of 2017?", "orders", "month"))
        self.assertEqual(title, "Number of Orders by Month in 2017")
        self.assertEqual(text_, "Number of orders was highest in November 2017 (7,544) and lowest in January 2017 (800).")

    def test_q12_breakdown_highest_and_lowest(self):
        # Model said: lowest is RO (5th highest). Lowest is SP.
        text_, _ = text_of(catalog_state("What is the average order value by customer state?",
                                         "average_order_value", "customer_state"))
        self.assertEqual(text_, "PB has the highest average order value (R$ 216.67) and SP the lowest (R$ 125.75).")

    def test_q13_month_breakdown_lowest(self):
        # Model said: lowest in January. Lowest was March.
        text_, _ = text_of(catalog_state("What was the average review score of orders purchased in each month of 2018?",
                                         "average_review_score", "month"))
        self.assertEqual(text_, "Average review score was highest in June 2018 (4.28) and lowest in March 2018 (3.75).")

    def test_share_comparison_and_total_templates(self):
        share, title = text_of(catalog_state("What share of payment value comes from each payment type?",
                                             "payment_value", "payment_type"))
        self.assertEqual(title, "Share of Payment Value by Payment Method")
        self.assertEqual(share, "Credit Card accounts for the largest share of payment value: 78.34% (R$ 12,542,084), "
                                "followed by Boleto (17.92%) and Voucher (2.37%). Together, these three account for "
                                "98.63% of the total.")
        compare, _ = text_of(catalog_state("Is there a relationship between delivery delays and review scores?",
                                           "average_review_score", "delivery_status"))
        self.assertEqual(compare, "The average review score is 4.29 for On-time deliveries, compared with 2.27 "
                                  "for Late deliveries.")
        total, title = text_of(catalog_state("What was the total freight cost in 2018?", "freight", None))
        self.assertEqual((title, total), ("Freight Cost in 2018", "The freight cost in 2018 was R$ 1,257,647."))

    def test_units_are_stated(self):
        text_, _ = text_of(catalog_state("Which customer states have the longest average delivery time?",
                                         "average_delivery_days", "customer_state"))
        self.assertEqual(text_, "RR has the highest average delivery time (29.39 days), followed by AP (27.19 days) "
                                "and AM (26.43 days).")

    def test_heading_fallback_without_metadata(self):
        self.assertEqual(heading_for(QUESTION), "Top 5 Product Categories by Sales Revenue")


class UserAnswerTests(unittest.TestCase):
    def test_answer_is_generated_from_data_never_from_model_text(self):
        reliable, md = user_answer(state())
        self.assertTrue(reliable)
        self.assertTrue(md.startswith("### Top 5 Product Categories by Sales Revenue"))
        self.assertIn("Health & Beauty has the highest revenue (R$ 1,258,681), followed by Watches & Gifts "
                      "(R$ 1,205,006) and Bed, Bath & Table (R$ 1,036,989).", md)
        self.assertNotIn("9,999,999", md)                                    # model text never shown
        self.assertNotIn("prefer", md)
        self.assertLess(md.index("has the highest"), md.index("| Rank"))    # summary first, then table
        assert_no_technical_text(self, md)

    def test_catalog_answer_uses_metric_heading(self):
        md = answer_markdown(catalog_state(QUESTION, "revenue", "category"))
        self.assertTrue(md.startswith("### Top 5 Product Categories by Revenue\n\nHealth & Beauty has the highest revenue"))
        self.assertIn("| 5 | Computers & Accessories | R$ 911,954 |", md)

    def test_info_issues_do_not_change_the_answer(self):
        _, md = user_answer(state(issues=[{"kind": "info", "message": "request corrected"}]))
        self.assertNotIn("corrected", md)

    def test_no_data_gives_short_message(self):
        self.assertEqual(user_answer(state(rows=[])), (False, NO_RELIABLE_ANSWER))

    def test_data_issue_blocks_the_answer_even_with_rows(self):
        reliable, md = user_answer(state(issues=[{"kind": "data", "message": "repeated labels"}]))
        self.assertFalse(reliable)
        self.assertEqual(md, NO_RELIABLE_ANSWER)

    def test_scope_issue_gives_its_reason(self):
        reliable, md = user_answer(state(rows=[], issues=[{"kind": "scope", "message": "No data for 2020."}]))
        self.assertFalse(reliable)
        self.assertEqual(md, "I can't answer this question reliably. No data for 2020.")

    def test_escape_markdown_and_safe_error(self):
        self.assertEqual(escape_markdown("R$ 1 and R$ 2"), r"R\$ 1 and R\$ 2")
        self.assertNotIn("secret", safe_error(RuntimeError("postgresql+psycopg://u:secret@host/db failed")))

    def test_run_question_streams_events_and_returns_final_state(self):
        events = []
        graph = scripted_graph(ScriptedLLM(route("data_analyst"), route("report")), ScriptedLLM(GOOD_SPEC))
        final = run_question(graph, QUESTION, on_event=events.append)
        self.assertEqual(len(events), len(final["trace"]))
        self.assertIn("issues", final)


class CategoryNameTests(unittest.TestCase):
    def test_every_database_category_has_a_readable_name(self):
        result = run_sql("SELECT DISTINCT category_reporting FROM warehouse.dim_products ORDER BY 1")
        self.assertTrue(result.ok, result.error)
        missing = [r["category_reporting"] for r in result.rows if r["category_reporting"] not in CATEGORY_NAMES]
        self.assertEqual(missing, [], "add these categories to retail_analyst/labels.py")

    def test_names_are_formatted_not_invented(self):
        self.assertEqual(display_label("health_beauty"), "Health & Beauty")
        self.assertEqual(display_label("bed_bath_table"), "Bed, Bath & Table")   # not "Towels"
        self.assertEqual(display_label("home_appliances_2"), "Home Appliances 2")  # separate category kept
        self.assertEqual(display_label("credit_card"), "Credit Card")            # non-category snake_case
        self.assertEqual(display_label("AL"), "AL")                              # codes unchanged
        for key, name in CATEGORY_NAMES.items():  # same words as the database, only formatted
            source = {w for w in key.split("_") if w not in ("and",)}
            shown = {w.lower().strip(":,&'") for w in name.replace("-", " ").split()}
            self.assertTrue({w for w in source if not w.isdigit()} <= shown | {"childrens", "dvds", "cds"}, key)


def scripted_graph(supervisor, data_analyst, report=MODEL_PROSE):
    return build_graph({"supervisor": supervisor, "data_analyst": data_analyst,
                        "analytics": ScriptedLLM(default=TOP_N), "report": ScriptedLLM(default=report)})


class AppTests(unittest.TestCase):
    def app(self, graph=None):
        at = AppTest.from_file(APP, default_timeout=60)
        if graph is not None:
            at.session_state["graph"] = graph
        return at.run()

    def ask(self, graph, question=QUESTION):
        at = self.app(graph)
        with self.assertLogs("retail_analyst.app", level="INFO") as logs:
            at.chat_input[0].set_value(question).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at, "\n".join(logs.output)

    def main_text(self, at):
        return "\n".join(m.value for m in at.main.markdown)

    def assert_clean_page(self, at):
        self.assertEqual(len(at.expander), 0)          # no trace / data / SQL expanders
        self.assertEqual(len(at.main.code), 0)         # no SQL or JSON blocks
        self.assertEqual(len(at.main.dataframe), 0)
        self.assertEqual(len(at.main.success) + len(at.main.warning) + len(at.main.error), 0)  # no status labels
        self.assertEqual(len(at.main.caption), 0)      # no timing / model captions
        assert_no_technical_text(self, self.main_text(at))

    def test_app_loads_without_technical_sidebar(self):
        at = self.app()
        self.assertFalse(at.exception)
        self.assertIn("Retail Data Analyst", at.title[0].value)
        sidebar = " ".join(m.value for m in at.sidebar.markdown)
        for marker in ("qwen", "retail_agent_ro", "read-only", "rows", "Safeguards"):
            self.assertNotIn(marker, sidebar)

    def test_successful_answer_shows_only_the_business_answer(self):
        graph = scripted_graph(ScriptedLLM(route("data_analyst"), route("report")), ScriptedLLM(GOOD_SPEC))
        at, logs = self.ask(graph)
        text_ = self.main_text(at)
        self.assertIn("### Top 5 Product Categories by Revenue", text_)
        self.assertIn(r"Health & Beauty has the highest revenue (R\$ 1,258,681)", text_)  # escaped for Streamlit
        self.assertIn(r"| 1 | Health & Beauty | R\$ 1,258,681 |", text_)
        self.assertNotIn("9,999,999", text_)
        self.assert_clean_page(at)
        self.assertIn("status: answered", logs)                          # diagnostics go to the log
        self.assertIn("warehouse.fact_order_lines", logs)                # ...including the real SQL

    def test_failed_question_shows_short_message(self):
        graph = scripted_graph(ScriptedLLM(default=route("data_analyst")), ScriptedLLM(default=bad_spec("nope")))
        at, logs = self.ask(graph)
        self.assertIn(escape_markdown(NO_RELIABLE_ANSWER.split("\n")[0]), self.main_text(at))
        self.assertNotIn("| Rank", self.main_text(at))
        self.assert_clean_page(at)
        self.assertIn("status: no_reliable_answer", logs)

    def test_unreliable_data_is_not_shown_as_an_answer(self):
        class FixedGraph:  # a workflow result whose data failed a reliability check
            def stream(self, *args, **kwargs):
                yield {"question": QUESTION, "trace": [], "answer_body": "",
                       "datasets": {"d1": {"rows": TOP_ROWS, "row_count": 3}}, "analyses": [],
                       "issues": [{"kind": "data", "message": "Dataset d1 repeats labels"}]}

        at, logs = self.ask(FixedGraph())
        self.assertIn(escape_markdown(NO_RELIABLE_ANSWER.split("\n")[0]), self.main_text(at))
        self.assertNotIn("| Rank", self.main_text(at))
        self.assert_clean_page(at)
        self.assertIn("repeats labels", logs)

    def test_out_of_scope_question_is_refused_with_its_reason(self):
        graph = scripted_graph(ScriptedLLM(default=route("data_analyst")), ScriptedLLM(default=GOOD_SPEC))
        at, logs = self.ask(graph, question="Why do customers prefer credit cards?")
        text_ = self.main_text(at)
        self.assertIn("I can't answer this question reliably", text_)
        self.assertIn("not why it happened", text_)
        self.assertNotIn("| Rank", text_)
        self.assert_clean_page(at)
        self.assertIn("status: refused_out_of_scope", logs)

    def test_system_error_shows_friendly_message(self):
        class BrokenGraph:
            def stream(self, *args, **kwargs):
                raise ConnectionError("database down at 127.0.0.1")
                yield  # pragma: no cover

        at, logs = self.ask(BrokenGraph())
        self.assertIn(SERVICE_UNAVAILABLE, self.main_text(at))
        self.assertNotIn("127.0.0.1", self.main_text(at))
        self.assert_clean_page(at)
        self.assertIn("database down", logs)

    def test_example_button_asks_the_question(self):
        graph = scripted_graph(ScriptedLLM(route("data_analyst"), route("report")), ScriptedLLM(GOOD_SPEC))
        at = self.app(graph)
        at.sidebar.button(key="example_0").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["messages"][0]["content"], QUESTION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
