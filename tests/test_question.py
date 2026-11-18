"""Deterministic question constraints: years, top N, direction, calculation and refusals."""
import unittest

from retail_analyst.question import parse_question


class ConstraintTests(unittest.TestCase):
    def test_years(self):
        self.assertEqual(parse_question("How did revenue grow during 2018?").years, (2018,))
        self.assertEqual(parse_question("Compare 2017 and 2018 revenue").years, (2017, 2018))
        self.assertEqual(parse_question("Which states sell the most?").years, ())

    def test_top_n_and_direction(self):
        c = parse_question("What are the top 5 product categories by sales revenue?")
        self.assertEqual((c.top_n, c.ascending, c.calculation), (5, False, "ranking"))
        c = parse_question("Which three states have the lowest review scores?")
        self.assertEqual((c.top_n, c.ascending, c.calculation), (3, True, "ranking"))
        self.assertEqual(parse_question("Show the 10 best selling categories").top_n, 10)
        self.assertIsNone(parse_question("Which customer states have the highest late delivery rate?").top_n)

    def test_calculations(self):
        self.assertEqual(parse_question("What share of payment value comes from each payment type?").calculation, "share")
        self.assertEqual(parse_question("How did monthly sales revenue grow during 2018?").calculation, "growth")
        self.assertEqual(parse_question("Show the revenue trend per month").calculation, "growth")
        self.assertEqual(parse_question("Which customer states have the highest late delivery rate?").calculation,
                         "ranking")
        self.assertEqual(parse_question("Is there a relationship between delivery delays and review scores?")
                         .calculation, "breakdown")


class RefusalTests(unittest.TestCase):
    REFUSED = {
        "Why do customers prefer credit cards?": "not why",
        "What is the reason for late deliveries in AL?": "not why",
        "What will revenue be in 2019?": "forecast",
        "Can you predict next month's orders?": "forecast",
        "What was revenue last quarter?": "relative period",
        "How many orders did we get this year?": "relative period",
        "What was the revenue in 2020?": "no data for 2020",
    }

    def test_unanswerable_questions_are_refused_with_a_reason(self):
        for question, reason in self.REFUSED.items():
            with self.subTest(question):
                refusal = parse_question(question).refusal
                self.assertIsNotNone(refusal)
                self.assertIn(reason, refusal)

    def test_answerable_questions_are_not_refused(self):
        for question in ("What are the top 5 product categories by sales revenue?",
                         "How did monthly sales revenue grow during 2018?",
                         "Which customer states have the highest late delivery rate?",
                         "What share of payment value comes from each payment type?",
                         "Is there a relationship between delivery delays and review scores?",
                         "Show the 10 best selling categories",                 # "best-selling" is allowed
                         "What is the average delivery time in days?",          # "in days" is not a grain
                         "In which month of 2018 were deliveries most often late?",
                         "Which states have the most customers?",
                         "What was the average review score of orders purchased in each month of 2018?"):
            with self.subTest(question):
                self.assertIsNone(parse_question(question).refusal)


class QualifierTests(unittest.TestCase):
    """Qualifiers and filters the catalog cannot represent (independent question set: ignoring them
    produced confident wrong answers)."""
    REFUSED = {
        "How many orders were cancelled?": "status",
        "How many returned orders were there in 2018?": "status",
        "How many new customers did we acquire each month in 2017?": "new and returning",
        "What share of customers come back for a second purchase?": "new and returning",
        "How many reviews did we receive in 2018?": "purchase date",
        "How many reviews were received in 2018?": "purchase date",
        "What is the most popular payment method?": "how often",
        "Which month of 2017 had the worst reviews?": "which measure",
        "What was revenue in May 2018?": "one specific period",
        "What was the revenue in Q2 2018?": "one specific period",
        "What is the daily revenue?": "months and years",
        "How much revenue on weekends?": "months and years",
        "How much did customers pay in total with vouchers?": "one payment method",
        "How much did we sell in the state of São Paulo?": "one state",
        "What was the revenue in Rio de Janeiro in 2017?": "one state",
        "How many late deliveries were there in SP?": "one state",
        "What was the revenue of health & beauty in 2018?": "one product category",
        "How many toys did we sell?": "one product category",
    }

    def test_unsupported_qualifiers_and_filters_are_refused(self):
        for question, reason in self.REFUSED.items():
            with self.subTest(question):
                refusal = parse_question(question).refusal
                self.assertIsNotNone(refusal)
                self.assertIn(reason, refusal)

    def test_reviews_received_is_an_event_date(self):
        refusal = parse_question("How many reviews did we receive in 2018?").refusal
        self.assertIsNotNone(refusal)
        self.assertIn("purchase date", refusal)
        self.assertIn("when something was received,", refusal)   # grammatical, not 'was "receive"'
        self.assertIn("when something was written,", parse_question("When were the reviews wrote?").refusal)

    def test_receiving_as_destination_is_not_refused(self):
        for question in ("Which states receive the most orders?",
                         "Which categories receive the most orders?"):
            with self.subTest(question):
                self.assertIsNone(parse_question(question).refusal)

    def test_new_customers_is_a_lifecycle_concept(self):
        refusal = parse_question("How many new customers did we acquire in 2017?").refusal
        self.assertIsNotNone(refusal)
        self.assertIn("new and returning", refusal)

    def test_new_alone_does_not_trigger_a_refusal(self):
        # Not refused because of "new". (This question is still refused later by the agents: no
        # catalog metric fits it.)
        self.assertIsNone(parse_question("What new marketing strategies should we consider?").refusal)

    def test_best_selling_is_allowed(self):
        constraints = parse_question("Which are the best-selling categories?")
        self.assertIsNone(constraints.refusal)
        self.assertEqual(constraints.calculation, "ranking")

    def test_lifecycle_and_event_phrases_in_context(self):
        refused = ["Which marketing channel brought the most new customers?",   # original benchmark Q16
                   "How many first-time buyers did we have in 2018?",
                   "What share of revenue comes from returning customers?",
                   "How many repeat purchases were made?",
                   "How many customers did we acquire in 2017?",
                   "When did customers receive their orders?",
                   "How many ratings were submitted in 2017?"]
        allowed = ["Which new product categories sold the most units?",
                   "Which seller states received the most revenue?",
                   "What is the average review score per month?"]
        for question in refused:
            with self.subTest(refused=question):
                self.assertIsNotNone(parse_question(question).refusal)
        for question in allowed:
            with self.subTest(allowed=question):
                self.assertIsNone(parse_question(question).refusal)

    def test_whole_dataset_range_is_no_filter(self):
        # Fresh question set: three answerable questions were refused because of "September".
        c = parse_question("Which Brazilian state(s) saw the highest number of orders during the period "
                           "September 2016 to October 2018?")
        self.assertIsNone(c.refusal)
        self.assertEqual(c.years, ())

    def test_whole_calendar_years_range_is_a_year_filter(self):
        c = parse_question("What was the revenue from January 2017 to December 2017?")
        self.assertIsNone(c.refusal)
        self.assertEqual(c.years, (2017,))

    def test_partial_range_and_endpoint_comparison_are_refused(self):
        self.assertIn("one specific period",
                      parse_question("What was the revenue from March 2017 to June 2017?").refusal)
        self.assertIn("compares two specific months",
                      parse_question("What was the percentage increase in orders from September 2016 to "
                                     "October 2018?").refusal)

    def test_worst_selling_reads_as_lowest(self):
        constraints = parse_question("Which are the 5 worst-selling categories?")
        self.assertIsNone(constraints.refusal)
        self.assertTrue(constraints.ascending)


if __name__ == "__main__":
    unittest.main()
