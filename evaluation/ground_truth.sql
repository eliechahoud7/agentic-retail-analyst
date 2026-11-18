-- Ground-truth answers for the five benchmark questions, written by hand and run with the
-- read-only retail_agent_ro login. Used to judge the agents' answers in evaluation/results/.
-- Revenue = items_price_total (item prices excluding freight), as documented in the database.

-- Q1. What are the top 5 product categories by sales revenue?
-- Expected: health_beauty 1258681.34, watches_gifts 1205005.68, bed_bath_table 1036988.68,
--           sports_leisure 988048.97, computers_accessories 911954.32
SELECT category_reporting, items_price_total
FROM analytics.category_performance
ORDER BY items_price_total DESC
LIMIT 5;

-- Q2. How did monthly sales revenue grow during 2018? (reliable window: up to 2018-08)
-- Expected: 2018-01 950030.36, 02 844178.71, 03 983213.44, 04 996647.75,
--           05 996517.68, 06 865124.31, 07 895507.22, 08 854686.33
SELECT year_month, ROUND(SUM(items_price_total), 2) AS revenue
FROM analytics.sales_daily
WHERE year = 2018 AND in_candidate_window
GROUP BY year_month
ORDER BY year_month;

-- Q3. Which customer states have the highest late delivery rate?
-- Expected: AL 0.2141, MA 0.1743, SE 0.1522, PI 0.1387, CE 0.1376
SELECT customer_state_std, late_rate, n_delivery_eligible
FROM analytics.delivery_by_state
ORDER BY late_rate DESC
LIMIT 5;

-- Q4. What share of payment value comes from each payment type?
-- Expected: credit_card 78.34 %, boleto 17.92 %, voucher 2.37 %, debit_card 1.36 %, not_defined 0.00 %
SELECT payment_type,
       ROUND(SUM(payment_value), 2) AS payment_value,
       ROUND(100 * SUM(payment_value) / SUM(SUM(payment_value)) OVER (), 2) AS share_pct
FROM warehouse.fact_payments
GROUP BY payment_type
ORDER BY payment_value DESC;

-- Q5. Is there a relationship between delivery delays and review scores?
-- Expected: late orders average 2.27 stars (n = 6,410), on-time orders 4.29 (n = 89,949);
--           53.7 % of reviews for late deliveries are 1 star.
SELECT o.is_late, ROUND(AVG(r.review_score), 3) AS avg_score, COUNT(*) AS n
FROM warehouse.fact_reviews r
JOIN warehouse.fact_orders o ON o.order_id = r.order_id
WHERE o.is_late IS NOT NULL
GROUP BY o.is_late;

SELECT delivery_class, review_score, review_share_pct
FROM dbt_marts.mart_review_scores_by_delivery
ORDER BY delivery_class, review_score;
