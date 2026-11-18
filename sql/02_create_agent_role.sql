-- Dedicated read-only login for the Agentic Retail Data Analyst.
-- Run once as the PostgreSQL admin. Does not touch powerbi_reader, data or table definitions;
-- the only side effect on existing objects is the added SELECT/USAGE/CONNECT grants below.
-- The password is set separately (never stored in this file).
BEGIN;

CREATE ROLE retail_agent_ro
  LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS
  CONNECTION LIMIT 5;

-- Session defaults (defense in depth; privileges below are the real guard).
ALTER ROLE retail_agent_ro SET default_transaction_read_only = on;
ALTER ROLE retail_agent_ro SET statement_timeout = '30s';
ALTER ROLE retail_agent_ro SET idle_in_transaction_session_timeout = '60s';

GRANT CONNECT ON DATABASE retail_intelligence TO retail_agent_ro;
GRANT USAGE ON SCHEMA warehouse, analytics, dbt_marts TO retail_agent_ro;

-- Business data only. Excluded: warehouse.load_run (pipeline metadata), dbt_staging (internal).
-- No default privileges: future tables need an explicit grant.
GRANT SELECT ON
  warehouse.dim_customers, warehouse.dim_dates, warehouse.dim_products, warehouse.dim_sellers,
  warehouse.fact_orders, warehouse.fact_order_lines, warehouse.fact_payments, warehouse.fact_reviews,
  analytics.category_performance, analytics.customer_performance, analytics.delivery_by_state,
  analytics.delivery_monthly, analytics.product_performance, analytics.sales_daily,
  analytics.seller_performance,
  dbt_marts.mart_customer_cohorts, dbt_marts.mart_payment_installments,
  dbt_marts.mart_payment_methods_monthly, dbt_marts.mart_review_scores_by_delivery,
  dbt_marts.mart_reviews_monthly
TO retail_agent_ro;

COMMIT;
