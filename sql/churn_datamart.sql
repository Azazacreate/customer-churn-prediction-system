CREATE TABLE IF NOT EXISTS customers (
    customer_id           TEXT PRIMARY KEY,
    customer_unique_id    TEXT,
    customer_zip_code_prefix TEXT,
    customer_city         TEXT,
    customer_state        TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id                  TEXT PRIMARY KEY,
    customer_id               TEXT,
    order_status              TEXT,
    order_purchase_timestamp  TEXT,
    order_approved_at         TEXT,
    order_delivered_carrier_date TEXT,
    order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT
);

CREATE TABLE IF NOT EXISTS order_items (
    order_id     TEXT,
    order_item_id TEXT,
    product_id   TEXT,
    seller_id    TEXT,
    shipping_limit_date TEXT,
    price        TEXT,
    freight_value TEXT
);

CREATE TABLE IF NOT EXISTS order_payments (
    order_id          TEXT,
    payment_sequential TEXT,
    payment_type      TEXT,
    payment_installments TEXT,
    payment_value     TEXT
);

CREATE TABLE IF NOT EXISTS order_reviews (
    review_id              TEXT,
    order_id               TEXT,
    review_score           TEXT,
    review_comment_title   TEXT,
    review_comment_message TEXT,
    review_creation_date   TEXT,
    review_answer_timestamp TEXT
);

-- ----------------------------------------------------------------------------
-- 1. ВИТРИНА ЗАКАЗОВ: обогащение заказов суммой чека, позициями и оценкой.
--    Используются CTE + агрегации + FILTER + касты типов.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_orders_enriched AS
WITH items_agg AS (
    SELECT
        order_id,
        COUNT(*)                       AS n_items,
        SUM(price::numeric)            AS items_value,
        SUM(freight_value::numeric)    AS freight_value
    FROM order_items
    GROUP BY order_id
),
pay_agg AS (
    SELECT
        order_id,
        SUM(payment_value::numeric)    AS payment_value,
        MAX(payment_installments::int) AS max_installments,
        COUNT(*) FILTER (WHERE payment_type = 'credit_card') AS n_credit_card
    FROM order_payments
    GROUP BY order_id
),
rev_agg AS (
    SELECT
        order_id,
        AVG(review_score::numeric)     AS review_score
    FROM order_reviews
    GROUP BY order_id
)
SELECT
    o.order_id,
    c.customer_unique_id,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp::timestamp AS order_purchase_timestamp,
    o.order_delivered_customer_date::timestamp AS order_delivered_customer_date,
    o.order_estimated_delivery_date::timestamp AS order_estimated_delivery_date,
    EXTRACT(EPOCH FROM (o.order_delivered_customer_date::timestamp
                        - o.order_purchase_timestamp::timestamp))/86400 AS delivery_days,
    EXTRACT(EPOCH FROM (o.order_delivered_customer_date::timestamp
                        - o.order_estimated_delivery_date::timestamp))/86400 AS delivery_delay_days,
    COALESCE(i.n_items, 0)         AS n_items,
    COALESCE(p.payment_value, 0)   AS payment_value,
    COALESCE(i.freight_value, 0)   AS freight_value,
    COALESCE(p.max_installments, 0) AS max_installments,
    r.review_score                 AS review_score
FROM orders o
JOIN customers c            ON c.customer_id = o.customer_id
LEFT JOIN items_agg i       ON i.order_id = o.order_id
LEFT JOIN pay_agg p         ON p.order_id = o.order_id
LEFT JOIN rev_agg r         ON r.order_id = o.order_id
WHERE o.order_status IN ('delivered', 'shipped', 'invoiced', 'processing');

-- ----------------------------------------------------------------------------
-- 2. КЛИЕНТСКАЯ ВИТРИНА (RFM) с оконными функциями.
--    recency / frequency / monetary + метка оттока (churn = recency > 180 дней).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_customer_features AS
WITH snapshot AS (
    SELECT MAX(order_purchase_timestamp) AS snap_ts FROM v_orders_enriched
),
per_order AS (
    SELECT
        customer_unique_id,
        order_id,
        order_purchase_timestamp,
        payment_value,
        n_items,
        freight_value,
        review_score,
        max_installments,
        customer_state,
        -- порядковый номер заказа клиента по времени (оконная функция)
        ROW_NUMBER() OVER (PARTITION BY customer_unique_id
                           ORDER BY order_purchase_timestamp) AS order_seq,
        -- интервал до предыдущего заказа клиента (оконная функция LAG)
        EXTRACT(EPOCH FROM (
            order_purchase_timestamp
            - LAG(order_purchase_timestamp) OVER (PARTITION BY customer_unique_id
                                                  ORDER BY order_purchase_timestamp)
        ))/86400 AS days_since_prev_order
    FROM v_orders_enriched
)
SELECT
    p.customer_unique_id,
    MAX(p.order_purchase_timestamp)                                    AS last_purchase,
    MIN(p.order_purchase_timestamp)                                    AS first_purchase,
    COUNT(DISTINCT p.order_id)                                         AS frequency,
    SUM(p.payment_value)                                               AS monetary,
    AVG(p.payment_value)                                               AS avg_order_value,
    SUM(p.n_items)                                                     AS total_items,
    AVG(p.freight_value)                                               AS avg_freight,
    AVG(p.review_score)                                                AS avg_review,
    MAX(p.max_installments)                                            AS max_installments,
    AVG(p.days_since_prev_order)                                       AS avg_days_between_orders,
    EXTRACT(EPOCH FROM (s.snap_ts - MAX(p.order_purchase_timestamp)))/86400 AS recency,
    EXTRACT(EPOCH FROM (MAX(p.order_purchase_timestamp)
                        - MIN(p.order_purchase_timestamp)))/86400      AS tenure,
    CASE
        WHEN EXTRACT(EPOCH FROM (s.snap_ts - MAX(p.order_purchase_timestamp)))/86400 > 180 THEN 1
        ELSE 0
    END                                                                AS churn
FROM per_order p
CROSS JOIN snapshot s
GROUP BY p.customer_unique_id, s.snap_ts;

-- ----------------------------------------------------------------------------
-- 3. ТОП клиентов с высоким риском (пример бизнес-запроса для retention-команды).
--    Предполагается, что модель записала скоры в таблицу churn_scores.
-- ----------------------------------------------------------------------------
-- SELECT
--     f.customer_unique_id,
--     f.recency,
--     f.frequency,
--     f.monetary,
--     sc.churn_proba,
--     RANK() OVER (ORDER BY sc.churn_proba DESC, f.monetary DESC) AS risk_rank
-- FROM v_customer_features f
-- JOIN churn_scores sc USING (customer_unique_id)
-- WHERE sc.churn_proba >= 0.5
-- ORDER BY sc.churn_proba DESC
-- LIMIT 100;
