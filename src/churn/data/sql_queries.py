"""SQL-запросы к PostgreSQL и ClickHouse.
Здесь собраны аналитические запросы с JOIN, агрегациями и оконными функциями,
которые формируют витрину признаков для модели оттока.
Запросы параметризованы и совместимы с psycopg2 / clickhouse-driver.
"""
from __future__ import annotations
CUSTOMER_BASE_SQL = """
-- Базовая витрина клиентов: подписка + платежи
SELECT
    c.customer_id,
    c.signup_date,
    c.contract_type,
    c.payment_method,
    c.region,
    s.monthly_charges,
    s.internet_service,
    COALESCE(SUM(p.amount), 0) AS total_charges
FROM customers c
LEFT JOIN subscriptions s
    ON s.customer_id = c.customer_id
LEFT JOIN payments p
    ON p.customer_id = c.customer_id
   AND p.status = 'paid'
GROUP BY
    c.customer_id, c.signup_date, c.contract_type, c.payment_method,
    c.region, s.monthly_charges, s.internet_service;
"""
ACTIVITY_WINDOW_SQL = """
WITH activity AS (
    SELECT
        customer_id,
        event_date,
        SUM(login_count) OVER (
            PARTITION BY customer_id
            ORDER BY event_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS logins_30d,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id ORDER BY event_date DESC
        ) AS rn
    FROM user_events
),
revenue_rank AS (
    SELECT
        customer_id,
        total_charges,
        RANK() OVER (
            PARTITION BY region ORDER BY total_charges DESC
        ) AS revenue_rank_in_region
    FROM customer_base
)
SELECT a.customer_id, a.logins_30d, r.revenue_rank_in_region
FROM activity a
JOIN revenue_rank r USING (customer_id)
WHERE a.rn = 1;
"""
SUPPORT_FEATURES_SQL = """
SELECT
    customer_id,
    COUNT(*)                                       AS tickets_90d,
    AVG(EXTRACT(EPOCH FROM (closed_at - opened_at)) / 3600.0) AS avg_resolution_hours,
    MAX(CASE WHEN priority = 'high' THEN 1 ELSE 0 END)         AS has_high_priority
FROM support_tickets
WHERE opened_at >= NOW() - INTERVAL '90 days'
GROUP BY customer_id;
"""
CLICKHOUSE_EVENTS_SQL = """
SELECT
    customer_id,
    countIf(event_type = 'login')      AS logins,
    countIf(event_type = 'purchase')   AS purchases,
    sum(bytes_used)                    AS traffic_bytes,
    dateDiff('day', max(event_time), now()) AS days_since_last_event
FROM analytics.user_events
WHERE event_time >= now() - INTERVAL 90 DAY
GROUP BY customer_id
HAVING logins > 0
ORDER BY days_since_last_event ASC
"""
CLICKHOUSE_FUNNEL_SQL = """
SELECT
    customer_id,
    windowFunnel(86400 * 30)(
        event_time,
        event_type = 'signup',
        event_type = 'activation',
        event_type = 'payment'
    ) AS funnel_stage
FROM analytics.user_events
GROUP BY customer_id
"""
def all_queries() -> dict[str, str]:
    """Вернуть именованные запросы (для тестов/документации)."""
    return {
        "customer_base": CUSTOMER_BASE_SQL.strip(),
        "activity_window": ACTIVITY_WINDOW_SQL.strip(),
        "support_features": SUPPORT_FEATURES_SQL.strip(),
        "clickhouse_events": CLICKHOUSE_EVENTS_SQL.strip(),
        "clickhouse_funnel": CLICKHOUSE_FUNNEL_SQL.strip(),
    }
