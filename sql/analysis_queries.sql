-- analysis_queries.sql
-- Feature-engineering queries that stitch together users, devices,
-- login_events, and transactions to build the signals used by the
-- anomaly-detection notebook.

-- 1. Per-user spending baseline (mean + stddev of transaction amount),
--    used later to flag amounts that are statistical outliers for THAT user.
CREATE VIEW user_spend_baseline AS
SELECT
    user_id,
    COUNT(*)              AS txn_count,
    AVG(amount)            AS avg_amount,
    -- SQLite has no native STDDEV; compute via variance formula
    SQRT(AVG(amount * amount) - AVG(amount) * AVG(amount)) AS stddev_amount
FROM transactions
GROUP BY user_id;

-- 2. Flag transactions made on a device that had NEVER been seen for that
--    user more than 10 minutes before the transaction (a "brand-new
--    device" signal -- the core of account-takeover detection).
CREATE VIEW new_device_transactions AS
SELECT
    t.transaction_id,
    t.user_id,
    t.device_id,
    t.txn_ts,
    t.amount,
    d.first_seen_ts,
    ROUND((JULIANDAY(t.txn_ts) - JULIANDAY(d.first_seen_ts)) * 24 * 60, 1) AS minutes_since_first_seen
FROM transactions t
JOIN devices d ON t.device_id = d.device_id
WHERE (JULIANDAY(t.txn_ts) - JULIANDAY(d.first_seen_ts)) * 24 * 60 <= 10;

-- 3. Flag logins from a country that differs from the user's home country
--    (cross-referencing users x login_events).
CREATE VIEW foreign_country_logins AS
SELECT
    l.login_id,
    l.user_id,
    l.device_id,
    l.login_ts,
    l.ip_country,
    u.home_country
FROM login_events l
JOIN users u ON l.user_id = u.user_id
WHERE l.ip_country != u.home_country;

-- 4. Account-takeover candidate signal: joins new-device transactions to
--    foreign logins on the same device within a 5-minute window, then
--    back to users for context. This is the multi-table join at the
--    heart of the ATO detection logic.
CREATE VIEW ato_candidates AS
SELECT
    ndt.transaction_id,
    ndt.user_id,
    u.home_country,
    fcl.ip_country,
    ndt.device_id,
    ndt.txn_ts,
    ndt.amount,
    ndt.minutes_since_first_seen
FROM new_device_transactions ndt
JOIN foreign_country_logins fcl
    ON ndt.user_id = fcl.user_id
    AND ndt.device_id = fcl.device_id
    AND ABS((JULIANDAY(ndt.txn_ts) - JULIANDAY(fcl.login_ts)) * 24 * 60) <= 5
JOIN users u ON ndt.user_id = u.user_id;

-- 5. Velocity signal: count of transactions per user in any 30-minute
--    rolling window, computed via a self-join on transactions.
CREATE VIEW velocity_flags AS
SELECT
    t1.user_id,
    t1.transaction_id AS anchor_txn,
    t1.txn_ts          AS window_start,
    COUNT(t2.transaction_id) AS txns_in_30min
FROM transactions t1
JOIN transactions t2
    ON t1.user_id = t2.user_id
    AND t2.txn_ts BETWEEN t1.txn_ts AND DATETIME(t1.txn_ts, '+30 minutes')
GROUP BY t1.transaction_id
HAVING COUNT(t2.transaction_id) >= 6;

-- 6. Amount-anomaly signal: transactions more than 5 standard deviations
--    above the user's own historical average (using the baseline view).
CREATE VIEW amount_anomalies AS
SELECT
    t.transaction_id,
    t.user_id,
    t.amount,
    b.avg_amount,
    b.stddev_amount,
    (t.amount - b.avg_amount) / NULLIF(b.stddev_amount, 0) AS z_score
FROM transactions t
JOIN user_spend_baseline b ON t.user_id = b.user_id
WHERE b.stddev_amount > 0
  AND (t.amount - b.avg_amount) / b.stddev_amount > 5;
