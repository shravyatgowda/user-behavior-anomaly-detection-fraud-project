-- schema.sql
-- Core tables for the user-behavior risk analytics dataset.

CREATE TABLE users (
    user_id       INTEGER PRIMARY KEY,
    signup_date   DATE,
    home_city     TEXT,
    home_country  TEXT,
    age_bracket   TEXT
);

CREATE TABLE devices (
    device_id      INTEGER PRIMARY KEY,
    user_id        INTEGER REFERENCES users(user_id),
    device_type    TEXT,
    first_seen_ts  DATETIME
);

CREATE TABLE login_events (
    login_id    INTEGER PRIMARY KEY,
    user_id     INTEGER REFERENCES users(user_id),
    device_id   INTEGER REFERENCES devices(device_id),
    login_ts    DATETIME,
    ip_country  TEXT
);

CREATE TABLE transactions (
    transaction_id     INTEGER PRIMARY KEY,
    user_id            INTEGER REFERENCES users(user_id),
    device_id          INTEGER REFERENCES devices(device_id),
    txn_ts             DATETIME,
    amount             REAL,
    merchant_category  TEXT
);

CREATE INDEX idx_txn_user ON transactions(user_id);
CREATE INDEX idx_txn_ts ON transactions(txn_ts);
CREATE INDEX idx_login_user ON login_events(user_id);
CREATE INDEX idx_device_user ON devices(user_id);
