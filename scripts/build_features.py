"""
build_features.py

Loads the generated CSVs into a SQLite database, applies schema.sql,
then runs analysis_queries.sql to build the risk-signal views. Finally
assembles a single per-user feature table (features.csv) that the
notebook uses for anomaly detection.

This mirrors a real analyst workflow: raw tables -> SQL joins -> a flat
feature table -> Python modeling.
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "risk_analytics.db"
DB_PATH.unlink(missing_ok=True)

conn = sqlite3.connect(DB_PATH)

# ---- load raw CSVs into SQLite tables --------------------------------
users = pd.read_csv(ROOT / "data" / "users.csv")
devices = pd.read_csv(ROOT / "data" / "devices.csv")
logins = pd.read_csv(ROOT / "data" / "login_events.csv")
transactions = pd.read_csv(ROOT / "data" / "transactions.csv")

users.to_sql("users", conn, index=False, if_exists="replace")
devices.to_sql("devices", conn, index=False, if_exists="replace")
logins.to_sql("login_events", conn, index=False, if_exists="replace")
transactions.to_sql("transactions", conn, index=False, if_exists="replace")

# ---- apply the analysis views -----------------------------------------
sql_text = (ROOT / "sql" / "analysis_queries.sql").read_text()
conn.executescript(sql_text)

# ---- pull each signal back into pandas ---------------------------------
ato = pd.read_sql("SELECT DISTINCT user_id FROM ato_candidates", conn)
ato["signal_ato"] = 1

velocity = pd.read_sql("SELECT DISTINCT user_id FROM velocity_flags", conn)
velocity["signal_velocity"] = 1

amount_anom = pd.read_sql("SELECT DISTINCT user_id FROM amount_anomalies", conn)
amount_anom["signal_amount_anomaly"] = 1

baseline = pd.read_sql("SELECT * FROM user_spend_baseline", conn)

# ---- assemble the per-user feature table --------------------------------
features = baseline.merge(ato, on="user_id", how="left") \
                    .merge(velocity, on="user_id", how="left") \
                    .merge(amount_anom, on="user_id", how="left")

for col in ["signal_ato", "signal_velocity", "signal_amount_anomaly"]:
    features[col] = features[col].fillna(0).astype(int)

features["rule_based_flag"] = (
    (features["signal_ato"] + features["signal_velocity"] + features["signal_amount_anomaly"]) > 0
).astype(int)

out_path = ROOT / "data" / "features.csv"
features.to_csv(out_path, index=False)

print(f"users scored: {len(features):,}")
print(f"flagged by rule-based signals: {features['rule_based_flag'].sum():,}")
print(f"  - account takeover signal: {features['signal_ato'].sum():,}")
print(f"  - velocity signal: {features['signal_velocity'].sum():,}")
print(f"  - amount anomaly signal: {features['signal_amount_anomaly'].sum():,}")
print(f"feature table written to: {out_path}")

conn.close()
