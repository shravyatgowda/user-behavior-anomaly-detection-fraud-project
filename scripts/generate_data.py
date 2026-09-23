"""
generate_data.py

Generates a synthetic, multi-table user-transaction dataset that mimics the
shape of real trust & safety data: users, devices, login events, and
transactions spread across related tables (~90K+ total rows, 4 tables --
comparable in structure to the 8-10 table joins common in production
risk-analytics warehouses, simplified here for a portfolio project).

Fraud is injected using three realistic patterns rather than random noise:
  1. Account takeover: a brand-new device logs in from a new country and
     immediately fires a high-value transaction.
  2. Velocity abuse: an unusually high number of transactions in a short
     time window on one account.
  3. Amount anomaly: a transaction far outside a user's own historical
     spending pattern.

Ground-truth fraud labels are kept ONLY in a held-out file
(labels_ground_truth.csv) so the analysis notebook can "discover" fraud
blind, then score itself against the truth at the end -- exactly like a
real detection workflow.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N_USERS = 5000
N_DEVICES = 8200
N_LOGINS = 60000
N_TRANSACTIONS = 92000
FRAUD_RATE = 0.025  # ~2.5% of accounts show fraud patterns

CITIES = ["Bengaluru", "Mumbai", "Jakarta", "Manila", "Bangkok", "Ho Chi Minh City",
          "Singapore", "Kuala Lumpur", "Hanoi", "Yangon"]
COUNTRIES = ["IN", "ID", "PH", "TH", "VN", "SG", "MY", "MM"]
MERCHANT_CATS = ["ride_hailing", "food_delivery", "groceries", "electronics",
                  "wallet_topup", "bill_payment", "entertainment"]
DEVICE_TYPES = ["android", "ios", "web"]

START = datetime(2025, 1, 1)
END = datetime(2025, 9, 1)


def random_ts(start=START, end=END, n=1):
    delta = (end - start).total_seconds()
    offsets = rng.random(n) * delta
    return [start + timedelta(seconds=float(o)) for o in offsets]


# ---------------------------------------------------------------- users ----
users = pd.DataFrame({
    "user_id": np.arange(1, N_USERS + 1),
    "signup_date": [d.date() for d in random_ts(START - timedelta(days=365), START, N_USERS)],
    "home_city": rng.choice(CITIES, N_USERS),
    "age_bracket": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], N_USERS,
                               p=[0.22, 0.38, 0.24, 0.11, 0.05]),
})
users["home_country"] = users["home_city"].map(dict(zip(CITIES, COUNTRIES + COUNTRIES[:2])))

# mark which users will exhibit fraud patterns
fraud_users = set(rng.choice(users["user_id"], size=int(N_USERS * FRAUD_RATE), replace=False))

# -------------------------------------------------------------- devices ----
device_user_ids = rng.choice(users["user_id"], N_DEVICES)
devices = pd.DataFrame({
    "device_id": np.arange(1, N_DEVICES + 1),
    "user_id": device_user_ids,
    "device_type": rng.choice(DEVICE_TYPES, N_DEVICES, p=[0.55, 0.35, 0.10]),
    "first_seen_ts": random_ts(START, END, N_DEVICES),
})

# give fraud users one extra "new" device right before their fraud event
extra_rows = []
next_device_id = N_DEVICES + 1
fraud_new_device = {}
for uid in fraud_users:
    fraud_new_device[uid] = next_device_id
    extra_rows.append({
        "device_id": next_device_id,
        "user_id": uid,
        "device_type": rng.choice(DEVICE_TYPES),
        "first_seen_ts": None,  # filled in after we know the fraud event time
    })
    next_device_id += 1
devices_extra = pd.DataFrame(extra_rows)

# ---------------------------------------------------------- login events ----
login_user_ids = rng.choice(users["user_id"], N_LOGINS)
login_device_ids = rng.choice(devices["device_id"], N_LOGINS)
login_ts = random_ts(START, END, N_LOGINS)
login_countries = []
for uid in login_user_ids:
    home = users.loc[users.user_id == uid, "home_country"].values[0]
    # 96% of logins are from the home country, 4% from elsewhere (normal travel/VPN noise)
    login_countries.append(home if rng.random() < 0.96 else rng.choice(COUNTRIES))

logins = pd.DataFrame({
    "login_id": np.arange(1, N_LOGINS + 1),
    "user_id": login_user_ids,
    "device_id": login_device_ids,
    "login_ts": login_ts,
    "ip_country": login_countries,
})

# --------------------------------------------------------- transactions ----
txn_user_ids = rng.choice(users["user_id"], N_TRANSACTIONS)
base_amounts = rng.gamma(shape=2.2, scale=180, size=N_TRANSACTIONS).round(2)  # normal spend distribution
txn_ts = random_ts(START, END, N_TRANSACTIONS)
txn_device_ids = rng.choice(devices["device_id"], N_TRANSACTIONS)
txn_merchant = rng.choice(MERCHANT_CATS, N_TRANSACTIONS)

transactions = pd.DataFrame({
    "transaction_id": np.arange(1, N_TRANSACTIONS + 1),
    "user_id": txn_user_ids,
    "device_id": txn_device_ids,
    "txn_ts": txn_ts,
    "amount": base_amounts,
    "merchant_category": txn_merchant,
})

# inject the three fraud patterns, tracked in ground truth
ground_truth = []

# Pattern 1: account takeover (new device + new country + high value txn)
ato_users = list(fraud_users)[: int(len(fraud_users) * 0.4)]
new_rows = []
for uid in ato_users:
    event_time = pd.Timestamp(random_ts(START + timedelta(days=30), END, 1)[0])
    devices_extra.loc[devices_extra.user_id == uid, "first_seen_ts"] = event_time - timedelta(minutes=3)
    foreign_country = rng.choice([c for c in COUNTRIES
                                   if c != users.loc[users.user_id == uid, "home_country"].values[0]])
    logins = pd.concat([logins, pd.DataFrame([{
        "login_id": logins.login_id.max() + 1,
        "user_id": uid, "device_id": fraud_new_device[uid],
        "login_ts": event_time - timedelta(minutes=2), "ip_country": foreign_country,
    }])], ignore_index=True)
    high_amount = float(rng.uniform(2500, 9000))
    new_rows.append({
        "transaction_id": transactions.transaction_id.max() + len(new_rows) + 1,
        "user_id": uid, "device_id": fraud_new_device[uid],
        "txn_ts": event_time, "amount": round(high_amount, 2),
        "merchant_category": rng.choice(["electronics", "wallet_topup"]),
    })
    ground_truth.append({"user_id": uid, "pattern": "account_takeover", "event_ts": event_time})

# Pattern 2: velocity abuse (many rapid transactions in <30 min window)
velocity_users = list(fraud_users)[int(len(fraud_users) * 0.4): int(len(fraud_users) * 0.7)]
for uid in velocity_users:
    window_start = pd.Timestamp(random_ts(START + timedelta(days=30), END, 1)[0])
    burst_size = int(rng.integers(8, 15))
    dev = int(rng.choice(devices.loc[devices.user_id == uid, "device_id"].values
                          if (devices.user_id == uid).any() else devices.device_id.values))
    for i in range(burst_size):
        new_rows.append({
            "transaction_id": transactions.transaction_id.max() + len(new_rows) + 1,
            "user_id": uid, "device_id": dev,
            "txn_ts": window_start + timedelta(minutes=int(rng.integers(0, 25))),
            "amount": round(float(rng.uniform(50, 400)), 2),
            "merchant_category": rng.choice(MERCHANT_CATS),
        })
    ground_truth.append({"user_id": uid, "pattern": "velocity_abuse", "event_ts": window_start})

# Pattern 3: amount anomaly (way outside the user's own historical average)
amount_users = list(fraud_users)[int(len(fraud_users) * 0.7):]
for uid in amount_users:
    event_time = pd.Timestamp(random_ts(START + timedelta(days=30), END, 1)[0])
    dev = int(rng.choice(devices.loc[devices.user_id == uid, "device_id"].values
                          if (devices.user_id == uid).any() else devices.device_id.values))
    spike_amount = float(rng.uniform(3000, 12000))
    new_rows.append({
        "transaction_id": transactions.transaction_id.max() + len(new_rows) + 1,
        "user_id": uid, "device_id": dev,
        "txn_ts": event_time, "amount": round(spike_amount, 2),
        "merchant_category": rng.choice(["electronics", "wallet_topup"]),
    })
    ground_truth.append({"user_id": uid, "pattern": "amount_anomaly", "event_ts": event_time})

transactions = pd.concat([transactions, pd.DataFrame(new_rows)], ignore_index=True)
devices = pd.concat([devices, devices_extra], ignore_index=True)
devices["first_seen_ts"] = pd.to_datetime(devices["first_seen_ts"])

ground_truth_df = pd.DataFrame(ground_truth)

# ------------------------------------------------------------- write out ----
users.to_csv("/home/claude/fraud-project/data/users.csv", index=False)
devices.to_csv("/home/claude/fraud-project/data/devices.csv", index=False)
logins.to_csv("/home/claude/fraud-project/data/login_events.csv", index=False)
transactions.to_csv("/home/claude/fraud-project/data/transactions.csv", index=False)
ground_truth_df.to_csv("/home/claude/fraud-project/data/labels_ground_truth.csv", index=False)

print(f"users: {len(users):,}")
print(f"devices: {len(devices):,}")
print(f"login_events: {len(logins):,}")
print(f"transactions: {len(transactions):,}")
print(f"fraud_users (ground truth): {ground_truth_df.user_id.nunique():,} "
      f"({ground_truth_df.user_id.nunique() / N_USERS:.2%} of users)")
