"""MatchVerse XGBoost recommender (training + generation).
Simple, readable script intended for demo / reproducibility.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("loading data...")
users = pd.read_csv("data/users.csv")
interactions = pd.read_csv("data/interactions.csv")
recommendations_template = pd.read_csv("data/recommendations.csv")

print(f"users: {len(users)} | interactions: {len(interactions)} | targets: {len(recommendations_template)}")

# encode categoricals
label_encoders = {}
cat_cols = ["Gender", "Marital_Status", "Sect", "Caste", "State"]
for c in cat_cols:
    enc = LabelEncoder()
    users[c] = users[c].fillna("Unknown")
    users[f"{c}_encoded"] = enc.fit_transform(users[c])
    label_encoders[c] = enc

user_features = users.set_index("Member_ID").to_dict("index")

print("building training samples...")

# Create positive samples from interactions
pos_rows = []
for _, r in tqdm(interactions.iterrows(), total=len(interactions), desc="interactions"):
    a = r["Member_ID"]
    b = r["Target_ID"]
    if a in user_features and b in user_features:
        ua = user_features[a]
        ub = user_features[b]
        pos_rows.append({
            "member_age": ua["Age"],
            "target_age": ub["Age"],
            "age_diff": abs(ua["Age"] - ub["Age"]),
            "member_gender": ua["Gender_encoded"],
            "target_gender": ub["Gender_encoded"],
            "same_state": int(ua["State_encoded"] == ub["State_encoded"]),
            "same_caste": int(ua["Caste_encoded"] == ub["Caste_encoded"]),
            "same_sect": int(ua["Sect_encoded"] == ub["Sect_encoded"]),
            "member_marital": ua["Marital_Status_encoded"],
            "target_marital": ub["Marital_Status_encoded"],
            "label": 1,
        })

print(f"generated {len(pos_rows)} positive rows")

# negative sampling: random opposite-gender pairs not in interactions
males = users[users["Gender"] == "Male"]["Member_ID"].to_numpy()
females = users[users["Gender"] == "Female"]["Member_ID"].to_numpy()
seen = set()
for _, r in interactions.iterrows():
    seen.add((r["Member_ID"], r["Target_ID"]))
    seen.add((r["Target_ID"], r["Member_ID"]))

neg_rows = []
np.random.seed(42)
needed = len(pos_rows)
for _ in tqdm(range(needed), desc="neg samples"):
    if np.random.random() > 0.5:
        a = int(np.random.choice(males, size=1)[0])
        b = int(np.random.choice(females, size=1)[0])
    else:
        a = int(np.random.choice(females, size=1)[0])
        b = int(np.random.choice(males, size=1)[0])
    if (a, b) in seen:
        continue
    if a in user_features and b in user_features:
        ua = user_features[a]
        ub = user_features[b]
        neg_rows.append({
            "member_age": ua["Age"],
            "target_age": ub["Age"],
            "age_diff": abs(ua["Age"] - ub["Age"]),
            "member_gender": ua["Gender_encoded"],
            "target_gender": ub["Gender_encoded"],
            "same_state": int(ua["State_encoded"] == ub["State_encoded"]),
            "same_caste": int(ua["Caste_encoded"] == ub["Caste_encoded"]),
            "same_sect": int(ua["Sect_encoded"] == ub["Sect_encoded"]),
            "member_marital": ua["Marital_Status_encoded"],
            "target_marital": ub["Marital_Status_encoded"],
            "label": 0,
        })
print(f"generated {len(neg_rows)} negative rows")

# Combine and create training dataset
all_rows = pos_rows + neg_rows
df_train = pd.DataFrame(all_rows)

# Prepare features and labels
features = [
    "member_age",
    "target_age",
    "age_diff",
    "member_gender",
    "target_gender",
    "same_state",
    "same_caste",
    "same_sect",
    "member_marital",
    "target_marital",
]

X = df_train[features]
y = df_train["label"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"train: {len(X_train)} | val: {len(X_val)}")
print("training model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)

from sklearn.metrics import accuracy_score, roc_auc_score
pred = model.predict(X_val)
pred_proba = model.predict_proba(X_val)[:, 1]
print(f"acc: {accuracy_score(y_val, pred):.4f} | auc: {roc_auc_score(y_val, pred_proba):.4f}")

print("creating recommendations...")

# Get members who need recommendations
targets = recommendations_template["Member_ID"].values
pop_count = interactions["Target_ID"].value_counts().to_dict()
results = []

for uid in tqdm(targets, desc="recommend"):
    if uid not in user_features:
        results.append({"ID": uid, "Member_ID": uid, "top_100_profiles": ""})
        continue
    u = user_features[uid]
    gender = u["Gender"]
    if gender == "Male":
        pool = users[users["Gender"] == "Female"]["Member_ID"].values
    else:
        pool = users[users["Gender"] == "Male"]["Member_ID"].values

    seen_targets = set(interactions[interactions["Member_ID"] == uid]["Target_ID"].values)
    pool = [c for c in pool if c not in seen_targets and c != uid]

    feats = []
    ids = []
    for v in pool:
        if v not in user_features:
            continue
        vfeat = user_features[v]
        feats.append([
            u["Age"],
            vfeat["Age"],
            abs(u["Age"] - vfeat["Age"]),
            u["Gender_encoded"],
            vfeat["Gender_encoded"],
            int(u["State_encoded"] == vfeat["State_encoded"]),
            int(u["Caste_encoded"] == vfeat["Caste_encoded"]),
            int(u["Sect_encoded"] == vfeat["Sect_encoded"]),
            u["Marital_Status_encoded"],
            vfeat["Marital_Status_encoded"],
        ])
        ids.append(v)

    if not feats:
        results.append({"ID": uid, "Member_ID": uid, "top_100_profiles": ""})
        continue

    Xc = pd.DataFrame(feats, columns=features)
    sc = model.predict_proba(Xc)[:, 1]
    pop = np.array([pop_count.get(i, 0) for i in ids])
    pop = pop / (pop.max() + 1)
    sc2 = sc * (1 - 0.1 * pop)

    top = np.argsort(sc2)[::-1][:100]
    top_ids = [ids[i] for i in top]
    while len(top_ids) < 100 and len(ids) > len(top_ids):
        rem = [c for c in ids if c not in top_ids]
        top_ids.extend(rem[: 100 - len(top_ids)])
    results.append({"ID": uid, "Member_ID": uid, "top_100_profiles": ",".join(map(str, top_ids[:100]))})

# Create final dataframe and save
df_final = pd.DataFrame(results)
df_final.to_csv("data/recommendations.csv", index=False)
print(f"wrote {len(df_final)} recommendations to data/recommendations.csv")
model.save_model("xgboost_matchmaking_model.json")
print("saved model -> xgboost_matchmaking_model.json")
