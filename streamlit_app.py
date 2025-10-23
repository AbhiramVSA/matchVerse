"""MatchVerse recommender demo (Streamlit UI)."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data")
MODEL_PATH = Path("xgboost_matchmaking_model.json")
FEATURES = [
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
CAT_COLS = ["Gender", "Marital_Status", "Sect", "Caste", "State"]

st.set_page_config(page_title="MatchVerse Demo", page_icon="", layout="wide")


@st.cache_data(show_spinner=False)
def load_users():
    df = pd.read_csv(DATA_DIR / "users.csv")
    for c in CAT_COLS:
        enc = LabelEncoder()
        df[c] = df[c].fillna("Unknown")
        df[f"{c}_encoded"] = enc.fit_transform(df[c])
    lookup = df.set_index("Member_ID").to_dict("index")
    lookup = {mid: {str(k): v for k, v in feats.items()} for mid, feats in lookup.items()}
    return df, lookup


@st.cache_data(show_spinner=False)
def load_interactions():
    df = pd.read_csv(DATA_DIR / "interactions.csv")
    hist = df.groupby("Member_ID")["Target_ID"].apply(lambda x: set(x)).to_dict()
    for a, b in zip(df["Target_ID"], df["Member_ID"]):
        hist.setdefault(a, set()).add(b)
    pop = df["Target_ID"].value_counts().to_dict()
    return df, hist, pop


@st.cache_data(show_spinner=False)
def load_template():
    df = pd.read_csv(DATA_DIR / "recommendations.csv")
    if "top_100_profiles" in df.columns:
        df = df.loc[:, ["ID", "Member_ID"]]
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    m = xgb.XGBClassifier()
    m.load_model(MODEL_PATH)
    return m


def get_pool(uid, users, lookup, hist):
    u = lookup.get(uid)
    if not u:
        return []
    g = u.get("Gender")
    if g == "Male":
        pool = users[users["Gender"] == "Female"]["Member_ID"].tolist()
    elif g == "Female":
        pool = users[users["Gender"] == "Male"]["Member_ID"].tolist()
    else:
        pool = users[users["Gender"] != g]["Member_ID"].tolist()
    blocked = hist.get(uid, set())
    return [c for c in pool if c != uid and c not in blocked]


def build_features(uid, pool, lookup):
    u = lookup[uid]
    rows = []
    ids = []
    for v in pool:
        vf = lookup.get(v)
        if not vf:
            continue
        rows.append([
            u.get("Age", 0),
            vf.get("Age", 0),
            abs(u.get("Age", 0) - vf.get("Age", 0)),
            u.get("Gender_encoded", 0),
            vf.get("Gender_encoded", 0),
            int(u.get("State_encoded") == vf.get("State_encoded")),
            int(u.get("Caste_encoded") == vf.get("Caste_encoded")),
            int(u.get("Sect_encoded") == vf.get("Sect_encoded")),
            u.get("Marital_Status_encoded", 0),
            vf.get("Marital_Status_encoded", 0),
        ])
        ids.append(v)
    return pd.DataFrame(rows, columns=FEATURES), ids


def score(ids, feats, model, pop):
    if feats.empty:
        return pd.Series(dtype=float)
    sc = model.predict_proba(feats)[:, 1]
    pcnt = np.array([pop.get(i, 0) for i in ids], dtype=float)
    pen = pcnt / (pcnt.max() + 1) if pcnt.size and pcnt.max() > 0 else np.zeros_like(sc)
    return pd.Series(sc * (1 - 0.1 * pen), index=ids)


def recommend(uid, top_k, users, lookup, hist, pop, model):
    pool = get_pool(uid, users, lookup, hist)
    if not pool:
        return pd.DataFrame(columns=["Member_ID", "Score"])
    feats, ids = build_features(uid, pool, lookup)
    scores = score(ids, feats, model, pop)
    if scores.empty:
        return pd.DataFrame(columns=["Member_ID", "Score"])
    top = scores.sort_values(ascending=False).head(top_k)
    recs = users[users["Member_ID"].isin(top.index)].copy()
    recs = recs.merge(top.rename("Score"), left_on="Member_ID", right_index=True)
    recs.sort_values(by="Score", ascending=False, inplace=True)
    return recs


st.title("💘 MatchVerse Recommender")
st.caption("Interactive demo: pick a profile, see top matches ranked by the XGBoost model.")

users, lookup = load_users()
_, hist, pop = load_interactions()
template = load_template()
model = load_model()

st.sidebar.header("Settings")
ids = template["Member_ID"].astype(int).tolist()
sel = st.sidebar.selectbox("Member ID", options=ids, format_func=lambda x: f"Member {x}")
if st.sidebar.button("Random"):
    sel = int(np.random.choice(ids))
k = st.sidebar.slider("Top N", min_value=10, max_value=100, value=20, step=5)

info = users[users["Member_ID"].astype(int) == sel]
if info.empty:
    st.error("Member not found in dataset.")
    st.stop()

cols = st.columns(5)
cols[0].metric("ID", sel)
cols[1].metric("Gender", info.iloc[0]["Gender"])
cols[2].metric("Age", int(info.iloc[0]["Age"]))
cols[3].metric("State", info.iloc[0]["State"])
cols[4].metric("Caste", info.iloc[0]["Caste"])

with st.spinner("scoring..."):
    recs = recommend(sel, k, users, lookup, hist, pop, model)

if recs.empty:
    st.warning("No matches found.")
    st.stop()

st.subheader("Top matches")
st.write("Ranked by model score with diversity penalty.")

display = ["Member_ID", "Score", "Gender", "Age", "Marital_Status", "Sect", "Caste", "State"]
st.dataframe(recs[display].reset_index(drop=True), use_container_width=True, hide_index=True)

st.download_button(
    label="Download CSV",
    data=recs[display].to_csv(index=False),
    file_name=f"member_{sel}_top_{k}.csv",
    mime="text/csv",
)

st.markdown("---")
with st.expander("How it works"):
    st.write("Model uses age, location, caste, sect compatibility + interaction history.")
    st.write("Diversity penalty reduces over-popular profiles slightly.")
