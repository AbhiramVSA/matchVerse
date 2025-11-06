"""MatchVerse implicit-feedback retrieval using Bayesian Personalised Ranking (BPR).

This script fits a collaborative-filtering retrieval model that complements the
existing gradient-boosted ranker. It trains on the implicit interaction matrix,
computes standard top-K retrieval metrics, and writes 100-profile candidate
lists for downstream re-ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

try:
    from implicit.bpr import BayesianPersonalizedRanking
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'implicit'. Install requirements.txt before running this script."
    ) from exc


RANDOM_STATE = 42
TOP_N = 100
METRIC_CUTOFFS = (10, 50, 100)
MODEL_PATH = Path("reports/bpr_model.npz")  # stores factors for inspection if desired
METRICS_PATH = Path("reports/bpr_metrics.json")
OUTPUT_PATH = Path("data/recommendations_bpr.csv")


def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv("data/users.csv")
    interactions = pd.read_csv("data/interactions.csv")
    template = pd.read_csv("data/recommendations.csv")
    interactions = interactions[["Member_ID", "Target_ID"]].dropna()
    interactions["Member_ID"] = interactions["Member_ID"].astype(int)
    interactions["Target_ID"] = interactions["Target_ID"].astype(int)
    return users, interactions, template


def build_mappings(users: pd.DataFrame, interactions: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int], np.ndarray]:
    user_ids = interactions["Member_ID"].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    all_member_ids = users["Member_ID"].unique()
    item_to_idx = {mid: idx for idx, mid in enumerate(all_member_ids)}
    idx_to_item = np.array(all_member_ids)
    return user_to_idx, item_to_idx, idx_to_item


def split_train_test(
    interactions: pd.DataFrame, user_to_idx: Dict[int, int]
) -> Tuple[List[Tuple[int, int]], Dict[int, List[int]]]:
    rng = np.random.default_rng(RANDOM_STATE)
    positives = interactions.groupby("Member_ID")["Target_ID"].apply(list)

    train_pairs: List[Tuple[int, int]] = []
    eval_holdout: Dict[int, List[int]] = {}

    for user_id, items in positives.items():
        rng.shuffle(items)
        mapped_user = user_to_idx.get(user_id)
        if mapped_user is None:
            continue
        if len(items) >= 2:
            eval_holdout[user_id] = items[:1]
            remaining = items[1:]
            if not remaining:
                # users with only the eval item cannot contribute to training
                continue
        else:
            remaining = items
        for item_id in remaining:
            train_pairs.append((user_id, item_id))
    return train_pairs, eval_holdout


def build_sparse_matrix(
    train_pairs: Iterable[Tuple[int, int]],
    user_to_idx: Dict[int, int],
    item_to_idx: Dict[int, int],
) -> sparse.csr_matrix:
    rows, cols, data = [], [], []
    for user_id, item_id in train_pairs:
        uidx = user_to_idx.get(user_id)
        iidx = item_to_idx.get(item_id)
        if uidx is None or iidx is None:
            continue
        rows.append(iidx)
        cols.append(uidx)
        data.append(1.0)

    if not rows:
        raise ValueError("No training pairs available. Check interaction data.")

    matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(item_to_idx), len(user_to_idx)),
        dtype=np.float32,
    )
    return matrix.tocsr()


def train_bpr_model(item_user: sparse.csr_matrix) -> BayesianPersonalizedRanking:
    model = BayesianPersonalizedRanking(
        factors=64,
        learning_rate=0.05,
        regularization=0.002,
        iterations=150,
        num_threads=0,
        random_state=RANDOM_STATE,
    )
    model.fit(item_user, show_progress=True)
    return model


def gender_compatible(
    source_gender: str, candidate_gender: str, member_id: int, candidate_id: int
) -> bool:
    if member_id == candidate_id:
        return False
    if source_gender in {"Male", "Female"} and candidate_gender in {"Male", "Female"}:
        return source_gender != candidate_gender
    return True


def recommend_for_user(
    model: BayesianPersonalizedRanking,
    user_idx: int,
    user_items_matrix: sparse.csr_matrix,
    idx_to_item: np.ndarray,
    member_gender: Dict[int, str],
    popularity: Dict[int, float],
    user_id: int,
) -> List[int]:
    user_row = user_items_matrix.getrow(user_idx)
    item_indices, scores = model.recommend(
        user_idx,
        None,
        N=TOP_N * 3,
        filter_already_liked_items=False,
        recalculate_user=False,
    )

    user_gender = member_gender.get(user_id, "Unknown")
    interacted_items = set(user_row.indices)
    seen_member_ids = {int(idx_to_item[idx]) for idx in interacted_items}
    filtered: List[int] = []
    for item_idx, _ in zip(item_indices, scores):
        candidate_id = int(idx_to_item[item_idx])
        candidate_gender = member_gender.get(candidate_id, "Unknown")
        if item_idx in interacted_items:
            continue
        if not gender_compatible(user_gender, candidate_gender, user_id, candidate_id):
            continue
        filtered.append(candidate_id)
        if len(filtered) >= TOP_N:
            break

    if len(filtered) < TOP_N:
        fallback_pool = [cid for cid, _ in sorted(popularity.items(), key=lambda x: -x[1])]
        for cid in fallback_pool:
            candidate_gender = member_gender.get(cid, "Unknown")
            if not gender_compatible(user_gender, candidate_gender, user_id, cid):
                continue
            if cid in seen_member_ids:
                continue
            if cid == user_id:
                continue
            if cid in filtered:
                continue
            filtered.append(cid)
            if len(filtered) >= TOP_N:
                break
    return filtered[:TOP_N]


def evaluate_model(
    model: BayesianPersonalizedRanking,
    eval_holdout: Dict[int, List[int]],
    user_to_idx: Dict[int, int],
    user_items_matrix: sparse.csr_matrix,
    idx_to_item: np.ndarray,
    member_gender: Dict[int, str],
    popularity: Dict[int, float],
) -> Dict[str, Dict[str, float]]:
    results = {str(k): {"recall": [], "map": []} for k in METRIC_CUTOFFS}
    evaluated_users = 0

    for user_id, test_items in tqdm(eval_holdout.items(), desc="eval"):
        user_idx = user_to_idx.get(user_id)
        if user_idx is None:
            continue
        recommended = recommend_for_user(
            model,
            user_idx,
            user_items_matrix,
            idx_to_item,
            member_gender,
            popularity,
            user_id,
        )
        if not recommended:
            continue
        evaluated_users += 1
        positives = set(test_items)
        for cutoff in METRIC_CUTOFFS:
            top_k = recommended[:cutoff]
            recall_value = int(bool(positives & set(top_k)))
            reciprocal_ranks = [1.0 / (top_k.index(item) + 1) for item in positives if item in top_k]
            average_precision = sum(reciprocal_ranks) / max(len(positives), 1)
            results[str(cutoff)]["recall"].append(recall_value)
            results[str(cutoff)]["map"].append(average_precision)

    for cutoff in METRIC_CUTOFFS:
        key = str(cutoff)
        recalls = results[key]["recall"]
        maps = results[key]["map"]
        results[key]["recall"] = float(np.mean(recalls)) if recalls else 0.0
        results[key]["map"] = float(np.mean(maps)) if maps else 0.0

    print(f"evaluated users: {evaluated_users}")
    for cutoff in METRIC_CUTOFFS:
        metrics = results[str(cutoff)]
        print(
            f"@{cutoff}: recall={metrics['recall']:.4f} | MAP={metrics['map']:.4f}"
        )
    return results


def build_popularity(interactions: pd.DataFrame, idx_to_item: np.ndarray) -> Dict[int, float]:
    counts = interactions["Target_ID"].value_counts()
    popularity = {int(k): float(v) for k, v in counts.items()}
    for member_id in idx_to_item:
        popularity.setdefault(int(member_id), 0.0)
    return popularity


def generate_recommendations(
    model: BayesianPersonalizedRanking,
    template: pd.DataFrame,
    user_to_idx: Dict[int, int],
    user_items_matrix: sparse.csr_matrix,
    idx_to_item: np.ndarray,
    member_gender: Dict[int, str],
    popularity: Dict[int, float],
) -> pd.DataFrame:
    rows = []
    for uid in tqdm(template["Member_ID"].astype(int), desc="candidates"):
        user_idx = user_to_idx.get(uid)
        if user_idx is None:
            ranked = []
        else:
            ranked = recommend_for_user(
                model,
                user_idx,
                user_items_matrix,
                idx_to_item,
                member_gender,
                popularity,
                uid,
            )
        rows.append(
            {
                "ID": uid,
                "Member_ID": uid,
                "top_100_profiles": ",".join(map(str, ranked)) if ranked else "",
            }
        )
    return pd.DataFrame(rows)


def persist_model_factors(model: BayesianPersonalizedRanking, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        path,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
    )


def main() -> None:
    users, interactions, template = load_tables()
    member_gender = users.set_index("Member_ID")["Gender"].to_dict()

    user_to_idx, item_to_idx, idx_to_item = build_mappings(users, interactions)
    train_pairs, eval_holdout = split_train_test(interactions, user_to_idx)
    item_user = build_sparse_matrix(train_pairs, user_to_idx, item_to_idx)
    user_items = item_user.T.tocsr()

    print(f"training pairs: {item_user.nnz} | users: {item_user.shape[1]} | items: {item_user.shape[0]}")
    model = train_bpr_model(item_user)

    popularity = build_popularity(interactions, idx_to_item)
    metrics = evaluate_model(
        model,
        eval_holdout,
        user_to_idx,
        user_items,
        idx_to_item,
        member_gender,
        popularity,
    )

    recommendations = generate_recommendations(
        model,
        template,
        user_to_idx,
        user_items,
        idx_to_item,
        member_gender,
        popularity,
    )
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    recommendations.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote retrieval recommendations -> {OUTPUT_PATH}")

    METRICS_PATH.parent.mkdir(exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump({"cutoffs": metrics}, fp, indent=2)
    print(f"saved metrics -> {METRICS_PATH}")

    persist_model_factors(model, MODEL_PATH)
    print(f"persisted latent factors -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
