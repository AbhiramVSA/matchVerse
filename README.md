## MatchVerse Recommender

End-to-end matchmaking recommender that learns compatibility patterns from user interactions and metadata, trains an XGBoost ranking model, and serves interactive recommendations through Streamlit.

---

### 1. Project Overview

- **Goal**: Recommend 100 high-quality profiles of the opposite gender for every member.
- **Core model**: Gradient boosted decision trees (`xgboost`) trained on positive and synthetic negative interaction pairs.
- **Primary signals**: Demographic alignment (age, caste, state, sect, marital status) and historical interaction patterns.
- **Deliverables**:
  - `train_xgboost_recommender.py`: Training + recommendation generation pipeline.
  - `data/recommendations.csv`: Kaggle-ready submission file.
  - `xgboost_matchmaking_model.json`: Persisted model for inference.
  - `streamlit_app.py`: Demo UI for exploring recommendations.
- `train_bpr_retrieval.py`: Implicit-feedback retrieval model producing top-100 candidate pools.
- `data/recommendations_bpr.csv`: BPR-generated candidate lists for every member.
- `reports/bpr_metrics.json`: Retrieval quality metrics (recall@K, MAP@K).

---

### 2. Data Assets

| File | Purpose |
| --- | --- |
| `data/users.csv` | Master user metadata (Member_ID, gender, age, marital status, sect, caste, state). |
| `data/interactions.csv` | Positive interactions (Member_ID -> Target_ID). |
| `data/recommendations.csv` | Template of member IDs requiring final recommendations. |
| `data/sample_*` | Lightweight samples for quick experiments. |

> All paths are relative to the repository root.

---

### 3. Environment Setup

1. Create and activate a Python 3.13+ virtual environment:
	```powershell
	python -m venv .venv
	.\.venv\Scripts\activate
	```
2. Install dependencies:
	```powershell
	pip install -r requirements.txt
	```
	or, if you prefer `uv`:
	```powershell
	uv pip install -r requirements.txt
	```

> The repository also ships a `pyproject.toml` with higher-version pins. Use the requirements file for reproducibility with the delivered artifacts.

---

### 4. Training Pipeline (`train_xgboost_recommender.py`)

1. **Load data**: `users.csv`, `interactions.csv`, and recommendation template.
2. **Encode metadata**: LabelEncode gender, marital status, sect, caste, state (missing values filled with `Unknown`).
3. **Positive samples**: Iterate over all recorded interactions.
4. **Negative sampling**:
	- Draw opposite-gender pairs not present in interaction history.
	- Maintain a 1:1 balance between positives and negatives.
5. **Feature engineering** (10 features):
	- Member/target age, absolute age difference.
	- Encoded gender, marital status.
	- Binary flags for same state, caste, sect.
6. **Model training**:
	```python
	xgb.XGBClassifier(
		 n_estimators=200,
		 max_depth=8,
		 learning_rate=0.1,
		 subsample=0.8,
		 colsample_bytree=0.8,
		 objective="binary:logistic",
		 random_state=42,
		 n_jobs=-1
	)
	```
	- Split: 80% train / 20% validation (stratified).
	- Monitored metric: log loss (implicit via early stopping). 
7. **Evaluation**:
	- Accuracy: ~0.815.
	- ROC-AUC: ~0.899 (probability a positive pair scores above a negative pair).
8. **Recommendation generation**:
	- For each member in template, gather opposite-gender candidates excluding past interactions.
	- Score using trained model, then apply a 10% popularity penalty to promote diversity.
	- Return top 100 IDs; save to `data/recommendations.csv`.
9. **Persistence**: Save trained model to `xgboost_matchmaking_model.json`.

> Run the full pipeline:
```powershell
python train_xgboost_recommender.py
```

Outputs:
- `data/recommendations.csv`
- `xgboost_matchmaking_model.json`

---

### 5. Retrieval Pipeline (`train_bpr_retrieval.py`)

1. **Load tables**: `users.csv`, `interactions.csv`, and the recommendation template.
2. **Interaction matrix**:
	- Map member IDs to compact integer indexes.
	- Hold out one positive target per user (when available) for evaluation.
	- Build a sparse item-user matrix for implicit feedback training.
3. **Model**: `implicit.bpr.BayesianPersonalizedRanking` with 64 latent factors and 150 iterations.
4. **Evaluation**:
	- Metrics: Recall@10/50/100 and MAP@10/50/100.
	- Outputs stored in `reports/bpr_metrics.json` and logged to stdout.
5. **Candidate generation**:
	- Produce up to 100 opposite-gender profiles per user via `model.recommend`.
	- Backfill from popularity priors if a user lacks sufficient results.
	- Persist to `data/recommendations_bpr.csv` for re-ranking or analysis.

Run the retrieval pipeline:
```powershell
python train_bpr_retrieval.py
```

Outputs:
- `data/recommendations_bpr.csv`
- `reports/bpr_metrics.json`
- `reports/bpr_model.npz` (latent factors for inspection)

---

### 6. Streamlit Demo (`streamlit_app.py`)

- Loads persisted model and cached datasets.
- Sidebar controls:
  - Pick member ID (or random member).
  - Select top-N (10 to 100).
- Displays member summary, ranked matches, and CSV download.
- Explains model signals via an expandable panel.

Launch the app:
```powershell
streamlit run streamlit_app.py
```

---

### 7. File Structure

```
matchVerse/
├── data/
│   ├── users.csv
│   ├── interactions.csv
│   ├── recommendations.csv
│   └── sample_*.csv
├── configs/
│   └── config.yaml
├── train_xgboost_recommender.py
├── streamlit_app.py
├── train_bpr_retrieval.py
├── xgboost_matchmaking_model.json
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

### 8. Key Design Choices

- **Balanced sampling** ensures the model sees both positive and negative signals.
- **Feature simplicity** keeps the model interpretable and fast to train.
- **Popularity penalty** discourages echo chamber recommendations.
- **Separate inference artifact** (JSON model) enables deployment without retraining.

---

### 9. Extending the Project

- Add temporal features (days since last interaction, recency weighting).
- Incorporate behavioral embeddings (e.g., collaborative filtering vectors or sentence-transformer profiles).
- Experiment with alternative ranking models (LightGBM, CatBoost, neural recommenders).
- Integrate A/B testing hooks in Streamlit to capture online feedback.

---

### 10. Troubleshooting

| Issue | Fix |
| --- | --- |
| `xgboost` build errors | Ensure you are on Python 3.12+ and install via `pip install xgboost==2.0.3`. |
| Memory pressure during training | Run on a machine with >=16 GB RAM or downsample negatives. |
| Streamlit UI cannot load model | Confirm `xgboost_matchmaking_model.json` exists (run training script first). |
| Old dependency conflicts | Use the provided virtual environment; avoid mixing with global site packages. |

---

### 11. License & Attribution

The dataset and competition guidelines belong to the original Kaggle organizers. This repository provides modeling and demo code for educational purposes. Please review license terms before distributing or deploying the data externally.
