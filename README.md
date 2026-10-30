# MatchVerse Recommender

An experimental profile recommender with an XGBoost ranker, a BPR retrieval model, and a Streamlit demo.

## Project map

| Path | Role |
|---|---|
| `train_xgboost_recommender.py` | Train an XGBoost classifier and produce ranked profile IDs. |
| `train_bpr_retrieval.py` | Train an implicit-feedback BPR model and produce candidate lists. |
| `streamlit_app.py` | Score and display profile recommendations. |
| `configs/config.yaml` | Experimental settings for a two-tower model, retrieval, and reranker. |
| `main.py` | Placeholder entry point that prints a greeting. |
| `reports/report.pdf` | Earlier project report. |

The two scripts use `data/users.csv`, `data/interactions.csv`, and `data/recommendations.csv`. The Streamlit app also needs `xgboost_matchmaking_model.json`.

The XGBoost script reads `data/recommendations.csv` as its template. It writes the result to that same path.

Back up the template before you run the script.

## Data and privacy

The public checkout includes four small `data/sample_*.csv` files. It does not include the full user tables, interaction history, or a trained model. The scripts do not read the sample files by default.

| Required input | Columns used by the scripts |
|---|---|
| `data/users.csv` | `Member_ID`, `Gender`, `Age`, `Marital_Status`, `Sect`, `Caste`, `State` |
| `data/interactions.csv` | `Member_ID`, `Target_ID` |
| `data/recommendations.csv` | `ID`, `Member_ID` |

The model code uses age, gender, marital status, sect, caste, state, and interaction history. Keep private user data and generated models out of public commits. Check the data source terms before you use or share any source data.

## Setup

Use Python 3.13 or newer. The `pyproject.toml` and `uv.lock` define the current environment:

```sh
uv sync --locked
```

`requirements.txt` pins a different, older dependency set. Use one dependency path at a time.

Place the required input tables in `data/`, then train the XGBoost model:

```sh
uv run python train_xgboost_recommender.py
```

The script writes `xgboost_matchmaking_model.json` and `data/recommendations.csv`. The BPR script writes `data/recommendations_bpr.csv` and `reports/bpr_metrics.json`:

```sh
uv run python train_bpr_retrieval.py
```

Start the demo after you create the input tables and XGBoost model:

```sh
uv run streamlit run streamlit_app.py
```

## Model notes

The XGBoost pipeline uses ten pair features: ages, age difference, gender codes, same-state, same-caste, same-sect, and marital-status codes. It samples non-interacted pairs as negatives and uses a stratified 80/20 train/validation split. It scores up to 100 candidates and applies a small popularity penalty.

The BPR model uses 64 latent factors and 150 iterations. It holds out one interaction per eligible user for evaluation and reports Recall and MAP at 10, 50, and 100. It can use popular profiles to fill a short candidate list.

The Streamlit demo lets you select a member or choose one at random. Set a result count from 10 to 100, inspect the ranked profiles, and download a CSV file.

The report in `reports/report.pdf` records 81.55% accuracy and 0.899 ROC-AUC for an earlier validation run with sampled negative pairs. Those metrics do not measure real-world match quality. The public sample files do not reproduce the reported results.

`configs/config.yaml` describes an experimental two-tower model, retrieval, and reranker. This checkout has no script that runs these settings.

## License and attribution

> The dataset and competition guidelines belong to the original Kaggle organizers. This repository provides modeling and demo code for educational purposes. Please review license terms before distributing or deploying the data externally.
