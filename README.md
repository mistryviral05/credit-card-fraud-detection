# Credit Card Fraud Detection Using Machine Learning

A college ML project that detects fraudulent credit card transactions with **XGBoost**.
It has a full training pipeline, an analysis notebook and **FraudShield**, an agent-style CLI
that checks a transaction and explains its decision.

Indian card networks are supported (RuPay, Visa, Mastercard, Amex, Diners) and amounts are in rupees (₹).

## Project structure

```
├── train.py                  # full pipeline: data → features → models → metrics → joblib
├── cli.py                    # FraudShield agent CLI (predict, metrics, demo)
├── notebooks/
│   └── analysis.ipynb        # EDA, feature engineering, model comparison, graphs
├── models/
│   └── fraud_model.joblib    # trained pipeline + threshold + metrics
├── reports/                  # graphs (PNG), metrics.json, metrics.md
├── data/                     # dataset CSVs (downloaded automatically, not committed)
└── pyproject.toml            # dependencies (managed with uv)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

## Usage

| Task | Command |
| --- | --- |
| Train the model | `uv run train.py` |
| Quick training run (20% data) | `uv run train.py --sample 0.2` |
| Check a transaction | `uv run cli.py` |
| Show model metrics | `uv run cli.py metrics` |
| Run sample transactions | `uv run cli.py demo` |
| Open the notebook | `uv run jupyter lab notebooks/analysis.ipynb` |

The dataset (about 200 MB) is downloaded from Kaggle on the first run. If the download fails,
get it from [kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
and put `fraudTrain.csv` and `fraudTest.csv` in `data/`.

## Dataset

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) by Kartik Shenoy.
It is simulated with the Sparkov data generator.

| File | Rows | Use |
| --- | --- | --- |
| `fraudTrain.csv` | 1,296,675 | Training, validation, cross validation |
| `fraudTest.csv` | 555,719 | Final test (never seen during training) |

Fraud is only about 0.5% of all rows, so the data is highly imbalanced.

## Features

| Feature | Description |
| --- | --- |
| `amount_inr` | Transaction amount in INR (USD × 83) |
| `hour`, `is_night` | Hour of the transaction, and whether it is between 22:00 and 03:59 |
| `day_of_week` | Day of the week |
| `age` | Cardholder age at the time of the transaction |
| `txn_count_24h` | Card's transactions in the previous 24 hours |
| `hours_since_last` | Hours since the card's previous transaction |
| `amt_to_avg_ratio` | Amount divided by the card's average past amount |
| `category` | Merchant category (one-hot encoded) |

History features only use earlier transactions of the same card, so there is no data leakage.
Location columns were dropped because they show no difference between fraud and normal rows in this dataset.

## Training pipeline (`train.py`)

1. Download the dataset if needed and engineer features.
2. Train Logistic Regression, Decision Tree, Random Forest and XGBoost on an 80/20 split of the training data.
3. Pick the best model by PR-AUC on the validation split.
4. Tune the decision threshold for the best F1 score.
5. Run 5-fold stratified cross validation.
6. Retrain on all training data and evaluate on the test file.
7. Save graphs, `metrics.json` and `metrics.md` to `reports/`.
8. Save the pipeline, threshold and metrics to `models/fraud_model.joblib` with joblib.

Class imbalance is handled with `class_weight="balanced"` and XGBoost's `scale_pos_weight`.

## Evaluation

| Metric | Meaning |
| --- | --- |
| Accuracy | Share of all predictions that are correct |
| Precision | Of the transactions flagged as fraud, how many really are fraud |
| Recall | Of all real fraud, how much the model catches |
| F1 Score | Balance of precision and recall |
| Specificity | Of all normal transactions, how many are correctly passed |
| ROC-AUC | How well the model separates fraud from normal |
| PR-AUC | Precision-recall area, the best single score for imbalanced data |

A model that always predicts "normal" gets about 99.5% accuracy but catches no fraud,
so recall, precision and PR-AUC matter most here.

Graphs saved in `reports/`:

| Graph | File |
| --- | --- |
| Confusion matrix | `confusion_matrix.png` |
| ROC curve | `roc_curve.png` |
| Precision-recall curve | `pr_curve.png` |
| Feature importance | `feature_importance.png` |
| Decision tree (depth 4) | `decision_tree.png` |
| Model comparison | `model_comparison.png` |
| Threshold tuning | `threshold_tuning.png` |

Full results are in [reports/metrics.md](reports/metrics.md).

## FraudShield CLI (`cli.py`)

The CLI asks for the transaction details step by step:

- Card number, validated with the Luhn check, network detected and masked on screen
- Amount (₹), merchant category, date and time
- Cardholder date of birth
- Card activity: usual spend, transactions in the last 24 hours, hours since the previous one

It then runs the model and shows the fraud probability, a risk meter, a LOW / MEDIUM / HIGH verdict,
the top factors behind the decision (XGBoost SHAP contributions) and a recommended action.

The card number is only validated and is never used as a model feature.

## Limitations

- The dataset is simulated and based on US transactions. Amounts are converted to INR, but the patterns are not from real Indian data.
- The model is for learning purposes and is not a production fraud system.

## License

See [LICENSE](../LICENSE).
