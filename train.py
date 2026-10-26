"""Credit card fraud detection: full training pipeline.

Downloads the dataset, engineers features, compares four models, tunes the
decision threshold, cross validates the best model, saves metrics and graphs
to reports/ and the final pipeline to models/fraud_model.joblib.

Usage:
    uv run train.py                 # full run
    uv run train.py --sample 0.2    # quick run on 20% of the training data
    uv run train.py --cv-folds 3    # fewer cross validation folds
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import date, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
REPORTS_DIR = ROOT / "reports"

KAGGLE_DATASET = "kartik2112/fraud-detection"
TRAIN_CSV = "fraudTrain.csv"
TEST_CSV = "fraudTest.csv"
# Location columns (lat, long, merch_lat, merch_long, city_pop) are left out
# on purpose: in this dataset they look the same for fraud and normal rows.
RAW_COLUMNS = ["trans_date_trans_time", "cc_num", "category", "amt", "dob", "is_fraud"]

USD_TO_INR = 83.0
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}
MAX_GAP_HOURS = 720.0  # cap "hours since last transaction" at 30 days
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42

NUM_FEATURES = [
    "amount_inr", "hour", "is_night", "day_of_week", "age",
    "txn_count_24h", "hours_since_last", "amt_to_avg_ratio",
]
CAT_FEATURES = ["category"]
FEATURES = NUM_FEATURES + CAT_FEATURES

FEATURE_LABELS = {
    "amount_inr": "Amount (INR)",
    "hour": "Hour of day",
    "is_night": "Night transaction",
    "day_of_week": "Day of week",
    "age": "Cardholder age",
    "txn_count_24h": "Transactions in last 24h",
    "hours_since_last": "Hours since last transaction",
    "amt_to_avg_ratio": "Amount vs usual spend",
    "category": "Merchant category",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "specificity": "Specificity",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}

console = Console()
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------- features --

def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw dataset rows into model features (plus the is_fraud label).

    History features (count in last 24h, gap, usual spend) are computed per
    card in time order, using only transactions before the current one.
    The original row index is kept.
    """
    df = raw.copy()
    df["trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "trans_time"])
    cards = df.groupby("cc_num")

    gap_hours = cards["trans_time"].diff().dt.total_seconds() / 3600
    prev_count = cards.cumcount()
    prev_avg = (cards["amt"].cumsum() - df["amt"]) / prev_count.replace(0, np.nan)
    count_24h = (
        df.set_index("trans_time").groupby("cc_num")["amt"].rolling("24h").count().to_numpy() - 1
    )
    hour = df["trans_time"].dt.hour

    out = pd.DataFrame({
        "amount_inr": df["amt"] * USD_TO_INR,
        "hour": hour,
        "is_night": hour.isin(NIGHT_HOURS).astype(int),
        "day_of_week": df["trans_time"].dt.day_of_week,
        "age": (df["trans_time"] - pd.to_datetime(df["dob"])).dt.days / 365.25,
        "txn_count_24h": count_24h,
        "hours_since_last": gap_hours.fillna(MAX_GAP_HOURS).clip(upper=MAX_GAP_HOURS),
        "amt_to_avg_ratio": (df["amt"] / prev_avg).fillna(1.0),
        "category": df["category"],
        "is_fraud": df["is_fraud"],
    })
    return out.sort_index()


def transaction_features(
    *,
    amount_inr: float,
    category: str,
    when: datetime,
    dob: date,
    txn_count_24h: int,
    hours_since_last: float,
    avg_amount_inr: float,
) -> pd.DataFrame:
    """Build the feature row for one live transaction (used by the CLI)."""
    row = {
        "amount_inr": amount_inr,
        "hour": when.hour,
        "is_night": int(when.hour in NIGHT_HOURS),
        "day_of_week": when.weekday(),
        "age": (when.date() - dob).days / 365.25,
        "txn_count_24h": txn_count_24h,
        "hours_since_last": min(hours_since_last, MAX_GAP_HOURS),
        "amt_to_avg_ratio": amount_inr / avg_amount_inr if avg_amount_inr > 0 else 1.0,
        "category": category,
    }
    return pd.DataFrame([row], columns=FEATURES)


# -------------------------------------------------------------------- data --

def ensure_dataset() -> tuple[Path, Path]:
    """Return local CSV paths, downloading from Kaggle if they are missing."""
    train_path, test_path = DATA_DIR / TRAIN_CSV, DATA_DIR / TEST_CSV
    if train_path.exists() and test_path.exists():
        return train_path, test_path

    import kagglehub

    try:
        with console.status(f"[cyan]Downloading {KAGGLE_DATASET} from Kaggle..."):
            src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:  # network or auth problems  # noqa: BLE001
        console.print(Panel(
            f"[red]Download failed:[/] {exc}\n\n"
            f"Download it manually from https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"and put [bold]{TRAIN_CSV}[/] and [bold]{TEST_CSV}[/] inside [bold]data/[/].",
            title="Dataset", border_style="red",
        ))
        raise typer.Exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    for name in (TRAIN_CSV, TEST_CSV):
        shutil.copy(next(src.rglob(name)), DATA_DIR / name)
    return train_path, test_path


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs and engineer features on them together.

    They are joined first so card history carries over from the train
    period into the test period.
    """
    train_path, test_path = ensure_dataset()
    train_raw = pd.read_csv(train_path, usecols=RAW_COLUMNS)
    test_raw = pd.read_csv(test_path, usecols=RAW_COLUMNS)
    features = engineer_features(pd.concat([train_raw, test_raw], ignore_index=True))
    n_train = len(train_raw)
    return features.iloc[:n_train], features.iloc[n_train:]


# ------------------------------------------------------------------ models --

def build_pipeline(model, scale: bool = True) -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def candidate_models(pos_weight: float) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, tree_method="hist", eval_metric="aucpr",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def feature_names(pipeline: Pipeline) -> list[str]:
    """Readable names for the columns coming out of the preprocessing step."""
    names = []
    for name in pipeline.named_steps["prep"].get_feature_names_out():
        name = name.split("__", 1)[1]
        if name.startswith("category_"):
            names.append(f"Category: {name.removeprefix('category_')}")
        else:
            names.append(FEATURE_LABELS.get(name, name))
    return names


def feature_importance(pipeline: Pipeline) -> pd.Series:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_[0])
    return pd.Series(values, index=feature_names(pipeline)).sort_values(ascending=False)


# ----------------------------------------------------------------- metrics --

def compute_metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_true, proba) -> float:
    """Threshold with the highest F1 on the given predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def cross_validate(model, X, y, threshold: float, folds: int) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for i, (tr, va) in enumerate(skf.split(X, y), start=1):
        with console.status(f"[cyan]Cross validation fold {i}/{folds}..."):
            pipe = build_pipeline(clone(model)).fit(X.iloc[tr], y.iloc[tr])
            m = compute_metrics(y.iloc[va], pipe.predict_proba(X.iloc[va])[:, 1], threshold)
        m.pop("confusion_matrix")
        rows.append(m)
        console.print(f"  [green]✓[/] Fold {i}/{folds}  [dim]PR-AUC {m['pr_auc']:.4f} · Recall {m['recall']:.4f}[/]")
    scores = pd.DataFrame(rows)
    return {"folds": folds, "mean": scores.mean().to_dict(), "std": scores.std().to_dict()}


# ------------------------------------------------------------------ tables --

def score_style(value: float) -> str:
    return "green" if value >= 0.9 else "yellow" if value >= 0.7 else "red"


def fmt_score(value: float) -> str:
    return f"[{score_style(value)}]{value:.4f}[/]"


def comparison_table(comparison: dict, best: str) -> Table:
    table = Table(title="Model comparison (validation set, threshold 0.5)", title_style="bold")
    table.add_column("Model", style="bold")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        table.add_column(METRIC_LABELS[key], justify="right")
    table.add_column("Train time", justify="right", style="dim")
    for name, m in comparison.items():
        label = f"{name} ⭐" if name == best else name
        table.add_row(
            label,
            *[fmt_score(m[k]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]],
            f"{m['fit_seconds']:.1f}s",
        )
    return table


def test_table(report: dict) -> Table:
    thr = report["threshold"]
    table = Table(title=f"{report['model_name']} on the test set", title_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column(f"Tuned threshold ({thr:.3f})", justify="right")
    table.add_column("Default threshold (0.5)", justify="right")
    table.add_column("Train set (tuned)", justify="right")
    for key, label in METRIC_LABELS.items():
        table.add_row(
            label,
            fmt_score(report["test"]["tuned"][key]),
            fmt_score(report["test"]["default"][key]),
            fmt_score(report["train"][key]),
        )
    return table


def cv_table(cv: dict) -> Table:
    table = Table(title=f"{cv['folds']}-fold cross validation (training data)", title_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right", style="dim")
    for key, label in METRIC_LABELS.items():
        table.add_row(label, fmt_score(cv["mean"][key]), f"± {cv['std'][key]:.4f}")
    return table


def confusion_table(cm: dict) -> Table:
    table = Table(title="Confusion matrix (test set, tuned threshold)", title_style="bold")
    table.add_column("", style="bold")
    table.add_column("Predicted normal", justify="right")
    table.add_column("Predicted fraud", justify="right")
    table.add_row("Actual normal", f"[green]{cm['tn']:,}[/] TN", f"[red]{cm['fp']:,}[/] FP")
    table.add_row("Actual fraud", f"[red]{cm['fn']:,}[/] FN", f"[green]{cm['tp']:,}[/] TP")
    return table


def render_report(report: dict) -> None:
    """Print every metric table for a training report."""
    console.print(comparison_table(report["comparison"], report["model_name"]))
    console.print(test_table(report))
    console.print(cv_table(report["cv"]))
    console.print(confusion_table(report["test"]["tuned"]["confusion_matrix"]))


# ------------------------------------------------------------------ graphs --

def save_plots(*, y_test, proba_test, threshold, best_pipe, comparison, y_val, proba_val, tree_pipe) -> list[Path]:
    REPORTS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved = []

    def save(fig, name):
        path = REPORTS_DIR / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(path)

    # Confusion matrix
    pred = (proba_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {threshold:.3f})")
    save(fig, "confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, proba_test):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(loc="lower right")
    save(fig, "roc_curve.png")

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, proba_test)
    idx = min(np.searchsorted(thresholds, threshold), len(thresholds) - 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {average_precision_score(y_test, proba_test):.4f}")
    ax.scatter(recall[idx], precision[idx], color="red", zorder=3, label=f"Chosen threshold {threshold:.3f}")
    ax.axhline(y_test.mean(), ls="--", color="grey", label="Random guess")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test set)")
    ax.legend(loc="lower left")
    save(fig, "pr_curve.png")

    # Feature importance
    importance = feature_importance(best_pipe).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importance.values, y=importance.index, ax=ax, color="#4C72B0")
    ax.set(xlabel="Importance", ylabel="", title="Top 15 features")
    save(fig, "feature_importance.png")

    # Decision tree (shallow tree on unscaled features so splits are readable)
    fig, ax = plt.subplots(figsize=(26, 11))
    plot_tree(tree_pipe.named_steps["model"], feature_names=feature_names(tree_pipe),
              class_names=["Normal", "Fraud"], filled=True, impurity=False,
              proportion=True, precision=1, fontsize=9, ax=ax)
    ax.set_title("Decision tree (depth 4)", fontsize=16)
    save(fig, "decision_tree.png")

    # Model comparison
    keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    data = pd.DataFrame(
        [{"Model": name, "Metric": METRIC_LABELS[k], "Score": m[k]} for name, m in comparison.items() for k in keys]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set(ylim=(0, 1.05), xlabel="", title="Model comparison (validation set)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "model_comparison.png")

    # Threshold tuning
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.plot(thresholds, f1[:-1], label="F1", lw=2)
    ax.axvline(threshold, ls="--", color="red", label=f"Chosen {threshold:.3f}")
    ax.axvline(0.5, ls=":", color="grey", label="Default 0.5")
    ax.set(xlabel="Threshold", ylabel="Score", title="Threshold tuning (validation set)")
    ax.legend()
    save(fig, "threshold_tuning.png")

    return saved


# ----------------------------------------------------------------- reports --

def write_markdown(report: dict, path: Path) -> None:
    thr = report["threshold"]
    lines = [
        "# Model Report",
        "",
        f"- Model: **{report['model_name']}** (v{report['version']})",
        f"- Trained on: {report['trained_on']}",
        f"- Training rows: {report['dataset']['train_rows']:,} · Test rows: {report['dataset']['test_rows']:,}",
        f"- Fraud rate: {report['dataset']['fraud_rate'] * 100:.2f}%",
        f"- Decision threshold: {thr:.3f}",
        "",
        "## Test set",
        "",
        f"| Metric | Tuned ({thr:.3f}) | Default (0.5) | Train (tuned) |",
        "| --- | --- | --- | --- |",
    ]
    for key, label in METRIC_LABELS.items():
        lines.append(
            f"| {label} | {report['test']['tuned'][key]:.4f} | "
            f"{report['test']['default'][key]:.4f} | {report['train'][key]:.4f} |"
        )

    cm = report["test"]["tuned"]["confusion_matrix"]
    lines += [
        "",
        "## Confusion matrix (test set, tuned threshold)",
        "",
        "| | Predicted normal | Predicted fraud |",
        "| --- | --- | --- |",
        f"| Actual normal | {cm['tn']:,} (TN) | {cm['fp']:,} (FP) |",
        f"| Actual fraud | {cm['fn']:,} (FN) | {cm['tp']:,} (TP) |",
        "",
        f"## {report['cv']['folds']}-fold cross validation",
        "",
        "| Metric | Mean | Std |",
        "| --- | --- | --- |",
    ]
    for key, label in METRIC_LABELS.items():
        lines.append(f"| {label} | {report['cv']['mean'][key]:.4f} | {report['cv']['std'][key]:.4f} |")

    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    lines += [
        "",
        "## Model comparison (validation set, threshold 0.5)",
        "",
        "| Model | " + " | ".join(METRIC_LABELS[k] for k in keys) + " | Train time |",
        "| --- " * (len(keys) + 2) + "|",
    ]
    for name, m in report["comparison"].items():
        star = " ⭐" if name == report["model_name"] else ""
        lines.append(
            f"| {name}{star} | " + " | ".join(f"{m[k]:.4f}" for k in keys) + f" | {m['fit_seconds']:.1f}s |"
        )

    lines += ["", "## Graphs", ""] + [f"![{p.stem}]({p.name})" for p in sorted(REPORTS_DIR.glob("*.png"))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------- main --

def step(message: str) -> None:
    console.print(f"[green]✓[/] {message}")


@app.command()
def main(
    sample: float = typer.Option(1.0, min=0.01, max=1.0, help="Fraction of training data to use."),
    cv_folds: int = typer.Option(5, min=2, max=10, help="Cross validation folds."),
) -> None:
    started = time.perf_counter()
    console.print(Panel.fit(
        "[bold cyan]FraudShield[/] · training pipeline\n"
        "[dim]Credit card fraud detection · Kaggle kartik2112/fraud-detection[/]",
        border_style="cyan",
    ))

    # 1. Data
    ensure_dataset()
    with console.status("[cyan]Loading data and engineering features..."):
        train_df, test_df = load_dataset()
    if sample < 1.0:
        train_df = train_df.groupby("is_fraud").sample(frac=sample, random_state=RANDOM_STATE)
    X_train, y_train = train_df[FEATURES], train_df["is_fraud"]
    X_test, y_test = test_df[FEATURES], test_df["is_fraud"]
    fraud_rate = float(y_train.mean())
    step(f"Data ready  [dim]train {len(X_train):,} rows · test {len(X_test):,} rows · "
         f"fraud {fraud_rate * 100:.2f}%[/]")

    # 2. Compare models on a validation split
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE,
    )
    pos_weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
    models = candidate_models(pos_weight)
    comparison, val_probas = {}, {}
    for name, model in models.items():
        with console.status(f"[cyan]Training {name}..."):
            t0 = time.perf_counter()
            pipe = build_pipeline(model).fit(X_fit, y_fit)
            fit_seconds = time.perf_counter() - t0
            val_probas[name] = pipe.predict_proba(X_val)[:, 1]
        m = compute_metrics(y_val, val_probas[name], 0.5)
        m["fit_seconds"] = fit_seconds
        comparison[name] = m
        step(f"{name} trained  [dim]{fit_seconds:.1f}s · PR-AUC {m['pr_auc']:.4f}[/]")

    best_name = max(comparison, key=lambda n: comparison[n]["pr_auc"])
    console.print(comparison_table(comparison, best_name))

    # 3. Tune threshold on validation predictions of the best model
    threshold = best_threshold(y_val, val_probas[best_name])
    step(f"Best model: [bold]{best_name}[/]  [dim]tuned threshold {threshold:.3f} (max F1 on validation)[/]")

    # 4. Cross validation
    cv = cross_validate(models[best_name], X_train, y_train, threshold, cv_folds)

    # 5. Final fit on all training data, evaluate on the untouched test set
    with console.status(f"[cyan]Training final {best_name} on all training data..."):
        best_pipe = build_pipeline(clone(models[best_name])).fit(X_train, y_train)
        proba_test = best_pipe.predict_proba(X_test)[:, 1]
        proba_train = best_pipe.predict_proba(X_train)[:, 1]
    step("Final model trained")

    report = {
        "model_name": best_name,
        "version": MODEL_VERSION,
        "trained_on": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "source": KAGGLE_DATASET,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "fraud_rate": fraud_rate,
            "usd_to_inr": USD_TO_INR,
        },
        "threshold": threshold,
        "test": {
            "tuned": compute_metrics(y_test, proba_test, threshold),
            "default": compute_metrics(y_test, proba_test, 0.5),
        },
        "train": compute_metrics(y_train, proba_train, threshold),
        "cv": cv,
        "comparison": comparison,
    }

    # 6. Graphs and reports
    with console.status("[cyan]Drawing graphs..."):
        tree_pipe = build_pipeline(
            DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE),
            scale=False,
        ).fit(X_fit, y_fit)
        plots = save_plots(
            y_test=y_test, proba_test=proba_test, threshold=threshold, best_pipe=best_pipe,
            comparison=comparison, y_val=y_val, proba_val=val_probas[best_name], tree_pipe=tree_pipe,
        )
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, REPORTS_DIR / "metrics.md")
    step(f"Saved {len(plots)} graphs, metrics.json and metrics.md  [dim]{REPORTS_DIR}[/]")

    # 7. Save the model bundle
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "pipeline": best_pipe,
        "threshold": threshold,
        "features": FEATURES,
        "categories": sorted(X_train["category"].unique()),
        "metrics": report,
        "model_name": best_name,
        "version": MODEL_VERSION,
        "trained_on": report["trained_on"],
    }, MODEL_PATH)
    step(f"Model saved  [dim]{MODEL_PATH}[/]")

    console.print(test_table(report))
    console.print(cv_table(cv))
    console.print(confusion_table(report["test"]["tuned"]["confusion_matrix"]))
    console.print(Panel.fit(
        f"[bold green]Done[/] in {time.perf_counter() - started:.0f}s\n"
        f"Next: [bold]uv run cli.py[/] to check a transaction",
        border_style="green",
    ))


if __name__ == "__main__":
    app()
