"""Streamlit app: score a test CSV with the saved shopper-purchase-intent classifiers.

The app never trains anything. It loads the pipelines saved in model/, runs them on the
uploaded (or bundled) test data, and reports the evaluation metrics for each model.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
CLASS_LABELS = ["no purchase", "purchase"]

st.set_page_config(page_title="Shopper Purchase Intent", page_icon="🛒", layout="wide")


# ----------------------------------------------------------------------------------------
# Loading saved artefacts
# ----------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading saved models...")
def load_artifacts():
    registry = json.loads((MODEL_DIR / "model_registry.json").read_text())
    schema = json.loads((MODEL_DIR / "feature_schema.json").read_text())
    models = {name: joblib.load(MODEL_DIR / info["file"]) for name, info in registry.items()}
    return registry, schema, models


def read_target(series: pd.Series) -> pd.Series:
    """Turn True/False, 'TRUE'/'FALSE' or 0/1 into integers (1 = purchase)."""
    if series.dtype == bool:
        return series.astype(int)
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "yes": 1, "no": 0})
    if mapped.isna().any():
        raise ValueError("The target column must contain only True/False (or 1/0) values.")
    return mapped.astype(int)


def positive_scores(model, X):
    """Score for the purchase class: probability if available, otherwise the SVM margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def score_model(model, X, y):
    pred = model.predict(X)
    score = positive_scores(model, X)
    try:
        auc = roc_auc_score(y, score)
    except ValueError:  # only one class present in the uploaded file
        auc = float("nan")
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred, score


@st.cache_data(show_spinner="Scoring all models...")
def evaluate_all(frame: pd.DataFrame, feature_cols: tuple, target_col: str):
    """Run every saved model on the data. Nothing is fitted here."""
    _, _, models = load_artifacts()
    X, y = frame[list(feature_cols)], read_target(frame[target_col])
    results = {}
    for name, model in models.items():
        metrics, pred, score = score_model(model, X, y)
        results[name] = {"metrics": metrics, "pred": pred, "score": score}
    return y.to_numpy(), results


# ----------------------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------------------
def draw_confusion(ax, y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", cbar=False, ax=ax,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    ax.set(xlabel="predicted", ylabel="actual", title=title)


def metric_cards(metrics: dict):
    cols = st.columns(len(METRIC_NAMES))
    for col, name in zip(cols, METRIC_NAMES):
        value = metrics[name]
        col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}")


# ----------------------------------------------------------------------------------------
# Sidebar: data source and model choice
# ----------------------------------------------------------------------------------------
registry, schema, models = load_artifacts()
feature_cols = tuple(schema["numeric_features"] + schema["categorical_features"])
target_col = schema["target"]

st.sidebar.title("Shopper purchase intent")
st.sidebar.caption("Scores saved classifiers on a test CSV. No model is retrained.")

st.sidebar.subheader("1. Test data")
uploaded = st.sidebar.file_uploader("Upload a test CSV", type="csv",
                                    help=f"Must contain the {len(feature_cols)} input columns and '{target_col}'.")
use_sample = st.sidebar.checkbox("Use the bundled test_data.csv", value=uploaded is None,
                                 disabled=uploaded is not None)

st.sidebar.subheader("2. Model")
model_name = st.sidebar.selectbox("Choose a model", list(registry.keys()), index=1)
st.sidebar.caption(f"Estimator: {registry[model_name]['estimator']}")
st.sidebar.json(registry[model_name]["best_params"], expanded=False)

# ----------------------------------------------------------------------------------------
# Load and validate the data
# ----------------------------------------------------------------------------------------
st.title("Will this shopping session end in a purchase?")
st.write(
    "Five classifiers trained on the UCI *Online Shoppers Purchasing Intention* data. "
    "Upload the test CSV in the sidebar to see how each saved model performs on it."
)

if uploaded is not None:
    data, source = pd.read_csv(uploaded), f"uploaded file `{uploaded.name}`"
elif use_sample:
    data, source = pd.read_csv(SAMPLE_CSV), "bundled `test_data.csv`"
else:
    st.info("Upload a test CSV or tick the bundled-data option in the sidebar to begin.")
    st.stop()

missing = [c for c in (*feature_cols, target_col) if c not in data.columns]
if missing:
    st.error(f"The file is missing required column(s): {', '.join(missing)}")
    st.stop()
if data[list(feature_cols) + [target_col]].isna().any().any():
    st.error("The file contains empty cells in the required columns. Please remove or fill them.")
    st.stop()

try:
    y_true, results = evaluate_all(data, feature_cols, target_col)
except ValueError as err:
    st.error(str(err))
    st.stop()

purchases = int(y_true.sum())
st.success(f"Scored **{len(data):,}** sessions from the {source}. "
           f"Purchases in this file: **{purchases:,}** ({purchases / len(y_true):.1%}).")

# ----------------------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------------------
tab_one, tab_all, tab_data = st.tabs(["Selected model", "All models compared", "Data preview"])

with tab_one:
    res = results[model_name]
    st.subheader(model_name)
    metric_cards(res["metrics"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        draw_confusion(ax, y_true, res["pred"])
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**ROC curve**")
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, res["score"])
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            ax.plot(fpr, tpr, lw=2, color="#6B7A1F", label=f"AUC {res['metrics']['AUC']:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set(xlabel="false positive rate", ylabel="true positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("The ROC curve needs both classes in the file.")

    st.markdown("**Classification report**")
    report = classification_report(y_true, res["pred"], labels=[0, 1], target_names=CLASS_LABELS,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

with tab_all:
    table = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T[METRIC_NAMES]
    table.index.name = "ML Model Name"
    st.subheader("Evaluation metrics, all five models")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#e3ebb8"),
                 width="stretch")
    best = table["MCC"].idxmax()
    st.caption(f"Highest value per column is shaded. Best MCC on this file: **{best}** ({table.loc[best, 'MCC']:.4f}).")

    long = table.reset_index().melt(id_vars="ML Model Name", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sns.barplot(data=long, x="metric", y="score", hue="ML Model Name", ax=ax, palette="viridis")
    ax.set(ylim=(0, 1), xlabel="", ylabel="score")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Confusion matrices")
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.6))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        draw_confusion(ax, y_true, r["pred"], title=name)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_data:
    st.write(f"{data.shape[0]:,} rows, {data.shape[1]} columns.")
    st.dataframe(data.head(100), width="stretch")
