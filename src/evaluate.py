"""Evaluation and explainability for the KOI exoplanet classification project.

Metrics (macro + per-class), confusion matrices, one-vs-rest ROC curves,
model comparison table, tree feature importances, and SHAP summary values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

FIGSIZE = (8, 6)


def score_summary(y_true, y_pred) -> dict:
    """Headline metrics. Macro averages weight all three classes equally,
    which matters here because CANDIDATE is the smallest and hardest class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def comparison_table(models: dict, X_test, y_test) -> pd.DataFrame:
    """Metrics for every fitted model on the held-out test set."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rows.append({"model": name, **score_summary(y_test, y_pred)})
    return pd.DataFrame(rows).set_index("model").sort_values("f1_macro", ascending=False)


def per_class_report(model, X_test, y_test, class_names) -> str:
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=class_names, digits=3)


def plot_confusion_matrix(model, X_test, y_test, class_names, title, save_path=None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_roc_curves(model, X_test, y_test_enc, class_names, title, save_path=None):
    """One-vs-rest ROC curve per class, with AUC in the legend."""
    y_score = model.predict_proba(X_test)
    n_classes = len(class_names)
    y_bin = label_binarize(y_test_enc, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_feature_importances(model, feature_names, top_n=20, title="Feature importances", save_path=None):
    """Top-N impurity-based importances for a fitted tree ensemble."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * top_n)))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#4878d0")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, top


def shap_summary(model, X_sample, class_names, save_path=None):
    """SHAP summary (bar) plot for a fitted tree model.

    Uses TreeExplainer on a sample of rows — exact for tree ensembles and far
    cheaper than KernelExplainer. Returns the explainer and shap values for
    further per-class plots in the notebook.
    """
    import shap  # imported here so evaluate.py stays usable without shap installed

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # Newer shap returns one (n_samples, n_features, n_classes) array for
    # multi-class; older versions return a list of per-class arrays. Normalize
    # to the list form that summary_plot expects for multi-class bar plots.
    if not isinstance(shap_values, list) and getattr(shap_values, "ndim", 0) == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        class_names=list(class_names),
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return explainer, shap_values


def save_figure(fig, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
