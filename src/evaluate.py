"""
evaluate.py — Evaluation utilities for the DDI GCN model.

Computes:
  - Accuracy, Precision, Recall, F1-score (per-class and macro)
  - Multi-class ROC-AUC (one-vs-rest)
  - Confusion matrix heatmap
  - Training history plots (loss & accuracy curves)
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)


def evaluate_model(model, data, test_idx, label_encoder, device, save_dir="models"):
    """
    Run the model on test edges and print full classification metrics.

    Args:
        model         : trained DDIGraphModel
        data          : PyG Data object
        test_idx      : LongTensor of test edge indices
        label_encoder : fitted LabelEncoder (for class names)
        device        : torch.device
        save_dir      : directory to save plots

    Returns:
        metrics : dict with accuracy, macro_f1, macro_precision, macro_recall, roc_auc
    """
    model.eval()
    data = data.to(device)

    # Batched inference to avoid memory issues on CPU
    batch_size = 8192
    test_idx_np = test_idx.numpy()
    all_probs = []
    all_preds = []
    all_true = []

    with torch.no_grad():
        z = model.encode(data.x.to(device), data.edge_index.to(device))

        for start in range(0, len(test_idx_np), batch_size):
            batch_idx = test_idx_np[start : start + batch_size]
            batch_idx_t = torch.tensor(batch_idx, dtype=torch.long, device=device)

            batch_edge_index = data.edge_index[:, batch_idx_t]
            batch_labels = data.edge_label[batch_idx_t]

            logits = model.decode(z, batch_edge_index)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(batch_labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)

    class_names = list(label_encoder.classes_)

    # ── Classification report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    report = classification_report(true, preds, target_names=class_names, digits=4)
    print(report)

    acc = accuracy_score(true, preds)

    # ── ROC-AUC (one-vs-rest, macro) ──────────────────────────────────────
    try:
        roc_auc = roc_auc_score(true, probs, multi_class="ovr", average="macro")
        print(f"ROC-AUC (macro, one-vs-rest): {roc_auc:.4f}")
    except ValueError:
        roc_auc = None
        print("ROC-AUC: could not compute (some classes may be missing in test set)")

    # ── Confusion matrix heatmap ──────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(true, preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — DDI Interaction Types")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {cm_path}")

    # ── Return metrics dict ───────────────────────────────────────────────
    report_dict = classification_report(true, preds, target_names=class_names,
                                        output_dict=True)
    metrics = {
        "accuracy": acc,
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "roc_auc": roc_auc,
    }
    return metrics


def plot_training_history(history, save_dir="models"):
    """
    Plot and save training/validation loss and accuracy curves.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(save_dir, "training_history.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Training history plot saved → {hist_path}")
