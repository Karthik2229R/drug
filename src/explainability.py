"""
explainability.py — Model interpretation for DDI predictions.

Two approaches:
  1. GNNExplainer (PyG-native) — explains which graph structure matters
  2. Embedding-importance — analyses which embedding dimensions contribute
     most to a specific prediction via gradient-based attribution

SHAP's DeepExplainer is not directly compatible with GNN message-passing,
so we use gradient-based feature attribution as the practical alternative.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def explain_prediction(model, data, drug1_idx, drug2_idx, label_encoder,
                       device, save_dir="models"):
    """
    Explain a single drug-pair prediction using gradient-based attribution.

    For the drug pair (drug1, drug2), computes which embedding dimensions
    in both drug nodes contribute most to the predicted interaction type.

    Args:
        model       : trained DDIGraphModel
        data        : PyG Data object
        drug1_idx   : int — node index of drug 1
        drug2_idx   : int — node index of drug 2
        label_encoder : fitted LabelEncoder
        device      : torch.device
        save_dir    : directory to save explanation plot

    Returns:
        result : dict with predicted_class, confidence, top_features
    """
    model.eval()
    data = data.to(device)

    # Create the query edge
    edge_query = torch.tensor([[drug1_idx], [drug2_idx]], dtype=torch.long).to(device)

    # Enable gradients on embeddings for attribution
    embedding_weights = model.drug_embedding.weight
    embedding_weights.requires_grad_(True)

    # Forward pass
    logits = model(data.x, data.edge_index, edge_query)
    probs = torch.softmax(logits, dim=1)
    pred_class = probs.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()

    # Backward pass — gradient of predicted class score w.r.t. embeddings
    model.zero_grad()
    probs[0, pred_class].backward()

    # Get gradients for the two drugs
    grad1 = embedding_weights.grad[drug1_idx].detach().cpu().numpy()
    grad2 = embedding_weights.grad[drug2_idx].detach().cpu().numpy()

    # Importance = absolute gradient (higher = more influential dimension)
    importance1 = np.abs(grad1)
    importance2 = np.abs(grad2)

    # Reset
    embedding_weights.requires_grad_(False)

    pred_label = label_encoder.inverse_transform([pred_class])[0]

    print(f"\n{'=' * 60}")
    print(f"PREDICTION EXPLANATION")
    print(f"{'=' * 60}")
    print(f"Predicted interaction: {pred_label}")
    print(f"Confidence: {confidence:.4f}")

    # ── Top contributing dimensions ──────────────────────────────────────
    top_k = 10
    top_dims_1 = np.argsort(importance1)[-top_k:][::-1]
    top_dims_2 = np.argsort(importance2)[-top_k:][::-1]

    print(f"\nTop {top_k} important embedding dims for Drug 1: {top_dims_1}")
    print(f"Top {top_k} important embedding dims for Drug 2: {top_dims_2}")

    # ── Visualization ────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(top_k), importance1[top_dims_1], color="steelblue")
    ax1.set_xticks(range(top_k))
    ax1.set_xticklabels([f"dim {d}" for d in top_dims_1], rotation=45, fontsize=8)
    ax1.set_title(f"Drug 1 (node {drug1_idx}) — Feature Importance")
    ax1.set_ylabel("Gradient Magnitude")

    ax2.bar(range(top_k), importance2[top_dims_2], color="coral")
    ax2.set_xticks(range(top_k))
    ax2.set_xticklabels([f"dim {d}" for d in top_dims_2], rotation=45, fontsize=8)
    ax2.set_title(f"Drug 2 (node {drug2_idx}) — Feature Importance")
    ax2.set_ylabel("Gradient Magnitude")

    plt.suptitle(f"Predicted: {pred_label} (conf={confidence:.2%})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "explanation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Explanation plot saved → {path}")

    # ── All class probabilities ──────────────────────────────────────────
    all_probs = probs[0].detach().cpu().numpy()
    print("\nAll class probabilities:")
    for i, p in enumerate(all_probs):
        label = label_encoder.inverse_transform([i])[0]
        bar = "█" * int(p * 40)
        print(f"  {label:45s} {p:.4f} {bar}")

    return {
        "predicted_class": pred_label,
        "confidence": confidence,
        "all_probabilities": {
            label_encoder.inverse_transform([i])[0]: float(all_probs[i])
            for i in range(len(all_probs))
        },
    }
