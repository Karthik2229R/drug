"""
main.py — End-to-end pipeline for Adverse Drug Interaction Prediction.

Orchestrates:
  1. Data loading and inspection
  2. Label encoding and class weight computation
  3. Graph construction (nodes=drugs, edges=interactions)
  4. Train/val/test edge splitting
  5. GCN model training with early stopping
  6. Test evaluation (metrics + confusion matrix)
  7. Example prediction with explanation
  8. Saving artifacts for the Streamlit UI
"""

import os
import json
import torch

from src.utils import set_seed, get_device, MODEL_DIR
from src.data_loader import (
    load_dataset,
    build_drug_index,
    build_drug_name_map,
    encode_labels,
    compute_class_weights,
)
from src.graph_builder import build_graph, split_edges
from src.model import DDIGraphModel
from src.train import train_model
from src.evaluate import evaluate_model, plot_training_history
from src.explainability import explain_prediction


def main():
    # ── 0. Setup ─────────────────────────────────────────────────────────
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # ── 1. Load and inspect dataset ──────────────────────────────────────
    df = load_dataset()

    # ── 2. Encode labels & build mappings ────────────────────────────────
    drug_to_idx = build_drug_index(df)
    drug_name_to_id = build_drug_name_map(df)
    labels, label_encoder, num_classes = encode_labels(df)
    class_weights = compute_class_weights(labels, num_classes)

    # ── 3. Build graph ───────────────────────────────────────────────────
    data = build_graph(df, drug_to_idx, labels)

    # ── 4. Split edges ───────────────────────────────────────────────────
    train_idx, val_idx, test_idx = split_edges(data)

    # ── 5. Initialize model ──────────────────────────────────────────────
    num_drugs = len(drug_to_idx)
    model = DDIGraphModel(
        num_drugs=num_drugs,
        num_classes=num_classes,
        embed_dim=128,
        hidden_dim=128,
        dropout=0.3,
    )
    print(f"\nModel architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── 6. Train ─────────────────────────────────────────────────────────
    model, history = train_model(
        model=model,
        data=data,
        train_idx=train_idx,
        val_idx=val_idx,
        class_weights=class_weights,
        device=device,
        epochs=300,
        lr=0.005,
        patience=20,
    )

    # ── 7. Evaluate on test set ──────────────────────────────────────────
    metrics = evaluate_model(model, data, test_idx, label_encoder, device,
                             save_dir=MODEL_DIR)
    plot_training_history(history, save_dir=MODEL_DIR)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")
    if metrics["roc_auc"]:
        print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")

    # ── 8. Example explanation ───────────────────────────────────────────
    # Pick first two drugs from the dataset for a demo explanation
    demo_drug1 = df.iloc[0]["drug1_id"]
    demo_drug2 = df.iloc[0]["drug2_id"]
    print(f"\nDemo explanation for: {df.iloc[0]['drug1_name']} ↔ {df.iloc[0]['drug2_name']}")

    explain_prediction(
        model=model,
        data=data,
        drug1_idx=drug_to_idx[demo_drug1],
        drug2_idx=drug_to_idx[demo_drug2],
        label_encoder=label_encoder,
        device=device,
        save_dir=MODEL_DIR,
    )

    # ── 9. Save artifacts for Streamlit ──────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save graph data (needed for inference)
    torch.save(data, os.path.join(MODEL_DIR, "graph_data.pt"))

    # Save mappings as JSON
    artifacts = {
        "drug_to_idx": drug_to_idx,
        "drug_name_to_id": drug_name_to_id,
        "label_classes": list(label_encoder.classes_),
        "num_drugs": num_drugs,
        "num_classes": num_classes,
    }
    with open(os.path.join(MODEL_DIR, "artifacts.json"), "w") as f:
        json.dump(artifacts, f, indent=2)

    print(f"\nArtifacts saved to {MODEL_DIR}/")
    print("To launch the web UI, run:")
    print("  streamlit run app/streamlit_app.py")
    print("\nDone!")


if __name__ == "__main__":
    main()
