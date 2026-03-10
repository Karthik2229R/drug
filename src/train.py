"""
train.py — Training loop for the DDI GCN model.

Features:
  - Class-weighted CrossEntropyLoss to handle imbalanced interaction types
  - Early stopping on validation loss (patience-based)
  - Best model checkpoint saved to models/best_model.pt
  - Per-epoch logging of train/val loss and accuracy
"""

import os
import torch
import torch.nn as nn
from src.utils import MODEL_DIR


def train_model(model, data, train_idx, val_idx, class_weights, device,
                epochs=100, lr=0.001, patience=10):
    """
    Train the DDI GCN model with early stopping.

    The GCN encoder runs once per epoch on the full graph (1868 nodes — fast).
    The decoder processes all edges at once (MLP is lightweight).
    """
    model = model.to(device)
    data = data.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_path = os.path.join(MODEL_DIR, "best_model.pt")

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"  Train edges: {len(train_idx)}")
    print(f"  Val edges  : {len(val_idx)}")

    for epoch in range(1, epochs + 1):
        # ── Train step ───────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        # Encode all nodes once (1868 nodes — fast even on CPU)
        z = model.encode(data.x, data.edge_index)

        # Decode ALL train edges at once (decoder is just an MLP)
        train_edge_index = data.edge_index[:, train_idx]
        train_labels = data.edge_label[train_idx]

        logits = model.decode(z, train_edge_index)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        train_acc = (logits.argmax(dim=1) == train_labels).float().mean().item()
        train_loss = loss.item()

        # ── Validation step ──────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
            val_edge_index = data.edge_index[:, val_idx]
            val_labels = data.edge_label[val_idx]

            val_logits = model.decode(z, val_edge_index)
            val_loss = criterion(val_logits, val_labels).item()
            val_acc = (val_logits.argmax(dim=1) == val_labels).float().mean().item()

        # ── Logging ──────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

        # ── Early stopping ───────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

    # Load best model
    model.load_state_dict(torch.load(best_path, weights_only=True))
    print(f"Best model loaded from {best_path} (val_loss={best_val_loss:.4f})")

    return model, history
