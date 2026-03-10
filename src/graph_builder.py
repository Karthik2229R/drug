"""
graph_builder.py — Construct a PyTorch Geometric graph from the DDI dataset.

Nodes  = unique drugs  (features via learnable Embedding in the model)
Edges  = drug–drug interactions (bidirectional)
Labels = interaction type per edge (multi-class classification target)

Also handles stratified train/val/test edge splitting.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def build_graph(df, drug_to_idx, labels):
    """
    Convert the DDI dataframe into a PyTorch Geometric Data object.

    Args:
        df          : DataFrame with drug1_id, drug2_id columns
        drug_to_idx : dict mapping drug_id → int node index
        labels      : np.ndarray of integer interaction-type labels

    Returns:
        data : torch_geometric.data.Data with:
            - x            : node index tensor (num_nodes,) for Embedding lookup
            - edge_index   : (2, 2*num_edges) — bidirectional
            - edge_label   : (2*num_edges,) — interaction type per edge
            - num_nodes    : int
            - num_classes  : int
    """
    num_nodes = len(drug_to_idx)

    # Map drug IDs to integer node indices
    src = df["drug1_id"].map(drug_to_idx).values
    dst = df["drug2_id"].map(drug_to_idx).values

    # Make edges bidirectional: (u→v) and (v→u)
    edge_src = np.concatenate([src, dst])
    edge_dst = np.concatenate([dst, src])
    edge_labels = np.concatenate([labels, labels])  # same label both directions

    edge_index = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
    edge_label = torch.tensor(edge_labels, dtype=torch.long)

    # Node feature placeholder — just node indices for nn.Embedding
    x = torch.arange(num_nodes, dtype=torch.long)

    num_classes = int(edge_label.max().item()) + 1

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_label=edge_label,
        num_nodes=num_nodes,
    )
    data.num_classes = num_classes

    print(f"\nGraph constructed:")
    print(f"  Nodes : {num_nodes}")
    print(f"  Edges : {edge_index.shape[1]} (bidirectional)")
    print(f"  Classes: {num_classes}")

    return data


def split_edges(data, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Stratified train/val/test split on edges (not nodes).
    Preserves class distribution across splits.

    Classes with fewer than 3 samples are placed entirely in the training set
    (too few for stratified splitting), then the rest is stratified normally.

    Returns:
        train_idx, val_idx, test_idx : LongTensors of edge indices
    """
    num_edges = data.edge_index.shape[1]
    indices = np.arange(num_edges)
    labels_np = data.edge_label.numpy()

    # Classes need enough samples for two stratified splits (train/temp, then
    # val/test).  A class with fewer than 10 edges risks having < 2 samples in
    # the temp set after the first split, which makes the second split fail.
    from collections import Counter
    class_counts = Counter(labels_np)
    min_count = 10  # conservative threshold
    rare_classes = {c for c, cnt in class_counts.items() if cnt < min_count}

    rare_mask = np.isin(labels_np, list(rare_classes))
    rare_indices = indices[rare_mask]
    common_indices = indices[~rare_mask]
    common_labels = labels_np[common_indices]

    if len(rare_indices) > 0:
        print(f"\n  {len(rare_classes)} rare classes ({len(rare_indices)} edges) "
              f"assigned directly to train set.")

    # Stratified split on the common (non-rare) edges
    train_common, temp_idx = train_test_split(
        common_indices, train_size=train_ratio,
        stratify=common_labels, random_state=seed
    )

    val_relative = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_relative,
        stratify=labels_np[temp_idx],
        random_state=seed,
    )

    # Combine rare edges into train
    train_idx_np = np.concatenate([train_common, rare_indices])

    train_idx = torch.tensor(train_idx_np, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    print(f"\nEdge split (stratified):")
    print(f"  Train : {len(train_idx)}")
    print(f"  Val   : {len(val_idx)}")
    print(f"  Test  : {len(test_idx)}")

    return train_idx, val_idx, test_idx
