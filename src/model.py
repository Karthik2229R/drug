"""
model.py — Graph Convolutional Network (GCN) for Drug–Drug Interaction prediction.

Architecture:
  1. nn.Embedding  — learnable drug node features (num_drugs → embed_dim)
  2. GCNConv × 2   — message-passing layers that aggregate neighbor info
  3. MLP decoder    — for each edge (u,v), concatenate node embeddings [z_u || z_v]
                      → Linear → ReLU → Linear → num_classes logits

This is a multi-class edge classification model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DDIGraphModel(nn.Module):
    """
    GCN-based model for predicting drug–drug interaction types.

    Args:
        num_drugs   : total number of unique drug nodes
        embed_dim   : dimensionality of learnable drug embeddings (default 128)
        hidden_dim  : hidden layer size in GCN and decoder (default 128)
        num_classes : number of interaction type classes
        dropout     : dropout probability (default 0.3)
    """

    def __init__(self, num_drugs, num_classes, embed_dim=128, hidden_dim=128,
                 dropout=0.3):
        super().__init__()

        # ── Node embedding ─────────────────────────────────────────────────
        self.drug_embedding = nn.Embedding(num_drugs, embed_dim)

        # ── GCN encoder (2 layers) ────────────────────────────────────────
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # ── Edge decoder (MLP) ────────────────────────────────────────────
        # Input: concatenated embeddings of the two drugs [z_u || z_v]
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = dropout

    def encode(self, x, edge_index):
        """
        Run node embeddings through GCN layers.

        Args:
            x          : (num_nodes,) LongTensor — node indices for Embedding
            edge_index : (2, num_edges) — graph connectivity

        Returns:
            z : (num_nodes, hidden_dim) — node embeddings after GCN
        """
        # Embedding lookup
        h = self.drug_embedding(x)  # (num_nodes, embed_dim)

        # GCN layer 1
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GCN layer 2
        h = self.conv2(h, edge_index)

        return h

    def decode(self, z, edge_label_index):
        """
        Predict interaction types for given drug pairs.

        Args:
            z                : (num_nodes, hidden_dim) — node embeddings
            edge_label_index : (2, num_query_edges) — drug pairs to classify

        Returns:
            logits : (num_query_edges, num_classes)
        """
        z_src = z[edge_label_index[0]]  # (num_edges, hidden_dim)
        z_dst = z[edge_label_index[1]]  # (num_edges, hidden_dim)
        edge_feat = torch.cat([z_src, z_dst], dim=-1)  # (num_edges, 2*hidden_dim)
        return self.decoder(edge_feat)

    def forward(self, x, edge_index, edge_label_index):
        """
        Full forward pass: encode all nodes, then decode specified edges.

        Args:
            x                : node index tensor
            edge_index       : full graph connectivity (used by GCN)
            edge_label_index : edges to predict labels for

        Returns:
            logits : (num_query_edges, num_classes)
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
