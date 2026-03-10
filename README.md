# Adverse Drug Interaction Prediction using Deep Learning

B.Tech AI & DS Capstone Project

## Overview

A Graph Neural Network (GCN)-based system that predicts whether a combination of two drugs will cause an adverse interaction, and what **type** of interaction is expected.

**Dataset**: Data of Multiple-Type Drug-Drug Interactions (Mendeley Data)  
- ~222,000 labeled drug pairs across ~16 interaction types

## Architecture

```
Drug Nodes (Embedding) → GCNConv × 2 → Node Embeddings → [z_u || z_v] → MLP → Interaction Type
```

- **Nodes**: Each unique drug (DrugBank IDs)
- **Edges**: Drug–drug interactions (bidirectional)
- **Task**: Multi-class edge classification

## Project Structure

```
Drug Interaction/
├── data/DDI_data.csv              # Dataset
├── src/
│   ├── data_loader.py             # Load CSV, encode labels, class weights
│   ├── graph_builder.py           # Build PyG graph, split edges
│   ├── model.py                   # GCN + MLP decoder
│   ├── train.py                   # Training with early stopping
│   ├── evaluate.py                # Metrics, confusion matrix, ROC-AUC
│   ├── explainability.py          # Gradient-based explanation
│   └── utils.py                   # Seed, device, paths
├── app/streamlit_app.py           # Web UI
├── models/                        # Saved checkpoints & plots
├── main.py                        # End-to-end pipeline
└── requirements.txt
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (adjust for your CUDA version)
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-geometric
```

## Usage

### Train the model
```bash
python main.py
```

This will:
1. Load and inspect the dataset
2. Build the drug interaction graph
3. Train the GCN model (with early stopping)
4. Evaluate on the test set (prints accuracy, F1, ROC-AUC)
5. Save confusion matrix and training curves to `models/`
6. Save model artifacts for the web UI

### Launch the Web UI
```bash
streamlit run app/streamlit_app.py
```

Select two drugs from the dropdowns → get predicted interaction type, confidence, and risk level.

## Evaluation Metrics

- **Accuracy, Precision, Recall, F1-score** (per-class and macro)
- **ROC-AUC** (one-vs-rest, macro)
- **Confusion Matrix** heatmap saved to `models/confusion_matrix.png`

## Tech Stack

| Component       | Technology                        |
|-----------------|-----------------------------------|
| Language        | Python 3.10+                      |
| Deep Learning   | PyTorch + PyTorch Geometric (GCN) |
| Preprocessing   | pandas, scikit-learn              |
| Class Imbalance | Weighted CrossEntropyLoss         |
| Explainability  | Gradient-based attribution        |
| Evaluation      | scikit-learn, seaborn             |
| Web UI          | Streamlit                         |
