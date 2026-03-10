"""
streamlit_app.py — Web UI for Adverse Drug Interaction Prediction.

Users select two drug names from dropdowns and get:
  - Predicted interaction type
  - Confidence score
  - Color-coded risk indicator
  - Class probability breakdown
"""

import os
import sys
import json
import torch
import pandas as pd
import streamlit as st

# Add project root to path so imports work when running from app/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model import DDIGraphModel
from src.utils import MODEL_DIR, get_device


# ── Risk classification for color coding ─────────────────────────────────────
HIGH_RISK_TYPES = {
    "risk or severity of bleeding",
    "risk or severity of adverse effects",
    "nephrotoxic activities",
    "respiratory depressant activities",
    "hypotensive, nephrotoxic, and hyperkalemic activities",
}

MODERATE_RISK_TYPES = {
    "serum concentration",
    "metabolism",
    "anticoagulant activities",
    "thrombogenic activities",
    "hypoglycemic activities",
    "anticholinergic activities",
}


def get_risk_level(interaction_type):
    """Return risk level and color for the predicted interaction."""
    if interaction_type in HIGH_RISK_TYPES:
        return "HIGH RISK", "🔴"
    elif interaction_type in MODERATE_RISK_TYPES:
        return "MODERATE RISK", "🟡"
    else:
        return "LOW RISK", "🟢"


@st.cache_resource
def load_artifacts():
    """Load model, graph data, and mappings (cached across reruns)."""
    artifacts_path = os.path.join(MODEL_DIR, "artifacts.json")
    if not os.path.exists(artifacts_path):
        st.error("Model artifacts not found. Please run `python main.py` first to train the model.")
        st.stop()

    with open(artifacts_path, "r") as f:
        artifacts = json.load(f)

    drug_to_idx = artifacts["drug_to_idx"]
    label_classes = artifacts["label_classes"]
    num_drugs = artifacts["num_drugs"]
    num_classes = artifacts["num_classes"]
    drug_names = sorted(artifacts["drug_name_to_id"].keys())
    drug_name_to_id = artifacts["drug_name_to_id"]

    # Load graph data
    graph_path = os.path.join(MODEL_DIR, "graph_data.pt")
    graph_data = torch.load(graph_path, weights_only=False)

    # Load model
    device = get_device()
    model = DDIGraphModel(num_drugs, num_classes)
    model_path = os.path.join(MODEL_DIR, "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model, graph_data, drug_to_idx, label_classes, drug_names, drug_name_to_id, device


def predict_interaction(model, graph_data, drug1_idx, drug2_idx, label_classes, device):
    """Run inference for a single drug pair."""
    graph_data = graph_data.to(device)
    edge_query = torch.tensor([[drug1_idx], [drug2_idx]], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index, edge_query)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = probs.argmax()
    pred_label = label_classes[pred_idx]
    confidence = probs[pred_idx]

    all_probs = {label_classes[i]: float(probs[i]) for i in range(len(label_classes))}
    return pred_label, confidence, all_probs


# ── Streamlit App ────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Drug Interaction Predictor",
        page_icon="💊",
        layout="wide",
    )

    st.title("💊 Adverse Drug Interaction Predictor")
    st.markdown(
        "Predict whether a combination of two drugs will cause an adverse "
        "interaction using a **Graph Convolutional Network (GCN)**."
    )

    # Load model and data
    model, graph_data, drug_to_idx, label_classes, drug_names, drug_name_to_id, device = load_artifacts()

    st.markdown("---")

    # ── Drug selection ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drug 1")
        drug1_name = st.selectbox("Select first drug:", drug_names, key="drug1")

    with col2:
        st.subheader("Drug 2")
        drug2_name = st.selectbox("Select second drug:", drug_names, key="drug2",
                                   index=min(1, len(drug_names) - 1))

    st.markdown("---")

    # ── Predict button ───────────────────────────────────────────────────
    if st.button("🔍 Predict Interaction", type="primary", use_container_width=True):
        if drug1_name == drug2_name:
            st.warning("Please select two different drugs.")
        else:
            drug1_id = drug_name_to_id[drug1_name]
            drug2_id = drug_name_to_id[drug2_name]

            if drug1_id not in drug_to_idx or drug2_id not in drug_to_idx:
                st.error("One or both drugs not found in the graph. Cannot predict.")
            else:
                drug1_idx = drug_to_idx[drug1_id]
                drug2_idx = drug_to_idx[drug2_id]

                pred_label, confidence, all_probs = predict_interaction(
                    model, graph_data, drug1_idx, drug2_idx, label_classes, device
                )

                risk_level, risk_icon = get_risk_level(pred_label)

                # ── Results ──────────────────────────────────────────
                st.markdown("### Prediction Result")

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Interaction Type", pred_label)
                with r2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with r3:
                    st.metric("Risk Level", f"{risk_icon} {risk_level}")

                # ── Probability breakdown ────────────────────────────
                st.markdown("### All Interaction Probabilities")
                prob_df = pd.DataFrame(
                    sorted(all_probs.items(), key=lambda x: -x[1]),
                    columns=["Interaction Type", "Probability"],
                )
                st.bar_chart(prob_df.set_index("Interaction Type"))
                st.dataframe(prob_df, width="stretch", hide_index=True)

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown("---")


if __name__ == "__main__":
    main()
