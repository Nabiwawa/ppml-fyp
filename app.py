# -----------------------------------------------------------------------------------
# PPML Demo: Upload/Built-in dataset + Technique selection
# Non-Private vs Different techniques (currently Differential Privacy) + Comparison
# -----------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from opacus import PrivacyEngine
from opacus.accountants.analysis import rdp as rdp_analysis


# -----------------------------
# 0) Streamlit basic setup
# -----------------------------
st.set_page_config(page_title="PPML Demo", layout="wide")
st.title("Privacy-Preserving ML Demo")
st.write(
    "Upload a CSV or use the built-in synthetic dataset, choose a technique, "
    "then train and visualise results. Compare **Non-Private** vs **Differential Privacy**."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1) Sidebar: dataset + technique
# -----------------------------
st.sidebar.header("Dataset")
ds_choice = st.sidebar.selectbox(
    "Select data source",
    [
        "Select data source",
        "Use built-in synthetic dataset",
        "Upload CSV (last column = label)"
    ],
    index=0
)

uploaded_file = None
if ds_choice == "Upload CSV (last column = label)":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Technique")
technique = st.sidebar.selectbox(
    "Select technique",
    [
        "Non-Private (Baseline)",
        "Differential Privacy (DP-SGD)",
        "Federated Learning (Coming Soon)",
        "Homomorphic Encryption (Coming Soon)"
    ],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.header("Training Settings")
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
np.random.seed(seed); torch.manual_seed(seed)

epochs = st.sidebar.slider("Epochs", 1, 20, 8, step=1)
batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)
lr = st.sidebar.selectbox("Learning rate", [0.1, 0.05, 0.02, 0.01], index=3)

st.sidebar.markdown("---")
st.sidebar.subheader("DP (only used when DP is selected)")
noise_multiplier = st.sidebar.selectbox("Noise multiplier", [0.5, 0.8, 1.0, 1.2, 1.5], index=2)
max_grad_norm = st.sidebar.selectbox("Max grad norm", [0.5, 1.0, 1.5, 2.0], index=1)
delta_str = st.sidebar.text_input("Delta", value="1e-5")

st.sidebar.info(f"Device: **{device}**")


# -----------------------------
# 2) Load data helper (CSV or built-in)
# -----------------------------
def load_dataset(ds_choice, uploaded_file, seed):
    """
    Returns: X_train, X_test, y_train, y_test (numpy arrays)
    CSV format assumption: last column = label (binary).
    """
    if ds_choice == "Upload CSV (last column = label)":
        if uploaded_file is None:
            st.info("Please upload a CSV to proceed (last column must be the label).")
            st.stop()
        df = pd.read_csv(uploaded_file)
        if df.shape[1] < 2:
            st.error("CSV must have at least 2 columns (features + label).")
            st.stop()

        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values

        # Auto-map non 0/1 labels if exactly two unique values
        unique_vals = pd.unique(y_raw)
        if len(unique_vals) != 2:
            st.error("Label column must be binary (exactly two classes).")
            st.stop()
        # Map to {0,1} deterministically
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        y = np.vectorize(mapping.get)(y_raw)

        # Standardise features
        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        source = f"Uploaded CSV: {uploaded_file.name}"
    else:
        # Built-in synthetic dataset
        n_samples = 1200
        n_features = 16
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(8, n_features),
            n_redundant=0,
            n_classes=2,
            random_state=seed
        )
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        source = "Built-in synthetic dataset"

    st.success(
        f"Dataset ready from **{source}** — Train: {len(y_train)}, Test: {len(y_test)}, Features: {X_train.shape[1]}"
    )
    return X_train, X_test, y_train, y_test



if ds_choice == "Select data source":
    st.warning("Please select a data source from the sidebar to continue.")
    st.stop()

X_train, X_test, y_train, y_test = load_dataset(ds_choice, uploaded_file, seed)


# To tensors + DataLoaders
def make_loaders(X_train, y_train, X_test, y_test, batch_size):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t, y_test_t)

    # Ensure batch size is valid
    bs = int(min(batch_size, max(2, len(train_ds))))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False)
    return train_ds, train_loader, test_loader, bs

train_ds, train_loader_base, test_loader, batch_size = make_loaders(
    X_train, y_train, X_test, y_test, batch_size
)


# -----------------------------
# 3) Simple model + train/eval helpers
# -----------------------------
class TinyMLP(nn.Module):
    def __init__(self, in_features, hidden=32, out_features=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.layers(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        preds.append(pred.cpu().numpy()); gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    return accuracy_score(gts, preds)


def run_baseline(X_train, y_train, train_loader, test_loader, epochs, lr, device):
    model = TinyMLP(in_features=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        l = train_one_epoch(model, train_loader, crit, opt, device)
        losses.append(l)
    acc = evaluate_acc(model, test_loader, device)
    return model, losses, acc


def run_dp(X_train, y_train, train_ds, test_loader, epochs, lr, batch_size, device, sigma, max_gn):
    model = TinyMLP(in_features=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    # Opacus DataLoader (must match batch size + drop_last)
    train_loader_dp = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    privacy_engine = PrivacyEngine()
    model, opt, train_loader_dp = privacy_engine.make_private(
        module=model,
        optimizer=opt,
        data_loader=train_loader_dp,
        noise_multiplier=float(sigma),
        max_grad_norm=float(max_gn),
    )
    losses = []
    for _ in range(epochs):
        l = train_one_epoch(model, train_loader_dp, crit, opt, device)
        losses.append(l)
    acc = evaluate_acc(model, test_loader, device)
    return model, losses, acc, train_loader_dp


def estimate_epsilon(batch_size, train_ds_len, epochs, train_loader_dp, sigma, delta_str):
    try:
        delta = float(delta_str)
        sample_rate = batch_size / train_ds_len
        steps = epochs * len(train_loader_dp)
        orders = np.linspace(1.25, 64.0, 100)
        rdp = rdp_analysis.compute_rdp(q=sample_rate, noise_multiplier=float(sigma), steps=steps, orders=orders)
        eps, best_order = rdp_analysis.get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
        return eps, best_order
    except Exception as e:
        return None, str(e)


# -----------------------------
# 4) Tabs: Single technique + Comparison
# -----------------------------
tab1, tab2 = st.tabs(["Run Selected Technique", "Comparison (Baseline vs DP)"])

# --- Tab 1: Run the selected technique ---
with tab1:
    st.header("Run Selected Technique")

    if technique == "Non-Private (Baseline)":
        with st.spinner("Training Non-Private model..."):
            _, np_losses, np_acc = run_baseline(X_train, y_train, train_loader_base, test_loader, epochs, lr, device)
        st.success(f"Non-Private Test Accuracy: {np_acc:.3f}")

        fig_np, ax_np = plt.subplots()
        ax_np.plot(range(1, epochs + 1), np_losses, marker="o", label="Non-Private Loss")
        ax_np.set_xlabel("Epoch"); ax_np.set_ylabel("Loss"); ax_np.set_title("Non-Private Loss per Epoch")
        ax_np.legend()
        st.pyplot(fig_np)

    elif technique == "Differential Privacy (DP-SGD)":
        with st.spinner("Training DP model (Opacus)..."):
            _, dp_losses, dp_acc, train_loader_dp = run_dp(
                X_train, y_train, train_ds, test_loader, epochs, lr, batch_size, device,
                sigma=noise_multiplier, max_gn=max_grad_norm
            )
        st.success(f"DP Test Accuracy: {dp_acc:.3f}")

        fig_dp, ax_dp = plt.subplots()
        ax_dp.plot(range(1, epochs + 1), dp_losses, marker="o", color="orange", label="DP Loss")
        ax_dp.set_xlabel("Epoch"); ax_dp.set_ylabel("Loss"); ax_dp.set_title("DP Loss per Epoch")
        ax_dp.legend()
        st.pyplot(fig_dp)

        # Epsilon estimate (DP only)
        eps, best_order = estimate_epsilon(batch_size, len(train_ds), epochs, train_loader_dp, noise_multiplier, delta_str)
        if eps is not None:
            st.info(f"Approx (ε, δ): ε ≈ {eps:.2f} at δ = {delta_str} (best order ≈ {best_order:.2f})")
        else:
            st.warning(f"Could not estimate ε: {best_order}")

    else:
        st.info("This technique is not yet implemented for this prototype and is **coming soon** and instead be implemented for FYP.")

# --- Tab 2: Comparison (train both) ---
with tab2:
    st.header("Baseline vs Differential Privacy (Side-by-side)")

    colA, colB = st.columns(2)

    with st.spinner("Training Baseline and DP for comparison..."):
        # Train baseline
        _, np_losses, np_acc = run_baseline(X_train, y_train, train_loader_base, test_loader, epochs, lr, device)
        # Train DP
        _, dp_losses, dp_acc, train_loader_dp = run_dp(
            X_train, y_train, train_ds, test_loader, epochs, lr, batch_size, device,
            sigma=noise_multiplier, max_gn=max_grad_norm
        )

    with colA:
        st.subheader("Non-Private")
        st.metric("Accuracy", f"{np_acc:.3f}")
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, epochs + 1), np_losses, marker="o", label="Non-Private Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Non-Private Loss")
        ax1.legend()
        st.pyplot(fig1)

    with colB:
        st.subheader("Differential Privacy")
        st.metric("Accuracy", f"{dp_acc:.3f}")
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, epochs + 1), dp_losses, marker="o", color="orange", label="DP Loss")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("DP Loss")
        ax2.legend()
        st.pyplot(fig2)

    # Combined loss plot
    st.subheader("Loss Comparison")
    fig3, ax3 = plt.subplots()
    ax3.plot(range(1, epochs + 1), np_losses, marker="o", label="Non-Private")
    ax3.plot(range(1, epochs + 1), dp_losses, marker="o", color="orange", label="DP")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss"); ax3.set_title("Loss: Non-Private vs DP")
    ax3.legend()
    st.pyplot(fig3)

    # DP epsilon
    eps, best_order = estimate_epsilon(batch_size, len(train_ds), epochs, train_loader_dp, noise_multiplier, delta_str)
    if eps is not None:
        st.info(f"Approx (ε, δ): ε ≈ {eps:.2f} at δ = {delta_str} (best order ≈ {best_order:.2f})")
    else:
        st.warning(f"Could not estimate ε: {best_order}")



