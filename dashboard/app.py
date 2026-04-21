import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# make project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.preprocess import load_etth1, train_val_test_split, scale_splits
from src.data.regime_dataset import RegimeAwareDataset
from src.models.regime_aware_model import RegimeAwarePatchTST
from src.evaluation.uncertainty import mc_dropout_predict
from src.evaluation.metrices import mae, rmse, smape


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inverse_transform_target(values, scaler, target_idx):
    return values * scaler.scale_[target_idx] + scaler.mean_[target_idx]


@st.cache_resource
def load_model_and_data():
    csv_path = "data/raw/ETTh1.csv"
    target_col = "OT"

    df, data, feature_cols, target_idx = load_etth1(csv_path, target_col=target_col)
    train, val, test = train_val_test_split(data)
    train, val, test, scaler = scale_splits(train, val, test)

    test_regimes = np.load("outputs/regimes/test_regime_labels.npy")

    input_len = 336
    pred_len = 96

    test_ds = RegimeAwareDataset(
        test,
        test_regimes,
        input_len=input_len,
        pred_len=pred_len,
        target_idx=target_idx
    )

    model = RegimeAwarePatchTST(
        input_dim=len(feature_cols),
        input_len=input_len,
        pred_len=pred_len,
        patch_len=16,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.2,
        num_regimes=3,
        regime_dim=16
    ).to(DEVICE)

    checkpoint_path = "checkpoints/best_regime_aware_patchtst.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_idx": target_idx,
        "scaler": scaler,
        "test_ds": test_ds,
        "model": model,
        "pred_len": pred_len,
        "test_regimes": test_regimes,
    }


@st.cache_data
def load_static_results():
    results = {
        "baseline_mae": 2.3176,
        "baseline_rmse": 2.8540,
        "baseline_smape": 32.29,
        "regime_mae": 2.2982,
        "regime_rmse": 2.9173,
        "regime_smape": 31.95,
        "unc_mae": 2.3470,
        "unc_rmse": 3.0172,
        "unc_smape": 32.33,
        "coverage_95": 31.84,
        "avg_width": 2.2220,
        "train_silhouette": 0.6144,
        "test_silhouette": 0.4473,
        "num_shifts": 199,
        "total_windows": 3054,
        "shift_rate": 6.52,
    }
    return results


def run_single_forecast(test_ds, model, scaler, target_idx, sample_idx, n_mc_samples):
    x, y, regime = test_ds[sample_idx]

    x_batch = x.unsqueeze(0).to(DEVICE)
    regime_batch = regime.unsqueeze(0).to(DEVICE)

    mean_pred, std_pred, _ = mc_dropout_predict(
        model,
        x_batch,
        regime_batch,
        n_samples=n_mc_samples
    )

    mean_pred = mean_pred.cpu().numpy().flatten()
    std_pred = std_pred.cpu().numpy().flatten()
    y = y.numpy().flatten()

    mean_pred = inverse_transform_target(mean_pred, scaler, target_idx)
    y = inverse_transform_target(y, scaler, target_idx)
    std_pred = std_pred * scaler.scale_[target_idx]

    lower_95 = mean_pred - 1.96 * std_pred
    upper_95 = mean_pred + 1.96 * std_pred

    return {
        "actual": y,
        "pred_mean": mean_pred,
        "std": std_pred,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "regime": int(regime.item()),
    }


def plot_forecast_with_uncertainty(actual, pred_mean, lower_95, upper_95):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual, label="Actual")
    ax.plot(pred_mean, label="Predicted Mean")
    ax.fill_between(
        np.arange(len(pred_mean)),
        lower_95,
        upper_95,
        alpha=0.25,
        label="95% Confidence Band"
    )
    ax.set_title("Forecast with Uncertainty")
    ax.set_xlabel("Forecast Step")
    ax.set_ylabel("OT")
    ax.grid(True)
    ax.legend()
    return fig


def plot_regime_timeline(regime_labels, sample_idx=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(regime_labels, drawstyle="steps-mid", label="Regime")
    if sample_idx is not None:
        ax.axvline(sample_idx, linestyle="--", alpha=0.8, label="Selected Window")
    ax.set_title("Regime Timeline")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Regime Label")
    ax.set_yticks(sorted(np.unique(regime_labels)))
    ax.grid(True)
    ax.legend()
    return fig


def build_shift_indices(regime_labels):
    shifts = []
    for i in range(1, len(regime_labels)):
        if regime_labels[i] != regime_labels[i - 1]:
            shifts.append(i)
    return np.array(shifts)


def plot_regime_shifts(regime_labels, shift_indices, sample_idx=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(regime_labels, drawstyle="steps-mid", label="Regime")

    for idx in shift_indices:
        ax.axvline(idx, color="red", alpha=0.08)

    if sample_idx is not None:
        ax.axvline(sample_idx, color="black", linestyle="--", alpha=0.9, label="Selected Window")

    ax.set_title("Regime Timeline with Shift Detections")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Regime Label")
    ax.set_yticks(sorted(np.unique(regime_labels)))
    ax.grid(True)
    ax.legend()
    return fig


def main():
    st.set_page_config(
        page_title="Regime-Aware Transformer Dashboard",
        layout="wide"
    )

    st.title("Regime-Aware Transformer Dashboard")
    st.write("Forecasting, regime detection, shift analysis, and uncertainty estimation on ETTh1.")

    if not os.path.exists("checkpoints/best_regime_aware_patchtst.pt"):
        st.error("Model checkpoint not found. Train the regime-aware model first.")
        return

    if not os.path.exists("outputs/regimes/test_regime_labels.npy"):
        st.error("Regime labels not found. Build regime labels first.")
        return

    bundle = load_model_and_data()
    results = load_static_results()

    test_ds = bundle["test_ds"]
    model = bundle["model"]
    scaler = bundle["scaler"]
    target_idx = bundle["target_idx"]
    test_regimes = bundle["test_regimes"]

    shift_indices = build_shift_indices(test_regimes)

    st.sidebar.header("Controls")
    sample_idx = st.sidebar.slider(
        "Select test window",
        min_value=0,
        max_value=len(test_ds) - 1,
        value=0,
        step=1
    )
    n_mc_samples = st.sidebar.slider(
        "MC Dropout samples",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

    out = run_single_forecast(
        test_ds=test_ds,
        model=model,
        scaler=scaler,
        target_idx=target_idx,
        sample_idx=sample_idx,
        n_mc_samples=n_mc_samples
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Regime", out["regime"])
    c2.metric("Shift Rate", f'{results["shift_rate"]:.2f}%')
    c3.metric("Train Silhouette", f'{results["train_silhouette"]:.4f}')
    c4.metric("Test Silhouette", f'{results["test_silhouette"]:.4f}')

    st.subheader("Model Comparison")
    df_metrics = pd.DataFrame({
        "Model": ["Baseline PatchTST", "Regime-Aware", "Uncertainty-Aware"],
        "MAE": [results["baseline_mae"], results["regime_mae"], results["unc_mae"]],
        "RMSE": [results["baseline_rmse"], results["regime_rmse"], results["unc_rmse"]],
        "sMAPE": [results["baseline_smape"], results["regime_smape"], results["unc_smape"]],
    })
    st.dataframe(df_metrics, use_container_width=True)

    st.subheader("Forecast with Uncertainty")
    fig1 = plot_forecast_with_uncertainty(
        out["actual"],
        out["pred_mean"],
        out["lower_95"],
        out["upper_95"]
    )
    st.pyplot(fig1)

    a, b = st.columns(2)

    with a:
        st.subheader("Regime Timeline")
        fig2 = plot_regime_timeline(test_regimes, sample_idx=sample_idx)
        st.pyplot(fig2)

    with b:
        st.subheader("Shift Detections")
        fig3 = plot_regime_shifts(test_regimes, shift_indices, sample_idx=sample_idx)
        st.pyplot(fig3)

    st.subheader("Uncertainty Summary")
    u1, u2, u3 = st.columns(3)
    u1.metric("95% Coverage", f'{results["coverage_95"]:.2f}%')
    u2.metric("Avg Interval Width", f'{results["avg_width"]:.4f}')
    u3.metric("Detected Shifts", f'{results["num_shifts"]} / {results["total_windows"]}')

    st.markdown("---")
    st.subheader("Project Summary")
    st.write(
        """
        This dashboard demonstrates a regime-aware transformer pipeline for non-stationary time series forecasting.
        The workflow includes:
        - PatchTST-style transformer forecasting
        - latent regime discovery from encoder embeddings
        - shift detection from regime transitions
        - regime-conditioned forecasting
        - Monte Carlo Dropout uncertainty estimation
        """
    )


if __name__ == "__main__":
    main()