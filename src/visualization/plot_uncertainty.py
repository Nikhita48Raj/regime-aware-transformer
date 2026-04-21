import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.preprocess import load_etth1, train_val_test_split, scale_splits
from src.data.regime_dataset import RegimeAwareDataset
from src.models.regime_aware_model import RegimeAwarePatchTST
from src.evaluation.uncertainty import mc_dropout_predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inverse_transform_target(values, scaler, target_idx):
    return values * scaler.scale_[target_idx] + scaler.mean_[target_idx]


@torch.no_grad()
def main():
    csv_path = "data/raw/ETTh1.csv"
    target_col = "OT"

    df, data, feature_cols, target_idx = load_etth1(csv_path, target_col=target_col)
    train, val, test = train_val_test_split(data)
    train, val, test, scaler = scale_splits(train, val, test)

    test_regimes = np.load("outputs/regimes/test_regime_labels.npy")

    input_len = 336
    pred_len = 96
    n_mc_samples = 50

    test_ds = RegimeAwareDataset(
        test,
        test_regimes,
        input_len=input_len,
        pred_len=pred_len,
        target_idx=target_idx
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

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

    model.load_state_dict(
        torch.load("checkpoints/best_regime_aware_patchtst.pt", map_location=DEVICE)
    )

    x, y, regime = next(iter(test_loader))
    x = x.to(DEVICE)
    regime = regime.to(DEVICE)

    mean_pred, std_pred, _ = mc_dropout_predict(
        model,
        x,
        regime,
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

    plt.figure(figsize=(12, 5))
    plt.plot(y, label="Actual")
    plt.plot(mean_pred, label="Predicted Mean")
    plt.fill_between(
        np.arange(len(mean_pred)),
        lower_95,
        upper_95,
        alpha=0.25,
        label="95% Confidence Band"
    )
    plt.title("Forecast with Uncertainty Bands")
    plt.xlabel("Forecast Step")
    plt.ylabel(f"Target Value ({target_col})")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()