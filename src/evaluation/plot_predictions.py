import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.preprocess import load_etth1, train_val_test_split, scale_splits
from src.data.dataset import TimeSeriesWindowDataset
from src.models.patchtst import PatchTSTSimple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inverse_transform_target(values, scaler, target_idx):
    """
    values: numpy array of shape [pred_len] or [N]
    Converts scaled target values back to original units.
    """
    return values * scaler.scale_[target_idx] + scaler.mean_[target_idx]


@torch.no_grad()
def main():
    csv_path = "data/raw/ETTh1.csv"
    target_col = "OT"

    df, data, feature_cols, target_idx = load_etth1(csv_path, target_col=target_col)
    train, val, test = train_val_test_split(data)
    train, val, test, scaler = scale_splits(train, val, test)

    input_len = 336
    pred_len = 96

    test_ds = TimeSeriesWindowDataset(
        test,
        input_len=input_len,
        pred_len=pred_len,
        target_idx=target_idx
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = PatchTSTSimple(
        input_dim=len(feature_cols),
        input_len=input_len,
        pred_len=pred_len,
        patch_len=16,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.2
    ).to(DEVICE)

    model.load_state_dict(
        torch.load("checkpoints/best_patchtst_simple.pt", map_location=DEVICE)
    )
    model.eval()

    x, y = next(iter(test_loader))
    x = x.to(DEVICE)

    pred = model(x).cpu().numpy().flatten()
    y = y.numpy().flatten()

    pred_original = inverse_transform_target(pred, scaler, target_idx)
    y_original = inverse_transform_target(y, scaler, target_idx)

    plt.figure(figsize=(12, 5))
    plt.plot(y_original, label="Actual")
    plt.plot(pred_original, label="Predicted")
    plt.title("Prediction vs Actual")
    plt.xlabel("Forecast Step")
    plt.ylabel(f"Target Value ({target_col})")
    plt.legend()
    plt.grid(True)
    import os

    os.makedirs("outputs/plots", exist_ok=True)

    plt.savefig(
        "outputs/plots/prediction_vs_actual.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()