import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.preprocess import load_etth1, train_val_test_split, scale_splits
from src.data.dataset import TimeSeriesWindowDataset
from src.models.patchtst import PatchTSTSimple
from src.evaluation.metrices import mae, rmse, smape

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

    input_len = 336
    pred_len = 96
    batch_size = 32

    test_ds = TimeSeriesWindowDataset(
        test,
        input_len=input_len,
        pred_len=pred_len,
        target_idx=target_idx
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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

    all_preds = []
    all_targets = []

    for x, y in test_loader:
        x = x.to(DEVICE)
        preds = model(x).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)      # [N, pred_len]
    all_targets = np.concatenate(all_targets, axis=0)  # [N, pred_len]

    all_preds = inverse_transform_target(all_preds, scaler, target_idx)
    all_targets = inverse_transform_target(all_targets, scaler, target_idx)

    print("Test Metrics")
    print(f"MAE   : {mae(all_targets, all_preds):.4f}")
    print(f"RMSE  : {rmse(all_targets, all_preds):.4f}")
    print(f"sMAPE : {smape(all_targets, all_preds):.2f}%")
if __name__ == "__main__":
    main()