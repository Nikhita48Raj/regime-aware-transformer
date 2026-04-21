import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.preprocess import load_etth1, train_val_test_split, scale_splits
from src.data.regime_dataset import RegimeAwareDataset
from src.models.regime_aware_model import RegimeAwarePatchTST
from src.evaluation.metrices import mae, rmse, smape
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
    batch_size = 16
    n_mc_samples = 30

    test_ds = RegimeAwareDataset(
        test,
        test_regimes,
        input_len=input_len,
        pred_len=pred_len,
        target_idx=target_idx
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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

    all_mean_preds = []
    all_std_preds = []
    all_targets = []

    for x, y, regime in test_loader:
        x = x.to(DEVICE)
        regime = regime.to(DEVICE)

        mean_pred, std_pred, _ = mc_dropout_predict(
            model,
            x,
            regime,
            n_samples=n_mc_samples
        )

        all_mean_preds.append(mean_pred.cpu().numpy())
        all_std_preds.append(std_pred.cpu().numpy())
        all_targets.append(y.numpy())

    all_mean_preds = np.concatenate(all_mean_preds, axis=0)
    all_std_preds = np.concatenate(all_std_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    all_mean_preds_unscaled = inverse_transform_target(all_mean_preds, scaler, target_idx)
    all_targets_unscaled = inverse_transform_target(all_targets, scaler, target_idx)
    all_std_unscaled = all_std_preds * scaler.scale_[target_idx]

    lower_95 = all_mean_preds_unscaled - 1.96 * all_std_unscaled
    upper_95 = all_mean_preds_unscaled + 1.96 * all_std_unscaled

    coverage = np.mean(
        (all_targets_unscaled >= lower_95) & (all_targets_unscaled <= upper_95)
    ) * 100

    avg_interval_width = np.mean(upper_95 - lower_95)

    print("Uncertainty-Aware Test Metrics")
    print(f"MAE              : {mae(all_targets_unscaled, all_mean_preds_unscaled):.4f}")
    print(f"RMSE             : {rmse(all_targets_unscaled, all_mean_preds_unscaled):.4f}")
    print(f"sMAPE            : {smape(all_targets_unscaled, all_mean_preds_unscaled):.2f}%")
    print(f"95% Coverage     : {coverage:.2f}%")
    print(f"Avg Interval Width: {avg_interval_width:.4f}")
if __name__ == "__main__":
    main()