import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.preprocess import load_etth1, train_val_test_split, scale_splits
from src.data.dataset import TimeSeriesWindowDataset
from src.models.patchtst import PatchTSTSimple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    os.makedirs("outputs/regimes", exist_ok=True)

    csv_path = "data/raw/ETTh1.csv"
    target_col = "OT"

    df, data, feature_cols, target_idx = load_etth1(csv_path, target_col=target_col)
    train, val, test = train_val_test_split(data)
    train, val, test, scaler = scale_splits(train, val, test)

    input_len = 336
    pred_len = 96
    batch_size = 64

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

    all_embeddings = []
    all_targets = []
    all_preds = []

    for x, y in test_loader:
        x = x.to(DEVICE)

        preds, embeddings = model(x, return_embedding=True)

        all_preds.append(preds.cpu().numpy())
        all_embeddings.append(embeddings.cpu().numpy())
        all_targets.append(y.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)   # [N, d_model]
    all_preds = np.concatenate(all_preds, axis=0)             # [N, pred_len]
    all_targets = np.concatenate(all_targets, axis=0)         # [N, pred_len]

    np.save("outputs/regimes/test_embeddings.npy", all_embeddings)
    np.save("outputs/regimes/test_preds.npy", all_preds)
    np.save("outputs/regimes/test_targets.npy", all_targets)

    print("Saved:")
    print(" - outputs/regimes/test_embeddings.npy", all_embeddings.shape)
    print(" - outputs/regimes/test_preds.npy", all_preds.shape)
    print(" - outputs/regimes/test_targets.npy", all_targets.shape)


if __name__ == "__main__":
    main()