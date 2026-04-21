import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data.preprocess import load_etth1, train_val_test_split, scale_splits
from src.data.dataset import TimeSeriesWindowDataset
from src.models.patchtst import PatchTSTSimple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def extract_embeddings(model, loader):
    all_embeddings = []
    for x, y in loader:
        x = x.to(DEVICE)
        _, embeddings = model(x, return_embedding=True)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


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

    train_ds = TimeSeriesWindowDataset(train, input_len=input_len, pred_len=pred_len, target_idx=target_idx)
    val_ds = TimeSeriesWindowDataset(val, input_len=input_len, pred_len=pred_len, target_idx=target_idx)
    test_ds = TimeSeriesWindowDataset(test, input_len=input_len, pred_len=pred_len, target_idx=target_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
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

    train_embeddings = extract_embeddings(model, train_loader)
    val_embeddings = extract_embeddings(model, val_loader)
    test_embeddings = extract_embeddings(model, test_loader)

    n_regimes = 3
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    train_labels = kmeans.fit_predict(train_embeddings)
    val_labels = kmeans.predict(val_embeddings)
    test_labels = kmeans.predict(test_embeddings)

    sil_score = silhouette_score(train_embeddings, train_labels)

    np.save("outputs/regimes/train_regime_labels.npy", train_labels)
    np.save("outputs/regimes/val_regime_labels.npy", val_labels)
    np.save("outputs/regimes/test_regime_labels.npy", test_labels)
    np.save("outputs/regimes/train_embeddings.npy", train_embeddings)
    np.save("outputs/regimes/val_embeddings.npy", val_embeddings)
    np.save("outputs/regimes/test_embeddings.npy", test_embeddings)

    print("Saved regime labels for train/val/test")
    print("Train embeddings:", train_embeddings.shape)
    print("Val embeddings  :", val_embeddings.shape)
    print("Test embeddings :", test_embeddings.shape)
    print("Train labels    :", train_labels.shape)
    print("Val labels      :", val_labels.shape)
    print("Test labels     :", test_labels.shape)
    print(f"Train silhouette score: {sil_score:.4f}")


if __name__ == "__main__":
    main()