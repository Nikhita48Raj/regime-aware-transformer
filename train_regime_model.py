import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.preprocess import load_etth1, train_val_test_split, scale_splits
from src.data.regime_dataset import RegimeAwareDataset
from src.models.regime_aware_model import RegimeAwarePatchTST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y, regime in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        regime = regime.to(DEVICE)

        optimizer.zero_grad()
        preds = model(x, regime)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for x, y, regime in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        regime = regime.to(DEVICE)

        preds = model(x, regime)
        loss = criterion(preds, y)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    set_seed(42)
    os.makedirs("checkpoints", exist_ok=True)

    csv_path = "data/raw/ETTh1.csv"
    target_col = "OT"

    df, data, feature_cols, target_idx = load_etth1(csv_path, target_col=target_col)
    train, val, test = train_val_test_split(data)
    train, val, test, scaler = scale_splits(train, val, test)

    train_regimes = np.load("outputs/regimes/train_regime_labels.npy")
    val_regimes = np.load("outputs/regimes/val_regime_labels.npy")
    test_regimes = np.load("outputs/regimes/test_regime_labels.npy")

    input_len = 336
    pred_len = 96
    batch_size = 32

    train_ds = RegimeAwareDataset(train, train_regimes, input_len=input_len, pred_len=pred_len, target_idx=target_idx)
    val_ds = RegimeAwareDataset(val, val_regimes, input_len=input_len, pred_len=pred_len, target_idx=target_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )
    criterion = torch.nn.MSELoss()

    epochs = 20
    best_val = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_regime_aware_patchtst.pt")
            print("Saved best regime-aware model.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/3")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    main()