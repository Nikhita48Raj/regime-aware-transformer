import torch


@torch.no_grad()
def mc_dropout_predict(model, x, regime, n_samples=30):
    """
    Monte Carlo Dropout prediction.

    Returns:
        mean_pred: [B, pred_len]
        std_pred : [B, pred_len]
        all_preds: [n_samples, B, pred_len]
    """
    model.train()  # important: enables dropout inside transformer layers

    preds = []

    for _ in range(n_samples):
        pred = model(x, regime)
        preds.append(pred.unsqueeze(0))

    all_preds = torch.cat(preds, dim=0)   # [n_samples, B, pred_len]
    mean_pred = all_preds.mean(dim=0)     # [B, pred_len]
    std_pred = all_preds.std(dim=0)       # [B, pred_len]

    model.eval()  # restore eval mode after sampling

    return mean_pred, std_pred, all_preds