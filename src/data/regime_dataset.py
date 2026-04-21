import torch
from torch.utils.data import Dataset


class RegimeAwareDataset(Dataset):
    def __init__(self, data, regime_labels, input_len=336, pred_len=96, target_idx=0):
        self.data = data
        self.regime_labels = regime_labels
        self.input_len = input_len
        self.pred_len = pred_len
        self.target_idx = target_idx

        expected_len = len(data) - input_len - pred_len + 1
        if len(regime_labels) != expected_len:
            raise ValueError(
                f"regime_labels length ({len(regime_labels)}) does not match "
                f"expected number of windows ({expected_len})"
            )

    def __len__(self):
        return len(self.regime_labels)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_len]
        y = self.data[
            idx + self.input_len: idx + self.input_len + self.pred_len,
            self.target_idx
        ]
        regime = self.regime_labels[idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        regime = torch.tensor(regime, dtype=torch.long)

        return x, y, regime