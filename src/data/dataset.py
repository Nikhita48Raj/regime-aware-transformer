import torch
from torch.utils.data import Dataset


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, data, input_len=336, pred_len=96, target_idx=0):
        self.data = data
        self.input_len = input_len
        self.pred_len = pred_len
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[
            idx + self.input_len : idx + self.input_len + self.pred_len,
            self.target_idx
        ]

        x = torch.tensor(x, dtype=torch.float32)              # [input_len, num_features]
        y = torch.tensor(y, dtype=torch.float32)              # [pred_len]

        return x, y