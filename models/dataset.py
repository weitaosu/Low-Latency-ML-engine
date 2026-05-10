"""ETTh1 dataset and split loader. Frozen at Step 3."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ETTDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int = 96, horizon: int = 96):
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, t):
        x = self.data[t : t + self.seq_len]
        y = self.data[t + self.seq_len : t + self.seq_len + self.horizon]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def load_splits(splits_path):
    splits = json.loads(Path(splits_path).read_text())
    csv_path = Path(splits_path).parent.parent / "data" / "ett_raw" / "ETTh1.csv"
    df = pd.read_csv(csv_path)
    data = df[splits["features"]].to_numpy(dtype=np.float32)
    mean = np.array(splits["normalization"]["mean"], dtype=np.float32)
    std  = np.array(splits["normalization"]["std"],  dtype=np.float32)
    s = splits["split_indices"]
    train = ((data[s["train"][0]:s["train"][1]] - mean) / std).astype(np.float32)
    val   = ((data[s["val"][0]:s["val"][1]]     - mean) / std).astype(np.float32)
    test  = ((data[s["test"][0]:s["test"][1]]   - mean) / std).astype(np.float32)
    return (
        ETTDataset(train, splits["seq_len"], splits["horizon"]),
        ETTDataset(val,   splits["seq_len"], splits["horizon"]),
        ETTDataset(test,  splits["seq_len"], splits["horizon"]),
    )
