"""Freeze the inputs / outputs / calibration sets used by every C++ stage.

Re-running with the same FREEZE_SEED produces byte-identical files.
Do not change FREEZE_SEED after the project starts using the .npy files.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path
import numpy as np
import torch

REPO       = Path(__file__).parent.parent
MODELS_DIR = REPO / "models"
DATA_DIR   = REPO / "data"
sys.path.insert(0, str(MODELS_DIR))
from model import build_model
from dataset import load_splits

FREEZE_SEED = 1234


def stack(ds, idx):
    return np.stack([ds[i][0].numpy() for i in idx]).astype(np.float32)


@torch.no_grad()
def predict_all(model, x_np, batch=64):
    outs = []
    for i in range(0, len(x_np), batch):
        outs.append(model(torch.from_numpy(x_np[i:i+batch])).numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def main():
    train_ds, _, test_ds = load_splits(MODELS_DIR / "splits.json")
    rng = np.random.default_rng(FREEZE_SEED)

    test_idx = np.sort(rng.choice(len(test_ds), 1000, replace=False))
    train_idx = np.sort(rng.choice(len(train_ds), 5000, replace=False))

    inputs_1k   = stack(test_ds, test_idx)
    inputs_test = stack(test_ds, list(range(len(test_ds))))
    calib_5k    = stack(train_ds, train_idx)

    np.save(DATA_DIR / "inputs_1k.npy",         inputs_1k)
    np.save(DATA_DIR / "inputs_test.npy",       inputs_test)
    np.save(DATA_DIR / "calib_5k.npy",          calib_5k)
    np.save(DATA_DIR / "inputs_1k_indices.npy", test_idx)
    np.save(DATA_DIR / "calib_5k_indices.npy",  train_idx)

    for size in ["small", "medium"]:
        ckpt = torch.load(MODELS_DIR / f"{size}.pt", map_location="cpu", weights_only=False)
        model = build_model(size); model.load_state_dict(ckpt["state_dict"]); model.eval()
        np.save(DATA_DIR / f"outputs_1k_{size}.npy", predict_all(model, inputs_1k))
        print(f"{size}: outputs_1k_{size}.npy written")

    print("\nFrozen sets written to", DATA_DIR)


if __name__ == "__main__":
    main()
