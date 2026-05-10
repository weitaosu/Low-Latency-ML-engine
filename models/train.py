"""Train small or medium TimeSeriesTransformer on ETTh1.

Usage:
    python models/train.py --size small --epochs 30
    python models/train.py --size medium --epochs 30

Or import as a function:
    from train import train
    train("small", epochs=30)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, CONFIGS
from dataset import load_splits

REPO = Path(__file__).parent.parent
SPLITS_PATH = REPO / "models" / "splits.json"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    loss_fn = nn.MSELoss(reduction='sum')
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += loss_fn(pred, y).item()
        total_n += y.numel()
    return total_loss / total_n


def train(size: str, epochs: int = 30, batch_size: int = 64,
          lr: float = 1e-4, weight_decay: float = 1e-4, patience: int = 7,
          seed: int = 42, device: str = None, save: bool = True, verbose: bool = True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    train_ds, val_ds, test_ds = load_splits(SPLITS_PATH)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val, best_state, since_best = float("inf"), None, 0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item() * y.size(0)
            epoch_n += y.size(0)
        sched.step()
        train_mse = epoch_loss / epoch_n
        val_mse = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
                        "lr": optim.param_groups[0]["lr"], "time_s": elapsed})

        improved = val_mse < best_val
        if improved:
            best_val = val_mse
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1

        if verbose:
            print(f"epoch {epoch:3d} | train {train_mse:.5f} | val {val_mse:.5f} "
                  f"| best {best_val:.5f} | {elapsed:5.1f}s {'*' if improved else ''}")

        if since_best >= patience:
            if verbose: print(f"early stop after {epoch+1} epochs")
            break

    model.load_state_dict(best_state)
    test_mse = evaluate(model, test_loader, device)

    if save:
        out_path = REPO / "models" / f"{size}.pt"
        torch.save({
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": CONFIGS[size],
            "seed": seed,
            "epochs_trained": len(history),
            "best_val_mse": best_val,
            "test_mse": test_mse,
            "history": history,
        }, out_path)
        if verbose:
            print(f"\nsaved: {out_path}")
            print(f"  best val MSE: {best_val:.6f}")
            print(f"  test MSE:     {test_mse:.6f}")

    return {"best_val_mse": best_val, "test_mse": test_mse, "history": history}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=["small", "medium"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(args.size, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, seed=args.seed, device=args.device)
