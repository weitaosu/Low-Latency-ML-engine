"""INT8 calibration. Records per-activation histograms (256 bins) and per-tensor /
per-output-channel weight stats. Emits calibration.json and quantized weights.

Usage: python scripts/calibrate.py --size small
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "models"))
from model import build_model, CONFIGS

N_BINS = 256


class ActivationRecorder:
    """Forward hook that records min, max, and a 256-bin histogram per activation."""
    def __init__(self):
        self.stats = {}    # name -> dict

    def hook(self, name):
        def fn(module, inp, out):
            x = inp[0].detach().cpu().numpy().ravel()
            s = self.stats.setdefault(name, {"min": +np.inf, "max": -np.inf,
                                             "hist": None, "hist_lo": None, "hist_hi": None})
            s["min"] = float(min(s["min"], x.min()))
            s["max"] = float(max(s["max"], x.max()))
            # On first batch, fix histogram range with a margin
            if s["hist"] is None:
                lo, hi = x.min(), x.max()
                margin = (hi - lo) * 0.5 + 1e-6
                s["hist_lo"], s["hist_hi"] = float(lo - margin), float(hi + margin)
                s["hist"] = np.zeros(N_BINS, dtype=np.int64)
            h, _ = np.histogram(x, bins=N_BINS, range=(s["hist_lo"], s["hist_hi"]))
            s["hist"] += h
        return fn


def quantize_weight_per_tensor(W: np.ndarray):
    """Symmetric INT8: W ≈ W_q * w_scale. Returns (W_q int8, w_scale float)."""
    m = float(np.abs(W).max())
    w_scale = m / 127.0 if m > 0 else 1e-8
    W_q = np.clip(np.round(W / w_scale), -127, 127).astype(np.int8)
    return W_q, w_scale


def quantize_weight_per_channel(W: np.ndarray):
    """Symmetric INT8 per output channel. W shape [out, in]. Returns (W_q, w_scale [out])."""
    m = np.abs(W).max(axis=1)
    w_scale = np.where(m > 0, m / 127.0, 1e-8).astype(np.float32)
    W_q = np.clip(np.round(W / w_scale[:, None]), -127, 127).astype(np.int8)
    return W_q, w_scale


def derive_activation_scale(stat: dict, mode: str = "minmax"):
    """Returns (a_scale, a_zp). Asymmetric uint8."""
    if mode == "minmax":
        lo, hi = stat["min"], stat["max"]
    elif mode == "percentile":
        # 99.9 percentile from histogram
        h = stat["hist"]; total = h.sum()
        edges = np.linspace(stat["hist_lo"], stat["hist_hi"], N_BINS + 1)
        cdf = np.cumsum(h) / total
        lo_idx = np.searchsorted(cdf, 0.0005)
        hi_idx = np.searchsorted(cdf, 0.9995)
        lo, hi = float(edges[lo_idx]), float(edges[hi_idx])
    else:
        raise ValueError(mode)
    a_scale = max((hi - lo) / 255.0, 1e-8)
    a_zp = int(round(-lo / a_scale))
    a_zp = max(0, min(255, a_zp))
    return float(a_scale), int(a_zp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=["small", "medium"])
    ap.add_argument("--mode", default="minmax", choices=["minmax", "percentile"])
    args = ap.parse_args()

    ckpt = torch.load(REPO / "models" / f"{args.size}.pt", map_location="cpu", weights_only=False)
    model = build_model(args.size); model.load_state_dict(ckpt["state_dict"]); model.eval()

    # Hook every Linear's input
    rec = ActivationRecorder()
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(rec.hook(name)))

    calib = np.load(REPO / "data" / "calib_5k.npy")
    with torch.no_grad():
        for i in range(0, len(calib), 64):
            model(torch.from_numpy(calib[i:i+64]))
    for h in handles: h.remove()

    # Build calibration JSON: per-tensor scales for 4a, per-channel for 4b
    out = {"size": args.size, "mode": args.mode, "linears": {}}
    weights_4a, weights_4b = {}, {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear): continue
        W = mod.weight.detach().cpu().numpy()
        b = mod.bias.detach().cpu().numpy() if mod.bias is not None else None

        # Activation scale (input to this Linear)
        a_scale, a_zp = derive_activation_scale(rec.stats[name], args.mode)

        # 4a: per-tensor weight
        Wq_a, ws_a = quantize_weight_per_tensor(W)
        # 4b: per-channel weight
        Wq_b, ws_b = quantize_weight_per_channel(W)

        # sum_W_per_col: precomputed for fast dequant (sum along input dim per output)
        sumWa = Wq_a.sum(axis=1).astype(np.int32)
        sumWb = Wq_b.sum(axis=1).astype(np.int32)

        out["linears"][name] = {
            "a_scale": a_scale, "a_zp": a_zp,
            "per_tensor":  {"w_scale": float(ws_a)},
            "per_channel": {"w_scale": ws_b.tolist()},
            "shape": list(W.shape),
        }
        weights_4a[name + ".weight_q"] = Wq_a;  weights_4a[name + ".sum_W"] = sumWa
        weights_4b[name + ".weight_q"] = Wq_b;  weights_4b[name + ".sum_W"] = sumWb
        if b is not None:
            weights_4a[name + ".bias"] = b.astype(np.float32)
            weights_4b[name + ".bias"] = b.astype(np.float32)

    out_dir = REPO / "models"
    (out_dir / f"calibration_{args.size}.json").write_text(json.dumps(out, indent=2))
    np.savez(out_dir / f"weights_4a_{args.size}.npz", **weights_4a)
    np.savez(out_dir / f"weights_4b_{args.size}.npz", **weights_4b)

    print(f"calibration_{args.size}.json: {len(out['linears'])} linears")
    print(f"weights_4a_{args.size}.npz, weights_4b_{args.size}.npz written")


if __name__ == "__main__":
    main()
