"""Compute per-stage test-set MSE for the Pareto plot.

Stages 0-3 and 5_fp32 are all bit-identical (FP32 path) — they share one MSE.
Stages 4a (per-tensor INT8) and 4b (per-channel INT8) use simulated quantization
in PyTorch with the same calibration parameters the C++ engines use, then
compute MSE on the full test set.

Output: results/tables/mse.json
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "models"))
from model import build_model

DATA = REPO / "data"
MODELS = REPO / "models"
OUT = REPO / "results" / "tables" / "mse.json"


def load_model(size):
    ckpt = torch.load(MODELS / f"{size}.pt", map_location="cpu", weights_only=False)
    m = build_model(size)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def quantize_per_tensor(W: torch.Tensor):
    """Symmetric INT8 per tensor; returns dequantized W'."""
    s = W.abs().max().item() / 127.0
    if s == 0: s = 1e-8
    Wq = (W / s).round().clamp(-127, 127)
    return Wq * s


def quantize_per_channel(W: torch.Tensor):
    """Symmetric INT8 per output channel (axis 0); returns dequantized W'."""
    s = W.abs().amax(dim=1) / 127.0
    s = torch.where(s == 0, torch.full_like(s, 1e-8), s)
    Wq = (W / s.unsqueeze(1)).round().clamp(-127, 127)
    return Wq * s.unsqueeze(1)


def quantize_act_dequantize(x: torch.Tensor, a_scale: float, a_zp: int):
    """Asymmetric uint8 round-trip on activation."""
    xq = (x / a_scale).round() + a_zp
    xq = xq.clamp(0, 255)
    return (xq - a_zp) * a_scale


@torch.no_grad()
def compute_int8_mse(size, mode, ref_y):
    """Simulate INT8 inference in PyTorch by replacing each Linear's weight with
    its quantized round-trip, and quantizing/dequantizing each Linear's input
    activation using the calibration scales."""
    cal = json.loads((MODELS / f"calibration_{size}.json").read_text())
    base = load_model(size)

    # Replace weights in-place with dequantized version
    quant_fn = quantize_per_tensor if mode == "4a" else quantize_per_channel
    for name, mod in base.named_modules():
        if isinstance(mod, nn.Linear) and name in cal["linears"]:
            with torch.no_grad():
                mod.weight.data = quant_fn(mod.weight.data)

    # Hook to quantize activations entering each Linear
    handles = []
    for name, mod in base.named_modules():
        if isinstance(mod, nn.Linear) and name in cal["linears"]:
            info = cal["linears"][name]
            a_scale = info["a_scale"]
            a_zp = int(info["a_zp"])
            def make_hook(a, z):
                def fn(m, inp):
                    return (quantize_act_dequantize(inp[0], a, z),)
                return fn
            handles.append(mod.register_forward_pre_hook(make_hook(a_scale, a_zp)))

    inputs = np.load(DATA / "inputs_test.npy")
    preds = []
    for i in range(0, len(inputs), 64):
        x = torch.from_numpy(inputs[i:i+64])
        preds.append(base(x).numpy())
    pred = np.concatenate(preds, axis=0)

    for h in handles: h.remove()

    # MSE between quantized engine output and the FROZEN PyTorch reference
    # (the same target the project trains/evaluates against would require y_true,
    # but the relevant comparison here is INT8-engine vs FP32-engine. For the
    # actual test MSE we'd need ground-truth y; we report drift here.)
    return float(np.mean((pred - ref_y) ** 2))


@torch.no_grad()
def compute_fp32_mse(size):
    """Test MSE of the FP32 model against ground-truth labels (the standard MSE)."""
    # Load the actual labels from the dataset
    sys.path.insert(0, str(MODELS))
    from dataset import load_splits
    _, _, test_ds = load_splits(MODELS / "splits.json")
    m = load_model(size)
    inputs = np.load(DATA / "inputs_test.npy")
    preds = []
    for i in range(0, len(inputs), 64):
        x = torch.from_numpy(inputs[i:i+64])
        preds.append(m(x).numpy())
    pred = np.concatenate(preds, axis=0)
    # Reconstruct labels by walking the test_ds (same indices as inputs_test)
    labels = np.stack([test_ds[i][1].numpy() for i in range(len(test_ds))])
    mse = float(np.mean((pred - labels) ** 2))
    return mse, pred


def main():
    out = {}
    for size in ["small", "medium"]:
        print(f"=== {size} ===")
        fp32_mse, ref_pred = compute_fp32_mse(size)
        print(f"  FP32 test MSE (vs ground truth): {fp32_mse:.5f}")

        # Stages 0-3 and 5_fp32 share this MSE (bit-identical FP32)
        for stage in ["stage0", "stage1", "stage2", "stage3", "stage5_fp32"]:
            out[f"{stage}_{size}"] = fp32_mse

        # INT8 stages: simulate then compute MSE vs ground truth too
        # (so they're directly comparable to FP32 numbers)
        for mode, key in [("4a", "stage4a"), ("4b", "stage4b")]:
            int8_pred_mse_vs_fp = compute_int8_mse(size, mode, ref_pred)
            print(f"  INT8 {mode} drift vs FP32 reference: {int8_pred_mse_vs_fp:.6f}")
            # We want MSE vs ground truth; recompute by replacing the weights again
            # (cheaper to rerun)
            sys.path.insert(0, str(MODELS))
            from dataset import load_splits
            _, _, test_ds = load_splits(MODELS / "splits.json")
            labels = np.stack([test_ds[i][1].numpy() for i in range(len(test_ds))])

            # Re-run quantized inference
            cal = json.loads((MODELS / f"calibration_{size}.json").read_text())
            qm = load_model(size)
            quant_fn = quantize_per_tensor if mode == "4a" else quantize_per_channel
            for name, mod in qm.named_modules():
                if isinstance(mod, nn.Linear) and name in cal["linears"]:
                    with torch.no_grad():
                        mod.weight.data = quant_fn(mod.weight.data)
            handles = []
            for name, mod in qm.named_modules():
                if isinstance(mod, nn.Linear) and name in cal["linears"]:
                    info = cal["linears"][name]
                    def make_hook(a, z):
                        def fn(m, inp): return (quantize_act_dequantize(inp[0], a, z),)
                        return fn
                    handles.append(mod.register_forward_pre_hook(make_hook(info["a_scale"], int(info["a_zp"]))))
            preds = []
            inputs = np.load(DATA / "inputs_test.npy")
            with torch.no_grad():
                for i in range(0, len(inputs), 64):
                    x = torch.from_numpy(inputs[i:i+64])
                    preds.append(qm(x).numpy())
            pred = np.concatenate(preds, axis=0)
            for h in handles: h.remove()
            int8_mse = float(np.mean((pred - labels) ** 2))
            print(f"  INT8 {mode} test MSE: {int8_mse:.5f}  (FP32: {fp32_mse:.5f}, "
                  f"degradation: {(int8_mse-fp32_mse)/fp32_mse*100:+.1f}%)")
            out[f"{key}_{size}"] = int8_mse

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
