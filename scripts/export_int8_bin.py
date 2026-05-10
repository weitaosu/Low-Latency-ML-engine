"""Combine the FP32 .bin with calibration.json + weights_4{a,b}.npz to produce
.bin files the C++ INT8 stages can mmap-load directly.

Output bins contain:
- All FP32 tensors from the original (unchanged)
- Per Linear that's quantized:
    {name}.weight_q       int8 [out, in]
    {name}.sum_W          int32 [out]    (sum of int8 weights along input dim)
    {name}.a_scale        fp32 [1]
    {name}.a_zp           fp32 [1]       (stored as fp32 for simplicity; convert at use)
    {name}.w_scale        fp32 [1] for 4a, [out] for 4b
- The original .weight tensors are KEPT (so the engine can fall back, and
  bias is reused). The C++ engine ignores .weight in favor of .weight_q at
  load time.

Usage:
    python scripts/export_int8_bin.py --size small --mode 4a
    python scripts/export_int8_bin.py --size small --mode 4b
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from export_bin import export_bin   # reuse the writer

MAGIC = b"LLML"


def load_existing_bin(path):
    """Read all tensors from a .bin into a dict of (name, numpy array)."""
    raw = path.read_bytes()
    pos = 0
    if raw[:4] != MAGIC: raise ValueError("bad magic")
    pos = 4
    version, n = struct.unpack_from("<II", raw, pos); pos += 8
    DT = {0: np.float32, 1: np.int8, 2: np.int32}
    out = {}
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", raw, pos); pos += 2
        nm = raw[pos:pos+nl].decode(); pos += nl
        dt, rk = struct.unpack_from("<BB", raw, pos); pos += 2
        sh = struct.unpack_from(f"<{rk}q", raw, pos); pos += 8 * rk
        pos += (-pos) % 32
        cnt = int(np.prod(sh))
        a = np.frombuffer(raw, dtype=DT[dt], count=cnt, offset=pos).reshape(sh).copy()
        pos += cnt * np.dtype(DT[dt]).itemsize
        out[nm] = a
    return out


def to_torch_dict(d):
    """The export_bin function expects torch tensors. Wrap numpy arrays."""
    import torch
    return {k: torch.from_numpy(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=["small", "medium"])
    ap.add_argument("--mode", required=True, choices=["4a", "4b"])
    args = ap.parse_args()

    fp32_bin = REPO / "models" / f"{args.size}.bin"
    calib_json = REPO / "models" / f"calibration_{args.size}.json"
    int8_npz = REPO / "models" / f"weights_{args.mode}_{args.size}.npz"
    out_bin = REPO / "models" / f"{args.size}_int8_{args.mode}.bin"

    for f in [fp32_bin, calib_json, int8_npz]:
        if not f.exists(): raise FileNotFoundError(f)

    fp32 = load_existing_bin(fp32_bin)
    calib = json.loads(calib_json.read_text())
    int8 = np.load(int8_npz)

    out = dict(fp32)   # start with all fp32 tensors

    for layer_name, info in calib["linears"].items():
        # int8 quantized weights
        out[f"{layer_name}.weight_q"] = int8[f"{layer_name}.weight_q"]
        out[f"{layer_name}.sum_W"]    = int8[f"{layer_name}.sum_W"]
        # scalars stored as 1-element fp32 tensors
        out[f"{layer_name}.a_scale"]  = np.array([info["a_scale"]], dtype=np.float32)
        out[f"{layer_name}.a_zp"]     = np.array([float(info["a_zp"])], dtype=np.float32)
        if args.mode == "4a":
            out[f"{layer_name}.w_scale"] = np.array(
                [info["per_tensor"]["w_scale"]], dtype=np.float32)
        else:  # 4b
            out[f"{layer_name}.w_scale"] = np.array(
                info["per_channel"]["w_scale"], dtype=np.float32)

    export_bin(to_torch_dict(out), out_bin)
    print(f"wrote {out_bin}  ({out_bin.stat().st_size/1024:.1f} KB, {len(out)} tensors)")


if __name__ == "__main__":
    main()
