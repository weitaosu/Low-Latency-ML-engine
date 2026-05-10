"""Export PyTorch state_dict to flat .bin format the C++ engine reads via mmap.

Format:
    magic     4 bytes  = b"LLML"
    version   u32  LE  = 1
    n_tensors u32  LE
    per tensor:
      name_len u16 LE
      name     utf-8
      dtype    u8       (0=fp32, 1=int8, 2=int32)
      rank     u8
      shape    int64 LE × rank
      pad      0..31 zero bytes to 32-byte alignment
      data     raw tensor bytes

Usage:
    python scripts/export_bin.py             # exports both small and medium
    python scripts/export_bin.py --size small
"""
import argparse
import struct
from pathlib import Path
import numpy as np
import torch

REPO       = Path(__file__).parent.parent
MODELS_DIR = REPO / "models"

MAGIC   = b"LLML"
VERSION = 1
DTYPE_TO_CODE = {torch.float32: 0, torch.int8: 1, torch.int32: 2}
CODE_TO_DTYPE = {0: np.float32, 1: np.int8, 2: np.int32}


def _pad_to(n, alignment=32):
    return (-n) % alignment


def export_bin(state_dict: dict, out_path: Path) -> None:
    items = list(state_dict.items())
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<II", VERSION, len(items))
    for name, tensor in items:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.detach().cpu().contiguous()
        if tensor.dtype not in DTYPE_TO_CODE:
            raise ValueError(f"Unsupported dtype {tensor.dtype} for {name}")
        dtype_code = DTYPE_TO_CODE[tensor.dtype]
        name_bytes = name.encode("utf-8")
        rank = tensor.dim()
        shape = list(tensor.shape)
        buf += struct.pack("<H", len(name_bytes))
        buf += name_bytes
        buf += struct.pack("<BB", dtype_code, rank)
        buf += struct.pack(f"<{rank}q", *shape)
        buf += b"\x00" * _pad_to(len(buf), 32)
        buf += tensor.numpy().tobytes()
    out_path.write_bytes(bytes(buf))


def load_bin(path: Path) -> dict:
    raw = path.read_bytes()
    pos = 0
    if raw[pos:pos+4] != MAGIC:
        raise ValueError("bad magic")
    pos += 4
    version, n = struct.unpack_from("<II", raw, pos); pos += 8
    assert version == VERSION
    out = {}
    for _ in range(n):
        (name_len,) = struct.unpack_from("<H", raw, pos); pos += 2
        name = raw[pos:pos+name_len].decode("utf-8"); pos += name_len
        dt, rank = struct.unpack_from("<BB", raw, pos); pos += 2
        shape = struct.unpack_from(f"<{rank}q", raw, pos); pos += 8 * rank
        pos += _pad_to(pos, 32)
        dtype_np = CODE_TO_DTYPE[dt]
        count = int(np.prod(shape))
        arr = np.frombuffer(raw, dtype=dtype_np, count=count, offset=pos).reshape(shape).copy()
        pos += count * np.dtype(dtype_np).itemsize
        out[name] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["small", "medium", "both"], default="both")
    args = ap.parse_args()
    sizes = ["small", "medium"] if args.size == "both" else [args.size]
    for size in sizes:
        ckpt = torch.load(MODELS_DIR / f"{size}.pt", map_location="cpu", weights_only=False)
        out = MODELS_DIR / f"{size}.bin"
        export_bin(ckpt["state_dict"], out)
        # Round-trip check
        rt = load_bin(out)
        for name, t in ckpt["state_dict"].items():
            assert np.array_equal(t.detach().cpu().numpy(), rt[name]), name
        print(f"{size}: {out}  ({out.stat().st_size/1024:.1f} KB, {len(rt)} tensors)")


if __name__ == "__main__":
    main()
