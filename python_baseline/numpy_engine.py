"""NumPy-only inference. np.matmul, manual LayerNorm/GELU."""
import os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from framework_baselines._bench_harness import measure_py, pin_to_core

MAGIC = b"LLML"
import struct

def load_bin(path):
    raw = Path(path).read_bytes(); pos = 0
    assert raw[:4] == MAGIC; pos = 4
    _, n = struct.unpack_from("<II", raw, pos); pos += 8
    out = {}
    DT = {0: np.float32, 1: np.int8, 2: np.int32}
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", raw, pos); pos += 2
        nm = raw[pos:pos+nl].decode(); pos += nl
        dt, rk = struct.unpack_from("<BB", raw, pos); pos += 2
        sh = struct.unpack_from(f"<{rk}q", raw, pos); pos += 8*rk
        pos += (-pos) % 32
        cnt = int(np.prod(sh))
        a = np.frombuffer(raw, dtype=DT[dt], count=cnt, offset=pos).reshape(sh).copy()
        pos += cnt * np.dtype(DT[dt]).itemsize
        out[nm] = a
    return out

def gelu_tanh(x):
    return 0.5*x*(1.0+np.tanh(0.7978845608*(x + 0.044715*x*x*x)))

def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def forward(x, W, cfg):
    T, n_vars = cfg["seq_len"], cfg["n_vars"]
    D, H, F = cfg["d_model"], cfg["n_heads"], cfg["d_ff"]; Dh = D // H
    h = x @ W["input_proj.weight"].T + W["input_proj.bias"]
    h = h + W["pos"]
    for L in range(cfg["n_layers"]):
        p = f"blocks.{L}."
        ht = layer_norm(h, W[p+"ln1.weight"], W[p+"ln1.bias"])
        qkv = ht @ W[p+"qkv.weight"].T + W[p+"qkv.bias"]
        q, k, v = qkv[..., :D], qkv[..., D:2*D], qkv[..., 2*D:]
        q = q.reshape(T, H, Dh).transpose(1, 0, 2)
        k = k.reshape(T, H, Dh).transpose(1, 0, 2)
        v = v.reshape(T, H, Dh).transpose(1, 0, 2)
        sc = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(Dh))
        a = (sc @ v).transpose(1, 0, 2).reshape(T, D)
        h = h + (a @ W[p+"attn_out.weight"].T + W[p+"attn_out.bias"])
        ht = layer_norm(h, W[p+"ln2.weight"], W[p+"ln2.bias"])
        ff = gelu_tanh(ht @ W[p+"ff1.weight"].T + W[p+"ff1.bias"])
        h = h + (ff @ W[p+"ff2.weight"].T + W[p+"ff2.bias"])
    pooled = h.mean(0)
    out = pooled @ W["head.weight"].T + W["head.bias"]
    return out.reshape(cfg["horizon"], cfg["n_vars"])

if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    CFG = {"small":  dict(n_vars=7,seq_len=96,horizon=96,d_model=128,n_heads=4,n_layers=2,d_ff=256),
           "medium": dict(n_vars=7,seq_len=96,horizon=96,d_model=192,n_heads=6,n_layers=4,d_ff=768)}
    W = load_bin(REPO / "models" / f"{size}.bin"); cfg = CFG[size]
    x = np.load(REPO / "data" / "inputs_1k.npy")[0]
    pin_to_core()
    def fn(): forward(x, W, cfg)
    measure_py(fn, stage="numpy", size=size, precision="fp32",
               warmup_iters=20, measure_iters=2000)
