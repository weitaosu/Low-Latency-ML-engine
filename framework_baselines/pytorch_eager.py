"""PyTorch eager-mode baseline. taskset -c 3 python framework_baselines/pytorch_eager.py small"""
import os, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "models"))
from framework_baselines._bench_harness import measure_py, pin_to_core
from model import build_model

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

size = sys.argv[1] if len(sys.argv) > 1 else "small"
ckpt = torch.load(REPO / "models" / f"{size}.pt", map_location="cpu", weights_only=False)
model = build_model(size); model.load_state_dict(ckpt["state_dict"]); model.eval()

x = torch.from_numpy(np.load(REPO / "data" / "inputs_1k.npy")[:1])
pin_to_core()

@torch.no_grad()
def fn(): _ = model(x)

measure_py(fn, stage="pytorch_eager", size=size, precision="fp32",
           warmup_iters=50, measure_iters=10000)
