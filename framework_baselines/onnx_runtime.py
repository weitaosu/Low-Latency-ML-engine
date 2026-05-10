"""ONNX Runtime baseline. Pins everything single-threaded."""
import sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "models"))
from framework_baselines._bench_harness import measure_py, pin_to_core
from model import build_model

size = sys.argv[1] if len(sys.argv) > 1 else "small"
onnx_path = REPO / "models" / f"{size}.onnx"

if not onnx_path.exists():
    ckpt = torch.load(REPO / "models" / f"{size}.pt", map_location="cpu", weights_only=False)
    model = build_model(size); model.load_state_dict(ckpt["state_dict"]); model.eval()
    dummy = torch.zeros(1, 96, 7)
    torch.onnx.export(model, dummy, onnx_path, input_names=["x"], output_names=["y"],
                      opset_version=17, dynamic_axes=None)
    print(f"exported {onnx_path}")

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                            providers=["CPUExecutionProvider"])

x = np.load(REPO / "data" / "inputs_1k.npy")[:1].astype(np.float32)
pin_to_core()

def fn(): sess.run(["y"], {"x": x})

measure_py(fn, stage="onnxruntime", size=size, precision="fp32",
           warmup_iters=200, measure_iters=10000)
