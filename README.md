# Low-Latency ML (LLML): Tail Latency Decomposition for CPU Transformer Inference

Final project for **Advanced ML Systems**, 2026. Author: Weitao Su (`ws535@cornell.edu`). Source: [github.com/weitaosu/Low-Latency-ML-engine](https://github.com/weitaosu/Low-Latency-ML-engine).

A six-stage ablation of a hand-coded C++ inference engine for an encoder-only Transformer on the ETTh1 time-series dataset, designed to attribute every increment of P99 tail-latency reduction to a single, named optimization (preallocation, fusion, cache tiling, INT8 quantization, AVX2 SIMD).

The full writeup is at [`paper/REPORT.pdf`](paper/REPORT.pdf).

## Headline result

Cumulative speedup vs naive Stage 0 at P99, after 100,000 measurement iterations on a kernel-isolated CPU core (Intel Alder Lake P-core, turbo off, swap off):


|                          | small (365K params) | medium (1.9M params) |
| ------------------------ | ------------------- | -------------------- |
| Stage 0 (naive FP32)     | 27.68 ms            | 85.56 ms             |
| Stage 5 (AVX2 FP32 SIMD) | **5.11 ms**         | **24.54 ms**         |
| Cumulative speedup       | **5.42×**          | **3.49×**           |
| ONNX Runtime (reference) | 3.89 ms             | 10.69 ms             |

The headline finding is the *overhead-versus-compute crossover*: preallocation (Stage 1) reduces small-model P99 by 2.18× but does nothing on medium, while cache tiling (Stage 3) shows the inverse pattern (−2.7% small, +10.5% medium). See `paper/REPORT.pdf` Section 4 for the complete six-stage table and interpretation.

## Repository layout

```
low_latency_ml/
├── README.md                            (this file)
│
├── paper/                               (final report)
│   ├── REPORT.pdf                       ← the deliverable
│   ├── REPORT.tex                       (ICML 2026 LaTeX source)
│   ├── references.bib
│   ├── icml2026.{sty,bst}, fancyhdr.sty, algorithm{,ic}.sty
│   └── build.sh                         (Tectonic-based build)
│
├── cpp/                                 (C++ inference engines, one per stage)
│   ├── CMakeLists.txt
│   ├── common/                          (tensor.h, weight loader, npy loader,
│   │                                     correctness, measure, alloc_guard)
│   ├── stage0_naive/
│   ├── stage1_prealloc/
│   ├── stage2_fused/
│   ├── stage3_tiled/
│   ├── stage4a_int8_pertensor/
│   ├── stage4b_int8_perchannel/
│   └── stage5_fp32_simd/                (AVX2 intrinsics)
│
├── models/                              (trained checkpoints + weight binaries)
│   ├── train.py, model.py, dataset.py
│   ├── splits.json                      (train/val/test split + normalization)
│   ├── small.pt, medium.pt              (PyTorch checkpoints)
│   ├── small.bin, medium.bin            (custom mmap-able binary format)
│   ├── small_int8_4{a,b}.bin            (INT8-quantized variants)
│   ├── weights_4{a,b}_{small,medium}.npz
│   ├── calibration_{small,medium}.json  (per-Linear scales + zero points)
│   └── small.onnx, medium.onnx          (for ONNX Runtime baseline)
│
├── data/                                (frozen inputs / outputs / calibration)
│   ├── ett_raw/                         (raw ETTh1.csv)
│   ├── inputs_1k.npy                    (correctness check inputs, 1000 samples)
│   ├── outputs_1k_{small,medium}.npy    (PyTorch reference outputs, frozen)
│   ├── inputs_test.npy                  (full test split, 2689 samples)
│   ├── calib_5k.npy                     (5000-sample INT8 calibration set)
│   └── *_indices.npy                    (sample indices for traceability)
│
├── scripts/                             (Python utilities)
│   ├── setup_env.sh, verify_rig.sh      (rig configuration & verification)
│   ├── freeze_data.py                   (regenerate frozen .npy files)
│   ├── export_bin.py                    (PyTorch → flat .bin)
│   ├── export_int8_bin.py               (calibration JSON + npz → INT8 .bin)
│   ├── calibrate.py                     (INT8 histogram calibration)
│   └── compute_per_stage_mse.py         (per-stage test-set MSE for Pareto plot)
│
├── analysis/
│   └── analysis.py                      (CSVs → ablation table + 4 figures)
│
├── framework_baselines/                 (Python framework comparisons)
│   ├── _bench_harness.py
│   ├── pytorch_eager.py
│   ├── torchscript.py
│   └── onnx_runtime.py
│
├── python_baseline/
│   └── numpy_engine.py                  (NumPy-only orchestration baseline)
│
└── results/
    ├── csvs/                            (raw 100K-sample timings, one per run)
    ├── figures/                         (noise_floor_cdf, cdf_overlay, gap_p99_p50, pareto)
    └── tables/                          (ablation.csv, mse.json)
```

## Reproducing the results

### 1. Lock the measurement rig (requires sudo + reboot)

The measurements assume the host is configured for low-jitter benchmarking. From a terminal:

```bash
# Append to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub:
#   isolcpus=3 nohz_full=3 rcu_nocbs=3
sudo update-grub
sudo cpupower frequency-set -g performance
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
sudo swapoff -a
sudo systemctl stop bluetooth NetworkManager-wait-online cron snapd packagekit
sudo reboot

# After reboot:
bash scripts/verify_rig.sh   # should print "rig OK"
```

### 2. Set up Python environment

```bash
conda create -n llml python=3.11 -y
conda activate llml
pip install torch numpy pandas matplotlib pytest python-docx onnx onnxruntime
```

### 3. Verify the noise floor

```bash
gcc -O2 -o cpp/common/noise_floor cpp/common/noise_floor.c
mkdir -p results/csvs
taskset -c 3 ./cpp/common/noise_floor > results/csvs/noise_floor.csv
```

Target: P50 ∈ [15, 30] ns; P99 within 2-3× P50; no microsecond-scale tail.

### 4. Build the C++ engines

```bash
cmake -B cpp/build -S cpp
cmake --build cpp/build
```

This produces seven binaries: `cpp/build/stage{0_naive,1_prealloc,2_fused,3_tiled,4a_int8_pertensor,4b_int8_perchannel,5_fp32_simd}/<name>`.

### 5. Run the benchmarks

```bash
# C++ stages (per-stage 100K iterations with full correctness check):
for stage in stage0_naive stage1_prealloc stage2_fused stage3_tiled \
             stage4a_int8_pertensor stage4b_int8_perchannel stage5_fp32_simd; do
    for size in small medium; do
        REPO=$(pwd) taskset -c 3 ./cpp/build/$stage/$stage $size
    done
done

# Python framework baselines:
for f in framework_baselines/{pytorch_eager,torchscript,onnx_runtime}.py \
         python_baseline/numpy_engine.py; do
    for size in small medium; do
        taskset -c 3 python3 $f $size
    done
done
```

Each run produces a self-documenting CSV in `results/csvs/` with the rig configuration in the header.

### 6. Generate the figures and tables

```bash
python3 scripts/compute_per_stage_mse.py     # writes results/tables/mse.json
python3 analysis/analysis.py                 # writes ablation table + 4 figures
```

### 7. Build the paper PDF

```bash
cd paper && ./build.sh                       # uses Tectonic; auto-installs if missing
```

Output: `paper/REPORT.pdf` (9 pages, ICML 2026 format).

## Verification

If you want to check that the data really comes from a clean run, the rig configuration is recorded inside every CSV header. For example:

```
$ head -2 results/csvs/stage0_small_fp32.csv
# stage=stage0 size=small precision=fp32 warmup=100 measure=100000 core=3
# isolated_cpus=3 no_turbo=1 perf_paranoid=1
```

Every numeric claim in `paper/REPORT.pdf` traces to a CSV in `results/csvs/` plus a row of `results/tables/ablation.csv`. Per-stage test MSE values come from `results/tables/mse.json`. The four published figures are regenerated by `analysis/analysis.py` deterministically from those CSVs.

## Limitations summary

- **CPU only**, batch size 1, no KV cache, encoder-only, single sequence length 96
- **Single dataset** (ETTh1)
- **Single machine** (Intel Alder Lake P-core with AVX-VNNI)
- **Stage 5 INT8 SIMD not implemented**: VNNI scaffold left for follow-up
- **Rust port not delivered**: scope-cut; see `paper/REPORT.pdf` Section 6

See `paper/REPORT.pdf` Section 6 for the full discussion.
