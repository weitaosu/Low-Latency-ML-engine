"""Mimic the C++ measure() rig: pin to core, mlock equivalent, capture timings."""
import os, time
from pathlib import Path
import numpy as np

ISOLATED_CORE = int(os.environ.get("ISOLATED_CORE", 3))
REPO = Path(__file__).parent.parent


def pin_to_core(core: int = ISOLATED_CORE):
    try: os.sched_setaffinity(0, {core})
    except AttributeError: pass


def measure_py(fn, *, stage: str, size: str, precision: str,
               warmup_iters: int = 100, measure_iters: int = 100000,
               csv_dir: Path = REPO / "results" / "csvs"):
    pin_to_core()
    for _ in range(warmup_iters): fn()
    timings = np.empty(measure_iters, dtype=np.uint64)
    perf_ns = time.perf_counter_ns
    for i in range(measure_iters):
        t0 = perf_ns(); fn(); timings[i] = perf_ns() - t0

    csv = csv_dir / f"{stage}_{size}_{precision}.csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    with open(csv, "w") as f:
        f.write(f"# stage={stage} size={size} precision={precision} "
                f"warmup={warmup_iters} measure={measure_iters} core={ISOLATED_CORE}\n")
        f.write("stage,size,precision,iteration,nanoseconds\n")
        for i, t in enumerate(timings):
            f.write(f"{stage},{size},{precision},{i},{t}\n")
    print(f"wrote {csv} (P50={np.percentile(timings,50):.0f}ns P99={np.percentile(timings,99):.0f}ns)")
