"""Analyze all CSVs in results/csvs/ -> ablation table + 4 figures + CIs.
Run: python analysis/analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).parent.parent
CSV_DIR = REPO / "results" / "csvs"
FIG_DIR = REPO / "results" / "figures"
TBL_DIR = REPO / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True); TBL_DIR.mkdir(parents=True, exist_ok=True)

STAGES_ORDER = ["stage0", "stage1", "stage2", "stage3",
                "stage4a", "stage4b", "stage5_fp32", "stage5_int8",
                "pytorch_eager", "torchscript", "onnxruntime", "numpy", "rust"]

STAGE_LABEL = {
    "stage0": "Stage 0 (naive FP32)",
    "stage1": "Stage 1 (prealloc)",
    "stage2": "Stage 2 (LN+Linear fused)",
    "stage3": "Stage 3 (tiled scalar)",
    "stage4a": "Stage 4a (INT8 per-tensor)",
    "stage4b": "Stage 4b (INT8 per-channel)",
    "stage5_fp32": "Stage 5 (FP32 AVX2 SIMD)",
}

STAGE_COLOR = {
    "stage0": "#d62728",
    "stage1": "#ff7f0e",
    "stage2": "#bcbd22",
    "stage3": "#2ca02c",
    "stage4a": "#17becf",
    "stage4b": "#1f77b4",
    "stage5_fp32": "#9467bd",
}

MIN_SAMPLES = 1000   # skip smoke-test CSVs


def load_all_csvs():
    rows = []
    for f in sorted(CSV_DIR.glob("*.csv")):
        if "noise_floor" in f.name: continue
        df = pd.read_csv(f, comment="#")
        if len(df) < MIN_SAMPLES:
            print(f"skipping {f.name} ({len(df)} samples < {MIN_SAMPLES})")
            continue
        df["file"] = f.name
        rows.append(df)
    if not rows: raise SystemExit("no CSVs found in " + str(CSV_DIR))
    return pd.concat(rows, ignore_index=True)


def bootstrap_p99_ci(timings, n_boot=1000, lo=2.5, hi=97.5, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(timings)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.percentile(timings[idx], 99)
    return float(np.percentile(boots, lo)), float(np.percentile(boots, hi))


def per_config_stats(df):
    out = []
    for (stage, size, prec), g in df.groupby(["stage", "size", "precision"]):
        ns = g["nanoseconds"].values
        p50 = np.percentile(ns, 50); p99 = np.percentile(ns, 99)
        p999 = np.percentile(ns, 99.9)
        ci_lo, ci_hi = bootstrap_p99_ci(ns)
        out.append(dict(stage=stage, size=size, precision=prec,
                        p50_ns=p50, p99_ns=p99, p999_ns=p999,
                        gap_p99_p50=p99 - p50,
                        ci_p99_lo=ci_lo, ci_p99_hi=ci_hi,
                        n_samples=len(ns)))
    s = pd.DataFrame(out)
    s["_order"] = s["stage"].apply(lambda x: STAGES_ORDER.index(x) if x in STAGES_ORDER else 99)
    return s.sort_values(["_order", "size", "precision"]).drop(columns="_order")


def ablation_table(stats):
    s0 = stats[stats.stage == "stage0"].set_index("size")["p99_ns"]
    speedup = (s0.reindex(stats["size"]).values / stats["p99_ns"].values)
    stats = stats.assign(speedup_vs_stage0=speedup)
    print("\n===== Ablation table =====")
    print(stats.to_string(index=False))
    stats.to_csv(TBL_DIR / "ablation.csv", index=False)


def noise_floor_cdf():
    """Single-panel CDF of clock_gettime jitter with one annotated percentile marker."""
    csv = CSV_DIR / "noise_floor.csv"
    if not csv.exists():
        print(f"skip noise_floor: {csv} missing"); return
    df = pd.read_csv(csv)
    ns = np.sort(df["nanoseconds"].values)
    cdf = np.arange(1, len(ns) + 1) / len(ns)

    p50 = np.percentile(ns, 50)
    p99 = np.percentile(ns, 99)
    p999 = np.percentile(ns, 99.9)
    p9999 = np.percentile(ns, 99.99)
    n_outliers = int((ns > 1000).sum())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.semilogx(ns, cdf, color="#1f77b4", lw=1.6)
    ax.axvline(p99, color="#888", ls="--", lw=1.0)
    ax.set_xlim(10, 2000)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("clock_gettime delta (ns, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("Noise floor: clock_gettime jitter (1M samples, isolated core 3)")
    ax.grid(True, alpha=0.3)

    txt = (f"P50  = {int(p50)} ns\n"
           f"P99  = {int(p99)} ns\n"
           f"P99.9  = {int(p999)} ns\n"
           f"P99.99 = {int(p9999)} ns\n"
           f"{n_outliers} samples in 1M > 1 us\n"
           f"max = {int(ns.max())} ns")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes,
            ha="right", va="bottom", family="monospace", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaa", alpha=0.95))
    ax.annotate(f"{n_outliers} outliers\n(max {int(ns.max())} ns)",
                xy=(ns.max(), 1.0), xytext=(900, 0.78),
                fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "noise_floor_cdf.png", dpi=140)
    plt.close()


def cdf_overlay(df):
    """Two-panel CDF (medium, small) with all seven stages colored consistently.
    Notes overlapping curves in the legend where they coincide visually.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    panel_order = [("small", axes[0]), ("medium", axes[1])]
    for size, ax in panel_order:
        sub = df[df["size"] == size]
        for stage in ["stage0", "stage1", "stage2", "stage3",
                      "stage4a", "stage4b", "stage5_fp32"]:
            g = sub[sub["stage"] == stage]
            if g.empty: continue
            ns = np.sort(g["nanoseconds"].values)
            cdf_y = np.arange(1, len(ns) + 1) / len(ns)
            ax.semilogx(ns / 1e6, cdf_y,
                        label=STAGE_LABEL[stage],
                        color=STAGE_COLOR[stage], lw=1.4)
        ax.set_xlabel("latency (ms, log scale)")
        ax.set_title(f"{size} model")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.92)
    axes[0].set_ylabel("CDF")
    plt.suptitle("Per-stage latency CDF (100K iterations each)", y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cdf_overlay.png", dpi=140, bbox_inches="tight")
    plt.close()


def gap_bar(stats):
    """Tail-gap (P99 - P50) by stage, log y-axis in microseconds.
    Stage 0 small is the outlier; log scale keeps the other six stages legible.
    """
    sub = stats[stats["stage"].str.startswith("stage") &
                ~stats["stage"].isin(["stage5_int8"])].copy()
    pivot = sub.pivot_table(index="stage", columns="size", values="gap_p99_p50")
    pivot = pivot.reindex([s for s in STAGES_ORDER if s in pivot.index])
    pivot_us = pivot / 1000.0  # ns -> us

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(pivot_us))
    width = 0.36
    bars_small = ax.bar(x - width/2, pivot_us["small"], width,
                        label="small", color="#ff7f0e", edgecolor="black", lw=0.4)
    bars_medium = ax.bar(x + width/2, pivot_us["medium"], width,
                         label="medium", color="#1f77b4", edgecolor="black", lw=0.4)
    ax.set_yscale("log")
    ax.set_ylim(5, 50000)
    ax.set_ylabel("P99 - P50 gap (us, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABEL[s].replace(" (", "\n(") for s in pivot_us.index],
                       fontsize=8)
    ax.set_title("Tail gap (P99 - P50) by stage and model size")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(title="model size")

    for bars in (bars_small, bars_medium):
        for b in bars:
            h = b.get_height()
            if np.isnan(h) or h <= 0: continue
            label = f"{int(round(h))}" if h >= 10 else f"{h:.1f}"
            ax.text(b.get_x() + b.get_width()/2, h * 1.15, label,
                    ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "gap_p99_p50.png", dpi=140)
    plt.close()


def pareto(stats, mse_path):
    """Latency-accuracy frontier per model size, drawn as two zoomed subplots.
    Each subplot draws the Pareto frontier (left-to-right: keep points whose
    MSE is the lowest so far).
    """
    if not mse_path.exists():
        print(f"skip pareto: {mse_path} not found"); return
    import json
    mse = json.loads(Path(mse_path).read_text())

    pts = []
    for _, r in stats.iterrows():
        if not r["stage"].startswith("stage"): continue
        if r["stage"] == "stage5_int8": continue
        key = f"{r['stage']}_{r['size']}"
        if key not in mse: continue
        pts.append(dict(stage=r["stage"], size=r["size"],
                        lat_us=r["p99_ns"] / 1000.0, mse=mse[key]))
    pdf = pd.DataFrame(pts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    label_offsets = {
        # stage -> (dx_log_units, dy_mse_units)
        "stage0":      (0.04, 0.0),
        "stage1":      (-0.05, 0.00009),
        "stage2":      (0.04, -0.00010),
        "stage3":      (0.04, 0.00009),
        "stage4a":     (0.04, 0.00009),
        "stage4b":     (0.04, -0.00010),
        "stage5_fp32": (0.04, 0.0),
    }

    for ax, size in zip(axes, ["small", "medium"]):
        cluster = pdf[pdf["size"] == size].sort_values("lat_us").reset_index(drop=True)

        # Pareto frontier: traverse left to right, keep points whose MSE is the
        # lowest observed so far.
        frontier_idx = []
        best_mse = float("inf")
        for i, r in cluster.iterrows():
            if r["mse"] < best_mse:
                frontier_idx.append(i); best_mse = r["mse"]
        front = cluster.loc[frontier_idx]

        # Step-style frontier line.
        ax.plot(front["lat_us"], front["mse"], color="#444",
                lw=1.0, ls="--", alpha=0.55, drawstyle="steps-post",
                label="Pareto frontier", zorder=2)

        for _, r in cluster.iterrows():
            on_front = r["stage"] in front["stage"].values
            ax.scatter(r["lat_us"], r["mse"],
                       s=180 if on_front else 110,
                       facecolor=STAGE_COLOR[r["stage"]],
                       edgecolor="black" if on_front else "#555",
                       linewidth=1.6 if on_front else 0.8,
                       zorder=4 if on_front else 3)
            dx_log, dy = label_offsets.get(r["stage"], (0.04, 0.0))
            ax.annotate(r["stage"].replace("stage", "S").replace("_fp32", "fp32"),
                        xy=(r["lat_us"], r["mse"]),
                        xytext=(r["lat_us"] * (10 ** dx_log), r["mse"] + dy),
                        fontsize=8.5, va="center",
                        ha="left" if dx_log >= 0 else "right",
                        color="#222", zorder=5)

        ax.set_xscale("log")
        ax.set_xlabel("P99 latency (us, log scale)")
        ax.set_ylabel("test MSE")
        ax.set_title(f"{size} model (lower-left is better)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        # Pad y range so labels fit.
        ymin, ymax = cluster["mse"].min(), cluster["mse"].max()
        pad = max((ymax - ymin) * 0.4, 0.0008)
        ax.set_ylim(ymin - pad, ymax + pad)
        xmin, xmax = cluster["lat_us"].min(), cluster["lat_us"].max()
        ax.set_xlim(xmin * 0.7, xmax * 1.6)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    plt.suptitle("Latency-accuracy frontier per model size", y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pareto.png", dpi=140, bbox_inches="tight")
    plt.close()


def main():
    df = load_all_csvs()
    stats = per_config_stats(df)
    ablation_table(stats)
    noise_floor_cdf()
    cdf_overlay(df)
    gap_bar(stats)
    pareto(stats, REPO / "results" / "tables" / "mse.json")
    print(f"\nfigures -> {FIG_DIR}\ntables -> {TBL_DIR}")


if __name__ == "__main__":
    main()
