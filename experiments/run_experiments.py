"""Comprehensive Experimentation Suite for Learned Page Replacement (CSE-307).

Executes shift benchmarks, cache sensitivity sweeps, calibration analysis,
and generates all publication-grade figures and result tables.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src is in search path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.workload import WorkloadGenerator
from src.simulator import Simulator


def setup_plot_style():
    """Configure clean publication styling for matplotlib figures."""
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.family": "sans-serif",
        "figure.titlesize": 13,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def run_all_experiments():
    print("=" * 70)
    print("CSE-307: LEARNING-AUGMENTED OS HEURISTICS EXPERIMENT SUITE")
    print("Track 1: Learned Page Replacement & Explanation Confidence Bonus")
    print("=" * 70)

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    setup_plot_style()

    # 1. Workload Generation
    print("\n[Step 1/5] Generating Workload Traces with Distribution Shift...")
    gen = WorkloadGenerator(seed=42)
    # Training trace for model fitting (2000 accesses with representative locality)
    train_trace, _, _ = gen.generate_shifted_workload(
        total_references=2500,
        num_pages=120,
        shift_ratio=0.8,  # predominantly locality-focused training
        phase1_type="locality_heavy",
        phase2_type="random_bursty",
    )

    # Evaluation trace (6000 accesses, sudden phase shift at t=3000)
    total_eval_refs = 6000
    eval_trace, shift_point, meta = gen.generate_shifted_workload(
        total_references=total_eval_refs,
        num_pages=120,
        shift_ratio=0.5,
        phase1_type="locality_heavy",
        phase2_type="random_bursty",
    )
    print(f"Generated evaluation trace: {len(eval_trace)} accesses, Shift at t={shift_point}.")
    print(f"Phase 1: Zipfian locality + loops (ticks 0 to {shift_point-1})")
    print(f"Phase 2: High-entropy uniform random + bursty thrashing (ticks {shift_point} to {total_eval_refs-1})")

    # 2. Main Simulation (Cache Capacity = 16)
    print("\n[Step 2/5] Training Models and Running Core Shift Benchmark (Capacity=16)...")
    cache_capacity = 16
    sim = Simulator(capacity=cache_capacity, seed=42)
    log_reg, tree = sim.train_models(train_trace)

    print("Model Weights (Logistic Regression normalized features):")
    feature_names = ["Recency (idle time)", "Frequency (accesses)", "Cache Age", "Prev Reuse Dist"]
    for f_name, w in zip(feature_names, log_reg.weights):
        print(f"  - {f_name:24s}: {w:+.4f}")
    print(f"  - Bias                    : {log_reg.bias:+.4f}")

    sim_results = sim.run_comparative_simulation(
        eval_trace=eval_trace,
        shift_point=shift_point,
        log_reg_model=log_reg,
        tree_model=tree,
        record_explanations=True,
    )

    summary_df = pd.DataFrame(sim_results["summary_table"])
    csv_path = os.path.join(results_dir, "hit_rates_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved primary metrics table to {csv_path}")

    # Generate Markdown Table
    md_table = summary_df.to_markdown(index=False)
    md_path = os.path.join(results_dir, "performance_table.md")
    with open(md_path, "w") as f:
        f.write("# Page Replacement Policy Comparison Under Workload Shift\n\n")
        f.write(f"**Trace Length:** {total_eval_refs} accesses | **Shift Point:** t={shift_point} | **Cache Capacity:** {cache_capacity} frames\n\n")
        f.write(md_table)
        f.write("\n\n*Note: Degradation = Phase 1 Hit Ratio - Phase 2 Hit Ratio.*\n")
    print(f"Saved markdown summary to {md_path}")
    print("\n" + md_table)

    # 3. Visualizations
    print("\n[Step 3/5] Generating Publication-Quality Figures...")

    # Figure 1: Hit Ratio Before vs After Shift
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    policies = summary_df["Policy"].tolist()
    p1_hr = summary_df["Phase1_HitRatio_%"].tolist()
    p2_hr = summary_df["Phase2_HitRatio_%"].tolist()
    ov_hr = summary_df["Overall_HitRatio_%"].tolist()

    x = np.arange(len(policies))
    width = 0.25

    rects1 = ax.bar(x - width, p1_hr, width, label="Phase 1 (Pre-Shift: Locality)", color="#2b5c8f")
    rects2 = ax.bar(x, p2_hr, width, label="Phase 2 (Post-Shift: Random/Burst)", color="#d95f02")
    rects3 = ax.bar(x + width, ov_hr, width, label="Overall Benchmark", color="#7570b3")

    ax.set_ylabel("Hit Ratio (%)")
    ax.set_title("Figure 1: Page Replacement Hit Ratio Across Workload Phase Shift (C = 16)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 100)

    # Add value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    fig1_path = os.path.join(results_dir, "figure1_hit_ratios.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Generated {fig1_path}")

    # Figure 2: Cumulative Faults Over Time
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    colors = {
        "FIFO": "#e41a1c",
        "LRU": "#377eb8",
        "Optimal (Belady)": "#4daf4a",
        "Learned (Logistic)": "#984ea3",
        "Learned (Tree)": "#ff7f00",
        "Adaptive-Hybrid": "#003f5c",
    }
    linestyles = {
        "FIFO": "--",
        "LRU": "-",
        "Optimal (Belady)": "-",
        "Learned (Logistic)": "-.",
        "Learned (Tree)": ":",
        "Adaptive-Hybrid": "-",
    }

    t_axis = np.arange(len(eval_trace))
    for name, faults_seq in sim_results["cumulative_faults"].items():
        lw = 2.2 if name in ["Optimal (Belady)", "Adaptive-Hybrid"] else 1.4
        ax.plot(t_axis, faults_seq, label=name, color=colors.get(name, "#333333"), linestyle=linestyles.get(name, "-"), linewidth=lw)

    ax.axvline(x=shift_point, color="#333333", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(shift_point + 60, 200, "Workload Shift\n(t = 3000)", fontsize=9, fontweight="bold", color="#222222")

    ax.set_xlabel("Time (Access Sequence Ticks)")
    ax.set_ylabel("Cumulative Page Faults")
    ax.set_title("Figure 2: Fault Accumulation Trajectory Over Time Showing Inflection Point")
    ax.legend(loc="upper left", framealpha=0.9)

    fig2_path = os.path.join(results_dir, "figure2_fault_timeline.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Generated {fig2_path}")

    # 4. Cache Sensitivity Sweep (C in [4, 8, 12, 16, 20, 24, 32])
    print("\n[Step 4/5] Executing Cache Capacity Sensitivity Sweep...")
    capacities = [4, 8, 12, 16, 20, 24, 32]
    sweep_data = {pol: [] for pol in policies}

    for cap in capacities:
        sweep_sim = Simulator(capacity=cap, seed=42)
        lr_m, tr_m = sweep_sim.train_models(train_trace)
        res = sweep_sim.run_comparative_simulation(
            eval_trace=eval_trace,
            shift_point=shift_point,
            log_reg_model=lr_m,
            tree_model=tr_m,
            record_explanations=False,
        )
        for row in res["summary_table"]:
            sweep_data[row["Policy"]].append(row["Overall_HitRatio_%"])

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    for pol_name in policies:
        lw = 2.2 if pol_name in ["Optimal (Belady)", "Adaptive-Hybrid"] else 1.4
        ax.plot(capacities, sweep_data[pol_name], marker="o", label=pol_name, color=colors.get(pol_name, "#333333"), linestyle=linestyles.get(pol_name, "-"), linewidth=lw)

    ax.set_xlabel("Cache Capacity (Frames)")
    ax.set_ylabel("Overall Hit Ratio (%)")
    ax.set_title("Figure 3: Cache Capacity Sensitivity Curve (Trace Length = 6,000)")
    ax.set_xticks(capacities)
    ax.legend(loc="lower right", framealpha=0.9)

    fig3_path = os.path.join(results_dir, "figure3_cache_sensitivity.png")
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Generated {fig3_path}")

    # 5. Bonus Track: Calibration Analysis & Explanations
    print("\n[Step 5/5] Analyzing Bonus Track: Explanation Confidence Calibration...")
    calib = sim_results["calibration_metrics"]
    print(f"Total Evictions Evaluated: {calib['total_evaluations']}")
    print(f"Overall Empirical Accuracy vs Belady: {calib['overall_accuracy']*100:.2f}%")
    print(f"Mean Self-Rated Confidence: {calib['mean_confidence']*100:.2f}%")
    print(f"Expected Calibration Error (ECE): {calib['ece']:.4f}")
    print(f"Brier Score: {calib['brier_score']:.4f}")

    # Figure 4: Reliability Diagram and Confidence Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Reliability diagram
    bin_confs = calib["bin_confidences"]
    bin_accs = calib["bin_accuracies"]
    bin_counts = calib["bin_counts"]

    # Filter non-empty bins for plotting
    valid_indices = [i for i, c in enumerate(bin_counts) if c > 0]
    plot_confs = [bin_confs[i] for i in valid_indices]
    plot_accs = [bin_accs[i] for i in valid_indices]

    ax1.plot([0.5, 1.0], [0.5, 1.0], "k--", label="Perfect Calibration", linewidth=1.5)
    ax1.scatter(plot_confs, plot_accs, color="#2b5c8f", s=50, zorder=5, label="Observed Accuracy")
    ax1.plot(plot_confs, plot_accs, color="#2b5c8f", linestyle="-", linewidth=1.5)
    ax1.fill_between(plot_confs, plot_accs, plot_confs, color="#2b5c8f", alpha=0.15, label=f"Calibration Gap (ECE={calib['ece']:.3f})")

    ax1.set_xlabel("Mean Confidence Score in Bin")
    ax1.set_ylabel("Empirical Accuracy vs Belady")
    ax1.set_title("(a) Reliability Diagram")
    ax1.set_xlim(0.48, 1.0)
    ax1.set_ylim(0.48, 1.0)
    ax1.legend(loc="upper left", framealpha=0.9)

    # Confidence Histogram
    conf_samples = [e["confidence"] for e in sim_results["explanations"]]
    ax2.hist(conf_samples, bins=15, color="#7570b3", edgecolor="black", alpha=0.75, rwidth=0.85)
    ax2.set_xlabel("Predicted Confidence Score")
    ax2.set_ylabel("Eviction Event Frequency")
    ax2.set_title("(b) Eviction Decision Confidence Distribution")

    fig4_path = os.path.join(results_dir, "figure4_calibration_curve.png")
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Generated {fig4_path}")

    # Export Sample Explanations
    sample_exp_path = os.path.join(results_dir, "sample_explanations.txt")
    with open(sample_exp_path, "w") as f:
        f.write("CSE-307 Term Paper - Bonus Track: Sample Natural Language Explanations\n")
        f.write("=" * 75 + "\n\n")
        # Sample high, medium, low confidence
        sorted_exps = sorted(sim_results["explanations"], key=lambda x: x["confidence"], reverse=True)
        high_confs = sorted_exps[:5]
        mid_confs = [e for e in sorted_exps if 0.65 <= e["confidence"] <= 0.75][:5]
        low_confs = sorted_exps[-5:]

        f.write("--- HIGH CONFIDENCE DECISIONS ---\n")
        for e in high_confs:
            f.write(e["explanation"] + "\n\n")

        f.write("--- MODERATE CONFIDENCE DECISIONS ---\n")
        for e in mid_confs:
            f.write(e["explanation"] + "\n\n")

        f.write("--- LOW CONFIDENCE DECISIONS (CLOSE MARGINS) ---\n")
        for e in low_confs:
            f.write(e["explanation"] + "\n\n")

    print(f"  -> Generated {sample_exp_path}")
    print("\nExperiment suite completed successfully!")


if __name__ == "__main__":
    run_all_experiments()
