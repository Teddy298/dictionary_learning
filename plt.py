import os
import json
import glob
import re
from collections import defaultdict

import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

BASE_DIR = "."

EXPERIMENTS = {
    "torch": os.path.join(BASE_DIR, "torch", "trainer_0"),
    "our": os.path.join(BASE_DIR, "our", "trainer_0"),
}

OUTPUT_DIR = "metric_plots"

# ============================================================
# Helpers
# ============================================================

STEP_PATTERN = re.compile(r"eval_results_step_(\d+)\.json")

# data[experiment][metric] = [(step, value), ...]
data = defaultdict(lambda: defaultdict(list))

# ============================================================
# Load metrics
# ============================================================

for exp_name, exp_path in EXPERIMENTS.items():

    files = glob.glob(
        os.path.join(exp_path, "eval_results_step_*.json")
    )

    for file_path in files:

        filename = os.path.basename(file_path)

        match = STEP_PATTERN.match(filename)

        if not match:
            continue

        step = int(match.group(1))

        with open(file_path, "r") as f:
            metrics = json.load(f)

        for metric_name, value in metrics.items():
            data[exp_name][metric_name].append((step, value))

# ============================================================
# Sort by evaluation step
# ============================================================

for exp_name in data:
    for metric_name in data[exp_name]:
        data[exp_name][metric_name].sort(key=lambda x: x[0])

# ============================================================
# Collect all metrics
# ============================================================

all_metrics = sorted(
    {
        metric
        for exp_data in data.values()
        for metric in exp_data.keys()
    }
)

# ============================================================
# Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Plot metrics
# ============================================================

for metric in all_metrics:

    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name in EXPERIMENTS.keys():

        if metric not in data[exp_name]:
            continue

        steps = [x[0] for x in data[exp_name][metric]]
        values = [x[1] for x in data[exp_name][metric]]

        ax.plot(
            steps,
            values,
            marker="o",
            linewidth=2,
            label=exp_name,
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_title(f"{metric} over evaluation steps")
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel(metric)

    # --------------------------------------------------------
    # Disable matplotlib +1 offset formatting
    # --------------------------------------------------------

    ax.ticklabel_format(
        useOffset=False,
        style="plain",
        axis="y",
    )

    # --------------------------------------------------------
    # Better ranges for metrics close to 1
    # --------------------------------------------------------

    near_one_metrics = {
        "frac_alive",
        "frac_recovered",
        "cossim",
        "relative_reconstruction_bias",
        "l2_ratio",
        "frac_variance_explained",
    }

    if metric in near_one_metrics:
        ax.set_ylim(0.95, 1.01)

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    ax.grid(True, alpha=0.3)
    ax.legend()

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{metric}.png",
    )

    plt.savefig(save_path, bbox_inches="tight")

    print(f"Saved: {save_path}")

    # --------------------------------------------------------
    # Show plot on screen
    # --------------------------------------------------------

    plt.show()

print(f"\nAll plots saved to: {OUTPUT_DIR}")
