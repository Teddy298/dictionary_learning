import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NeurIPS-ready paired BatchTopK figures for torch vs our."
    )
    parser.add_argument("--torch-train-csv", required=True)
    parser.add_argument("--our-train-csv", required=True)
    parser.add_argument("--torch-eval-csv", required=True)
    parser.add_argument("--our-eval-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--output-dir", default="figures/batch_topk_fixed_neurips")
    parser.add_argument("--ignore-first-train-steps", type=int, default=20)
    return parser.parse_args()


def load_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    converted = []
    for row in rows:
        converted.append({k: _try_float(v) for k, v in row.items()})
    return converted


def _try_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.2,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def add_plain_format(ax) -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def filtered_rows(rows: list[dict], min_step: int) -> list[dict]:
    return [row for row in rows if float(row["step"]) > min_step]


def median_metric(rows: list[dict], metric: str) -> float:
    return float(statistics.median(float(row[metric]) for row in rows))


def plot_train_metric(
    torch_rows: list[dict],
    our_rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    stem: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots()
    ax.plot([row["step"] for row in torch_rows], [row[metric] for row in torch_rows], label="torch", color="#0072B2")
    ax.plot([row["step"] for row in our_rows], [row[metric] for row in our_rows], label="our", color="#D55E00")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    add_plain_format(ax)
    save_figure(fig, output_dir, stem)


def plot_eval_metric(
    torch_rows: list[dict],
    our_rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    stem: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots()
    ax.plot([row["step"] for row in torch_rows], [row[metric] for row in torch_rows], marker="o", label="torch", color="#0072B2")
    ax.plot([row["step"] for row in our_rows], [row[metric] for row in our_rows], marker="o", label="our", color="#D55E00")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    add_plain_format(ax)
    save_figure(fig, output_dir, stem)


def plot_summary_bar(
    torch_rows: list[dict],
    our_rows: list[dict],
    output_dir: Path,
) -> None:
    metrics = [
        ("step_time_ms", "Step time (ms)"),
        ("topk_time_ms", "Top-k time (ms)"),
        ("samples_per_sec", "Samples / sec"),
        ("topk_frac_of_step", "Top-k fraction"),
    ]
    torch_vals = [median_metric(torch_rows, metric) for metric, _ in metrics]
    our_vals = [median_metric(our_rows, metric) for metric, _ in metrics]
    labels = [label for _, label in metrics]

    x = range(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar([i - width / 2 for i in x], torch_vals, width=width, label="torch", color="#0072B2")
    ax.bar([i + width / 2 for i in x], our_vals, width=width, label="our", color="#D55E00")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Paired Speed Summary")
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "figure_speed_summary_bar")


def plot_speedup_annotation(
    torch_rows: list[dict],
    our_rows: list[dict],
    output_dir: Path,
) -> None:
    step_speedup = median_metric(torch_rows, "step_time_ms") / median_metric(our_rows, "step_time_ms")
    topk_speedup = median_metric(torch_rows, "topk_time_ms") / median_metric(our_rows, "topk_time_ms")
    throughput_speedup = median_metric(our_rows, "samples_per_sec") / median_metric(torch_rows, "samples_per_sec")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    labels = ["Step speedup", "Top-k speedup", "Throughput speedup"]
    values = [step_speedup, topk_speedup, throughput_speedup]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    ax.bar(labels, values, color=colors)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Ratio")
    ax.set_title("Speedup Ratios (torch vs our)")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}x", ha="center", va="bottom")
    save_figure(fig, output_dir, "figure_speedup_ratios")


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    output_dir = Path(args.output_dir)
    torch_train = filtered_rows(load_csv(args.torch_train_csv), args.ignore_first_train_steps)
    our_train = filtered_rows(load_csv(args.our_train_csv), args.ignore_first_train_steps)
    torch_eval = load_csv(args.torch_eval_csv)
    our_eval = load_csv(args.our_eval_csv)

    plot_train_metric(
        torch_train, our_train, "step_time_ms", "Step time (ms)",
        "End-to-End Training Step Time", "figure_train_step_time", output_dir
    )
    plot_train_metric(
        torch_train, our_train, "topk_time_ms", "Top-k time (ms)",
        "Top-k Kernel Time", "figure_train_topk_time", output_dir
    )
    plot_train_metric(
        torch_train, our_train, "topk_frac_of_step", "Top-k fraction of step",
        "How Much of Training is Top-k?", "figure_train_topk_fraction", output_dir
    )
    plot_train_metric(
        torch_train, our_train, "samples_per_sec", "Samples / sec",
        "Training Throughput", "figure_train_samples_per_sec", output_dir
    )

    plot_eval_metric(
        torch_eval, our_eval, "nmse", "NMSE",
        "Normalized MSE During Training", "figure_eval_nmse", output_dir
    )
    plot_eval_metric(
        torch_eval, our_eval, "ce_degradation", "Cross-entropy degradation",
        "Downstream CE Degradation During Training", "figure_eval_ce_degradation", output_dir
    )
    plot_eval_metric(
        torch_eval, our_eval, "active_latents_mean", "Average active latents / sample",
        "Average Sparsity During Training", "figure_eval_active_latents_mean", output_dir
    )
    plot_eval_metric(
        torch_eval, our_eval, "active_latents_std", "Std. of active latents / sample",
        "Per-sample Active-Latent Variability", "figure_eval_active_latents_std", output_dir
    )
    plot_eval_metric(
        torch_eval, our_eval, "frac_variance_explained", "Fraction variance explained",
        "Variance Explained During Training", "figure_eval_frac_variance_explained", output_dir
    )
    plot_eval_metric(
        torch_eval, our_eval, "loss_reconstructed", "Reconstructed CE loss",
        "Reconstructed LM Loss During Training", "figure_eval_loss_reconstructed", output_dir
    )

    plot_summary_bar(torch_train, our_train, output_dir)
    plot_speedup_annotation(torch_train, our_train, output_dir)

    if args.summary_json:
        with open(args.summary_json) as f:
            summary = json.load(f)
        with open(output_dir / "figure_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
