import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NeurIPS-ready BatchTopK figures from sweep and eval outputs."
    )
    parser.add_argument(
        "--pair-results",
        required=True,
        help="Path to pair_results.csv from sweep_batch_topk_speed.py",
    )
    parser.add_argument(
        "--best-params",
        default="",
        help="Optional path to best_params.json for annotation.",
    )
    parser.add_argument(
        "--torch-eval-csv",
        default="",
        help="Optional eval_history.csv for the torch baseline run.",
    )
    parser.add_argument(
        "--our-eval-csv",
        default="",
        help="Optional eval_history.csv for the our run.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/batch_topk_neurips",
        help="Directory where figures will be saved.",
    )
    return parser.parse_args()


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def convert_rows(rows: list[dict[str, str]]) -> list[dict]:
    converted = []
    for row in rows:
        converted.append({key: maybe_float(value) for key, value in row.items()})
    return converted


def load_best(path: str | Path) -> dict | None:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


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
            "lines.markersize": 6,
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


def annotate_best_xy(ax, x: float | None, y: float | None, label: str) -> None:
    if x is None or y is None:
        return
    ax.scatter([x], [y], color="black", marker="*", s=120, zorder=5)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )


def annotate_best(ax, best_summary: dict | None, x_key: str, y_key: str) -> None:
    if best_summary is None:
        return
    best = best_summary.get("best", {})
    x = best.get(x_key)
    y = best.get(y_key)
    if x is None or y is None:
        return
    label = (
        f"best: k={int(best['k'])}, d={int(best['dict_size'])}, "
        f"b={int(best['sae_batch_size'])}"
    )
    annotate_best_xy(ax, x, y, label)


def add_plain_number_format(ax) -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def plot_sweep_step_speedup(rows: list[dict], best_summary: dict | None, output_dir: Path) -> None:
    fig, ax = plt.subplots()
    colors = {12: "#E69F00", 16: "#D55E00", 32: "#0072B2", 64: "#009E73", 96: "#CC79A7"}
    for k in sorted({int(row["k"]) for row in rows}):
        subset = [row for row in rows if int(row["k"]) == k]
        x = [float(row["dict_size"]) * float(row["sae_batch_size"]) for row in subset]
        y = [float(row["step_speedup_ratio"]) for row in subset]
        ax.scatter(x, y, label=f"k={k}", color=colors.get(k))

    if best_summary is not None:
        best = best_summary.get("best", {})
        x_best = float(best["dict_size"]) * float(best["sae_batch_size"])
        y_best = float(best["step_speedup_ratio"])
        label = (
            f"best: k={int(best['k'])}, d={int(best['dict_size'])}, "
            f"b={int(best['sae_batch_size'])}"
        )
        annotate_best_xy(ax, x_best, y_best, label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Flattened vector size N = dict_size × batch_size")
    ax.set_ylabel("End-to-end step speedup (torch / our)")
    ax.set_title("End-to-End Training Speedup")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output_dir, "figure_step_speedup_vs_N")


def plot_sweep_topk_speedup(rows: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots()
    for batch_size in sorted({int(row["sae_batch_size"]) for row in rows}):
        subset = [row for row in rows if int(row["sae_batch_size"]) == batch_size]
        x = [float(row["dict_size"]) * float(row["sae_batch_size"]) for row in subset]
        y = [float(row["topk_speedup_ratio"]) for row in subset]
        ax.scatter(x, y, label=f"batch={batch_size}")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Flattened vector size N = dict_size × batch_size")
    ax.set_ylabel("Top-k kernel speedup (torch / our)")
    ax.set_title("Top-k Kernel Speedup")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output_dir, "figure_topk_speedup_vs_N")


def plot_speedup_tradeoff(rows: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots()
    x = [float(row["topk_speedup_ratio"]) for row in rows]
    y = [float(row["step_speedup_ratio"]) for row in rows]
    c = [math.log2(float(row["dict_size"]) * float(row["sae_batch_size"])) for row in rows]
    sc = ax.scatter(x, y, c=c, cmap="viridis")
    ax.set_xlabel("Top-k kernel speedup (torch / our)")
    ax.set_ylabel("End-to-end step speedup (torch / our)")
    ax.set_title("Kernel Speedup vs Training Speedup")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log2(N)")
    save_figure(fig, output_dir, "figure_speedup_tradeoff")


def plot_topk_fraction(rows: list[dict], output_dir: Path) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda row: (float(row["dict_size"]) * float(row["sae_batch_size"]), float(row["k"])),
    )
    labels = [
        f"k={int(row['k'])}\nd={int(row['dict_size'])}\nb={int(row['sae_batch_size'])}"
        for row in rows_sorted
    ]
    x = list(range(len(rows_sorted)))
    torch_vals = [float(row["torch_topk_frac_of_step"]) for row in rows_sorted]
    our_vals = [float(row["our_topk_frac_of_step"]) for row in rows_sorted]

    fig, ax = plt.subplots(figsize=(max(8.0, len(rows_sorted) * 0.32), 4.2))
    ax.plot(x, torch_vals, marker="o", label="torch")
    ax.plot(x, our_vals, marker="o", label="our")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Top-k fraction of total step time")
    ax.set_title("How Much of Training is Top-k?")
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "figure_topk_fraction")


def plot_eval_pair(
    torch_rows: list[dict],
    our_rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots()
    if torch_rows:
        ax.plot(
            [float(row["step"]) for row in torch_rows],
            [float(row[metric]) for row in torch_rows],
            marker="o",
            label="torch",
            color="#0072B2",
        )
    if our_rows:
        ax.plot(
            [float(row["step"]) for row in our_rows],
            [float(row[metric]) for row in our_rows],
            marker="o",
            label="our",
            color="#D55E00",
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    add_plain_number_format(ax)
    save_figure(fig, output_dir, stem)


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    output_dir = Path(args.output_dir)
    pair_rows = convert_rows(load_csv_rows(args.pair_results))
    best_summary = load_best(args.best_params)

    plot_sweep_step_speedup(pair_rows, best_summary, output_dir)
    plot_sweep_topk_speedup(pair_rows, output_dir)
    plot_speedup_tradeoff(pair_rows, output_dir)
    plot_topk_fraction(pair_rows, output_dir)

    torch_eval_rows = (
        convert_rows(load_csv_rows(args.torch_eval_csv)) if args.torch_eval_csv else []
    )
    our_eval_rows = (
        convert_rows(load_csv_rows(args.our_eval_csv)) if args.our_eval_csv else []
    )

    if torch_eval_rows or our_eval_rows:
        plot_eval_pair(
            torch_eval_rows,
            our_eval_rows,
            metric="nmse",
            ylabel="NMSE",
            title="Normalized MSE During Training",
            output_dir=output_dir,
            stem="figure_eval_nmse",
        )
        plot_eval_pair(
            torch_eval_rows,
            our_eval_rows,
            metric="ce_degradation",
            ylabel="Cross-entropy degradation",
            title="Downstream CE Degradation During Training",
            output_dir=output_dir,
            stem="figure_eval_ce_degradation",
        )
        plot_eval_pair(
            torch_eval_rows,
            our_eval_rows,
            metric="active_latents_mean",
            ylabel="Average active latents per sample",
            title="Average Sparsity During Training",
            output_dir=output_dir,
            stem="figure_eval_active_latents_mean",
        )
        plot_eval_pair(
            torch_eval_rows,
            our_eval_rows,
            metric="active_latents_std",
            ylabel="Std. of active latents per sample",
            title="Per-sample Active-Latent Variability",
            output_dir=output_dir,
            stem="figure_eval_active_latents_std",
        )

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
