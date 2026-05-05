import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed BatchTopK benchmark pair (torch vs our) and make NeurIPS-ready plots."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--dataset-name", default="openwebtext")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--threshold-start-step", type=int, default=1000)
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--sae-batch-size", type=int, default=1024)
    parser.add_argument("--n-ctxs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-n-batches", type=int, default=10)
    parser.add_argument("--eval-llm-batch-size", type=int, default=100)
    parser.add_argument("--eval-sae-batch-size", type=int, default=4096)
    parser.add_argument("--eval-n-ctxs", type=int, default=256)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--ignore-first-train-steps", type=int, default=20)
    parser.add_argument("--save-dir", default="runs/batch_topk_fixed_benchmark")
    parser.add_argument("--figure-dir", default="figures/batch_topk_fixed_neurips")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="batch_topk")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def run_variant(args: argparse.Namespace, *, our: bool) -> None:
    cmd = [
        sys.executable,
        "train_batch_topk.py",
        "--device", args.device,
        "--model-name", args.model_name,
        "--dataset-name", args.dataset_name,
        "--layer", str(args.layer),
        "--steps", str(args.steps),
        "--k", str(args.k),
        "--lr", str(args.lr),
        "--warmup-steps", str(args.warmup_steps),
        "--threshold-start-step", str(args.threshold_start_step),
        "--expansion-factor", str(args.expansion_factor),
        "--llm-batch-size", str(args.llm_batch_size),
        "--sae-batch-size", str(args.sae_batch_size),
        "--n-ctxs", str(args.n_ctxs),
        "--save-dir", args.save_dir,
        "--log-steps", str(args.log_steps),
        "--run-eval",
        "--skip-final-eval",
        "--eval-every", str(args.eval_every),
        "--eval-n-batches", str(args.eval_n_batches),
        "--eval-llm-batch-size", str(args.eval_llm_batch_size),
        "--eval-sae-batch-size", str(args.eval_sae_batch_size),
        "--eval-n-ctxs", str(args.eval_n_ctxs),
    ]
    if args.use_wandb:
        cmd.extend(
            [
                "--use-wandb",
                "--wandb-project",
                args.wandb_project,
                "--wandb-entity",
                args.wandb_entity,
            ]
        )
    if our:
        cmd.append("--our")

    subprocess.run(cmd, check=True)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    converted = []
    for row in rows:
        converted.append({k: float(v) for k, v in row.items()})
    return converted


def median_metric(rows: list[dict], metric: str, min_step: int) -> float:
    filtered = [row[metric] for row in rows if row["step"] > min_step]
    return float(statistics.median(filtered))


def build_summary(
    args: argparse.Namespace,
    torch_train: list[dict],
    our_train: list[dict],
    dict_size: int,
) -> dict:
    step_speedup_ratio = median_metric(torch_train, "step_time_ms", args.ignore_first_train_steps) / median_metric(
        our_train, "step_time_ms", args.ignore_first_train_steps
    )
    topk_speedup_ratio = median_metric(torch_train, "topk_time_ms", args.ignore_first_train_steps) / median_metric(
        our_train, "topk_time_ms", args.ignore_first_train_steps
    )
    throughput_speedup_ratio = median_metric(
        our_train, "samples_per_sec", args.ignore_first_train_steps
    ) / median_metric(torch_train, "samples_per_sec", args.ignore_first_train_steps)

    return {
        "config": {
            "k": args.k,
            "dict_size": dict_size,
            "sae_batch_size": args.sae_batch_size,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "dataset_name": args.dataset_name,
            "model_name": args.model_name,
            "layer": args.layer,
        },
        "summary_metrics": {
            "torch_step_time_ms_median": median_metric(torch_train, "step_time_ms", args.ignore_first_train_steps),
            "our_step_time_ms_median": median_metric(our_train, "step_time_ms", args.ignore_first_train_steps),
            "step_speedup_ratio": step_speedup_ratio,
            "torch_topk_time_ms_median": median_metric(torch_train, "topk_time_ms", args.ignore_first_train_steps),
            "our_topk_time_ms_median": median_metric(our_train, "topk_time_ms", args.ignore_first_train_steps),
            "topk_speedup_ratio": topk_speedup_ratio,
            "torch_topk_frac_of_step_median": median_metric(
                torch_train, "topk_frac_of_step", args.ignore_first_train_steps
            ),
            "our_topk_frac_of_step_median": median_metric(
                our_train, "topk_frac_of_step", args.ignore_first_train_steps
            ),
            "torch_samples_per_sec_median": median_metric(
                torch_train, "samples_per_sec", args.ignore_first_train_steps
            ),
            "our_samples_per_sec_median": median_metric(
                our_train, "samples_per_sec", args.ignore_first_train_steps
            ),
            "throughput_speedup_ratio": throughput_speedup_ratio,
        },
    }


def main() -> None:
    args = parse_args()
    run_variant(args, our=False)
    run_variant(args, our=True)

    run_root = Path(args.save_dir)
    torch_dir = run_root / f"layer_{args.layer}_torch" / "trainer_0"
    our_dir = run_root / f"layer_{args.layer}_our" / "trainer_0"

    torch_train = load_csv(torch_dir / "train_history.csv")
    our_train = load_csv(our_dir / "train_history.csv")
    with open(torch_dir / "config.json") as f:
        trainer_config = json.load(f)["trainer"]
    summary = build_summary(args, torch_train, our_train, int(trainer_config["dict_size"]))

    summary_path = run_root / "fixed_benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if not args.skip_plots:
        plot_cmd = [
            sys.executable,
            "plot_batch_topk_fixed_neurips.py",
            "--torch-train-csv", str(torch_dir / "train_history.csv"),
            "--our-train-csv", str(our_dir / "train_history.csv"),
            "--torch-eval-csv", str(torch_dir / "eval_history.csv"),
            "--our-eval-csv", str(our_dir / "eval_history.csv"),
            "--summary-json", str(summary_path),
            "--output-dir", args.figure_dir,
            "--ignore-first-train-steps", str(args.ignore_first_train_steps),
        ]
        subprocess.run(plot_cmd, check=True)

    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
