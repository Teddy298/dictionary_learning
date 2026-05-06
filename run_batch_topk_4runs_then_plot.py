import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch 4 separate BatchTopK runs in parallel "
            "(optimal/paper × torch/our), then build figures and summaries."
        )
    )
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:2,cuda:3",
        help="Comma-separated devices for the 4 runs: optimal-torch, optimal-our, paper-torch, paper-our.",
    )
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--dataset-name", default="openwebtext")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--threshold-start-step", type=int, default=1000)
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--n-ctxs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-n-batches", type=int, default=10)
    parser.add_argument("--eval-llm-batch-size", type=int, default=100)
    parser.add_argument("--eval-sae-batch-size", type=int, default=4096)
    parser.add_argument("--eval-n-ctxs", type=int, default=256)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--ignore-first-train-steps", type=int, default=20)
    parser.add_argument("--runs-root", default="runs/batch_topk_experiments")
    parser.add_argument("--figures-root", default="figures/batch_topk_experiments")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="batch_topk")
    parser.add_argument("--wandb-entity", default="")
    return parser.parse_args()


def preset_configs() -> dict[str, dict[str, int]]:
    return {
        "optimal": {
            "k": 12,
            "expansion_factor": 32,
            "sae_batch_size": 1024,
        },
        "paper": {
            "k": 32,
            "expansion_factor": 24,
            "sae_batch_size": 4096,
        },
    }


def variant_specs(args: argparse.Namespace) -> list[dict]:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) != 4:
        raise ValueError(
            "--devices must provide exactly 4 entries for "
            "optimal-torch, optimal-our, paper-torch, paper-our."
        )

    presets = preset_configs()
    return [
        {"preset": "optimal", "our": False, "device": devices[0], **presets["optimal"]},
        {"preset": "optimal", "our": True, "device": devices[1], **presets["optimal"]},
        {"preset": "paper", "our": False, "device": devices[2], **presets["paper"]},
        {"preset": "paper", "our": True, "device": devices[3], **presets["paper"]},
    ]


def build_train_command(args: argparse.Namespace, spec: dict) -> list[str]:
    save_dir = Path(args.runs_root) / spec["preset"]
    cmd = [
        sys.executable,
        "train_batch_topk.py",
        "--device",
        spec["device"],
        "--model-name",
        args.model_name,
        "--dataset-name",
        args.dataset_name,
        "--layer",
        str(args.layer),
        "--steps",
        str(args.steps),
        "--k",
        str(spec["k"]),
        "--lr",
        str(args.lr),
        "--warmup-steps",
        str(args.warmup_steps),
        "--threshold-start-step",
        str(args.threshold_start_step),
        "--expansion-factor",
        str(spec["expansion_factor"]),
        "--llm-batch-size",
        str(args.llm_batch_size),
        "--sae-batch-size",
        str(spec["sae_batch_size"]),
        "--n-ctxs",
        str(args.n_ctxs),
        "--save-dir",
        str(save_dir),
        "--log-steps",
        str(args.log_steps),
        "--run-eval",
        "--skip-final-eval",
        "--eval-every",
        str(args.eval_every),
        "--eval-n-batches",
        str(args.eval_n_batches),
        "--eval-llm-batch-size",
        str(args.eval_llm_batch_size),
        "--eval-sae-batch-size",
        str(args.eval_sae_batch_size),
        "--eval-n-ctxs",
        str(args.eval_n_ctxs),
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
    if spec["our"]:
        cmd.append("--our")
    return cmd


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        out.append({k: float(v) for k, v in row.items()})
    return out


def median_metric(rows: list[dict], metric: str, min_step: int) -> float:
    filtered = [row[metric] for row in rows if row["step"] > min_step]
    return float(statistics.median(filtered))


def build_summary(
    run_root: Path,
    layer: int,
    ignore_first_train_steps: int,
    preset: dict,
) -> dict:
    torch_dir = run_root / f"layer_{layer}_torch" / "trainer_0"
    our_dir = run_root / f"layer_{layer}_our" / "trainer_0"
    torch_train = load_csv(torch_dir / "train_history.csv")
    our_train = load_csv(our_dir / "train_history.csv")

    with open(torch_dir / "config.json") as f:
        trainer_config = json.load(f)["trainer"]

    step_speedup_ratio = median_metric(
        torch_train, "step_time_ms", ignore_first_train_steps
    ) / median_metric(our_train, "step_time_ms", ignore_first_train_steps)
    topk_speedup_ratio = median_metric(
        torch_train, "topk_time_ms", ignore_first_train_steps
    ) / median_metric(our_train, "topk_time_ms", ignore_first_train_steps)
    throughput_speedup_ratio = median_metric(
        our_train, "samples_per_sec", ignore_first_train_steps
    ) / median_metric(torch_train, "samples_per_sec", ignore_first_train_steps)

    return {
        "config": {
            "k": int(trainer_config["k"]),
            "dict_size": int(trainer_config["dict_size"]),
            "sae_batch_size": int(preset["sae_batch_size"]),
            "layer": int(trainer_config["layer"]),
            "model_name": trainer_config["lm_name"],
        },
        "summary_metrics": {
            "torch_step_time_ms_median": median_metric(
                torch_train, "step_time_ms", ignore_first_train_steps
            ),
            "our_step_time_ms_median": median_metric(
                our_train, "step_time_ms", ignore_first_train_steps
            ),
            "step_speedup_ratio": step_speedup_ratio,
            "torch_topk_time_ms_median": median_metric(
                torch_train, "topk_time_ms", ignore_first_train_steps
            ),
            "our_topk_time_ms_median": median_metric(
                our_train, "topk_time_ms", ignore_first_train_steps
            ),
            "topk_speedup_ratio": topk_speedup_ratio,
            "torch_topk_frac_of_step_median": median_metric(
                torch_train, "topk_frac_of_step", ignore_first_train_steps
            ),
            "our_topk_frac_of_step_median": median_metric(
                our_train, "topk_frac_of_step", ignore_first_train_steps
            ),
            "torch_samples_per_sec_median": median_metric(
                torch_train, "samples_per_sec", ignore_first_train_steps
            ),
            "our_samples_per_sec_median": median_metric(
                our_train, "samples_per_sec", ignore_first_train_steps
            ),
            "throughput_speedup_ratio": throughput_speedup_ratio,
        },
    }
def make_plots(args: argparse.Namespace, preset_name: str) -> None:
    run_root = Path(args.runs_root) / preset_name
    figure_root = Path(args.figures_root) / preset_name
    plot_cmd = [
        sys.executable,
        "plot_batch_topk_fixed_neurips.py",
        "--torch-train-csv",
        str(run_root / f"layer_{args.layer}_torch" / "trainer_0" / "train_history.csv"),
        "--our-train-csv",
        str(run_root / f"layer_{args.layer}_our" / "trainer_0" / "train_history.csv"),
        "--torch-eval-csv",
        str(run_root / f"layer_{args.layer}_torch" / "trainer_0" / "eval_history.csv"),
        "--our-eval-csv",
        str(run_root / f"layer_{args.layer}_our" / "trainer_0" / "eval_history.csv"),
        "--summary-json",
        str(run_root / "fixed_benchmark_summary.json"),
        "--output-dir",
        str(figure_root),
        "--ignore-first-train-steps",
        str(args.ignore_first_train_steps),
    ]
    subprocess.run(plot_cmd, check=True)


def main() -> None:
    args = parse_args()
    specs = variant_specs(args)

    processes: list[tuple[dict, subprocess.Popen]] = []
    for spec in specs:
        cmd = build_train_command(args, spec)
        print(
            f"Launching {spec['preset']}-{'our' if spec['our'] else 'torch'} "
            f"on {spec['device']}"
        )
        processes.append((spec, subprocess.Popen(cmd)))

    failures = []
    for spec, proc in processes:
        returncode = proc.wait()
        if returncode != 0:
            failures.append((spec, returncode))

    if failures:
        messages = [
            f"{spec['preset']}-{'our' if spec['our'] else 'torch'} exited with {code}"
            for spec, code in failures
        ]
        raise RuntimeError("Some runs failed: " + "; ".join(messages))

    presets = preset_configs()
    for preset_name in ["optimal", "paper"]:
        run_root = Path(args.runs_root) / preset_name
        summary = build_summary(
            run_root, args.layer, args.ignore_first_train_steps, presets[preset_name]
        )
        with open(run_root / "fixed_benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        make_plots(args, preset_name)
        print(f"Saved summary and figures for {preset_name}")


if __name__ == "__main__":
    main()
