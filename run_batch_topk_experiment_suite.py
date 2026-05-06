import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run named BatchTopK benchmark presets (optimal, paper) with separate "
            "runs/ and figures/ folders."
        )
    )
    parser.add_argument(
        "--which",
        choices=["optimal", "paper", "both"],
        default="both",
        help="Which preset(s) to run.",
    )
    parser.add_argument("--device", default="cuda:0")
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
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def preset_configs() -> dict[str, dict[str, int]]:
    return {
        "optimal": {
            "k": 12,
            "expansion_factor": 32,  # dict_size = 16384 for hidden size 512
            "sae_batch_size": 1024,
        },
        "paper": {
            # Assumption: representative GPT-2-Small-style BatchTopK setting from
            # the paper's reported regime.
            "k": 32,
            "expansion_factor": 24,  # dict_size = 12288 for hidden size 512
            "sae_batch_size": 4096,
        },
    }


def build_command(args: argparse.Namespace, preset_name: str, preset: dict[str, int]) -> list[str]:
    save_dir = Path(args.runs_root) / preset_name
    figure_dir = Path(args.figures_root) / preset_name

    cmd = [
        sys.executable,
        "run_batch_topk_fixed_benchmark.py",
        "--device",
        args.device,
        "--model-name",
        args.model_name,
        "--dataset-name",
        args.dataset_name,
        "--layer",
        str(args.layer),
        "--steps",
        str(args.steps),
        "--k",
        str(preset["k"]),
        "--lr",
        str(args.lr),
        "--warmup-steps",
        str(args.warmup_steps),
        "--threshold-start-step",
        str(args.threshold_start_step),
        "--expansion-factor",
        str(preset["expansion_factor"]),
        "--llm-batch-size",
        str(args.llm_batch_size),
        "--sae-batch-size",
        str(preset["sae_batch_size"]),
        "--n-ctxs",
        str(args.n_ctxs),
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
        "--log-steps",
        str(args.log_steps),
        "--ignore-first-train-steps",
        str(args.ignore_first_train_steps),
        "--save-dir",
        str(save_dir),
        "--figure-dir",
        str(figure_dir),
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
    if args.skip_plots:
        cmd.append("--skip-plots")

    return cmd


def main() -> None:
    args = parse_args()
    presets = preset_configs()

    if args.which == "both":
        selected = ["optimal", "paper"]
    else:
        selected = [args.which]

    for preset_name in selected:
        preset = presets[preset_name]
        print(
            f"Running preset '{preset_name}' with "
            f"k={preset['k']}, "
            f"expansion_factor={preset['expansion_factor']}, "
            f"sae_batch_size={preset['sae_batch_size']}"
        )
        cmd = build_command(args, preset_name, preset)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
