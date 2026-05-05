import argparse
import csv
import itertools
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

from nnsight import LanguageModel

from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.utils import hf_dataset_to_generator


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep BatchTopK parameters to maximize end-to-end training-step speedup "
            "of topk='our' over topk='torch'."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--dataset-name", default="openwebtext")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--threshold-start-step", type=int, default=1000)
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--n-ctxs", type=int, default=100)
    parser.add_argument("--k-values", default="16,32,64")
    parser.add_argument("--dict-sizes", default="4096,8192")
    parser.add_argument("--sae-batch-sizes", default="2048,4096")
    parser.add_argument("--ignore-first-steps", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--save-dir", default="runs/batch_topk_speed_sweep")
    parser.set_defaults(use_wandb=True)
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project", default="batch_topk")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--log-steps", type=int, default=25)
    return parser.parse_args()


def make_buffer(
    dataset_name: str,
    model: LanguageModel,
    submodule,
    activation_dim: int,
    device: str,
    llm_batch_size: int,
    sae_batch_size: int,
    n_ctxs: int,
) -> ActivationBuffer:
    data = hf_dataset_to_generator(dataset_name)
    return ActivationBuffer(
        data=data,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        n_ctxs=n_ctxs,
        device=device,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
    )


def median_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def run_single_config(
    *,
    args: argparse.Namespace,
    model: LanguageModel,
    submodule,
    activation_dim: int,
    topk_impl: str,
    k: int,
    dict_size: int,
    sae_batch_size: int,
    run_root: Path,
) -> dict[str, float | int | str]:
    buffer = make_buffer(
        dataset_name=args.dataset_name,
        model=model,
        submodule=submodule,
        activation_dim=activation_dim,
        device=args.device,
        llm_batch_size=args.llm_batch_size,
        sae_batch_size=sae_batch_size,
        n_ctxs=args.n_ctxs,
    )

    run_name = f"BatchTopKSpeed-{topk_impl}-k{k}-d{dict_size}-b{sae_batch_size}"
    step_times: list[float] = []
    topk_times: list[float] = []
    topk_fracs: list[float] = []
    throughputs: list[float] = []

    trainer_cfg = {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": activation_dim,
        "dict_size": dict_size,
        "lr": args.lr,
        "device": args.device,
        "steps": args.steps,
        "layer": args.layer,
        "lm_name": args.model_name,
        "warmup_steps": args.warmup_steps,
        "threshold_start_step": args.threshold_start_step,
        "k": k,
        "topk": topk_impl,
        "wandb_name": run_name,
        "submodule_name": f"mlp_layer_{args.layer}",
    }

    def post_step_callback(step: int, trainers: list, log_queues: list) -> None:
        del log_queues
        if step <= args.ignore_first_steps:
            return
        trainer = trainers[0]
        step_times.append(float(trainer.step_time_ms))
        topk_times.append(float(trainer.topk_time_ms))
        topk_fracs.append(float(trainer.topk_frac_of_step))
        throughputs.append(float(trainer.samples_per_sec))

    trainSAE(
        data=buffer,
        trainer_configs=[trainer_cfg],
        steps=args.steps,
        save_dir=None,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        log_steps=args.log_steps,
        run_cfg={
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "sweep_type": "batch_topk_speed",
            "topk_impl": topk_impl,
            "k": k,
            "dict_size": dict_size,
            "sae_batch_size": sae_batch_size,
        },
        post_step_callback=post_step_callback,
    )

    return {
        "topk_impl": topk_impl,
        "k": k,
        "dict_size": dict_size,
        "sae_batch_size": sae_batch_size,
        "median_step_time_ms": median_or_zero(step_times),
        "median_topk_time_ms": median_or_zero(topk_times),
        "median_topk_frac_of_step": median_or_zero(topk_fracs),
        "median_samples_per_sec": median_or_zero(throughputs),
        "measured_steps": len(step_times),
    }


def write_pair_results(csv_path: Path, results: list[dict]) -> None:
    if not results:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def build_best_summary(
    results: list[dict], args: argparse.Namespace, start_time: float
) -> dict:
    best = max(
        results,
        key=lambda row: (row["step_speedup_ratio"], row["step_time_delta_ms"]),
    )
    return {
        "selection_metric": "max step_speedup_ratio, then max step_time_delta_ms",
        "max_hours": args.max_hours,
        "steps_per_run": args.steps,
        "ignore_first_steps": args.ignore_first_steps,
        "best": best,
        "tested_pairs": len(results),
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def main() -> None:
    args = parse_args()
    k_values = parse_int_list(args.k_values)
    dict_sizes = parse_int_list(args.dict_sizes)
    sae_batch_sizes = parse_int_list(args.sae_batch_sizes)

    model = LanguageModel(args.model_name, device_map=args.device)
    submodule = model.gpt_neox.layers[args.layer].mlp
    activation_dim = model.config.hidden_size

    for dict_size in dict_sizes:
        if dict_size % activation_dim != 0:
            raise ValueError(
                f"dict_size={dict_size} must be divisible by activation_dim={activation_dim}"
            )

    run_root = Path(args.save_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    all_pair_results: list[dict] = []
    start_time = time.perf_counter()
    time_budget_seconds = args.max_hours * 3600.0
    completed_pairs = 0

    combos = list(itertools.product(k_values, dict_sizes, sae_batch_sizes))
    total_combos = len(combos)

    for combo_index, (k, dict_size, sae_batch_size) in enumerate(combos, start=1):
        elapsed = time.perf_counter() - start_time
        if elapsed >= time_budget_seconds:
            print("Time budget exhausted before starting next combo.")
            break

        if completed_pairs > 0:
            avg_pair_time = elapsed / completed_pairs
            remaining = time_budget_seconds - elapsed
            if remaining < avg_pair_time * 1.1:
                print(
                    "Stopping before next combo because the estimated next paired run "
                    "would likely exceed the time budget."
                )
                break

        print(
            f"\n[{combo_index}/{total_combos}] Running paired benchmark for "
            f"k={k}, dict_size={dict_size}, sae_batch_size={sae_batch_size}"
        )

        order = ("torch", "our") if combo_index % 2 == 1 else ("our", "torch")
        run_metrics: dict[str, dict] = {}

        for topk_impl in order:
            run_metrics[topk_impl] = run_single_config(
                args=args,
                model=model,
                submodule=submodule,
                activation_dim=activation_dim,
                topk_impl=topk_impl,
                k=k,
                dict_size=dict_size,
                sae_batch_size=sae_batch_size,
                run_root=run_root,
            )

        torch_metrics = run_metrics["torch"]
        our_metrics = run_metrics["our"]

        torch_step = float(torch_metrics["median_step_time_ms"])
        our_step = float(our_metrics["median_step_time_ms"])
        torch_topk = float(torch_metrics["median_topk_time_ms"])
        our_topk = float(our_metrics["median_topk_time_ms"])
        torch_sps = float(torch_metrics["median_samples_per_sec"])
        our_sps = float(our_metrics["median_samples_per_sec"])

        pair_result = {
            "k": k,
            "dict_size": dict_size,
            "sae_batch_size": sae_batch_size,
            "torch_step_time_ms": torch_step,
            "our_step_time_ms": our_step,
            "step_speedup_ratio": (torch_step / our_step) if our_step > 0 else 0.0,
            "step_time_delta_ms": torch_step - our_step,
            "torch_topk_time_ms": torch_topk,
            "our_topk_time_ms": our_topk,
            "topk_speedup_ratio": (torch_topk / our_topk) if our_topk > 0 else 0.0,
            "torch_topk_frac_of_step": float(torch_metrics["median_topk_frac_of_step"]),
            "our_topk_frac_of_step": float(our_metrics["median_topk_frac_of_step"]),
            "torch_samples_per_sec": torch_sps,
            "our_samples_per_sec": our_sps,
            "throughput_speedup_ratio": (our_sps / torch_sps) if torch_sps > 0 else 0.0,
            "torch_measured_steps": int(torch_metrics["measured_steps"]),
            "our_measured_steps": int(our_metrics["measured_steps"]),
        }
        all_pair_results.append(pair_result)
        completed_pairs += 1

        write_pair_results(run_root / "pair_results.csv", all_pair_results)
        with open(run_root / "pair_results.json", "w") as f:
            json.dump(all_pair_results, f, indent=2)
        with open(run_root / "best_params.json", "w") as f:
            json.dump(build_best_summary(all_pair_results, args, start_time), f, indent=2)

        print(json.dumps(pair_result, indent=2))

    if not all_pair_results:
        raise RuntimeError("No paired results were produced.")

    summary = build_best_summary(all_pair_results, args, start_time)

    with open(run_root / "best_params.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBest configuration found:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
