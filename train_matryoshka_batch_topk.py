import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

from nnsight import LanguageModel

from dictionary_learning import ActivationBuffer
from dictionary_learning.evaluation import evaluate
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
from dictionary_learning.utils import hf_dataset_to_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Matryoshka Batch Top-K SAE with torch.topk or the Triton path."
    )
    parser.add_argument("--our", action="store_true", help="Use topk='our' instead of topk='torch'.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--dataset-name", default="openwebtext")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--steps", type=int, default=244141)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--threshold-start-step", type=int, default=1000)
    parser.add_argument("--dict-size", type=int, default=12288)
    parser.add_argument("--group-fractions", default="0.25,0.25,0.5")
    parser.add_argument("--group-weights", default="")
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--sae-batch-size", type=int, default=4096)
    parser.add_argument("--n-ctxs", type=int, default=100)
    parser.add_argument("--save-dir", default="runs/matryoshka_batch_topk")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="dictionary_learning")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-n-batches", type=int, default=10)
    parser.add_argument("--eval-llm-batch-size", type=int, default=100)
    parser.add_argument("--eval-sae-batch-size", type=int, default=4096)
    parser.add_argument("--eval-n-ctxs", type=int, default=256)
    parser.add_argument("--eval-log-file", default="eval_history.jsonl")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def make_buffer(
    args: argparse.Namespace,
    model: LanguageModel,
    submodule,
    activation_dim: int,
    llm_batch_size: int,
    sae_batch_size: int,
    n_ctxs: int,
) -> ActivationBuffer:
    data = hf_dataset_to_generator(args.dataset_name)
    return ActivationBuffer(
        data=data,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        n_ctxs=n_ctxs,
        device=args.device,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
    )


def run_eval(
    args: argparse.Namespace,
    model: LanguageModel,
    submodule,
    activation_dim: int,
    dictionary: MatryoshkaBatchTopKSAE,
) -> dict[str, float]:
    eval_buffer = make_buffer(
        args=args,
        model=model,
        submodule=submodule,
        activation_dim=activation_dim,
        llm_batch_size=args.eval_llm_batch_size,
        sae_batch_size=args.eval_sae_batch_size,
        n_ctxs=args.eval_n_ctxs,
    )
    return evaluate(
        dictionary=dictionary,
        activations=eval_buffer,
        batch_size=args.eval_llm_batch_size,
        device=args.device,
        n_batches=args.eval_n_batches,
    )


def clone_dictionary_for_eval(
    source: MatryoshkaBatchTopKSAE,
    topk_impl: str,
    device: str,
    dtype,
) -> MatryoshkaBatchTopKSAE:
    dictionary = MatryoshkaBatchTopKSAE(
        activation_dim=source.activation_dim,
        dict_size=source.dict_size,
        k=int(source.k.item()),
        group_sizes=source.group_sizes.tolist(),
        topk=topk_impl,
    ).to(device)
    dictionary.load_state_dict(source.state_dict())
    dictionary = dictionary.to(dtype=dtype)
    dictionary.eval()
    return dictionary


def main() -> None:
    args = parse_args()
    topk_impl = "our" if args.our else "torch"
    group_fractions = parse_float_list(args.group_fractions)
    group_weights = parse_float_list(args.group_weights) if args.group_weights else None

    model = LanguageModel(args.model_name, device_map=args.device)
    submodule = model.gpt_neox.layers[args.layer].mlp
    activation_dim = model.config.hidden_size

    buffer = make_buffer(
        args=args,
        model=model,
        submodule=submodule,
        activation_dim=activation_dim,
        llm_batch_size=args.llm_batch_size,
        sae_batch_size=args.sae_batch_size,
        n_ctxs=args.n_ctxs,
    )

    trainer_cfg = {
        "trainer": MatryoshkaBatchTopKTrainer,
        "dict_class": MatryoshkaBatchTopKSAE,
        "activation_dim": activation_dim,
        "dict_size": args.dict_size,
        "group_fractions": group_fractions,
        "group_weights": group_weights,
        "lr": args.lr,
        "device": args.device,
        "steps": args.steps,
        "layer": args.layer,
        "lm_name": args.model_name,
        "warmup_steps": args.warmup_steps,
        "threshold_start_step": args.threshold_start_step,
        "k": args.k,
        "topk": topk_impl,
        "wandb_name": f"MatryoshkaBatchTopKSAE-{topk_impl}",
        "submodule_name": f"mlp_layer_{args.layer}",
    }

    save_dir = Path(args.save_dir) / f"layer_{args.layer}_{topk_impl}"
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training MatryoshkaBatchTopK SAE with topk={topk_impl}")
    print(f"Saving outputs to {save_dir}")

    trainer_dir = save_dir / "trainer_0"
    eval_history_path = trainer_dir / args.eval_log_file
    eval_history_csv_path = trainer_dir / "eval_history.csv"

    def append_eval_record(step: int, eval_results: dict[str, float]) -> None:
        trainer_dir.mkdir(parents=True, exist_ok=True)
        record = {"step": step, **eval_results}
        with open(eval_history_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        csv_exists = eval_history_csv_path.exists()
        with open(eval_history_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not csv_exists:
                writer.writeheader()
            writer.writerow(record)

    def post_step_callback(step: int, trainers: list, log_queues: list) -> None:
        del log_queues
        if not args.run_eval or args.eval_every <= 0 or step % args.eval_every != 0:
            return

        eval_path = trainer_dir / f"eval_results_step_{step}.json"
        eval_dictionary = clone_dictionary_for_eval(
            source=trainers[0].ae,
            topk_impl=topk_impl,
            device=args.device,
            dtype=model.dtype,
        )
        eval_results = run_eval(
            args=args,
            model=model,
            submodule=submodule,
            activation_dim=activation_dim,
            dictionary=eval_dictionary,
        )

        print(f"Evaluation results at step {step}:")
        print(json.dumps(eval_results, indent=2, sort_keys=True))

        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2, sort_keys=True)
        append_eval_record(step, eval_results)

    trainSAE(
        data=buffer,
        trainer_configs=[trainer_cfg],
        steps=args.steps,
        save_dir=str(save_dir),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        log_steps=args.log_steps,
        run_cfg={
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "topk_impl": topk_impl,
            "save_dir": str(save_dir),
        },
        post_step_callback=post_step_callback,
    )

    if args.run_eval:
        ae_path = trainer_dir / "ae.pt"
        eval_path = trainer_dir / "eval_results.json"

        dictionary = MatryoshkaBatchTopKSAE.from_pretrained(
            str(ae_path),
            k=args.k,
            device=args.device,
            topk=topk_impl,
        ).to(dtype=model.dtype)
        eval_results = run_eval(
            args=args,
            model=model,
            submodule=submodule,
            activation_dim=activation_dim,
            dictionary=dictionary,
        )

        print("Evaluation results:")
        print(json.dumps(eval_results, indent=2, sort_keys=True))

        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2, sort_keys=True)
        append_eval_record(args.steps, eval_results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
