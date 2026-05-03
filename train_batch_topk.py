import argparse
import json
import multiprocessing as mp
from pathlib import Path

from nnsight import LanguageModel
import wandb

from dictionary_learning import ActivationBuffer
from dictionary_learning.evaluation import evaluate
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.utils import hf_dataset_to_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Batch Top-K SAE with torch.topk or the Triton path."
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
    parser.add_argument("--expansion-factor", type=int, default=16)
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--sae-batch-size", type=int, default=4096)
    parser.add_argument("--n-ctxs", type=int, default=100)
    parser.add_argument("--save-dir", default="runs/batch_topk")
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
    return parser.parse_args()


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
    dictionary: BatchTopKSAE,
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


def main() -> None:
    args = parse_args()
    topk_impl = "our" if args.our else "torch"

    model = LanguageModel(args.model_name, device_map=args.device)
    submodule = model.gpt_neox.layers[args.layer].mlp
    activation_dim = model.config.hidden_size
    dict_size = args.expansion_factor * activation_dim

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
        "k": args.k,
        "topk": topk_impl,
        "wandb_name": f"BatchTopKSAE-{topk_impl}",
        "submodule_name": f"mlp_layer_{args.layer}",
    }

    save_dir = Path(args.save_dir) / f"layer_{args.layer}_{topk_impl}"
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training BatchTopK SAE with topk={topk_impl}")
    print(f"Saving outputs to {save_dir}")
    if args.use_wandb:
        print(
            f"W&B enabled: project={args.wandb_project}, "
            f"entity={args.wandb_entity or '<default>'}, log_steps={args.log_steps}"
        )

    def post_step_callback(step: int, trainers: list, log_queues: list) -> None:
        if not args.run_eval or args.eval_every <= 0 or step % args.eval_every != 0:
            return

        trainer_dir = save_dir / "trainer_0"
        eval_path = trainer_dir / f"eval_results_step_{step}.json"
        eval_results = run_eval(
            args=args,
            model=model,
            submodule=submodule,
            activation_dim=activation_dim,
            dictionary=trainers[0].ae.to(dtype=model.dtype),
        )

        print(f"Evaluation results at step {step}:")
        print(json.dumps(eval_results, indent=2, sort_keys=True))

        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2, sort_keys=True)

        if args.use_wandb and log_queues:
            log_queues[0].put(
                {
                    "_step": step,
                    "step": step,
                    "eval_at_step": step,
                    **{f"eval/{k}": v for k, v in eval_results.items()},
                }
            )

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
        trainer_dir = save_dir / "trainer_0"
        ae_path = trainer_dir / "ae.pt"
        eval_path = trainer_dir / "eval_results.json"

        dictionary = BatchTopKSAE.from_pretrained(
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



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
