import argparse
from pathlib import Path

from nnsight import LanguageModel

from dictionary_learning import ActivationBuffer
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topk_impl = "our" if args.our else "torch"

    model = LanguageModel(args.model_name, device_map=args.device)
    submodule = model.gpt_neox.layers[args.layer].mlp
    activation_dim = model.config.hidden_size
    dict_size = args.expansion_factor * activation_dim

    data = hf_dataset_to_generator(args.dataset_name)

    buffer = ActivationBuffer(
        data=data,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        n_ctxs=args.n_ctxs,
        device=args.device,
        refresh_batch_size=args.llm_batch_size,
        out_batch_size=args.sae_batch_size,
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

    trainSAE(
        data=buffer,
        trainer_configs=[trainer_cfg],
        steps=args.steps,
        save_dir=str(save_dir),
    )


if __name__ == "__main__":
    main()
