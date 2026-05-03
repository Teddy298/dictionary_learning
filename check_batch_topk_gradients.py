import argparse

import torch as t

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, triton


def clone_sae(sae: BatchTopKSAE, topk: str) -> BatchTopKSAE:
    other = BatchTopKSAE(
        activation_dim=sae.activation_dim,
        dict_size=sae.dict_size,
        k=int(sae.k.item()),
        topk=topk,
    ).to(next(sae.parameters()).device)
    other.load_state_dict(sae.state_dict())
    return other


def run_one(sae: BatchTopKSAE, x: t.Tensor) -> dict[str, float]:
    sae.zero_grad(set_to_none=True)
    x_local = x.detach().clone().requires_grad_(True)

    encoded = sae.encode(x_local, use_threshold=False)
    loss = sae.decode(encoded).pow(2).mean()
    loss.backward()

    encoder_grad = sae.encoder.weight.grad
    decoder_grad = sae.decoder.weight.grad

    return {
        "loss": float(loss.item()),
        "input_grad_norm": float(x_local.grad.norm().item()) if x_local.grad is not None else 0.0,
        "encoder_grad_norm": float(encoder_grad.norm().item()) if encoder_grad is not None else 0.0,
        "decoder_grad_norm": float(decoder_grad.norm().item()) if decoder_grad is not None else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare gradient flow for BatchTopK torch vs our implementations."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--activation-dim", type=int, default=64)
    parser.add_argument("--dict-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if "cuda" in args.device and not t.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if triton is None:
        raise RuntimeError("Triton is not installed, cannot check topk='our'.")

    t.manual_seed(args.seed)
    if "cuda" in args.device:
        t.cuda.manual_seed_all(args.seed)

    base = BatchTopKSAE(
        activation_dim=args.activation_dim,
        dict_size=args.dict_size,
        k=args.k,
        topk="torch",
    ).to(args.device)
    sae_torch = clone_sae(base, "torch")
    sae_our = clone_sae(base, "our")

    x = t.randn(args.batch_size, args.activation_dim, device=args.device)

    torch_stats = run_one(sae_torch, x)
    our_stats = run_one(sae_our, x)

    print("torch:", torch_stats)
    print("our:  ", our_stats)
    print()
    print("Interpretation:")
    print("- If 'our' gradient norms are zero while torch's are non-zero, autograd is broken.")
    print("- If losses match but gradient norms differ a lot, backward behavior differs.")


if __name__ == "__main__":
    main()
