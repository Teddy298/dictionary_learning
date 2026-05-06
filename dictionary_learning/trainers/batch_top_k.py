import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import time
from collections import namedtuple
from typing import Callable, Optional

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


if triton is not None:

    @triton.jit
    def _filter_topk_kernel(
        x_ptr,
        out_vals_ptr,
        out_idx_ptr,
        count_ptr,
        threshold,
        n_elements,
        max_out,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        read_mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=read_mask, other=-float("inf"))

        pass_mask = (x >= threshold) & read_mask

        pass_int = pass_mask.to(tl.int32)
        n_pass = tl.sum(pass_int)

        if n_pass > 0:
            global_offset = tl.atomic_add(count_ptr, n_pass)
            local_offset = tl.cumsum(pass_int) - 1
            write_idx = global_offset + local_offset
            write_mask = pass_mask & (write_idx < max_out)

            tl.store(out_vals_ptr + write_idx, x, mask=write_mask)
            tl.store(out_idx_ptr + write_idx, offsets, mask=write_mask)

    @triton.jit
    def _filter_topk_kernel_new(
        x_ptr,
        out_vals_ptr,
        out_idx_ptr,
        count_ptr,
        threshold,
        n_elements,
        max_out,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        read_mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=read_mask, other=-float("inf"))
        pass_mask = (x >= threshold) & read_mask
        pass_int = pass_mask.to(tl.int32)
        n_pass = tl.sum(pass_int)

        if n_pass > 0:
            global_offset = tl.atomic_add(count_ptr, n_pass)
            local_offset = tl.cumsum(pass_int) - 1
            write_idx = global_offset + local_offset
            write_mask = pass_mask & (write_idx < max_out)

            tl.store(out_vals_ptr + write_idx, x, mask=write_mask)
            tl.store(out_idx_ptr + write_idx, offsets, mask=write_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        ],
        key=["n_elements"],
        reset_to_zero=["count_ptr"],
    )
    @triton.jit
    def _filter_topk_batched_kernel(
        x_ptr,
        out_vals_ptr,
        out_idx_ptr,
        count_ptr,
        thresholds_ptr,
        stride_xb,
        stride_ob,
        n_elements,
        max_out,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_chunk = tl.program_id(axis=0)
        pid_batch = tl.program_id(axis=1)

        block_start = pid_chunk * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        read_mask = offsets < n_elements

        x_offset = pid_batch * stride_xb + offsets
        x = tl.load(x_ptr + x_offset, mask=read_mask, other=-float("inf"))
        thresh = tl.load(thresholds_ptr + pid_batch)

        pass_mask = (x >= thresh) & read_mask
        pass_int = pass_mask.to(tl.int32)
        n_pass = tl.sum(pass_int)

        if n_pass > 0:
            global_offset = tl.atomic_add(count_ptr + pid_batch, n_pass)
            local_offset = tl.cumsum(pass_int) - 1
            write_idx = global_offset + local_offset
            write_mask = pass_mask & (write_idx < max_out)

            out_offset = pid_batch * stride_ob + write_idx
            tl.store(out_vals_ptr + out_offset, x, mask=write_mask)
            tl.store(
                out_idx_ptr + out_offset, offsets.to(tl.int64), mask=write_mask
            )


def prof_k_params(
    n: int,
    k: int,
    z: float = 7.0,
    s_min: int = 65_536,
    s_max: int = 262_144,
) -> tuple[int, int, int]:
    alpha = k / n

    s_opt = int(2.30 * (k * (n - k)) ** (1 / 3))
    s = max(s_min, min(s_max, s_opt))

    f = 1.0 - s / n
    z_a = z / 2.0
    t_rank = int(
        math.ceil(alpha * s + z_a * math.sqrt(alpha * (1 - alpha) * s * f))
    )
    t_rank = max(1, min(s, t_rank))

    delta = z * math.sqrt(k * (n - k) / s * f)
    max_out = int(math.ceil(k + delta))
    max_out = max(int(1.2 * k), min(n, max_out))
    return s, t_rank, max_out


def ks_topk_triton(
    x: t.Tensor, k: int, eps: float = 1e-3
) -> tuple[t.Tensor, t.Tensor]:
    del eps

    if triton is None:
        raise RuntimeError("topk='our' requires Triton to be installed.")
    if not x.is_cuda:
        raise RuntimeError("topk='our' requires CUDA tensors.")

    n = x.numel()
    if k <= 0 or k > n:
        raise ValueError(f"k={k} must satisfy 0 < k <= n={n}")
    if n < 2:
        post_topk = x.topk(k, sorted=False, dim=-1)
        return post_topk.values, post_topk.indices

    sample_size = int((2 / 3) ** (2 / 3) * (n * math.log(n)) ** (2 / 3))
    sample_size = max(1, min(sample_size, n))
    sample = x[t.randint(0, n, (sample_size,), device=x.device)]

    alpha = k / n
    sigma = math.sqrt(sample_size * alpha * (1 - alpha))
    margin = max(1, int(2.0 * sigma))

    k_sample = min(int(sample_size * alpha) + margin, sample_size)

    sample_top, _ = t.topk(sample, k_sample)
    threshold = sample_top[-1].item()

    max_out = min(n, int(k * 1.05) + 50_000)

    out_idx = t.empty(max_out, dtype=t.int64, device=x.device)
    count = t.zeros(1, dtype=t.int32, device=x.device)

    block_size = 4096
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _filter_topk_kernel[grid](
        x,
        t.empty(max_out, dtype=x.dtype, device=x.device),
        out_idx,
        count,
        threshold,
        n,
        max_out,
        BLOCK_SIZE=block_size,
    )

    total_passed = count.item()
    if total_passed < k:
        post_topk = x.topk(k, sorted=False, dim=-1)
        return post_topk.values, post_topk.indices

    valid_count = min(total_passed, max_out)
    valid_idx = out_idx[:valid_count]
    # Re-gather values from the original tensor so autograd can propagate
    # gradients back through the selected entries.
    valid_vals = x[valid_idx]

    final_topk = t.topk(valid_vals, k)
    return final_topk.values, valid_idx[final_topk.indices]


def ks_topk_triton_new(
    x: t.Tensor, k: int, eps: float = 1e-3
) -> tuple[t.Tensor, t.Tensor]:
    del eps

    if triton is None:
        raise RuntimeError("topk='our_new' requires Triton to be installed.")
    if not x.is_cuda:
        raise RuntimeError("topk='our_new' requires CUDA tensors.")

    if x.dim() != 1:
        x = x.reshape(-1)

    n = x.numel()
    if k <= 0 or k > n:
        raise ValueError(f"k={k} must satisfy 0 < k <= n={n}")
    if n < 2:
        post_topk = x.topk(k, sorted=False, dim=-1)
        return post_topk.values, post_topk.indices

    sample_size, t_rank, max_out = prof_k_params(n, k)
    stride = max(1, (n - sample_size + 1) // sample_size)
    sample = x[::stride]
    if sample.numel() > sample_size:
        sample = sample[:sample_size]
    if sample.numel() < t_rank:
        post_topk = x.topk(k, sorted=False, dim=-1)
        return post_topk.values, post_topk.indices

    sample_top, _ = t.topk(sample, t_rank, sorted=False)
    threshold = sample_top[-1].item()

    out_vals = t.full((max_out,), -float("inf"), dtype=x.dtype, device=x.device)
    out_idx = t.empty(max_out, dtype=t.int64, device=x.device)
    count = t.zeros(1, dtype=t.int32, device=x.device)

    block_size = 2048
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _filter_topk_kernel_new[grid](
        x,
        out_vals,
        out_idx,
        count,
        threshold,
        n,
        max_out,
        BLOCK_SIZE=block_size,
    )

    total_passed = count.item()
    if total_passed < k or total_passed > max_out:
        post_topk = x.topk(k, sorted=False, dim=-1)
        return post_topk.values, post_topk.indices

    valid_out_vals = out_vals[:total_passed]
    valid_idx = out_idx[:total_passed]
    final_topk = t.topk(valid_out_vals, k, sorted=False)
    return x[valid_idx[final_topk.indices]], valid_idx[final_topk.indices]


def ks_topk_triton_fixed(
    x: t.Tensor,
    k: int,
    s: int,
    t_rank: int,
    max_out: int,
) -> tuple[t.Tensor, t.Tensor]:
    if triton is None:
        raise RuntimeError("fixed Triton topk variants require Triton to be installed.")
    if not x.is_cuda:
        raise RuntimeError("fixed Triton topk variants require CUDA tensors.")

    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = x.contiguous()
    batch_size, n = x.shape
    if k <= 0 or k > n:
        raise ValueError(f"k={k} must satisfy 0 < k <= n={n}")

    stride = max(1, (n - s + 1) // s)
    sample = x[:, ::stride]
    if sample.size(1) > s:
        sample = sample[:, :s]

    effective_t = min(t_rank, sample.size(1))
    if effective_t <= 0:
        post_topk = x.topk(k, dim=1)
        return post_topk.values.reshape(-1), post_topk.indices.reshape(-1)

    sample_top, _ = t.topk(sample, effective_t, dim=1, sorted=False)
    thresholds = sample_top[:, -1].contiguous()

    out_vals = t.full(
        (batch_size, max_out), -float("inf"), dtype=x.dtype, device=x.device
    )
    out_idx = t.empty((batch_size, max_out), dtype=t.int64, device=x.device)
    counts = t.zeros(batch_size, dtype=t.int32, device=x.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), batch_size)
    _filter_topk_batched_kernel[grid](
        x,
        out_vals,
        out_idx,
        counts,
        thresholds,
        x.stride(0),
        out_vals.stride(0),
        n,
        max_out,
    )

    final_top_vals, final_local_idx = t.topk(out_vals, k, dim=1, sorted=False)
    if t.isneginf(final_top_vals[:, -1]).any():
        post_topk = x.topk(k, dim=1)
        return post_topk.values.reshape(-1), post_topk.indices.reshape(-1)

    final_top_idx = out_idx.gather(1, final_local_idx)
    gathered_vals = x.gather(1, final_top_idx)
    return gathered_vals.reshape(-1), final_top_idx.reshape(-1)


def ks_topk_triton_1(x: t.Tensor, k: int) -> tuple[t.Tensor, t.Tensor]:
    return ks_topk_triton_fixed(x, k, s=32768, t_rank=110, max_out=201167)


def ks_topk_triton_2(x: t.Tensor, k: int) -> tuple[t.Tensor, t.Tensor]:
    return ks_topk_triton_fixed(x, k, s=32768, t_rank=118, max_out=230234)


def ks_topk_triton_3(x: t.Tensor, k: int) -> tuple[t.Tensor, t.Tensor]:
    return ks_topk_triton_fixed(x, k, s=43099, t_rank=150, max_out=196608)


def ks_topk_triton_fixed_nofallback(
    x: t.Tensor,
    k: int,
    s: int,
    t_rank: int,
    max_out: int,
) -> tuple[t.Tensor, t.Tensor]:
    if triton is None:
        raise RuntimeError("fixed Triton topk variants require Triton to be installed.")
    if not x.is_cuda:
        raise RuntimeError("fixed Triton topk variants require CUDA tensors.")

    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = x.contiguous()
    batch_size, n = x.shape
    if k <= 0 or k > n:
        raise ValueError(f"k={k} must satisfy 0 < k <= n={n}")

    stride = max(1, (n - s + 1) // s)
    sample = x[:, ::stride]
    if sample.size(1) > s:
        sample = sample[:, :s]

    effective_t = min(t_rank, sample.size(1))
    if effective_t <= 0:
        post_topk = x.topk(k, dim=1)
        return post_topk.values.reshape(-1), post_topk.indices.reshape(-1)

    sample_top, _ = t.topk(sample, effective_t, dim=1, sorted=False)
    thresholds = sample_top[:, -1].contiguous()

    out_vals = t.full(
        (batch_size, max_out), -float("inf"), dtype=x.dtype, device=x.device
    )
    out_idx = t.empty((batch_size, max_out), dtype=t.int64, device=x.device)
    counts = t.zeros(batch_size, dtype=t.int32, device=x.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), batch_size)
    _filter_topk_batched_kernel[grid](
        x,
        out_vals,
        out_idx,
        counts,
        thresholds,
        x.stride(0),
        out_vals.stride(0),
        n,
        max_out,
    )

    final_top_vals, final_local_idx = t.topk(out_vals, k, dim=1, sorted=False)
    final_top_idx = out_idx.gather(1, final_local_idx)
    gathered_vals = x.gather(1, final_top_idx)
    return gathered_vals.reshape(-1), final_top_idx.reshape(-1)


def ks_topk_triton_4(x: t.Tensor, k: int) -> tuple[t.Tensor, t.Tensor]:
    return ks_topk_triton_fixed_nofallback(x, k, s=32768, t_rank=110, max_out=201167)


def ks_topk_triton_5(x: t.Tensor, k: int) -> tuple[t.Tensor, t.Tensor]:
    return ks_topk_triton_fixed_nofallback(x, k, s=32768, t_rank=118, max_out=230234)


def ks_topk_triton_6(x: t.Tensor, k: int) -> tuple[t.Tensor, t.Tensor]:
    return ks_topk_triton_fixed_nofallback(x, k, s=43099, t_rank=150, max_out=196608)


class BatchTopKSAE(Dictionary, nn.Module):
    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        topk: str = "torch",
        batch_topk: Optional[Callable[[t.Tensor, int], tuple[t.Tensor, t.Tensor]]] = None,
        batch_topk_name: str = "torch.topk",
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.topk = topk
        self.batch_topk_name = batch_topk_name

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        if topk not in {"torch", "our", "our_new", "our_1", "our_2", "our_3", "our_4", "our_5", "our_6"}:
            raise ValueError(
                f"topk={topk!r} must be 'torch', 'our', 'our_new', 'our_1', 'our_2', 'our_3', 'our_4', 'our_5', or 'our_6'"
            )
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        if batch_topk is not None:
            self.batch_topk = batch_topk
            self.batch_topk_name = batch_topk_name
        elif topk == "our":
            self.batch_topk = ks_topk_triton
            self.batch_topk_name = "our"
        elif topk == "our_new":
            self.batch_topk = ks_topk_triton_new
            self.batch_topk_name = "our_new"
        elif topk == "our_1":
            self.batch_topk = ks_topk_triton_1
            self.batch_topk_name = "our_1"
        elif topk == "our_2":
            self.batch_topk = ks_topk_triton_2
            self.batch_topk_name = "our_2"
        elif topk == "our_3":
            self.batch_topk = ks_topk_triton_3
            self.batch_topk_name = "our_3"
        elif topk == "our_4":
            self.batch_topk = ks_topk_triton_4
            self.batch_topk_name = "our_4"
        elif topk == "our_5":
            self.batch_topk = ks_topk_triton_5
            self.batch_topk_name = "our_5"
        elif topk == "our_6":
            self.batch_topk = ks_topk_triton_6
            self.batch_topk_name = "our_6"
        else:
            self.batch_topk = None
            self.batch_topk_name = "torch.topk"

        self.last_topk_time_ms = 0.0

    def select_batch_topk(
        self, flattened_acts: t.Tensor, k: int
    ) -> tuple[t.Tensor, t.Tensor]:
        if flattened_acts.is_cuda:
            start_event = t.cuda.Event(enable_timing=True)
            end_event = t.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        if self.batch_topk is None:
            post_topk = flattened_acts.topk(k, sorted=False, dim=-1)
            values, indices = post_topk.values, post_topk.indices
        else:
            post_topk = self.batch_topk(flattened_acts, k)

            if hasattr(post_topk, "values") and hasattr(post_topk, "indices"):
                values = post_topk.values
                indices = post_topk.indices
            else:
                values, indices = post_topk

        if flattened_acts.is_cuda:
            end_event.record()
            end_event.synchronize()
            self.last_topk_time_ms = start_event.elapsed_time(end_event)
        else:
            self.last_topk_time_ms = (time.perf_counter() - start_time) * 1000.0

        return values, indices

    def encode(
        self, x: t.Tensor, return_active: bool = False, use_threshold: bool = True
    ):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
        else:
            # Flatten and perform batch top-k
            flattened_acts = post_relu_feat_acts_BF.flatten()
            topk_count = self.k.item() * x.size(0)
            topk_values, topk_indices = self.select_batch_topk(flattened_acts, topk_count)

            encoded_acts_BF = (
                t.zeros_like(post_relu_feat_acts_BF.flatten())
                .scatter_(-1, topk_indices, topk_values)
                .reshape(post_relu_feat_acts_BF.shape)
            )

        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(cls, path, k=None, device=None, **kwargs) -> "BatchTopKSAE":
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = cls(
            activation_dim,
            dict_size,
            k,
            topk=kwargs.get("topk", "torch"),
            batch_topk=kwargs.get("batch_topk"),
            batch_topk_name=kwargs.get("batch_topk_name", "torch.topk"),
        )
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class BatchTopKTrainer(SAETrainer):
    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        dict_class: type = BatchTopKSAE,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        k_anneal_steps: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "BatchTopKSAE",
        submodule_name: Optional[str] = None,
        topk: str = "torch",
        batch_topk: Optional[Callable[[t.Tensor, int], tuple[t.Tensor, t.Tensor]]] = None,
        batch_topk_name: str = "torch.topk",
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.k_anneal_steps = k_anneal_steps
        self.topk = topk

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.batch_topk_name = batch_topk_name
        self.ae = dict_class(
            activation_dim,
            dict_size,
            k,
            topk=topk,
            batch_topk=batch_topk,
            batch_topk_name=batch_topk_name,
        )

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.logging_parameters = [
            "effective_l0",
            "dead_features",
            "pre_norm_auxk_loss",
            "step_time_ms",
            "topk_time_ms",
            "topk_frac_of_step",
            "samples_per_sec",
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
        self.step_time_ms = 0.0
        self.topk_time_ms = 0.0
        self.topk_frac_of_step = 0.0
        self.samples_per_sec = 0.0

        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
    ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None:
            return

        assert 0 <= k_anneal_steps < self.steps, (
            "k_anneal_steps must be >= 0 and < steps."
        )
        # self.k is the target k set for the trainer, not the dictionary's current k
        assert activation_dim > self.k, "activation_dim must be greater than k"

        step = min(step, k_anneal_steps)
        ratio = step / k_anneal_steps
        annealed_value = activation_dim * (1 - ratio) + self.k * ratio

        # Update in-place
        self.ae.k.fill_(int(annealed_value))

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, f: t.Tensor):
        device_type = "cuda" if f.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = f[f > 0]

            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=t.float32)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(self, x, step=None, logging=False):
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=False
        )
        # l0 = (f != 0).float().sum(dim=-1).mean().item()

        if step > self.threshold_start_step:
            self.update_threshold(f)

        x_hat = self.ae.decode(f)

        e = x - x_hat

        self.effective_l0 = self.ae.k.item()

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x):
        use_cuda_timing = isinstance(self.device, str) and "cuda" in self.device
        if use_cuda_timing:
            step_start_event = t.cuda.Event(enable_timing=True)
            step_end_event = t.cuda.Event(enable_timing=True)
            step_start_event.record()
        else:
            step_start_time = time.perf_counter()

        if step == 0:
            median = self.geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step)
        self.topk_time_ms = float(self.ae.last_topk_time_ms)
        loss.backward()

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        if use_cuda_timing:
            step_end_event.record()
            step_end_event.synchronize()
            self.step_time_ms = step_start_event.elapsed_time(step_end_event)
        else:
            self.step_time_ms = (time.perf_counter() - step_start_time) * 1000.0

        if self.step_time_ms > 0:
            self.topk_frac_of_step = self.topk_time_ms / self.step_time_ms
            self.samples_per_sec = x.size(0) / (self.step_time_ms / 1000.0)
        else:
            self.topk_frac_of_step = 0.0
            self.samples_per_sec = 0.0

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "BatchTopKTrainer",
            "dict_class": "BatchTopKSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "topk": self.topk,
            "batch_topk_name": self.batch_topk_name,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = t.zeros_like(guess)
        weights = t.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break

        return guess
