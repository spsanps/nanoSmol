"""
Distributed Data Parallel (DDP) setup utilities.

Handles NCCL process group initialization for multi-GPU training via
``torchrun`` / ``torch.distributed.launch``.  Falls back gracefully to
single-GPU when ``RANK`` is not set in the environment.

Usage in train.py::

    rank, world_size, device = setup_distributed()
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    ...
    cleanup_distributed()
"""

import os
import torch
import torch.distributed as dist


def setup_distributed():
    """
    Initialize the distributed process group.

    Reads ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` from the environment
    (set automatically by ``torchrun``).  If these variables are absent,
    assumes single-GPU training and returns ``(0, 1, cuda:0)``.

    Returns
    -------
    rank : int
        Global rank of this process (0-indexed).
    world_size : int
        Total number of processes in the group.
    device : torch.device
        CUDA device assigned to this rank.
    """
    if "RANK" not in os.environ:
        # Single-GPU fallback -- no process group needed.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # NCCL is fastest for GPU-to-GPU collectives; gloo is the fallback for
    # CPU-only runs (unlikely in practice but keeps tests running).
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    # Synchronize all ranks before returning so that model loading on rank 0
    # (which may print diagnostics) does not race ahead of other ranks.
    dist.barrier()

    return rank, world_size, device


def cleanup_distributed():
    """
    Destroy the process group.

    Safe to call even when no process group was initialized (single-GPU path).
    Should be called at the very end of training, after the final checkpoint
    has been saved.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """
    Return ``True`` if this is rank 0 or if distributed training is not active.

    Use this to gate operations that should happen on only one process:
    logging, checkpoint saving, evaluation, wandb, etc.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    """
    Return the global rank of the current process.

    Returns 0 when distributed training is not active.
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """
    Return the total number of processes.

    Returns 1 when distributed training is not active.
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_mean(tensor):
    """
    All-reduce a scalar tensor by averaging across all ranks.

    When distributed training is not active the tensor is returned unchanged.
    This is useful for aggregating loss values before logging.

    Parameters
    ----------
    tensor : torch.Tensor
        A scalar (0-d or 1-element) tensor on the correct device.

    Returns
    -------
    torch.Tensor
        The mean-reduced tensor (same value on every rank after the call).
    """
    if not dist.is_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    return t
