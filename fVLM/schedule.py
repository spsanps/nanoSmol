"""
Learning rate schedule: cosine decay with linear warmup.

The standard recipe for vision-language model fine-tuning:

    LR
    ^
    |     /\\
    |    /  \\___
    |   /       \\___
    |  /             \\___
    | /                   \\___
    |/                         \\__ min_lr
    +------------------------------> step
    0   warmup     ...       total

During warmup (steps 0..num_warmup_steps), the LR increases linearly from
0 to the base LR.  After warmup, it decays following a cosine curve down to
``min_lr_ratio * base_lr``.

Compatible with PyTorch's ``LambdaLR`` scheduler so it plays nicely with
gradient scaler and checkpoint resume (``scheduler.state_dict()`` /
``scheduler.load_state_dict()``).
"""

import math
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
):
    """
    Create a cosine-decay LR scheduler with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose parameter groups get the schedule applied.
    num_warmup_steps : int
        Number of steps for the linear warmup phase.  During warmup the LR
        grows from 0 to the base LR set in the optimizer.
    num_training_steps : int
        Total number of training steps (warmup + decay).  At step
        ``num_training_steps`` the LR reaches ``min_lr_ratio * base_lr``.
    min_lr_ratio : float
        The LR floor as a fraction of the base LR.  Default 0.1 means the
        LR never drops below 10% of its peak value.  Set to 0.0 for full
        decay to zero.
    last_epoch : int
        Used by PyTorch for resume.  Leave at -1 for new training runs.

    Returns
    -------
    LambdaLR
        A scheduler that should be stepped once per optimizer step::

            optimizer.step()
            scheduler.step()
    """
    if num_warmup_steps < 0:
        raise ValueError(
            f"num_warmup_steps must be >= 0, got {num_warmup_steps}"
        )
    if num_training_steps <= 0:
        raise ValueError(
            f"num_training_steps must be > 0, got {num_training_steps}"
        )
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(
            f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}"
        )

    def lr_lambda(current_step: int) -> float:
        # Phase 1: linear warmup from 0 to 1.
        if current_step < num_warmup_steps:
            return current_step / max(num_warmup_steps, 1)

        # Phase 2: cosine decay from 1 to min_lr_ratio.
        if current_step >= num_training_steps:
            return min_lr_ratio

        progress = (current_step - num_warmup_steps) / max(
            num_training_steps - num_warmup_steps, 1
        )
        # Cosine annealing: 0.5 * (1 + cos(pi * progress)) decays from 1 to 0
        # We scale it to decay from 1 to min_lr_ratio instead.
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_converging_schedule(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    target_lr: float,
    last_epoch: int = -1,
):
    """
    Per-group converging LR schedule for Stage 1.

    Connector starts at high LR (100:1 ratio) and decays toward target_lr.
    Backbone groups start at low LR and rise toward the same target_lr.
    By end of training, all groups converge to target_lr (1:1 ratio).

    This lets the randomly-initialized connector catch up fast, then all
    components train at matched rates — ready for Stage 2+ flat LR.

    LR (connector)          LR (backbone)
    ^                       ^
    |  /\\                   |          ___________
    | /  \\___              |        /
    |/       \\___  target  |      /     target
    |             \\___/    |    /
    |                      |  /
    +-----------> step     +-----------> step

    Uses cosine interpolation for smooth convergence.
    PyTorch LambdaLR with per-group lambdas.
    """
    lambdas = []
    for group in optimizer.param_groups:
        base_lr = group["lr"]
        # Ratio: lambda * base_lr = effective_lr
        # At end: lambda * base_lr = target_lr → lambda = target_lr / base_lr
        final_ratio = target_lr / base_lr

        if base_lr > target_lr:
            # Connector: decay from 1.0 to final_ratio (< 1.0)
            def make_lambda(fr):
                def lr_lambda(step, _fr=fr):
                    if step < num_warmup_steps:
                        return step / max(num_warmup_steps, 1)
                    if step >= num_training_steps:
                        return _fr
                    progress = (step - num_warmup_steps) / max(
                        num_training_steps - num_warmup_steps, 1
                    )
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return _fr + (1.0 - _fr) * cosine
                return lr_lambda
            lambdas.append(make_lambda(final_ratio))
        else:
            # Backbone: warmup from 1.0, then rise to final_ratio (>= 1.0)
            def make_lambda(fr):
                def lr_lambda(step, _fr=fr):
                    if step < num_warmup_steps:
                        return step / max(num_warmup_steps, 1)
                    if step >= num_training_steps:
                        return _fr
                    progress = (step - num_warmup_steps) / max(
                        num_training_steps - num_warmup_steps, 1
                    )
                    # Inverse cosine: rises from 1.0 to final_ratio
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return _fr + (1.0 - _fr) * cosine  # same formula works both ways
                return lr_lambda
            lambdas.append(make_lambda(final_ratio))

    return LambdaLR(optimizer, lambdas, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    """
    Constant LR after linear warmup.  Ideal for scaling law runs where
    intermediate checkpoints should all be comparable (no schedule-dependent
    loss artifact).

    LR
    ^
    |     ___________________________
    |    /
    |   /
    |  /
    | /
    |/
    +------------------------------> step
    0   warmup     ...
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(num_warmup_steps, 1)
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
