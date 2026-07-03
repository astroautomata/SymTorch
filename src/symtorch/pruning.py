"""Progressive dimensionality-reduction schedules and importance ranking."""

import math
from typing import Dict

import torch


def make_pruning_schedule(
    initial_dim: int,
    target_dim: int,
    total_steps: int,
    decay_rate: str = "cosine",
    end_step_frac: float = 0.5,
) -> Dict[int, int]:
    """
    Create step-based pruning schedule.

    Args:
        initial_dim (int): Initial number of dimensions
        target_dim (int): Target number of dimensions after pruning
        total_steps (int): Total number of training steps
        decay_rate (str): Type of decay schedule ('exp', 'linear', 'cosine')
        end_step_frac (float): Fraction of steps to complete pruning by

    Returns:
        dict: Mapping from step number to target dimensions
    """

    prune_end_step = int(end_step_frac * total_steps)
    prune_steps = prune_end_step

    dims_to_prune = initial_dim - target_dim
    schedule_dict = {}

    # Different pruning schedules
    # Exponential decay
    if decay_rate == "exp":
        decay_rate_val = 3.0
        max_decay = 1 - math.exp(-decay_rate_val)

        for step in range(prune_end_step):
            progress = step / prune_steps
            raw_decay = 1 - math.exp(-decay_rate_val * progress)
            decay_factor = raw_decay / max_decay

            dims_pruned = math.ceil(dims_to_prune * decay_factor)
            target_dims = max(initial_dim - dims_pruned, target_dim)
            schedule_dict[step] = target_dims

    # Linear decay
    elif decay_rate == "linear":
        for step in range(prune_end_step):
            progress = step / prune_steps
            dims_pruned = math.ceil(dims_to_prune * progress)
            target_dims = max(initial_dim - dims_pruned, target_dim)
            schedule_dict[step] = target_dims

    # Cosine decay
    elif decay_rate == "cosine":
        for step in range(prune_end_step):
            progress = step / prune_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            dims_pruned = math.ceil(dims_to_prune * (1 - cosine_decay))
            target_dims = max(initial_dim - dims_pruned, target_dim)
            schedule_dict[step] = target_dims

    # Keep target_dim for the last part of training
    for step in range(prune_end_step, total_steps):
        schedule_dict[step] = target_dim

    return schedule_dict


def rank_dimensions(output_array: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Indices of the n_keep most important dimensions by activation std."""
    importance = output_array.std(dim=0)
    return torch.argsort(importance, descending=True)[:n_keep]
