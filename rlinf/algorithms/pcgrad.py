from __future__ import annotations

from typing import List, Optional

import torch


def _dot_and_norm2(
    grads_i: List[Optional[torch.Tensor]],
    grads_j: List[Optional[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute <g_i, g_j> and ||g_j||^2 over a list of per-parameter gradients."""
    device = None
    dot = None
    norm2 = None

    for gi, gj in zip(grads_i, grads_j):
        if gi is None or gj is None:
            continue
        if device is None:
            device = gi.device
        gi_flat = gi.view(-1)
        gj_flat = gj.view(-1)
        if dot is None:
            dot = torch.dot(gi_flat, gj_flat)
            norm2 = torch.dot(gj_flat, gj_flat)
        else:
            dot = dot + torch.dot(gi_flat, gj_flat)
            norm2 = norm2 + torch.dot(gj_flat, gj_flat)

    if dot is None:
        if device is None:
            device = torch.device("cpu")
        dot = torch.tensor(0.0, device=device)
        norm2 = torch.tensor(0.0, device=device)
    return dot, norm2


def apply_pcgrad(
    per_task_grads: List[List[Optional[torch.Tensor]]],
) -> List[Optional[torch.Tensor]]:
    """Apply PCGrad to a list of per-task gradients.

    Uses incremental projection + aggregation: only one task's working gradient
    copy is held at a time instead of cloning all T tasks upfront, reducing
    peak memory by ~(T-1)/T of the gradient copy overhead.

    Args:
        per_task_grads: list of length T (num_tasks). Each element is a list
            of gradients for each parameter (can contain None if a parameter
            does not receive gradient for that task).

    Returns:
        List[Optional[torch.Tensor]]: projected gradients for each parameter
        (None where no task had a gradient, e.g. FSDP shard). Caller should
        skip assigning for None entries.
    """
    if not per_task_grads:
        return []

    num_tasks = len(per_task_grads)
    num_params = len(per_task_grads[0])

    # Random permutation for projection order (same order for all tasks).
    perm = torch.randperm(num_tasks).tolist()

    # Incremental aggregation: process one task at a time instead of cloning
    # all T tasks upfront. Saves ~(T-1)/T of gradient copy memory.
    agg_grads: List[Optional[torch.Tensor]] = [None] * num_params
    agg_counts: List[int] = [0] * num_params

    for i in perm:
        task_grads = per_task_grads[i]
        # Single working copy for task i (instead of cloning all T tasks).
        projected_i = [
            g.clone() if g is not None else None for g in task_grads
        ]

        # PCGrad: project g_i against other tasks' gradients when conflicting.
        # Use original per_task_grads[j] for j != i (read-only).
        # Random permutation of other tasks for projection order (as in original).
        j_perm = torch.randperm(num_tasks).tolist()
        for j in j_perm:
            if j == i:
                continue
            gj = per_task_grads[j]
            dot, norm2 = _dot_and_norm2(projected_i, gj)
            if norm2.item() == 0.0:
                continue
            if dot.item() < 0.0:
                coeff = dot / (norm2 + 1e-12)
                for k, (gik, gjk) in enumerate(zip(projected_i, gj)):
                    if gik is None or gjk is None:
                        continue
                    projected_i[k] = gik - coeff * gjk

        # Incremental aggregation: add projected_i to accumulator.
        for p_idx in range(num_params):
            g = projected_i[p_idx]
            if g is None:
                continue
            if agg_grads[p_idx] is None:
                agg_grads[p_idx] = g.clone()
            else:
                agg_grads[p_idx].add_(g)
            agg_counts[p_idx] += 1

    # Normalize by count (mean).
    final_grads: List[Optional[torch.Tensor]] = []
    for p_idx in range(num_params):
        agg = agg_grads[p_idx]
        count = agg_counts[p_idx]
        if agg is None or count == 0:
            # No task had a gradient for this parameter (e.g. FSDP shard).
            final_grads.append(None)
        else:
            agg.div_(float(count))
            final_grads.append(agg)
    return final_grads

