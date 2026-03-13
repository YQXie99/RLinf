from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


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


def _param_shapes_from_params(params: Sequence[torch.Tensor]) -> List[Tuple[int, ...]]:
    """Return list of shapes for each parameter."""
    return [tuple(p.shape) for p in params]


def flatten_grad_list(
    grad_list: List[Optional[torch.Tensor]],
    param_shapes: List[Tuple[int, ...]],
    device: torch.device,
) -> torch.Tensor:
    """Flatten a per-parameter gradient list into a single contiguous tensor.

    None entries are replaced by zeros of the corresponding param shape.
    """
    parts = []
    for g, shape in zip(grad_list, param_shapes):
        if g is not None:
            parts.append(g.reshape(-1).to(device=device, dtype=g.dtype))
        else:
            parts.append(torch.zeros(int(torch.Size(shape).numel()), device=device))
    return torch.cat(parts) if parts else torch.tensor([], device=device)


def unflatten_grad_list(
    flat: torch.Tensor,
    param_shapes: List[Tuple[int, ...]],
    device: torch.device,
) -> List[Optional[torch.Tensor]]:
    """Unflatten a contiguous tensor back into a per-parameter gradient list."""
    out: List[Optional[torch.Tensor]] = []
    offset = 0
    for shape in param_shapes:
        numel = int(torch.Size(shape).numel())
        if numel == 0:
            out.append(None)
            continue
        seg = flat[offset : offset + numel].view(shape)
        offset += numel
        out.append(seg)
    return out


def aggregate_per_task_grads_across_ranks(
    per_task_grads_local: List[List[Optional[torch.Tensor]]],
    local_task_ids: List[int],
    params: Sequence[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
) -> List[List[Optional[torch.Tensor]]]:
    """Aggregate per-task gradients from all ranks so every rank has full per_task_grads.

    Each rank has computed gradients only for tasks present in its batch. This
    function allgathers task ids and allreduces gradients per task, so that
    all ranks end up with the same aggregated per_task_grads (list ordered by
    global sorted task id).

    Args:
        per_task_grads_local: list of length = len(local_task_ids). Each element
            is a list of gradients per parameter for that task.
        local_task_ids: task ids corresponding to per_task_grads_local.
        params: model parameters (same order on all ranks; used for shapes).
        group: process group for collective. Default world.

    Returns:
        List of length T_global (number of unique tasks across ranks). Each
        element is a list of gradient tensors per parameter (aggregated sum).
    """
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        # Return in sorted order by task id for consistency with multi-rank path.
        sorted_pairs = sorted(zip(local_task_ids, per_task_grads_local))
        return [g for _, g in sorted_pairs]

    device = params[0].device
    param_shapes = _param_shapes_from_params(params)
    total_numel = sum(int(torch.Size(s).numel()) for s in param_shapes)
    dtype = params[0].dtype

    # Allgather task ids so we have global_task_list (sorted unique).
    local_task_ids_list: List[Optional[List[int]]] = [None] * world_size
    dist.all_gather_object(local_task_ids_list, local_task_ids, group=group)
    global_task_list = sorted(
        set(tid for lst in local_task_ids_list if lst for tid in lst)
    )

    if not global_task_list:
        return []

    local_grad_by_task = dict(zip(local_task_ids, per_task_grads_local))

    # For each global task id, contribute flattened grad (or zeros) and allreduce.
    per_task_grads_aggregated: List[List[Optional[torch.Tensor]]] = []
    for tid in global_task_list:
        grad_list = local_grad_by_task.get(tid)
        if grad_list is None:
            flat = torch.zeros(total_numel, device=device, dtype=dtype)
        else:
            flat = flatten_grad_list(grad_list, param_shapes, device)
            if flat.numel() != total_numel:
                flat = torch.zeros(total_numel, device=device, dtype=dtype)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        per_task_grads_aggregated.append(
            unflatten_grad_list(flat, param_shapes, device)
        )
    return per_task_grads_aggregated


def _compute_projected_grad_for_task_index(
    task_index: int,
    per_task_grads: List[List[Optional[torch.Tensor]]],
    num_params: int,
) -> List[Optional[torch.Tensor]]:
    """Compute PCGrad projected gradient for a single task index (in perm order)."""
    projected_i = [
        g.clone() if g is not None else None for g in per_task_grads[task_index]
    ]
    num_tasks = len(per_task_grads)
    j_perm = torch.randperm(num_tasks).tolist()
    for j in j_perm:
        if j == task_index:
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
    return projected_i


def apply_pcgrad_distributed(
    per_task_grads: List[List[Optional[torch.Tensor]]],
    params: Sequence[torch.Tensor],
    rank: int,
    world_size: int,
    seed: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> List[Optional[torch.Tensor]]:
    """Apply PCGrad with task-wise work split across ranks.

    Each rank computes projected gradients only for task indices
    i where i % world_size == rank, then allreduces the summed projected
    gradient so all ranks get the same final gradient. This reduces per-rank
    memory and compute for the projection step.

    Args:
        per_task_grads: list of length T (num_tasks). Each element is a list
            of gradients per parameter. Must be the same on all ranks (e.g.
            after aggregate_per_task_grads_across_ranks).
        params: model parameters (for shapes and device).
        rank: this rank's index.
        world_size: total number of ranks.
        seed: random seed for permutation (same on all ranks for determinism).
        group: process group. Default world.
    """
    if not per_task_grads:
        return []

    num_tasks = len(per_task_grads)
    num_params = len(per_task_grads[0])
    param_shapes = _param_shapes_from_params(params)
    device = params[0].device

    # Deterministic permutation so all ranks agree on task order.
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    perm = torch.randperm(num_tasks, generator=g).tolist()

    # This rank handles task indices i where i % world_size == rank.
    my_task_indices = [i for i in range(num_tasks) if i % world_size == rank]

    agg_grads: List[Optional[torch.Tensor]] = [None] * num_params
    for i in perm:
        if i not in my_task_indices:
            continue
        projected_i = _compute_projected_grad_for_task_index(
            i, per_task_grads, num_params
        )
        for p_idx in range(num_params):
            g = projected_i[p_idx]
            if g is None:
                continue
            if agg_grads[p_idx] is None:
                agg_grads[p_idx] = g.clone()
            else:
                agg_grads[p_idx].add_(g)

    # Flatten, allreduce (sum of projected grads across ranks), then normalize by num_tasks.
    if agg_grads[0] is not None:
        flat = flatten_grad_list(agg_grads, param_shapes, device)
    else:
        flat = torch.zeros(
            sum(int(torch.Size(s).numel()) for s in param_shapes),
            device=device,
            dtype=params[0].dtype,
        )
    total_numel = sum(int(torch.Size(s).numel()) for s in param_shapes)
    if flat.numel() != total_numel:
        flat = torch.zeros(total_numel, device=device, dtype=params[0].dtype)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    flat.div_(float(num_tasks))
    return unflatten_grad_list(flat, param_shapes, device)

