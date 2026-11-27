"""Replicated models, globally summed gradients, unchanged chronological windows.

Every rank advances the same detached memory history. Only the owner of a
microbatch evaluates the action loss and backpropagates. This partitions work
without multiplying the two global windows or breaking episode chronology.
"""
import os
import torch
import torch.distributed as dist


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def setup():
    # Required for bitwise replay after rebuilding CUDA modules. CUDA attention
    # backward may otherwise use atomic reductions even with identical RNG.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")


def barrier():
    if dist.is_initialized():
        dist.barrier()


def gather(value):
    if world_size() == 1:
        return [value]
    values = [None] * world_size()
    dist.all_gather_object(values, value)
    return values


def sum_tensor(value):
    if world_size() > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def sum_gradients(parameters, bucket_bytes=32 * 1024 * 1024):
    """SUM because local losses already use the global action/state denominators.

    Preserve globally unused parameters as None: AdamW must not decay them.
    Equal collectives on idle/tail ranks prevent short-window hangs.
    """
    if world_size() == 1:
        return
    parameters = list(parameters)
    device = parameters[0].device
    used = torch.tensor([p.grad is not None for p in parameters], device=device, dtype=torch.int32)
    sum_tensor(used)
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket])
        sum_tensor(flat)
        offset = 0
        for p in bucket:
            n = p.numel()
            if p.grad is None:
                p.grad = flat[offset:offset+n].view_as(p).clone()
            else:
                p.grad.copy_(flat[offset:offset+n].view_as(p))
            offset += n

    for p, active in zip(parameters, used.tolist()):
        if not active:
            continue
        n = p.numel() * p.element_size()
        if bucket and size + n > bucket_bytes:
            flush(); bucket, size = [], 0
        bucket.append(p); size += n
    flush()
