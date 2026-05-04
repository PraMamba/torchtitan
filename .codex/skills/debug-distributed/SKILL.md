---
name: debug-distributed
description: Guide for debugging distributed training issues in torchtitan. Use when encountering hangs, wrong results, OOM, or communication errors.
---


## Codex Compatibility Note

Migrated from `.claude/skills/debug-distributed/SKILL.md`. This skill preserves the original Claude-oriented workflow
for information fidelity, with the following Codex adaptation rules:

- Treat Claude-specific tool names or UI mechanisms as historical references.
- Use available Codex shell/MCP/subagent tools that satisfy the same workflow step.
- Keep destructive or external side-effect steps gated by the current Codex
  permissions and project policy.
- If a step references Claude-only cache paths or invocation syntax, treat that
  as source context unless the user explicitly asks to reproduce it.

---


# Debug Distributed Training

Debugging guide for distributed training issues in torchtitan (FSDP2, TP, PP, CP, EP).

## When to Use

- Training hangs or deadlocks
- Results differ across ranks or are numerically wrong
- OOM errors in distributed settings
- NCCL/communication errors or device mesh issues

## Debugging Principles

### Minimal Reproduction

**Always create minimal reproduction:**

1. Remove unrelated model components
2. Use small tensor sizes
3. Reduce world_size to minimum (e.g., 2 GPUs)
4. Remove torch.compile if possible
5. Disable activation checkpointing

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()

# Reproduce the exact operation that fails
tensor = torch.ones(10).cuda()
dist.all_reduce(tensor)
print(f"Rank {rank}: {tensor}")
```

## Step-by-Step Debugging

### 1. Hang Debugging (Deadlocks, Synchronization)

**Environment Variables:**
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_LOGS="+dynamo,recompiles"
export TORCHDYNAMO_VERBOSE=1
```

**Dump Call Stack (for hung processes):**
```bash
ps aux | grep python
py-spy dump --pid <PID>
py-spy record -o profile.svg --pid <PID> --duration 30
```

**Common Causes:**
1. Mismatched collectives — one rank calls all_reduce, another doesn't
2. Wrong process group — using wrong group for collective
3. Tensor shape mismatch across ranks
4. PP schedule deadlock — check stage assignment

**Debug Steps:**
```python
# Verify group membership
mesh = parallel_dims.get_mesh("tp")
group = mesh.get_group()
print(f"Rank {dist.get_rank()}: group size = {dist.get_world_size(group)}")

# Verify tensor shapes across ranks
print(f"Rank {dist.get_rank()}: tensor.shape = {tensor.shape}")
dist.barrier()
```

### 2. Wrong Results (Gradient, Reduction Issues)

**Check DTensor Placements:**
```python
from torch.distributed.tensor import DTensor
for name, param in model.named_parameters():
    if isinstance(param, DTensor):
        print(f"{name}: placements={param.placements}, mesh={param.device_mesh}")
```

**Verify Gradient Reduction:**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_sum = param.grad.sum().item()
        print(f"Rank {dist.get_rank()}: {name} grad_sum={grad_sum}")
```

**Check Gradient Clipping:**
```python
# torchtitan uses custom clip_grad_norm_ that handles DTensor + PP + EP
# Verify EP/non-EP parameter separation
from torchtitan.distributed.utils import clip_grad_norm_
```

**Numerical Validation:**
```bash
# Run with deterministic mode
torchrun --nproc_per_node=4 -m torchtitan.train \
  --module llama3 --config llama3_debugmodel \
  --debug.seed=42 --debug.deterministic
```

### 3. OOM Issues

**Check Memory Usage:**
```python
print(f"Rank {dist.get_rank()}: "
      f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB, "
      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
```

**Check FSDP Coverage:**
```python
for name, param in model.named_parameters():
    is_dtensor = isinstance(param, DTensor)
    print(f"{name}: is_dtensor={is_dtensor}, shape={param.shape}")
```

**Common Fixes:**
- Enable activation checkpointing (`ac_config.mode="selective"`)
- Increase FSDP sharding (reduce `dp_replicate`, increase `dp_shard`)
- Check `reshard_after_forward` policy
- For PP: default disables resharding to avoid per-microbatch all-gathers

### 4. Communication Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `NCCL WARN Cuda failure` | GPU communication | Check NCCL version, GPU topology |
| `RuntimeError: Timed out` | Rank sync issue | Increase timeout, check code paths |
| `Invalid device mesh` | Mesh config | Verify `dp * tp * pp == world_size` |
| `ProcessGroup not initialized` | Missing init | Check `dist.init_process_group()` |

## Key Files Reference

| Component | File |
|-----------|------|
| ParallelDims | `torchtitan/distributed/parallel_dims.py` |
| Tensor Parallel | `torchtitan/distributed/tensor_parallel.py` |
| Pipeline Parallel | `torchtitan/distributed/pipeline_parallel.py` |
| Context Parallel | `torchtitan/distributed/context_parallel.py` |
| Expert Parallel | `torchtitan/distributed/expert_parallel.py` |
| Gradient Clipping | `torchtitan/distributed/utils.py` |
| Activation Checkpoint | `torchtitan/distributed/activation_checkpoint.py` |

## Debugging Tools

| Tool | Purpose |
|------|---------|
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | Detailed distributed logging |
| `NCCL_DEBUG=INFO` | NCCL communication logging |
| `CUDA_LAUNCH_BLOCKING=1` | Synchronous CUDA (slow, for debugging) |
| `py-spy dump --pid <PID>` | Call stack of hung process |
| `torch.cuda.memory_summary()` | Detailed memory breakdown |
| `scripts/loss_compare.py` | Compare loss curves across runs |
