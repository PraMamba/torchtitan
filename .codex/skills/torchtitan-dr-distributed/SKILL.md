---
name: torchtitan-dr-distributed
description: Use when changing TorchTitan distributed parallelism behavior, especially device-mesh construction, activation checkpointing, FSDP or TP rules, context or pipeline parallel flow, MoE expert routing, or distributed runtime utilities.
---

# TorchTitan Distributed

## Overview

`torchtitan/distributed` is the infrastructure layer that turns model-local modules into distributed training graphs. It owns device-mesh construction, context/tensor/pipeline/expert parallel wrappers, activation-checkpoint and `torch.compile` helpers, FSDP reshard policy resolution, DualPipeV overlap hooks, and runtime utilities for distributed initialization, determinism, process-group timeouts, and gradient clipping. The main design pattern is that model families define *what* modules should be parallelized, while this package defines *how* those modules are wrapped, sharded, scheduled, and synchronized.

## Public Surface

- `/home/scbjtfy/torchtitan/torchtitan/distributed/__init__.py`
  - Re-exports `ParallelDims`, the main mesh/topology descriptor used by runtime and model code.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/parallel_dims.py`
  - `ParallelDims`: validates degree combinations, creates world/dense/sparse/dataloading meshes, and exposes `get_mesh(...)` helpers plus enabled-dimension properties.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/activation_checkpoint.py`
  - `apply_ac(...)`: applies full, selective, or memory-budget activation checkpointing to transformer layers.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/compile.py`
  - `apply_compile_dense(...)`, `apply_compile_sparse(...)`, `apply_compile_dense_rl(...)`: compile dense, MoE, and RL-specific block structures.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/context_parallel.py`
  - `apply_cp_to_attention_module(...)`, `prepare_context_parallel_input(...)`, `cp_shard(...)`: CP wrapping and input/mask sharding.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/pipeline_parallel.py`
  - `pipeline_llm(...)`, `build_pipeline_schedule(...)`, `generate_llm_fqn_per_model_part(...)`, `pipeline_module_split(...)`: PP stage construction and schedule setup.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/expert_parallel.py`
  - `TensorParallel`, `ExpertParallel`, `ExpertTensorParallel`, `ReordererSequenceParallel`, `DeepEPExpertParallel`: MoE expert-weight sharding and token dispatch/combine strategies.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/dual_pipe_v.py`
  - `get_dual_pipe_v_flag(...)`, `DualPipeExpertParallel`, `overlap_callback(...)`: PP+EP overlap guardrails and synchronization hooks.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/tensor_parallel.py`
  - `NoParallel`, `ColwiseParallelWithGradPlacement`, `maybe_enable_async_tp(...)`: TP-adjacent wrapper styles and async-TP toggle.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/fsdp.py`
  - `get_fsdp_reshard_after_forward_policy(...)`: resolves the string policy used by model parallelization code.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/utils.py`
  - Distributed collectives wrappers, determinism seeding, fake/backend initialization, PG timeout updates, AMP gating, and PP-aware gradient clipping.

## Design Logic

- `ParallelDims` is the anchor object. Most other files assume mesh names like `pp`, `fsdp`, `tp`, `ep`, `efsdp`, and `loss` already exist and mean consistent things across dense and sparse subgraphs.
- Parallelism styles are deliberately decomposed by concern rather than fused into one mega-wrapper. For example, TP-specific layout conversion lives in `tensor_parallel.py`, EP dispatch/combine lives in `expert_parallel.py`, and PP stage carving lives in `pipeline_parallel.py`.
- Dense and sparse compile paths are separated in `/home/scbjtfy/torchtitan/torchtitan/distributed/compile.py` because MoE experts introduce graph breaks and wrapper ordering problems that do not exist for plain transformer blocks.
- Context parallelism is split into two layers: `apply_cp_to_attention_module(...)` wires module-level dispatching, while `prepare_context_parallel_input(...)` and `cp_shard(...)` transform runtime inputs, labels, positions, and masks into the sequence-sharded form the wrapped attention modules expect.
- EP implementations preserve state between dispatch and combine on purpose. `ExpertParallel` stores `input_splits`, `output_splits`, `input_shape`, and `permuted_indices`; `DeepEPExpertParallel` stores a backend-specific `_state`. That stateful design lets dispatch and combine stay paired without threading extra metadata through model code.
- DualPipeV overlap is opt-in and heavily guarded. `get_dual_pipe_v_flag(...)` rejects the unsupported `EP + DualPipeV + activation checkpointing` combination early, and `SyncHook` / `HookCoordinator` in `dual_pipe_v.py` enforce the forward/backward alternation needed for EP/PP overlap.
- Runtime utilities in `utils.py` intentionally know about mesh structure. Functions like `set_determinism(...)`, `set_pg_timeouts(...)`, and `clip_grad_norm_(...)` are not generic torch helpers; they encode TorchTitan’s assumptions about PP, SPMD meshes, EP-specific gradients, and fake backend modes.

## Core Data Structures

- `ParallelDims` in `/home/scbjtfy/torchtitan/torchtitan/distributed/parallel_dims.py`
  - Holds degree counts (`dp_replicate`, `dp_shard`, `cp`, `tp`, `pp`, `ep`, `etp`, `world_size`) and lazily-built meshes (`_world_mesh`, `_meshes`).
- `BaseExpertParallel` in `/home/scbjtfy/torchtitan/torchtitan/distributed/expert_parallel.py`
  - Abstract contract for MoE wrappers: `_partition_fn(...)`, `_token_dispatch(...)`, `_token_combine(...)`.
- `ExpertParallel` and `DeepEPExpertParallel`
  - Stateful EP implementations. `ExpertParallel` stores split/permutation metadata. `DeepEPExpertParallel` stores backend dispatch state and backend selection knobs.
- `NoParallel` and `ColwiseParallelWithGradPlacement` in `/home/scbjtfy/torchtitan/torchtitan/distributed/tensor_parallel.py`
  - TP-adjacent `ParallelStyle` subclasses that control how tensors become `DTensor`s and how gradient placements behave on backward.
- `HookCoordinator` and `SyncHook` in `/home/scbjtfy/torchtitan/torchtitan/distributed/dual_pipe_v.py`
  - Global coordination primitives that synchronize PP forward/backward threads around MoE communication.
- Pipeline schedule products from `/home/scbjtfy/torchtitan/torchtitan/distributed/pipeline_parallel.py`
  - `PipelineStage` instances, model chunks, and a `_PipelineSchedule` subclass chosen from schedule name or CSV.

## State Flow

1. Mesh bootstrap:
   `ParallelDims.build_mesh()` validates degree arithmetic, creates the world mesh via `init_device_mesh(...)`, unflattens it into dataloading/dense/sparse global meshes, and registers named one-dimensional views like `pp`, `batch`, `fsdp`, `tp`, and `ep`.
2. Runtime initialization:
   `init_distributed(...)` in `/home/scbjtfy/torchtitan/torchtitan/distributed/utils.py` decides whether to use fake mode, local tensor mode, torchcomms, or the normal backend path, then sets NCCL flight-recorder env vars and calls `torch.distributed.init_process_group(...)`.
3. Model wrapping:
   Model-specific `parallelize.py` files call into this package to apply TP/EP/CP/PP/FSDP/compile/AC in a defined order. For example, `pipeline_llm(...)` deep-copies the whole model per stage, trims each copy to the requested module FQNs, then runs the model family’s `parallelize_fn(...)` on each stage chunk before building a schedule.
4. Context and expert routing:
   `prepare_context_parallel_input(...)` creates `positions`, calls `cp_shard(...)`, and mutates `extra_kwargs` with sharded positions and masks. For MoE, EP wrappers intercept module input/output through `distribute_module(...)`; `ExpertParallel._token_dispatch(...)` performs all-to-all, `_permute(...)`, and padding metadata setup, then `_token_combine(...)` reverses that with `_unpermute(...)` and all-to-all on the way out.
5. Execution-time coordination:
   `build_pipeline_schedule(...)` computes microbatches and instantiates the right schedule. When DualPipeV overlap is enabled, `overlap_callback(...)` waits on pending P2P receives, launches backward in a companion thread, runs forward on the main stream, and uses `SyncHook` barriers to interleave EP communication with PP compute.
6. Runtime services:
   `set_determinism(...)` computes rank-specific seeds from requested mesh dimensions, `set_pg_timeouts(...)` barriers then updates all PG timeouts, and `clip_grad_norm_(...)` reduces across DTensor/PP/EP boundaries before clipping.

## Error Handling And Side Effects

- `ParallelDims._validate()` rejects invalid degree products and unsupported `etp` settings when `ep > 1`.
- `context_parallel.py` raises on unsupported attention types, invalid load-balancer names, and `ptrr` usage without a single `BlockMask`.
- `activation_checkpoint.py` raises on invalid AC mode and enables memory-budget visualization/output directories as a side effect when requested.
- `compile.py` raises if async TP is requested without model compilation enabled and mutates global `torch._dynamo` / `torch._inductor` config flags.
- `pipeline_parallel.py` raises for impossible stage distributions, missing `n_layers`, bad microbatch divisibility, missing CSV schedule files, and invalid schedule geometry for looped versus single-stage schedules.
- `dual_pipe_v.py` raises `NotImplementedError` for `EP + DualPipeV + AC`.
- `utils.py` mutates process-global environment variables, deterministic flags, process-group timeouts, RNG seeds, and flight-recorder settings.

## Common Modification Scenarios

- Change how meshes are named or combined:
  Start in `/home/scbjtfy/torchtitan/torchtitan/distributed/parallel_dims.py`. `build_mesh()` and `get_optional_mesh(...)` are the central places that define which mesh names exist and which global meshes can answer multi-dimension lookups.
- Add a new activation-checkpoint save/recompute rule:
  Edit `_get_save_ops()` or `_get_custom_policy()` in `/home/scbjtfy/torchtitan/torchtitan/distributed/activation_checkpoint.py`. That is where expensive compute ops, comm ops, forced-recompute linear shapes, and CUDA-to-CPU transfers are classified.
- Modify PP stage carving or schedule defaults:
  Edit `generate_llm_fqn_per_model_part(...)`, `pipeline_module_split(...)`, and `build_pipeline_schedule(...)` in `/home/scbjtfy/torchtitan/torchtitan/distributed/pipeline_parallel.py`. These functions together define stage contents, virtual-stage arithmetic, and schedule instantiation.
- Change EP token movement or add a new backend:
  Work in `/home/scbjtfy/torchtitan/torchtitan/distributed/expert_parallel.py` and, if overlap matters, `/home/scbjtfy/torchtitan/torchtitan/distributed/dual_pipe_v.py`. New backends need partition logic, dispatch/combine state handling, and possibly SAC-visible custom op imports.
- Adjust distributed init, fake modes, or gradient clipping semantics:
  Update `/home/scbjtfy/torchtitan/torchtitan/distributed/utils.py`. That file owns backend selection, fake/local-tensor setup, timeout behavior, seed policy, and PP/EP-aware norm reduction.

## File Map

- `/home/scbjtfy/torchtitan/torchtitan/distributed/parallel_dims.py`: mesh algebra and accessors.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/activation_checkpoint.py`: SAC/full/memory-budget AC application.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/compile.py`: dense/sparse/RL compile wrappers.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/context_parallel.py`: CP attention wrapping and input sharding.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/expert_parallel.py`: MoE EP/ETP/DeepEP token routing.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/dual_pipe_v.py`: EP/PP overlap synchronization.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/pipeline_parallel.py`: stage splitting and schedule building.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/tensor_parallel.py`: custom TP-adjacent styles.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/fsdp.py`: FSDP reshard policy resolver.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/utils.py`: distributed runtime and reduction helpers.

## See Also

- `reference.md` in this directory for class/function inventory, key parameters, and file-by-file responsibilities.
