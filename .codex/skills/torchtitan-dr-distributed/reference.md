# Distributed Reference

## File Index

- `/home/scbjtfy/torchtitan/torchtitan/distributed/__init__.py`
  - Re-exports `ParallelDims`.
- `/home/scbjtfy/torchtitan/torchtitan/distributed/activation_checkpoint.py`
  - `_get_save_ops()`
  - `_apply_op_sac(...)`
  - `_apply_full_ac(...)`
  - `_apply_ac_to_transformer_block(...)`
  - `apply_ac(...)`
- `/home/scbjtfy/torchtitan/torchtitan/distributed/compile.py`
  - `apply_compile_dense(...)`
  - `apply_compile_sparse(...)`
  - `apply_compile_dense_rl(...)`
- `/home/scbjtfy/torchtitan/torchtitan/distributed/context_parallel.py`
  - `apply_cp_to_attention_module(...)`
  - `prepare_context_parallel_input(...)`
  - `cp_shard(...)`
- `/home/scbjtfy/torchtitan/torchtitan/distributed/dual_pipe_v.py`
  - `get_dual_pipe_v_flag(...)`
  - `DualPipeExpertParallel`
  - `HookCoordinator`
  - `SyncHook`
  - `_count_moe_modules(...)`
  - `overlap_callback(...)`
- `/home/scbjtfy/torchtitan/torchtitan/distributed/expert_parallel.py`
  - `BaseExpertParallel`
  - `TensorParallel`
  - `ExpertParallel`
  - `ExpertTensorParallel`
  - `ReordererSequenceParallel`
  - `DeepEPExpertParallel`
- `/home/scbjtfy/torchtitan/torchtitan/distributed/fsdp.py`
  - `get_fsdp_reshard_after_forward_policy(...)`
- `/home/scbjtfy/torchtitan/torchtitan/distributed/parallel_dims.py`
  - `ParallelDims`
- `/home/scbjtfy/torchtitan/torchtitan/distributed/pipeline_parallel.py`
  - `pipeline_llm(...)`
  - `build_pipeline_schedule(...)`
  - `generate_llm_fqn_per_model_part(...)`
  - `pipeline_module_split(...)`
- `/home/scbjtfy/torchtitan/torchtitan/distributed/tensor_parallel.py`
  - `NoParallel`
  - `ColwiseParallelWithGradPlacement`
  - `maybe_enable_async_tp(...)`
- `/home/scbjtfy/torchtitan/torchtitan/distributed/utils.py`
  - `_dist_reduce(...)`
  - `dist_max(...)`, `dist_sum(...)`, `dist_mean(...)`
  - `set_determinism(...)`
  - `get_train_context(...)`
  - `maybe_enable_amp(...)`
  - `init_fake_mode(...)`
  - `init_distributed(...)`
  - `set_pg_timeouts(...)`
  - `clip_grad_norm_(...)`
  - `_clip_grad_norm_with_ep(...)`

## Key Types And Responsibilities

### `ParallelDims`

- Fields:
  - Degree fields: `dp_replicate`, `dp_shard`, `cp`, `tp`, `pp`, `ep`, `etp`, `world_size`
  - Cached mesh state: `_meshes`, `_world_mesh`
- Core methods:
  - `_validate()`: checks degree arithmetic and derives `dp_shard` when `-1`
  - `build_mesh()`: creates world mesh plus dataloading/dense/sparse unflattened views
  - `_validate_meshes()`: asserts runtime mesh sizes match expectations
  - `get_optional_mesh(...)` / `get_mesh(...)`: lookup by single dim or compatible dim set
  - `get_all_one_dimensional_meshes()`: enumerates active 1-D meshes for timeout updates and PG access
- Important properties:
  - `fsdp_enabled`, `tp_enabled`, `pp_enabled`, `ep_enabled`, `etp_enabled`
  - `fsdp_gradient_divide_factor`
  - `non_data_parallel_size`
  - `seq_len_divisor`

### Activation Checkpointing

- `_get_save_ops()` builds the op set for selective AC from:
  - default compute-intensive ATen ops
  - SDPA/FlexAttention/custom attention ops
  - communication ops like reduce-scatter/all-to-all
  - optional DeepEP/HybridEP custom ops when registered
- `_apply_op_sac(...)`:
  - creates a per-op policy
  - supports force-recompute of selected `nn.Linear` RHS shapes via `per_op_sac_force_recompute_mm_shapes_by_fqns`
  - treats CUDA-to-CPU copies as `MUST_SAVE`
- `apply_ac(...)`:
  - disables Dynamo LRU cache to avoid SAC+PP+Flex recompilation mismatch
  - either configures memory-budget mode globally or wraps each `layers.<id>` transformer block

### Compile Helpers

- `apply_compile_dense(...)`
  - compiles each transformer block as a whole
  - sets `skip_fwd_side_effects_in_bwd_under_checkpoint`
- `apply_compile_sparse(...)`
  - compiles MoE submodules individually to avoid FSDP(GroupedExperts) graph breaks
  - optionally patches grouped expert kernel wrapper for dynamic token counts when EP is enabled
- `apply_compile_dense_rl(...)`
  - uses in-place `Module.compile()` to avoid `_orig_mod` renaming that would break trainer/vLLM weight-name matching

### Context Parallel

- `apply_cp_to_attention_module(...)`
  - constructs `_ContextParallel` plans for `"flex"` or `"sdpa"`
  - enables the CP dispatcher for SDPA
- `prepare_context_parallel_input(...)`
  - synthesizes `positions`
  - calls `cp_shard(...)`
  - mutates `extra_kwargs` with sharded `positions` and `attention_masks`
- `cp_shard(...)`
  - supports optional load balancers `"headtail"` and `"ptrr"`
  - shards all input tensors on a configurable sequence dimension
  - shards only the query sequence axis of `BlockMask` data

### Expert Parallel

- `TensorParallel`
  - expert-local TP for grouped experts only
  - shards `w1`/`w3` on output dimension and `w2` on input/output as row-wise
- `ExpertParallel`
  - shards experts on expert dimension
  - dispatch path:
    - all-to-all exchange of `num_tokens_per_expert`
    - CPU materialization of split sizes
    - all-to-all on routed tokens
    - `_permute(...)` to local-expert layout plus padding alignment
  - combine path:
    - `_unpermute(...)`
    - reverse all-to-all
- `ExpertTensorParallel`
  - same dispatch/combine logic, but partitioning uses a 2-D `[ep, tp]` mesh
  - dispatch/combine explicitly use `device_mesh["ep"]`
- `ReordererSequenceParallel`
  - sequence-parallel helper for `ETP=1`
  - slices routing metadata along token dimension and restores global token offsets on output
- `DeepEPExpertParallel`
  - imports backend modules eagerly so custom ops exist before SAC classification
  - dispatches via `deepep.dispatch_tokens(...)` or `hybridep.dispatch_tokens(...)`
  - stores backend `_state` and consumes it in combine

### DualPipeV Overlap

- `get_dual_pipe_v_flag(...)`
  - returns `False` unless both `ep` and `pp` are enabled and schedule is `dualpipev`
  - rejects activation checkpointing with DualPipeV overlap
- `DualPipeExpertParallel`
  - wraps another `BaseExpertParallel`
  - inserts autograd hooks around dispatch/combine boundaries in order `A -> dispatch -> B -> module -> C -> combine -> D`
- `HookCoordinator`
  - owns a `threading.Barrier(2)` and cycle counter
- `overlap_callback(...)`
  - called by runtime pipeline schedules on overlap actions
  - waits on forward/backward recv ops
  - counts MoE modules on both stages to limit coordination cycles
  - runs backward in a thread and forward on the main stream

### Pipeline Parallel

- `pipeline_llm(...)`
  - computes virtual-stage count from schedule type and optional `pipeline_parallel_layers_per_stage`
  - auto-generates stage module FQNs when config does not provide them
  - deep-copies and trims stage models through `pipeline_module_split(...)`
  - applies model-family `parallelize_fn(...)` to each stage chunk
  - returns `pp_schedule`, `model_parts`, `has_first_stage`, `has_last_stage`
- `build_pipeline_schedule(...)`
  - resolves schedule from CSV or schedule name
  - validates `local_batch_size % pipeline_parallel_microbatch_size == 0`
  - warns when microbatch count is lower than total stage count
  - registers `overlap_callback` for `ScheduleDualPipeV` when EP overlap is enabled
- `generate_llm_fqn_per_model_part(...)`
  - distributes effective layers across stages
  - treats embeddings and output stack as weighted pseudo-layers
- `pipeline_module_split(...)`
  - deep-copies the whole model per stage
  - deletes or empties modules not listed in the stage FQN set
  - creates `PipelineStage` objects and supports looped or V-style local-stage layouts

### Tensor Parallel Helpers

- `NoParallel`
  - keeps parameters on the TP mesh as `DTensor`s without sharding compute
  - useful for replicated router/gate paths that still need mesh-consistent parameter placement
- `ColwiseParallelWithGradPlacement`
  - extends `ColwiseParallel`
  - passes `grad_placements` into `DTensor.from_local(...)` to control backward placement explicitly
- `maybe_enable_async_tp(...)`
  - requires both compile enabled and `"model"` in compile components
  - toggles `torch._inductor.config._micro_pipeline_tp`

### Runtime Utilities

- `_dist_reduce(...)`
  - materializes `DTensor` to full tensor when needed
  - can reduce across an extra process group before the mesh reduction
- `set_determinism(...)`
  - enables deterministic algorithms and CuBLAS/CuDNN settings
  - seeds PP and SPMD dimensions differently by folding local mesh ranks into a shared base seed
- `maybe_enable_amp(...)`
  - disables AMP when FSDP/DDP handles precision internally or when TP/PP are used without those wrappers
- `init_fake_mode(...)`
  - initializes fake PG and optional `LocalTensorMode`
- `init_distributed(...)`
  - supports `fake_backend`, `local_tensor`, normal distributed mode, and `torchcomms`
  - mutates flight-recorder environment variables and timeout-related NCCL env vars
- `set_pg_timeouts(...)`
  - barriers before updating all mesh PGs plus the default group
- `clip_grad_norm_(...)`
  - reduces DTensor norms to local tensors
  - optionally all-reduces norms across PP stages
- `_clip_grad_norm_with_ep(...)`
  - separates parameters by whether their mesh dim names include `"ep"`
  - combines EP and non-EP norms manually before clipping both parameter groups

## Modification Checklist

- If a change affects mesh naming or shape assumptions, update `ParallelDims` first and then audit every `get_mesh(...)` caller.
- If a change adds custom comm ops that SAC should preserve, add them in `_get_save_ops()` before relying on activation checkpointing.
- If a change adds a new PP schedule or overlap mode, update both `build_pipeline_schedule(...)` and `pipeline_module_split(...)`; stage construction and schedule execution assumptions are coupled.
- If a change alters EP token formats, update both dispatch and combine methods plus any DualPipeV wrapper that assumes hook boundaries around them.
- If a change alters distributed initialization behavior, verify both fake/local-tensor modes and normal PG setup paths in `init_distributed(...)`.
