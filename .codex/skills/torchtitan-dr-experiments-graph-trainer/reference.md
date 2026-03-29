# GraphTrainer Reference

## File Index
- `torchtitan/experiments/graph_trainer/README.md`
  - User-facing overview, supported compile modes, and feature matrix.
- `torchtitan/experiments/graph_trainer/configs.py`
  - `GraphTrainerCompileConfig`, `GraphTrainerConfig`, `to_graph_trainer_config(...)`.
- `torchtitan/experiments/graph_trainer/trainer.py`
  - `FwdBwdStepModule`, `GraphTrainer`, `_make_fx_forward_backward_step(...)`.
- `torchtitan/experiments/graph_trainer/compile.py`
  - `_apply_jit_compile(...)`, `_apply_aot_compile(...)`, `_apply_aot_compile_load(...)`, `apply_compile(...)`.
- `torchtitan/experiments/graph_trainer/graph_utils.py`
  - `export_joint(...)`, `joint_graph_builder(...)`, `CompiledModule`, pass resolution helpers.
- `torchtitan/experiments/graph_trainer/make_fx_tracer.py`
  - tensor-subclass flattening, patched backward tracing, traced graph execution.
- `torchtitan/experiments/graph_trainer/passes.py`
  - all named compiler and joint passes.
- `torchtitan/experiments/graph_trainer/simple_fsdp.py`
  - graph-friendly DP/FSDP/HSDP wrapper.
- `torchtitan/experiments/graph_trainer/common_utils.py`
  - annotations, AC-region helpers, extra FSDP process-group creation.
- `torchtitan/experiments/graph_trainer/cudagraph.py`
  - shared CUDA graph manager and wrapper.
- `torchtitan/experiments/graph_trainer/reshard_after_forward.py`
  - joint-graph recompute/save annotations for SimpleFSDP all-gathers.
- `torchtitan/experiments/graph_trainer/storage.py`
  - abstract storage adapter plus local disk implementation.
- `torchtitan/experiments/graph_trainer/precompile.py`
  - artifact serialization, fingerprinting, and lazy load wrapper.
- `torchtitan/experiments/graph_trainer/jit_backend.py`
  - JIT backend assembly for auto and manual bucketing.
- `torchtitan/experiments/graph_trainer/llama3/*`
  - graph-trainer Llama3 model wrapper, registry, configs, and parallelization hooks.
- `torchtitan/experiments/graph_trainer/deepseek_v3/*`
  - graph-trainer DeepSeek-V3 model wrapper, registry, configs, and parallelization hooks.
- `torchtitan/experiments/graph_trainer/tests/*`
  - integration matrix plus targeted numerics, pass, precompile, and tracing tests.

## Config And Registry Details
- `GraphTrainerCompileConfig.mode`
  - `jit`: standard `torch.compile()` with one optional bucketing pass through `jit_backend.py`.
  - `aot`: export a joint graph, run joint passes, partition, then compile.
  - `aot_fx_trace`: skip model-level wrapping and trace `model + loss + autograd.grad` from inside `GraphTrainer.forward_backward_step()`.
- `GraphTrainerCompileConfig.passes`
  - Compiler pass names for partitioned forward/backward graphs.
- `GraphTrainerCompileConfig.joint_passes`
  - Joint-graph-only passes before partitioning.
- `GraphTrainerCompileConfig.precompile`
  - If true, AOT mode attempts to load a saved artifact first and otherwise compiles with `serializable=True`.
- `to_graph_trainer_config(...)`
  - Copies all base `Trainer.Config` fields, swaps `model_spec`, drops the old compile config, and forces non-`none` activation checkpoint settings to `selective` because GraphTrainer only supports graph-based SAC.

## Compilation Pipeline Details

### `compile.py`
- `_get_precompile_storage_and_key(...)`
  - Uses `DiskStorageAdapter` rooted at `precompile_artifact_dir` and names artifacts by rank, e.g. `default_rank0`.
- `_apply_jit_compile(...)`
  - Builds a custom backend with `get_compile_backend_with_passes(...)`, then calls `torch.compile(..., fullgraph=True)`.
- `_make_precompile_callback(...)`
  - Computes a fingerprint and returns a callback that saves the compiled function plus metadata after AOT compilation finishes.
- `_apply_aot_compile(...)`
  - Registers `BlockMask` as a pytree node, optionally tries `precompile_load(...)`, resolves joint and compiler passes, constructs AOT compilers through `make_compiler_with_passes(...)`, and returns a `CompiledModule`.
- `_apply_aot_compile_load(...)`
  - Loads a serialized callable and wraps the model in `CompiledModule` with a dummy builder so live graph capture is never used.
- `apply_compile(...)`
  - Central validation point for compile mode, serializable-pass requirements, and pass dispatch.

### `graph_utils.py`
- `export_joint(...)`
  - Uses `dynamo_graph_capture_for_export(...)` to capture a graph module, preserves metadata, and exports a joint graph with descriptors.
- `joint_graph_builder(...)`
  - Runs optional joint passes, then `aot_compile_joint_with_descriptors(...)`; if `on_compile` is present it gets the compiled function plus `out_spec`.
- `CompiledModule.forward(...)`
  - Converts inputs with `parallelize_inputs(...)`, lazily builds or loads the compiled function, and executes it.
- `validate_pass_names(...)`
  - Enforces ordering and dependencies:
  - `cudagraph` must be last.
  - `auto_bucketing` and `transformer_block_bucketing` are mutually exclusive.
  - `full_inductor_compilation` requires `inductor_decomposition` and must be last or second-to-last before `cudagraph`.
- `get_compiler_passes_from_config(...)`
  - Special-cases `transformer_block_bucketing` to create an extra FSDP process group and prepend `reassign_to_pg_pass(...)`.
- `get_joint_custom_passes_from_config(...)`
  - Adds `validate_flex_attn_annotation_pass(...)` unless full Inductor is active, skips direct insertion of `inductor_decomposition`, and always appends `fsdp_reshard_after_fwd_pass(...)`.

## Pass Inventory
- `autobucketing_reordering_pass(...)`
  - Uses Inductor overlap scheduling to reorder collectives and compute.
- `transformer_block_bucketing_reordering_pass(...)`
  - Applies manual overlap bucketing with model-derived bucket plans.
- `regional_inductor_pass(...)`
  - Runs `regional_inductor(...)`; when serializable, it forces autograd cache and broadens the allowed op filter to include `_c10d_functional`.
- `cudagraph_pass(...)`
  - Wraps a graph module's `forward` in `CUDAGraphWrapper`.
- `validate_flex_attn_annotation_pass(...)`
  - Ensures FlexAttention HOP nodes carry `compile_with_inductor` metadata.
- `apply_sac_pass(...)`
  - Annotates joint-graph nodes with `CheckpointPolicy`, region IDs, and special handling for `mm`, `getitem`, and `wait_tensor`.
- `fsdp_reshard_after_fwd_pass(...)`
  - Marks SimpleFSDP all-gather nodes as recompute or save based on `reshard_after_forward`.
- `inductor_decomposition_pass(...)`
  - Retraces the joint graph with Inductor decompositions while preserving placeholder metadata and names.
- `full_inductor_compilation_pass(...)`
  - Calls `compile_fx_inner(...)`.
- `reassign_to_pg_pass(...)`
  - Rewrites all-gather nodes from one process-group name to another so AG and RS can run on separate NCCL communicators.
- `tlparse_log_graph_pass(...)`
  - Emits a readable graph artifact through `trace_structured(...)`.

## Tracing And Subclass Handling
- `_patch_engine_run_backward()`
  - Temporarily patches autograd's engine entrypoint so traced backward nodes receive `seq_nr` metadata.
- `_copy_fwd_metadata_to_bw_nodes(...)`
  - Uses shared `seq_nr` values to propagate forward annotations to backward nodes.
- `_unwrap_subclass(...)` and `_wrap_to_subclass(...)`
  - Flatten and restore nested tensor subclasses such as DTensor.
- `_remove_cpu_shadow_chains(...)`
  - Deletes dead CPU `empty_strided` metadata chains left by DTensor shadow ops.
- `trace_module(...)`
  - Lifts parameters and buffers into a pure traced function, traces it with `make_fx`, preserves metadata, then returns `TracedResult`.
- `run_traced_module(...)`
  - Replays the traced graph with the caller's current parameter and buffer snapshot, then rewraps outputs into their original subclass layouts.

## SimpleFSDP Details
- `data_parallel(...)`
  - Supports `replicate`, `fully_shard`, and `hybrid_shard`.
- `_distribute_dtensor(...)`
  - Handles nested `DTensor` distribution so DP can wrap tensors already sharded by TP or EP.
- `_register_parametrization(...)`
  - Generates a dynamic subclass per wrapped module so parameter access routes through `ReplicateComputation(...)`.
- `ReplicateComputation.replicate_compute(...)`
  - Redistributes sharded parameters to replicated compute placements, with support for nested non-DP meshes and optional full-DTensor output.
- `disable_active_parametrization()`
  - Used by graph-trainer model subclasses during `init_weights()` so initialization runs on the actual parameter objects, not their replicated views.

## Model-Specific Hooks
- `llama3/parallelize.py`
  - Annotates FlexAttention plus AC regions, applies TP with optional async TP, adds graph SAC, wraps simple FSDP or HSDP, and compiles.
- `deepseek_v3/parallelize.py`
  - Adds MoE dispatch/combine/compute annotations, validates CP only for SDPA attention, applies non-MoE TP plus MoE EP/ETP, graph SAC, expert-specific data parallel, then compile.
- Both model folders use `model_registry(...)` to swap in graph-trainer model subclasses while keeping the original loss builders, pipeline functions, and state-dict adapters.

## Precompile Mechanics
- `compute_config_fingerprint(...)`
  - Includes parameter and buffer names, shapes, dtypes, non-private `ParallelDims` fields, compile mode/backend/pass lists, PyTorch version, and CUDA capability.
- `_unwrap_serializable(...)`
  - Walks the `__wrapped__` chain until it finds a `BundledAOTAutogradSerializableCallable`.
- `precompile_save(...)`
  - Serializes compiled artifacts plus parameter/buffer specs and writes them with `StorageAdapter.save(...)`.
- `precompile_load(...)`
  - Validates model compatibility, checks or warns on fingerprints, lazily deserializes on first call, reconstructs the flat input layout as `params + buffers + user args`, and unflattens outputs with the saved `out_spec`.

## Test Coverage Map
- `tests/integration_tests.py`
  - Defines the supported CLI matrices for Llama3 and DeepSeek-V3 across JIT, AOT, `aot_fx_trace`, TP, DP, PP, EP, FlexAttention, cudagraph, and checkpoint resume.
- `tests/test_numerics.py`
  - Verifies graph-trainer numerics against eager baselines through `scripts/loss_compare.py`; also compares `simple_fsdp` against composable FSDP2.
- `tests/test_passes.py`
  - Covers process-group reassignment and selective AC annotation semantics.
- `tests/test_precompile.py`
  - Covers disk storage, artifact serialization, fingerprint sensitivity, and save/load validation errors.
- `tests/test_trace_module.py`
  - Covers basic tracing, DTensor tracing, metadata propagation, model-family bitwise tests, FlexAttention annotations, and FSDP traces with real collectives.

## Modification Checklist
- When adding a pass:
  - Update `passes.py`.
  - Register it in the right registry.
  - Add validation in `graph_utils.validate_pass_names(...)` if ordering matters.
  - Extend `test_passes.py` or `test_trace_module.py`.
- When changing `simple_fsdp.py`:
  - Recheck `test_numerics.py` and the FSDP portions of `test_trace_module.py`.
- When changing precompile behavior:
  - Recheck `test_precompile.py`.
- When changing model annotations or compile configuration:
  - Update the model-specific `parallelize.py` and verify `integration_tests.py` still reflects supported combinations.
