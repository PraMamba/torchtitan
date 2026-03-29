---
name: torchtitan-dr-experiments-graph-trainer
description: Use when changing TorchTitan's graph_trainer experiment, especially when debugging AOT or JIT graph capture, compiler pass selection, SimpleFSDP behavior, precompile artifact loading, or the model-specific graph_trainer Llama3 and DeepSeek-V3 integrations.
---

# TorchTitan GraphTrainer

## Overview
`torchtitan/experiments/graph_trainer` is TorchTitan's compiler-centric training stack. It reuses the normal `Trainer` runtime and model families, but swaps eager parallelization for graph capture, FX graph rewriting, and compiler-driven execution. The module covers three execution modes: JIT `torch.compile()` with a custom backend, AOT joint forward-backward graph export with configurable joint and per-graph passes, and `aot_fx_trace`, which traces the entire forward-loss-backward step with `make_fx` at runtime. It also includes its own `simple_fsdp` data-parallel wrapper, optional precompiled artifact save/load, model-specific graph_trainer registries for Llama3 and DeepSeek-V3, and a large test suite that defines the supported combinations and bitwise-equivalence expectations.

## Public Surface
- `torchtitan.experiments.graph_trainer.trainer.GraphTrainer`: `Trainer` subclass that adds `GraphTrainerCompileConfig` support and the `aot_fx_trace` execution path.
- `torchtitan.experiments.graph_trainer.configs.GraphTrainerCompileConfig`: compile-mode config with `mode`, `backend`, `passes`, `joint_passes`, and precompile settings.
- `torchtitan.experiments.graph_trainer.compile.apply_compile(...)`: top-level dispatcher that chooses JIT, AOT, or `aot_fx_trace`.
- `torchtitan.experiments.graph_trainer.graph_utils.CompiledModule`: wrapper that lazily builds or loads a compiled graph callable and delegates parameter/state methods to the inner model.
- `torchtitan.experiments.graph_trainer.simple_fsdp.data_parallel(...)`: graph-friendly FSDP/DDP/HSDP wrapper implemented through DTensor redistribution and dynamic parametrization.
- `torchtitan.experiments.graph_trainer.precompile.precompile_save(...)` / `precompile_load(...)`: save and reload serializable AOT artifacts.
- `torchtitan.experiments.graph_trainer.llama3.model_registry(...)` and `torchtitan.experiments.graph_trainer.deepseek_v3.model_registry(...)`: model-spec entrypoints for the experiment.

## Design Logic
- GraphTrainer keeps the core training lifecycle in `Trainer` and only replaces the compilation and graph-execution seams. That is why `GraphTrainer` in `trainer.py` subclasses `Trainer` instead of rebuilding the runtime.
- AOT mode centers the entire system around a joint forward-backward graph. `graph_utils.export_joint()` plus `joint_graph_builder()` let the experiment annotate, rewrite, partition, and optionally serialize the exact graph that spans both phases.
- JIT and AOT share a pass vocabulary but use different application points. JIT mode installs one pass into a custom backend in `jit_backend.py`, while AOT mode resolves named passes in `graph_utils.get_compiler_passes_from_config()` and `get_joint_custom_passes_from_config()` and runs them on joint or partitioned graphs.
- `simple_fsdp.py` avoids standard FSDP wrappers because GraphTrainer needs compiler-friendly parameter access. It converts parameters to DTensors, then dynamically exposes replicated local tensors through generated properties so the forward graph still looks like dense tensor math.
- The module treats annotations as a control plane. `annotate_fn(...)` metadata and `custom` node metadata drive regional inductor, selective activation checkpointing region boundaries, and MoE debugging tags.
- Precompile mode is deliberately conservative. `precompile.py` fingerprints model structure, parallel dims, compile passes, PyTorch version, and CUDA capability to reject stale or cross-machine-incompatible artifacts.

## Core Data Structures
- `GraphTrainerCompileConfig` in `torchtitan/experiments/graph_trainer/configs.py`
  - Extends `CompileConfig` with `mode`, `passes`, `joint_passes`, `precompile`, and `precompile_artifact_dir`.
- `GraphTrainer.Config` in `torchtitan/experiments/graph_trainer/trainer.py`
  - Replaces the base trainer's compile config with `GraphTrainerCompileConfig`.
- `TracedResult`, `SubclassMeta`, and `SubclassLayout` in `torchtitan/experiments/graph_trainer/make_fx_tracer.py`
  - Hold traced FX state plus enough metadata to unwrap and rewrap DTensor or other tensor subclasses.
- `PrecompiledArtifact` in `torchtitan/experiments/graph_trainer/precompile.py`
  - Stores serialized compiled code, parameter and buffer ordering, output pytree spec, metadata, and a config fingerprint.
- `CompiledModule` in `torchtitan/experiments/graph_trainer/graph_utils.py`
  - Lazy wrapper that compiles on first forward or loads a precompiled callable.
- `MixedPrecisionPolicy` and `ReplicateComputation` in `torchtitan/experiments/graph_trainer/simple_fsdp.py`
  - Define how graph-friendly data parallel redistributes parameters and gradients across DP meshes.

## State Flow
1. Config creation:
   `llama3/config_registry.py` and `deepseek_v3/config_registry.py` convert normal trainer configs with `to_graph_trainer_config(...)`, swap in graph-trainer model registries, and default `compile.enable=True`.
2. Trainer construction:
   `GraphTrainer.__init__()` in `trainer.py` calls the base `Trainer` setup, then validates that `aot_fx_trace` is not combined with pipeline parallelism.
3. Model preparation:
   The model-specific `parallelize.py` file annotates the model, applies TP or EP/ETP, optionally injects graph selective AC via `apply_graph_ac(...)`, wraps data parallel with `simple_fsdp.data_parallel(...)`, and finally calls `apply_compile(...)`.
4. Compilation:
   `compile.apply_compile(...)` validates mode and pass combinations, resolves FSDP reshard policy, optionally checks for saved precompile artifacts, and then chooses `_apply_jit_compile(...)`, `_apply_aot_compile(...)`, or no-op for `aot_fx_trace`.
5. AOT graph build:
   `CompiledModule.forward()` calls `joint_graph_builder(...)` on first invocation. `graph_utils.export_joint()` captures a joint graph, optional joint passes mutate it, and `aot_compile_joint_with_descriptors(...)` partitions and compiles it into a callable.
6. `aot_fx_trace` path:
   `GraphTrainer.forward_backward_step()` constructs `FwdBwdStepModule`, traces it once with `trace_module(...)`, then runs the traced graph each step with `run_traced_module(...)` and manually accumulates the returned gradients into parameters.
7. Runtime graph reuse:
   `CompiledModule` caches the compiled callable in `joint_graph_module`; `CUDAGraphWrapper` caches warmup, capture, and replay state; `precompile_load(...)` defers deserialization until first execution on the correct CUDA device.
8. Shutdown:
   `GraphTrainer.close()` calls `cudagraph_teardown()` after the normal trainer shutdown to release CUDA graph references that otherwise block process-group destruction.

## Error Handling And Side Effects
- `GraphTrainer.__init__()` raises if `compile.mode == "aot_fx_trace"` is used with pipeline parallelism.
- `compile.apply_compile(...)` raises on unknown modes, invalid precompile pass combinations, and unsupported JIT pass layouts.
- `common_utils.apply_graph_ac(...)` only accepts `activation_checkpoint.mode == "selective"` or `"none"`.
- `graph_utils.validate_pass_names(...)` enforces pass ordering, including the requirement that `full_inductor_compilation` depends on `inductor_decomposition`.
- `precompile_load(...)` raises `ValueError` for parameter, buffer, or fingerprint mismatches and only allows bypass through `TORCHTITAN_SKIP_FINGERPRINT_CHECK=1`.
- `DiskStorageAdapter.save(...)` writes through a temp file plus atomic rename so a crash cannot leave a partial artifact in place.
- `simple_fsdp._register_parametrization(...)` mutates module classes at runtime to expose parametrized properties, which is intentional and required for pickling and graph serialization.

## Common Modification Scenarios
- Add a new graph-trainer model family:
  Mirror the pattern in `torchtitan/experiments/graph_trainer/llama3/` or `torchtitan/experiments/graph_trainer/deepseek_v3/`. You need a graph-specific model subclass, a graph-trainer `model_registry(...)`, a config registry that uses `to_graph_trainer_config(...)`, and a `parallelize.py` that applies annotations, graph AC, data parallel, and `apply_compile(...)`.
- Add or change a compiler pass:
  Update `torchtitan/experiments/graph_trainer/passes.py`, then register the pass name in `AVAILABLE_COMPILER_PASSES` or `AVAILABLE_JOINT_PASSES`. If the pass has ordering or dependency constraints, update `graph_utils.validate_pass_names(...)` too.
- Change precompile artifact behavior:
  Start in `torchtitan/experiments/graph_trainer/precompile.py` and `storage.py`, then wire any new save/load conditions through `_apply_aot_compile(...)` in `compile.py`.
- Debug DTensor or subclass tracing failures:
  The key logic is in `torchtitan/experiments/graph_trainer/make_fx_tracer.py`, especially `_unwrap_subclass(...)`, `_wrap_to_subclasses(...)`, `_patch_engine_run_backward()`, and `_remove_cpu_shadow_chains(...)`.
- Change SimpleFSDP sharding or mixed-precision behavior:
  Edit `torchtitan/experiments/graph_trainer/simple_fsdp.py`. The important pieces are `_distribute_dtensor(...)`, `ReplicateComputation.replicate_compute(...)`, and `data_parallel(...)`.
- Add new integration coverage:
  Extend `torchtitan/experiments/graph_trainer/tests/integration_tests.py` for supported run matrices, then update focused behavioral tests in `test_numerics.py`, `test_passes.py`, `test_precompile.py`, or `test_trace_module.py`.

## File Map
- `torchtitan/experiments/graph_trainer/configs.py`: graph-trainer config extensions and adapter from base trainer config.
- `torchtitan/experiments/graph_trainer/trainer.py`: `GraphTrainer` runtime overrides and `aot_fx_trace` execution.
- `torchtitan/experiments/graph_trainer/compile.py`: mode dispatcher and precompile integration.
- `torchtitan/experiments/graph_trainer/graph_utils.py`: joint graph export, pass wiring, and `CompiledModule`.
- `torchtitan/experiments/graph_trainer/make_fx_tracer.py`: make-fx tracer and DTensor/subclass handling.
- `torchtitan/experiments/graph_trainer/passes.py`: named graph passes and registries.
- `torchtitan/experiments/graph_trainer/simple_fsdp.py`: graph-friendly data parallel wrapper.
- `torchtitan/experiments/graph_trainer/llama3/` and `deepseek_v3/`: model-specific experiment integration.
- `torchtitan/experiments/graph_trainer/tests/`: numerics, tracing, pass, precompile, and integration coverage.

## See Also
- `reference.md`: function-by-function inventory, pass registry details, model-specific hooks, and test coverage map.
