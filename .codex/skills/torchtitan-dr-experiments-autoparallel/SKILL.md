---
name: torchtitan-dr-experiments-autoparallel
description: Use when working on TorchTitan's AutoParallel experiment, especially when changing automatic sharding placement, model registration, or DeepSeek MoE handling
---

# Torchtitan Experiments Autoparallel

## Overview

`torchtitan/experiments/autoparallel` is the bridge between TorchTitan's normal `ModelSpec` and `Trainer` flow and PyTorch's external `autoparallel` package. It adds an `AutoParallelConfig`/`AutoParallelCompileConfig` overlay, registers three experiment entrypoints (`autoparallel.llama3`, `autoparallel.deepseek_v3`, `autoparallel.local_map_deepseek_v3`), and replaces the usual hand-written parallelization rules with `AutoParallel.optimize_placement()` plus explicit mesh/input/output constraints. The dense Llama path is simple; the DeepSeek paths exist mainly to make MoE routing and metadata survive AutoParallel wrapping.

## When To Use

Use this module when:
- a TorchTitan run uses `--module autoparallel.llama3`, `autoparallel.deepseek_v3`, or `autoparallel.local_map_deepseek_v3`
- you need to change how AutoParallel sees mesh dimensions, tensor placements, or compile settings
- you are debugging why AutoParallel-wrapped DeepSeek MoE blocks lost attributes like `moe_enabled` or `load_balance_coeff`
- you need to add another AutoParallel experiment family without changing core `torchtitan/trainer.py`

Do not use this module to change TorchTitan's general distributed runtime. That logic lives in `torchtitan/distributed/` and the model-family packages under `torchtitan/models/`.

## Purpose And Capabilities

This experiment package exposes a thin public API:
- `AutoParallelCompileConfig` and `AutoParallelConfig` in `torchtitan/experiments/autoparallel/configs.py`
- one `model_registry(flavor)` per experiment family in `llama3/__init__.py`, `deepseek_v3/__init__.py`, and `local_map_deepseek_v3/__init__.py`
- one debug config builder per family in each `config_registry.py`
- one AutoParallel-backed `parallelize_*` function per family
- one integration-test suite builder in `tests/integration_tests.py`

Externally, the module's job is to keep TorchTitan's existing `ModelSpec` contract intact while swapping the implementation of `parallelize_fn`. That means the rest of training still expects normal TorchTitan concepts: `Trainer.Config`, `ModelSpec`, pipeline function, loss builder, checkpoint config, and optional post-optimizer hooks.

## Design Logic

The core design choice is "adapt TorchTitan to AutoParallel, not the other way around." Each experiment family still returns a normal `ModelSpec`, and each config builder still returns a `Trainer.Config` subclass. The only unusual fields live under `compile`, via `AutoParallelCompileConfig`, and the only unusual runtime behavior lives inside the `parallelize_*` implementations.

The Llama path in `torchtitan/experiments/autoparallel/llama3/parallelize_llama.py` is intentionally minimal. It:
- forces some Inductor flags (`force_disable_caches`, `allow_buffer_reuse`)
- forwards `compile_config.comms_bucket_reorder_strategy` into `configure_inductor_for_autobucketing()`
- builds a "dense" mesh from the enabled `dp_replicate`, `fsdp`, and `tp` dimensions
- defines synthetic CUDA input tensors in `input_fn()`
- feeds mesh-aware placement constraints to `AutoParallel`
- optionally patches the output into a 1D `DTensor` for loss-parallel compatibility

The DeepSeek path is more invasive because AutoParallel does not preserve all of TorchTitan's module structure and the MoE execution path is not directly trace-friendly. `torchtitan/experiments/autoparallel/deepseek_v3/parallelize_deepseekv3.py` therefore:
- rewrites MoE execution through `_moe_forward()`, `moe_forward()`, and `create_functional_router_forward()`
- applies `local_map` to `_moe_forward()` inside `monkey_patch_local_map_moe()`
- restores TorchTitan-facing flags through `set_torchtitan_fields()` and `_preserve_moe_attributes()`

The `local_map_deepseek_v3` variant is a compatibility path for AutoParallel's own testing model (`autoparallel._testing.models.dsv3`). It exists because the standard TorchTitan DeepSeek model and AutoParallel's local-map assumptions are not fully aligned. This variant keeps the TorchTitan `BaseModel` contract by wrapping AutoParallel's test model in `local_map_deepseek_v3/model.py` and reusing TorchTitan's registry/state-dict wiring.

## Core Data Structures

The key structures are:
- `AutoParallelCompileConfig` in `torchtitan/experiments/autoparallel/configs.py`
  - extends `CompileConfig`
  - adds `comms_bucket_reorder_strategy` and `autop_force_bf16`
- `AutoParallelConfig` in `torchtitan/experiments/autoparallel/configs.py`
  - extends `Trainer.Config`
  - replaces `compile` with `AutoParallelCompileConfig`
- `ModelSpec` instances returned by each family's `model_registry()`
  - these bind TorchTitan's model config, AutoParallel `parallelize_fn`, `pipeline_llm`, loss builder, and state-dict adapter
- `DeepSeekV3ModelArgs` in `local_map_deepseek_v3/args.py`
  - inherits both AutoParallel's `_DeepSeekV3ModelArgs` and TorchTitan `BaseModel.Config`
- `DeepSeekV3Model` in `local_map_deepseek_v3/model.py`
  - inherits both AutoParallel's test model and TorchTitan `BaseModel`

The file-by-file API index and function signatures are in `reference.md`.

## State Flow

### Normal entry path

1. CLI selects an experiment module, for example `--module autoparallel.llama3`.
2. The corresponding `config_registry.py` function returns an `AutoParallelConfig` with a `model_spec`, optimizer/LR scheduler config, HF dataset config, metrics config, and a debug-oriented training setup.
3. The family `model_registry()` returns a `ModelSpec` whose `parallelize_fn` points at one of the AutoParallel wrappers.
4. During trainer setup, TorchTitan calls that `parallelize_fn` with the model, `ParallelDims`, training config, model-converter config, compile config, and activation-checkpoint config.
5. The wrapper builds a mesh, defines `input_fn()`, configures AutoParallel constraints, runs `optimize_placement()`, and returns the transformed module.
6. For DeepSeek variants, post-processing restores non-graph metadata that later optimizer/load-balancing code still expects.

### Error and guard paths

This module relies heavily on assertions instead of graceful fallback:
- `parallelize_llama()` rejects DDP, CP, and PP with assertions.
- `parallelize_deepseekv3()` also rejects DDP, CP, and PP.
- `local_map_deepseek_v3/parallelize_deepseekv3.py` asserts the sparse mesh is exactly 2D.
- the local-map variant asserts loss parallel is off.
- `monkey_patch_checks()` asserts specific DeepSeek MoE invariants before monkey patching.

Those assertions are the real compatibility contract. If they fire, the intended fix is usually to expand AutoParallel support in the wrapper, not to suppress the assertion.

## Modification Scenarios

### Add a new AutoParallel model family

Create a new subpackage mirroring `llama3/` or `deepseek_v3/`:
- add `__init__.py` with `model_registry(flavor) -> ModelSpec`
- add `config_registry.py` with at least one `AutoParallelConfig` builder
- add a `parallelize_*.py` that receives the standard TorchTitan parallelization signature
- add an entry to `torchtitan/experiments/__init__.py` elsewhere in the repo so CLI discovery can find it
- add an integration-test variant in `torchtitan/experiments/autoparallel/tests/integration_tests.py`

### Change the placement search or mesh semantics

Edit the relevant `parallelize_*` file:
- mesh composition is controlled by `dense_names` in `llama3/parallelize_llama.py`
- sparse MoE mesh composition is controlled by `sparse_names` in both DeepSeek wrappers
- input/output sharding constraints come from `possible_input_shardings`, `possible_output_shardings`, or direct tuples like `x_sharding = (Shard(0), Shard(0))`

If you change mesh names, also update any `dense_mesh["tp"]` or `sparse_mesh["etp"]` lookups used for the loss-parallel output hook.

### Change compile/autobucketing behavior

The only experiment-specific compile knobs live in `torchtitan/experiments/autoparallel/configs.py`. If you want a new CLI-visible setting, add it to `AutoParallelCompileConfig` and then thread it through the relevant `parallelize_*` function. Current compile behavior fans out in two places:
- `configure_inductor_for_autobucketing(compile_config.comms_bucket_reorder_strategy)`
- the local-map variant's `should_compile` gating plus assertions on `compile_config.components` and `compile_config.backend`

### Debug MoE metadata or load-balancing regressions

Start in `deepseek_v3/parallelize_deepseekv3.py` or `local_map_deepseek_v3/parallelize_deepseekv3.py`:
- `set_torchtitan_fields()` restores `moe_enabled`
- `_preserve_moe_attributes()` restores `load_balance_coeff`
- `monkey_patch_local_map_moe()` replaces MoE forward behavior and patches every MoE block

If optimizer hooks stop recognizing MoE blocks after AutoParallel wrapping, this is the first place to inspect.

### Update or re-enable tests

Edit `torchtitan/experiments/autoparallel/tests/integration_tests.py`. The suite currently only runs the Llama case; the DeepSeek case is present but commented out with a TODO. This file is also where GPU count and override CLI arguments are defined.

## File Guide

- `torchtitan/experiments/autoparallel/configs.py`: experiment-specific config extension point
- `torchtitan/experiments/autoparallel/llama3/`: dense Llama integration
- `torchtitan/experiments/autoparallel/deepseek_v3/`: TorchTitan DeepSeek integration with MoE monkey patching
- `torchtitan/experiments/autoparallel/local_map_deepseek_v3/`: AutoParallel test-model-based DeepSeek path
- `torchtitan/experiments/autoparallel/tests/`: integration test matrix

## Common Mistakes

- Assuming this module changes trainer behavior globally. It does not; it only swaps the `parallelize_fn` attached to a `ModelSpec`.
- Editing only `config_registry.py` when adding support for a new feature. Most real behavior lives in `parallelize_*.py`.
- Forgetting that DeepSeek support depends on copied metadata after AutoParallel rewrites the module structure.
- Treating the commented DeepSeek test as active coverage. It is disabled.
