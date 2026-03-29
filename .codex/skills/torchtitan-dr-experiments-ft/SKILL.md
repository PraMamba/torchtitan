---
name: torchtitan-dr-experiments-ft
description: Use when changing TorchTitan fault-tolerant training behavior, especially TorchFT setup, replica-group data partitioning, semi-synchronous DiLoCo or LocalSGD flow, or the FT Llama3 trainer and optimizer wiring.
---

# TorchTitan Experiments FT

## Overview

`torchtitan/experiments/ft` adds fault-tolerant and semi-synchronous training on top of the core TorchTitan runtime. The module centers on three things: `FTManager` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/manager.py` for TorchFT process-group and quorum management, `FaultTolerantTrainer` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/trainer.py` for trainer-level integration, and `FTOptimizersContainer` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/optimizer.py` for checkpoint-safe optimizer wrapping around TorchFT. The shipped model integration is `ft.llama3`, defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/llama3/__init__.py`, with config presets in `llama3/config_registry.py` and optional model fragmentation helpers under `diloco/`.

## Public Surface

- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/__init__.py`
  - Exports `FTManager`, `has_torchft`, and `maybe_semi_sync_training`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/manager.py`
  - `FTManager.Config`: TorchFT enablement, replica-group, process-group, and semi-sync controls.
  - `FTManager`: builds TorchFT manager state and exposes `get_dp_info()`, `maybe_set_all_reduce_hook()`, and `loss_sync_pg`.
  - `maybe_semi_sync_training(...)`: returns a `nullcontext`, `torchft.local_sgd.LocalSGD`, or `torchft.local_sgd.DiLoCo` context manager.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/trainer.py`
  - `FaultTolerantTrainer.Config`: extends `Trainer.Config` with `fault_tolerance`.
  - `FaultTolerantTrainer`: FT-aware trainer initialization, distributed bootstrap, train-step loss reduction, and semi-sync training lifecycle.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/optimizer.py`
  - `FTOptimizersContainer`: TorchFT-aware optimizer wrapper with cached state-dict semantics.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/config/job_config.py`
  - `FaultTolerance`: extended FT config for Streaming DiLoCo fragmentation, quantization, and sync cadence.
  - `JobConfig`: wraps `fault_tolerance`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/llama3/__init__.py`
  - `model_registry(flavor)`: returns a `FaultTolerantModelSpec` with `fragment_fn=fragment_llm`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/llama3/config_registry.py`
  - `llama3_ft_debugmodel()`: debug preset for FT Llama3 runs.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/diloco/utils.py`
  - `module_split(...)` and `fragment_llm(...)`: fragment a decoder-style model for DiLoCo outer synchronization.

## Design Logic

- Fault tolerance is implemented as an extension of the existing trainer, not a separate training stack. `FaultTolerantTrainer` reuses most of `torchtitan.trainer.Trainer` but overrides the places where replica-group semantics matter: distributed init, data partitioning, optimizer construction, checkpoint manager wiring, profiling folder layout, and loss synchronization.
- `FTManager` separates two modes deliberately. If `semi_sync_method is None`, it enables asynchronous quorum and registers an extra managed replicate process group; if a semi-sync algorithm is chosen, per-step quorum is disabled and synchronization is delegated to the semi-sync context returned by `maybe_semi_sync_training(...)`.
- `FTOptimizersContainer` caches optimizer state eagerly because TorchFT's optimizer wrapper and DCP optimizer state helpers can otherwise trigger state initialization at bad times. Its `state_dict()` intentionally returns a cached snapshot rather than recomputing live state on every call.
- DiLoCo fragmentation is opt-in and model-aware. `fragment_llm(...)` either returns the whole model, uses explicit `module_fqns_per_model_fragment`, or derives fragments via `generate_llm_fqn_per_model_part(...)`. The FT experiment avoids baking fragmentation rules into the trainer itself.
- The FT Llama3 integration stays minimal: it reuses the base Llama3 model config, parallelizer, pipeline function, loss builder, and state-dict adapter, then swaps only the model spec type and fragment function. That keeps FT-specific behavior in the experiment layer instead of forking the model implementation.

## Core Data Structures

- `FaultTolerance` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/config/job_config.py`
  - Extends `FTManager.Config` with `sync_steps`, `should_quantize`, `fragment_sync_delay`, `fragment_update_alpha`, `module_fqns_per_model_fragment`, and `num_fragments`.
- `FTManager.Config` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/manager.py`
  - Core FT knobs: `enable`, `process_group`, `process_group_timeout_ms`, `replica_id`, `group_size`, `min_replica_size`, `semi_sync_method`.
- `FaultTolerantTrainer.Config` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/trainer.py`
  - Adds `fault_tolerance: FaultTolerance` on top of the base runtime config tree.
- `FTOptimizersContainer` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/optimizer.py`
  - Holds `optimizers`, `cache_state_dict`, `_ft_optimizer`, and `_use_ft_optimizer` dispatch state.
- `FaultTolerantModelSpec` instance from `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/llama3/__init__.py`
  - Carries the usual model hooks plus `fragment_fn=fragment_llm`.

See `/home/scbjtfy/torchtitan/.codex/skills/torchtitan-dr-experiments-ft/reference.md` for field-level details and file-by-file API inventory.

## State Flow

1. Config selection:
   `llama3_ft_debugmodel()` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/llama3/config_registry.py` builds a `FaultTolerantTrainer.Config` that swaps in `FTOptimizersContainer.Config`, `FaultTolerance(...)`, and `model_registry("debugmodel")`.
2. Distributed bootstrap:
   `FaultTolerantTrainer.__init__()` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/trainer.py` sets the device first, then calls `self.init_distributed()`. `init_distributed()` optionally restricts `dist_utils.init_distributed(...)` to ranks belonging to the current FT replica group and then builds `self.ft_manager = config.fault_tolerance.build()`.
3. Data partitioning:
   After core DP mesh info is computed, `self.ft_manager.get_dp_info(batch_degree, batch_rank)` expands the effective data-parallel degree and remaps rank IDs so the dataloader partitions data across replica groups instead of only within the local DP mesh.
4. Model and runtime wiring:
   The trainer builds tokenizer, dataloader, model, converters, metrics, loss, and model parts much like the base trainer, but passes FT-specific extras to `build_loss_fn(...)`, `metrics.build(...)`, `checkpoint.build(...)`, and optionally `FTOptimizersContainer.build(...)`.
5. Async-quorum path:
   If FT is enabled and no semi-sync method is selected, `FTManager.maybe_set_all_reduce_hook(...)` installs a hook on each `FSDPModule` so replicated reductions go through `self.replicate_pg`, and `loss_sync_pg` is later used in metric reduction.
6. Semi-sync path:
   `FaultTolerantTrainer.train()` wraps the whole training loop inside `maybe_semi_sync_training(...)`. That helper creates `torchft.local_sgd.LocalSGD` or `torchft.local_sgd.DiLoCo`, optionally fragments the model via `fragment_fn`, and for DiLoCo builds one outer SGD optimizer per fragment.
7. Train step:
   `FaultTolerantTrainer.train_step()` mirrors the base accumulation flow, but when `parallel_dims.dp_cp_enabled` it reduces average loss, max loss, and `n_tokens_seen` through `dist_utils.dist_sum/dist_max(..., ft_pg=self.ft_manager.loss_sync_pg)`.
8. Integration tests:
   `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/tests/integration_tests.py` launches FT runs via `run_train.sh`, one command per replica group, and treats the module's end-to-end contract as “8 GPUs, TorchFT enabled, replicated launch arguments, shared dump folder root”.

## Error Handling And Side Effects

- `FTManager.__init__()` raises `ImportError` if FT is enabled without `torchft` installed.
- `FTManager.__init__()` raises `ValueError` for unsupported `process_group` values.
- `maybe_semi_sync_training(...)` asserts the FT manager is enabled before constructing LocalSGD/DiLoCo and raises `ValueError` for unknown `semi_sync_method`.
- `fragment_llm(...)` asserts `num_fragments > 0`; `module_split(...)` mutates fragment models by reusing submodules from the original model.
- `FTOptimizersContainer.load_state_dict()` clears `cache_state_dict` first to avoid stale references and memory leakage because underlying optimizer load uses assignment semantics.
- `tests/integration_tests.py` refuses to run unless the output directory is empty and skips when fewer than 8 GPUs are available.

## Common Modification Scenarios

- Add a new FT-enabled model family:
  Start in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/<model>/__init__.py` with a `FaultTolerantModelSpec`, then add presets like `llama3_ft_debugmodel()` in a `config_registry.py`. Reuse the base model's `parallelize_fn`, `pipelining_fn`, and adapter where possible, and decide whether the model needs a custom `fragment_fn`.
- Change TorchFT replica-group behavior or quorum rules:
  Edit `FTManager.Config` and `FTManager.__init__()` in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/manager.py`. That is where process-group type, quorum mode, replica IDs, and managed-process-group registration are decided.
- Change DiLoCo fragmentation or outer-sync behavior:
  Edit `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/diloco/utils.py` and `maybe_semi_sync_training(...)` in `manager.py`. Fragment boundaries come from `module_fqns_per_model_fragment` or `num_fragments`, while outer optimizer choice and sync cadence are hard-coded in the DiLoCo branch.
- Change FT-aware optimizer checkpoint behavior:
  Edit `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/optimizer.py`. The key logic is the cached `state_dict()`, `load_state_dict()` invalidation, and `_use_ft_optimizer` gating in `step()` and `zero_grad()`.
- Change how FT training affects metrics, profiling, or dataloader sharding:
  Edit `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/trainer.py`. `get_dp_info(...)`, metrics builder kwargs (`ft_enable`, `ft_replica_id`), profiling `leaf_folder`, and `loss_sync_pg` usage are all wired there.
- Expand or debug end-to-end FT coverage:
  Edit `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/tests/integration_tests.py` and `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/torchft.md`. The test matrix is small and currently assumes one 8-GPU machine, so new replica-group topologies and commands belong there.
