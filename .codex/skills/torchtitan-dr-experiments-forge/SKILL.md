---
name: torchtitan-dr-experiments-forge
description: Use when working on TorchTitan's lightweight Forge engine, especially when changing shared trainer bootstrap, post-training engine setup, example-trainer wiring, or checkpoint and optimizer construction without the full runtime stack.
---

# TorchTitan Experiments Forge

## Overview

`torchtitan/experiments/forge` is a stripped-down training bootstrap layer built around [`ForgeEngine`](../../../torchtitan/experiments/forge/engine.py). It extracts the constructor-heavy part of the full runtime into a reusable engine that initializes distributed state, builds a model on meta tensors, applies parallelization, creates optimizers and schedulers, and wires checkpoint state, but deliberately stops short of owning dataloading, metrics, validation, or the full train loop. [`example_train.py`](../../../torchtitan/experiments/forge/example_train.py) shows how a downstream trainer subclasses `ForgeEngine` to add those pieces back for a pretraining-style workflow.

## Public Surface

- [`torchtitan/experiments/forge/__init__.py`](../../../torchtitan/experiments/forge/__init__.py)
  - Re-exports `ForgeEngine`.
- [`torchtitan/experiments/forge/engine.py`](../../../torchtitan/experiments/forge/engine.py)
  - `ForgeEngine`: reusable constructor-time engine that is both `Stateful` and `Configurable`.
  - `ForgeEngine.Config`: minimal runtime config bundle for model construction, parallelism, optimizer, scheduler, checkpointing, and distributed settings.
- [`torchtitan/experiments/forge/example_train.py`](../../../torchtitan/experiments/forge/example_train.py)
  - `Trainer(ForgeEngine)`: example subclass that adds tokenizer, dataloader, metrics, validation, and the train loop.
  - `main(custom_trainer_class=None)`: example executable entrypoint compatible with `ConfigManager`.

## Design Logic

- `ForgeEngine` exists to isolate constructor logic from the full [`Trainer`](../../../torchtitan/trainer.py). Its contract is: "build a distributed-ready model stack and the core training components, but leave data flow and outer-loop policy to the caller."
- The engine keeps the same meta-device initialization strategy as the full runtime. `model_config.build()` runs under `torch.device("meta")`, then the chosen parallelization or pipelining function wraps the model before `to_empty()` and `init_weights()`. That keeps initialization sharding-friendly and avoids materializing dense weights too early.
- The config type is intentionally narrower than the full runtime config. `ForgeEngine.Config` includes the pieces needed for model construction and checkpoint state, but omits tokenizer, dataloader, metrics, profiling, and validation. The example trainer demonstrates the intended extension pattern by accepting the larger `TitanTrainer.Config` superset and passing it through `super().__init__(config)`.
- Pipeline parallelism is treated as a first-class branch during construction. `ForgeEngine.__init__()` either calls `train_spec.pipelining_fn(...)` and manages `model_parts` plus PP schedule metadata, or it calls `train_spec.parallelize_fn(...)` and keeps a single-model `model_parts = [model]`. Downstream trainers are expected to respect that split.
- Checkpoint state is deliberately minimal at the engine layer. `ForgeEngine` only persists engine-owned state via the `states={"train_state": self}` handoff to `CheckpointManager`; concrete subclasses decide what extra counters should exist and override `state_dict()` / `load_state_dict()` as needed.

## Core Data Structures

- `ForgeEngine.Config` in [`engine.py`](../../../torchtitan/experiments/forge/engine.py)
  - Holds `hf_assets_path`, `dump_folder`, `model_spec`, optimizer and LR scheduler configs, `training`, `parallelism`, `checkpoint`, activation checkpointing, compile settings, model converters, `comm`, and `debug`.
  - `to_dict()` returns a deep `asdict(...)` rendering for logging and downstream metrics consumers.
- `ForgeEngine`
  - Runtime fields include `config`, `parallel_dims`, `train_spec`, `model_parts`, `loss_fn`, `optimizers`, `lr_schedulers`, `checkpointer`, `device`, `gc_handler`, `gradient_accumulation_steps`, `train_context`, `dp_degree`, `dp_rank`, `model_config`, `num_flops_per_token`, `model_param_count`, and `global_batch_size`.
  - PP-only state includes `pp_schedule`, `pp_has_first_stage`, and `pp_has_last_stage`.
- `Trainer` in [`example_train.py`](../../../torchtitan/experiments/forge/example_train.py)
  - Extends `ForgeEngine` with `tokenizer`, `dataloader`, `validator`, `metrics_processor`, and `step`.
  - Shows the intended subclass pattern for adding runtime policy and IO around the core engine.

See [`reference.md`](./reference.md) for the full file inventory and method-level detail.

## State Flow

1. Entry and config parsing:
   `main()` in [`example_train.py`](../../../torchtitan/experiments/forge/example_train.py) initializes logging, logs `torchtitan.__version__`, parses CLI config through `ConfigManager.parse_args()`, and either returns early for `config.comm.mode == "local_tensor"` or builds a trainer through `config.build()` or `custom_trainer_class(config)`.
2. Engine construction:
   `ForgeEngine.__init__()` in [`engine.py`](../../../torchtitan/experiments/forge/engine.py) sets the local device from `LOCAL_RANK`, initializes distributed state with `dist_utils.init_distributed(...)`, derives a `ParallelDims` object from `ParallelismConfig`, computes DP rank/degree, installs `GarbageCollection`, sets determinism, and stores `train_spec = config.model_spec`.
3. Meta-model build and sizing:
   The engine pulls `model_config = train_spec.model`, mutates it through `model_config.update_from_config(trainer_config=config)`, builds the model under a meta device and default dtype context, then computes `model_param_count` and `num_flops_per_token`.
4. Parallelization branch:
   If PP is enabled, `train_spec.pipelining_fn(...)` returns `pp_schedule`, `model_parts`, and first/last-stage flags; each model part is then `to_empty(...)`, `init_weights(...)`, and set to train mode. Otherwise the engine calls `train_spec.parallelize_fn(...)`, materializes the single model, initializes weights, and stores it as the sole entry in `model_parts`.
5. Core component build:
   The engine builds `loss_fn`, `optimizers`, optional `post_optimizer_build_fn`, `lr_schedulers`, and `checkpointer`. It also computes `global_batch_size`, validates that it divides `local_batch_size * dp_degree`, derives `gradient_accumulation_steps`, and configures `train_context` plus AMP through distributed helpers.
6. Example trainer extension:
   `Trainer.__init__()` in [`example_train.py`](../../../torchtitan/experiments/forge/example_train.py) logs config, builds the tokenizer and dataloader, creates `MetricsProcessor`, logs model-size and memory facts, initializes `step = 0`, and conditionally builds a validator with PP metadata if validation is enabled.
7. Example training path:
   `Trainer.train()` loads checkpoint state, wraps the loop in profiling and memory-snapshot contexts, creates `data_iterator = self.batch_generator(self.dataloader)`, and iterates until `training.steps`. Each step runs GC, executes `train_step()`, optionally validates, saves checkpoints, advances profilers, and reduces process-group timeouts after the first successful step.
8. Microbatch execution:
   `train_step()` collects `gradient_accumulation_steps` CPU microbatches, counts valid tokens, all-reduces them across the batch mesh when DP is enabled, moves each microbatch to device, and calls `forward_backward_step()`. That method routes either through `pp_schedule.step(...)` or a single-model forward under `train_context` and AMP, then normalizes the loss by global valid tokens before backward.
9. Shutdown:
   `Trainer.close()` closes metrics first and then delegates to `ForgeEngine.close()`, which closes the checkpointer. `main()` then destroys the process group after a clean close path.

## Error Handling And Side Effects

- `ForgeEngine.__init__()` assumes `LOCAL_RANK` and `WORLD_SIZE` are present in the environment; it reads them directly with `os.environ[...]`.
- Global batch-size validation is strict. The engine asserts `global_batch_size > 0` and asserts that it is divisible by `local_batch_size * dp_degree`.
- Enabling PP without `train_spec.pipelining_fn` raises `RuntimeError`.
- `main()` enforces the same seed-checkpoint invariants as the full runtime: `WORLD_SIZE == 1` and `config.checkpoint.enable` must hold before `trainer.checkpointer.save(curr_step=0, last_step=True)`.
- `batch_generator()` converts `StopIteration` into `DataloaderExhaustedError`, and `Trainer.train()` handles that by canceling the unfinished step rather than saving partial progress.
- Engine-side side effects include initializing distributed process groups, setting the device, mutating model config via `update_from_config(...)`, materializing and training-enabling model parts, advancing optimizer and scheduler state, and writing checkpoint state under `dump_folder`.

## Common Modification Scenarios

- Add a new post-training engine consumer:
  Start in [`engine.py`](../../../torchtitan/experiments/forge/engine.py) and subclass `ForgeEngine` the way [`example_train.py`](../../../torchtitan/experiments/forge/example_train.py) does. Keep constructor-time model/optimizer/checkpoint logic in the base engine, then add tokenizer, dataloader, metrics, replay buffers, generators, or custom actor state in the subclass.
- Change what the engine considers "core" versus downstream-owned:
  Edit `ForgeEngine.Config` and `ForgeEngine.__init__()` in [`engine.py`](../../../torchtitan/experiments/forge/engine.py). If a subsystem should exist for all engine consumers, add its config and build path there; if it is specific to one downstream trainer, keep it in the subclass.
- Modify pipeline-parallel construction:
  Edit the PP branch in `ForgeEngine.__init__()` in [`engine.py`](../../../torchtitan/experiments/forge/engine.py). That is where `pipelining_fn(...)` is invoked, PP stage flags are stored, and model parts are materialized.
- Change example-loop input shaping or validation behavior:
  Edit `Trainer.post_dataloading_process()`, `forward_backward_step()`, and `train_step()` in [`example_train.py`](../../../torchtitan/experiments/forge/example_train.py). Those methods jointly define how extra inputs, attention masks, CP preprocessing, PP forwarding, and metric logging behave.
- Extend checkpointed training state in the example trainer:
  Update `Trainer.state_dict()` and `Trainer.load_state_dict()` in [`example_train.py`](../../../torchtitan/experiments/forge/example_train.py), and make sure the new attributes are initialized before `self.checkpointer.load(...)` runs in `Trainer.train()`.
- Add a new example executable behavior or custom trainer injection point:
  Edit `main()` in [`example_train.py`](../../../torchtitan/experiments/forge/example_train.py). That is the only place where CLI config is parsed, `custom_trainer_class` is honored, seed-checkpoint mode is handled, and process-group teardown happens.

## File Map

- [`torchtitan/experiments/forge/__init__.py`](../../../torchtitan/experiments/forge/__init__.py): module export surface.
- [`torchtitan/experiments/forge/engine.py`](../../../torchtitan/experiments/forge/engine.py): reusable constructor-time engine.
- [`torchtitan/experiments/forge/example_train.py`](../../../torchtitan/experiments/forge/example_train.py): example trainer subclass and executable entrypoint.
- [`torchtitan/experiments/forge/README.md`](../../../torchtitan/experiments/forge/README.md): high-level intended use and scope boundaries.

## See Also

- [`reference.md`](./reference.md): detailed method index, field inventory, and cross-file relationships for the forge module.
