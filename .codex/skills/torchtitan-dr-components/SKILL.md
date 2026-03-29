---
name: torchtitan-dr-components
description: Use when changing TorchTitan checkpointing, dataloader state, loss or learning-rate plumbing, metrics logging, tokenizer loading, validation flow, or quantization converter behavior.
---

# TorchTitan Components

## Overview

`torchtitan/components` is TorchTitan's reusable training-subsystem layer. It packages the pieces that the runtime wires together around a model: checkpoint save/load orchestration, stateful data loading, loss builders, optimizer and LR scheduler containers, metrics backends, tokenizer loading and chat-template rendering, validation loops, and quantization converters. Most of these classes are `Configurable`, so `Trainer.Config` can construct them with uniform `.build(...)` calls while keeping model-specific logic elsewhere.

Externally, this module exposes the concrete machinery that `torchtitan/trainer.py` depends on after config parsing is complete:
- checkpoint persistence through `CheckpointManager` in `torchtitan/components/checkpoint.py`
- distributed-aware dataloading via `ParallelAwareDataloader` in `torchtitan/components/dataloader.py`
- loss builders in `torchtitan/components/loss.py`
- optimizer and scheduler multiplexers in `torchtitan/components/optimizer.py` and `torchtitan/components/lr_scheduler.py`
- logging and throughput/memory reporting through `MetricsProcessor` in `torchtitan/components/metrics.py`
- tokenizer bootstrapping through `HuggingFaceTokenizer` in `torchtitan/components/tokenizer.py`
- evaluation wiring through `Validator` in `torchtitan/components/validate.py`
- float8 / MXFP8 conversion helpers under `torchtitan/components/quantization/`

## Design Logic

The module is designed around "small reusable runtime subsystems" rather than one monolithic trainer helper file.

- Stateful wrappers hide multi-part training topology details.
  `OptimizersContainer`, `LRSchedulersContainer`, and `ModelWrapper` flatten multiple per-model-part objects into one trainer-facing interface. This is critical for pipeline parallel and multi-chunk models, where state dictionaries would otherwise collide or become awkward to resume.
- Checkpointing treats DCP as the source of truth, with optional Hugging Face adaptation layered on top.
  `CheckpointManager.dcp_save()` and `dcp_load()` in `torchtitan/components/checkpoint.py` use standard DCP paths for native checkpoints, then delegate to a `BaseStateDictAdapter` only when importing or exporting HF-format safetensors.
- Components are built for resharding and topology changes on resume.
  Optimizer state uses `flatten_optimizer_state_dict=True`; LR scheduler state assumes identical schedulers across shards; dataloader state is saved only once per DP rank. The common pattern is "save the minimal portable state that can survive a mesh change."
- Validation intentionally mirrors training's dataloading post-processing instead of inventing a separate evaluation contract.
  `Validator.post_dataloading_process()` duplicates the same attention-mask and context-parallel shaping logic used in the runtime so validation measures the real model path, including PP and CP branches.
- Quantization conversion preserves TorchTitan's `Module` protocol after torchao swaps classes.
  The helpers in `quantization/module_utils.py` capture custom attributes and dynamically inject protocol mixins back into converted modules so later code can still rely on `torchtitan.models.common.linear.Linear` behavior.

## Core Data Structures

The most important types are:

- `CheckpointManager.Config` in `torchtitan/components/checkpoint.py`
  Encodes checkpoint cadence, seed-checkpoint mode, async mode, HF import/export mode, FT dataloader checkpointing, retention, and load filtering.
- `ModelWrapper` in `torchtitan/components/checkpoint.py`
  Adapts one module or a list of model chunks into a flat model `state_dict()` / `load_state_dict()` interface.
- `BaseDataLoader` and `ParallelAwareDataloader` in `torchtitan/components/dataloader.py`
  Define the stateful dataloader contract and the DP-rank-aware implementation used by HF datasets.
- `OptimizersContainer` and `OptimizersInBackwardContainer` in `torchtitan/components/optimizer.py`
  Wrap one optimizer per model part or one optimizer per parameter.
- `LRSchedulersContainer` in `torchtitan/components/lr_scheduler.py`
  Wraps multiple `LambdaLR` instances and enforces the "all schedulers are equivalent" assumption needed for portable resume.
- `MetricsProcessor` in `torchtitan/components/metrics.py`
  Tracks throughput, memory, log cadence, metric backends, and validation logging.
- `BaseTokenizer` and `HuggingFaceTokenizer` in `torchtitan/components/tokenizer.py`
  Provide the repo's tokenizer abstraction and HF file-format loader.
- `BaseValidator` and `Validator` in `torchtitan/components/validate.py`
  Define validation cadence plus the actual eval loop for dense and PP models.
- `QuantizationConverter`, `Float8LinearConverter`, `Float8GroupedMMConverter`, and `MXFP8Converter`
  Represent model-mutating converter objects that swap module implementations or parameter behaviors after the model is built.

Detailed symbol inventory and file map are in `reference.md`.

## State Flow

### Runtime construction
1. `Trainer.__init__()` constructs tokenization, dataloading, loss, metrics, optimizers, schedulers, validation, converters, and checkpointing from config objects.
2. Dataloaders and tokenizers feed CPU-side batch structures into the runtime.
3. Loss, optimizer, scheduler, and quantization pieces are attached after model construction and parallelization.
4. Metrics and validation observe the live training objects after all topology decisions are made.

### Checkpoint flow
1. `CheckpointManager.__init__()` receives model parts, optimizers, LR schedulers, dataloader, extra state, and an optional `BaseStateDictAdapter`.
2. For save, `save()` decides whether to skip, stage asynchronously, or write synchronously.
3. Native DCP saves flatten model state via `_flattened_model_states_sd()`. HF exports pass through `sd_adapter.to_hf(...)` inside `dcp_save()`.
4. For load, `load()` chooses an initial path or latest step, derives the right subset with `_states_to_load()`, then calls `dcp_load()`.
5. FT mode adds `_ft_save()` / `_ft_load()` for per-replica dataloader checkpoints and installs state-dict callbacks into the FT manager.

### Batch and validation flow
1. `ParallelAwareDataloader.state_dict()` stores one serialized loader state per DP rank key like `dp_rank_0`.
2. Training consumes these batches; validation reconstructs the same forward inputs in `Validator.post_dataloading_process()`.
3. `Validator.validate()` handles dense or PP eval, counts valid tokens with `IGNORE_INDEX`, optionally all-reduces counts over DP, and logs the aggregate through `MetricsProcessor.log_validation()`.

### Metrics flow
1. `MetricsProcessor` is built with logging backend settings plus `ParallelDims`.
2. The runtime populates `num_flops_per_token`, optimizer, scheduler, and model references later.
3. Training updates `ntokens_since_last_log` and `data_loading_times`.
4. `log()` computes tps, TFLOPs, MFU, and memory peaks from `DeviceMemoryMonitor`, emits backend metrics, then resets counters and peak stats.

### Quantization flow
1. Converters are constructed from config after the model exists but before normal training loops begin.
2. `convert(model)` mutates the model in place via torchao helpers.
3. Before conversion, `verify_module_protocol()` and `capture_module_attrs()` snapshot expectations and custom attributes.
4. After conversion, `inject_module_protocol()` restores TorchTitan's module protocol and `post_optimizer_hook()` performs any required scale precomputation.

## Error Handling And Side Effects

- `ParallelAwareDataloader._validate_kwargs()` rejects invalid `StatefulDataLoader` kwargs and strips `persistent_workers` / `prefetch_factor` when `num_workers == 0`.
- `CheckpointManager.__init__()` raises on `keep_latest_k == 1`, invalid async modes, and missing `sd_adapter` for HF export.
- `CheckpointManager.load()` raises for invalid initial paths, contradictory HF settings, or requested load steps whose checkpoint directory is missing.
- `CheckpointManager._states_to_load()` raises if `exclude_from_loading` mentions a key that is not in the managed state map.
- `OptimizersContainer._resolve_optimizer_cls()` rejects unsupported optimizers; only `Adam` and `AdamW` are accepted.
- `LRSchedulersContainer.Config.build()` warns and clamps when warmup or decay spans exceed total steps, rather than failing hard.
- `HuggingFaceTokenizer._load_tokenizer_from_path()` has explicit migration and format errors, including the deprecated `./assets/tokenizer` path.
- `Validator.Config.__post_init__()` rejects invalid validation step counts.
- Quantization converters raise if `torchao` is missing, hardware capability is insufficient, or unsupported parallelism is enabled for prototype features.

The major side effects are:
- creating and closing DCP async resources, gloo process groups, purge threads, and optional staging infrastructure
- mutating the live model in place during quantization conversion
- writing TensorBoard or WandB logs under the dump folder
- mutating expert-bias tensors in `register_moe_load_balancing_hook()`

## Common Modification Scenarios

- Add a new checkpointing policy or export format:
  Start in `torchtitan/components/checkpoint.py`. Most policy changes land in `CheckpointManager.Config`, `load()`, `_states_to_load()`, `_save_last_step()`, or `dcp_save()` / `dcp_load()`. If the format is not native DCP, you will likely also need a new `BaseStateDictAdapter` implementation outside this module.
- Add a new optimizer implementation or optimizer-wide hook:
  Edit `OptimizersContainer._resolve_optimizer_cls()` and `_build_optimizer_kwargs()` in `torchtitan/components/optimizer.py`. If the new optimizer changes state-dict behavior, also revisit `state_dict()` / `load_state_dict()` and checkpoint resume assumptions.
- Change LR schedule semantics:
  Update `LRSchedulersContainer.Config.build()` in `torchtitan/components/lr_scheduler.py`. The nested `linear_warmup_stable_decay()` function is the single place that encodes warmup, stable span, decay shape, and `min_lr_factor`.
- Change how tokenizer special tokens or chat templates are inferred:
  Edit `HuggingFaceTokenizer.__init__()`, `_infer_special_tokens()`, `_infer_should_add_bos_eos()`, and `_load_tokenizer_from_path()` in `torchtitan/components/tokenizer.py`. Those methods collectively define file precedence, BOS/EOS injection policy, and supported tokenizer formats.
- Change validation input shaping or PP eval behavior:
  Edit `Validator.post_dataloading_process()` and `Validator.validate()` in `torchtitan/components/validate.py`. These are the hotspots for attention-mask construction, context-parallel preprocessing, PP stage inputs, and loss normalization.
- Add a new quantization converter:
  Follow the pattern in `quantization/float8.py` or `quantization/mx.py`: subclass `QuantizationConverter`, validate hardware in `__init__()`, mutate in `convert()`, and restore protocol invariants with `capture_module_attrs()`, `inject_module_protocol()`, and `verify_module_protocol()`.
- Change logging destinations or metric selection:
  Edit `MetricsProcessor._build_metric_logger()`, `log()`, and `log_validation()` in `torchtitan/components/metrics.py`. Rank-selection logic is shared with pipeline-parallel visibility rules through `_get_metrics_rank()` and `ensure_pp_loss_visible()`.

## File Map

- `torchtitan/components/checkpoint.py`: checkpoint policy, save/load orchestration, FT dataloader checkpointing, HF conversion hooks
- `torchtitan/components/dataloader.py`: stateful dataloader base classes and DP-aware loader state
- `torchtitan/components/loss.py`: cross-entropy and MSE loss builders with optional compile
- `torchtitan/components/lr_scheduler.py`: warmup-stable-decay scheduler container
- `torchtitan/components/metrics.py`: memory monitor, backend loggers, training and validation metric aggregation
- `torchtitan/components/optimizer.py`: optimizer containers and MoE load-balancing update hook
- `torchtitan/components/tokenizer.py`: tokenizer file loading, BOS/EOS inference, chat templates
- `torchtitan/components/validate.py`: validation loop and eval-side dataloading post-processing
- `torchtitan/components/quantization/__init__.py`: quantization base class and shared constants
- `torchtitan/components/quantization/float8.py`: float8 linear and grouped-GEMM converters
- `torchtitan/components/quantization/mx.py`: MXFP8 grouped-GEMM converter
- `torchtitan/components/quantization/module_utils.py`: protocol-preservation helpers after module swapping
- `torchtitan/components/quantization/utils.py`: FQN and shape-based module filtering

## See Also

- `reference.md` for function-by-function responsibilities, state-dict behavior, and a denser API index
