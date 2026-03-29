# TorchTitan Components Reference

## File Index

### `torchtitan/components/checkpoint.py`
- `AsyncMode`: checkpoint async policy enum with `disabled`, `async`, `async_with_pinned_mem`
- `ModelWrapper`: flattens one module or many model parts into one model state interface
- `purge_thread(...)`: background deletion worker for stale checkpoints
- `CheckpointManager.Config`
  - checkpoint enablement, cadence, retention, FT dataloader checkpointing
  - initial HF/native load policy
  - final-step model-only vs full-save policy
  - async and staging behavior
- `CheckpointManager.__init__(...)`
  - installs managed states
  - configures FT manager callbacks
  - initializes async checkpoint process group and purge thread
- `dcp_save(...)`
  - native DCP save or HF safetensors export
  - optionally consolidates distributed HF shards
- `dcp_load(...)`
  - native DCP load or HF safetensors import through `BaseStateDictAdapter`
- `save(curr_step, last_step=False)`
  - public save entrypoint; chooses FT save, async staging, or sync save
- `load(step=-1)`
  - public load entrypoint; resolves initial path vs latest step
- `_find_load_step(...)`
  - scans `step-*` folders and checks `.metadata` or `model.safetensors.index.json`
- `_states_to_load(model_only)`
  - computes the subset of managed state that should be loaded
- `_save_last_step(curr_step)`
  - applies final model-only / dtype conversion / HF export rules

Key managed state keys:
- `MODEL`
- `OPTIMIZER`
- `LR_SCHEDULER`
- `DATALOADER`
- `TRAIN_STATE`

Important invariants:
- `keep_latest_k` cannot be `1`
- HF save/load requires a non-`None` `sd_adapter`
- quantized HF load requires `initial_load_in_hf=True`
- FT per-replica checkpoints only store dataloader state

### `torchtitan/components/dataloader.py`
- `DataloaderExhaustedError`: special exception intentionally not derived from `StopIteration`
- `BaseDataLoader.Config`
  - `dataset`
  - `dataset_path`
- `ParallelAwareDataloader.Config`
  - `num_workers`
  - `persistent_workers`
  - `pin_memory`
  - `prefetch_factor`
- `ParallelAwareDataloader.state_dict()`
  - saves only one DP-rank-specific payload under `dp_rank_<rank>`
  - preserves backward compatibility with `pickle.dumps(...)`
- `ParallelAwareDataloader.load_state_dict()`
  - validates DP world-size consistency
  - warns if the rank-specific key is missing

### `torchtitan/components/loss.py`
- `IGNORE_INDEX = -100`
- `cross_entropy_loss(pred, labels)`
  - flattens batch/time dims and uses `reduction="sum"`
- `build_cross_entropy_loss(compile_config, **kwargs)`
  - optionally wraps the loss in `torch.compile`
- `mse_loss(pred, labels)`
  - sum-reduced float MSE with detached labels
- `build_mse_loss(...)`

### `torchtitan/components/lr_scheduler.py`
- `LRSchedulersContainer.Config`
  - `warmup_steps`
  - `total_steps`
  - `decay_ratio`
  - `decay_type`
  - `min_lr_factor`
- `Config.build(optimizers, training_steps)`
  - computes effective total, warmup, stable, and decay spans
  - returns a `LRSchedulersContainer`
- `LRSchedulersContainer`
  - builds one `LambdaLR` per optimizer
  - persists only the first scheduler state because all schedulers are assumed identical

Decay behavior:
- `linear`
- `sqrt`
- `cosine`

### `torchtitan/components/metrics.py`
- `DeviceMemoryMonitor`
  - snapshots memory capacity
  - reports peak active/reserved bytes and allocator retry / OOM counts
- `build_device_memory_monitor()`
- `BaseLogger`
- `TensorBoardLogger`
- `WandBLogger`
- `LoggerContainer`
- `ensure_pp_loss_visible(...)`
  - warns if `LOG_RANK` hides the PP-visible loss rank
- `_get_metrics_rank(...)`
  - rank 0 for non-PP or `ZBVZeroBubble`
  - first rank of the last PP stage otherwise
- `MetricsProcessor.Config`
  - log cadence
  - TensorBoard/WandB toggles
  - color printing
  - folder layout
  - rank fan-out policy
- `MetricsProcessor.log(...)`
  - computes throughput, TFLOPs, MFU, memory peaks, and data-loading time percentages
- `MetricsProcessor.log_validation(...)`
  - validation-specific loss / throughput / memory logging

Late-bound fields the runtime must set:
- `num_flops_per_token`
- `optimizers`
- `lr_schedulers`
- `model_parts`

### `torchtitan/components/optimizer.py`
- `OptimizersContainer.Config`
  - `name`
  - `lr`
  - `beta1`, `beta2`
  - `eps`
  - `weight_decay`
  - `implementation`
- `OptimizersContainer`
  - creates one optimizer per model part
  - uses DCP flattening in `state_dict()` / `load_state_dict()`
- `OptimizersInBackwardContainer`
  - creates one optimizer per parameter
  - registers `post_accumulate_grad_hook` to step/zero grads in backward
- `register_moe_load_balancing_hook(...)`
  - attaches a step pre-hook that updates `moe.expert_bias`
  - aggregates `tokens_per_expert` across the loss mesh
  - compensates for full activation-checkpoint recomputation by dividing counts by 2

Supported optimizer classes:
- `torch.optim.Adam`
- `torch.optim.AdamW`

### `torchtitan/components/tokenizer.py`
- `BaseTokenizer`
  - abstract `encode`, `decode`, `get_vocab_size`
  - `set_chat_template(...)`
  - `apply_chat_template(messages, **kwargs)`
- `HuggingFaceTokenizer`
  - file precedence:
    1. `tokenizer.json`
    2. `vocab.json` or `vocab.txt` + optional `merges.txt`
  - chat template precedence:
    1. `chat_template.jinja`
    2. inline `chat_template` in `tokenizer_config.json`
- `_infer_special_tokens()`
  - reads both top-level special-token keys and `added_tokens_decoder`
- `_infer_should_add_bos_eos()`
  - blends config flags with empirical detection on `encode("")`
- `encode(...)`
  - adds BOS/EOS only if requested and not already auto-inserted by the underlying tokenizer

Operational pitfall:
- deprecated `./assets/tokenizer` paths intentionally raise a migration-focused `FileNotFoundError`

### `torchtitan/components/validate.py`
- `BaseValidator.Config.freq`
- `Validator.Config`
  - `enable`
  - `steps`
  - `dataloader`
- `Validator.__init__(...)`
  - clones the validation dataloader config with `infinite=config.steps != -1`
- `post_dataloading_process(...)`
  - removes `positions` unless using `block_causal` attention
  - asks the model for attention masks if supported
  - applies context-parallel input sharding
- `validate(model_parts, step)`
  - handles dense eval and PP eval
  - normalizes by global valid-token count
  - restores train mode at the end

### `torchtitan/components/quantization/__init__.py`
- `QuantizationConverter`
  - base marker class for converter objects
- shared constants
  - `FP8_GROUP_ALIGNMENT_SIZE = 16`
  - `MXFP8_GROUP_ALIGNMENT_SIZE = 32`

### `torchtitan/components/quantization/float8.py`
- `Float8LinearConverter.Config`
  - `enable_fsdp_float8_all_gather`
  - `precompute_float8_dynamic_scale_for_fsdp`
  - `recipe_name`
  - `filter_fqns`
  - `emulate`
- `Float8LinearConverter.convert(model)`
  - verifies all `nn.Linear` already satisfy TorchTitan `Linear`
  - captures `_init_mean` / `_init_std`
  - uses `torchao.float8.convert_to_float8_training(...)`
  - reinjects protocol afterward
- `Float8LinearConverter.post_optimizer_hook(...)`
  - optionally precomputes dynamic scales for FSDP
- `Float8GroupedMMConverter`
  - prototype MoE grouped-GEMM float8 path
  - forbids PP and CP
  - forces token-group alignment size `16`
- `find_float8_linear_config(converters)`

Special filter behavior:
- `AUTO_FILTER_SMALL_KN_FLAG = "auto_filter_small_kn"`

### `torchtitan/components/quantization/mx.py`
- `MXFP8Converter.Config`
  - `recipe_name`
  - `fqns`
  - `pad_token_groups_for_grouped_mm`
- `MXFP8Converter.__init__(...)`
  - requires `torchao`
  - requires SM100+ capability
  - warns if `torch.compile` is disabled
- `MXFP8Converter.convert(model)`
  - quantizes targeted MoE modules using `MXFP8TrainingOpConfig`
  - restores TorchTitan module protocol after conversion

### `torchtitan/components/quantization/module_utils.py`
- `capture_module_attrs(...)`
  - saves attributes by fully-qualified module name
- `inject_module_protocol(...)`
  - dynamically creates patched subclasses that mix in TorchTitan's `Module`
- `verify_module_protocol(...)`
  - raises loudly if any target module no longer satisfies the expected protocol

### `torchtitan/components/quantization/utils.py`
- `module_filter_fn(mod, fqn, filter_fqns)`
  - only converts `nn.Linear`
  - requires both dimensions divisible by 16
  - skips FQNs containing any filter substring

## Cross-File Relationships

- `CheckpointManager` depends on:
  - `BaseDataLoader`
  - `OptimizersContainer`
  - `LRSchedulersContainer`
  - `BaseStateDictAdapter`
- `Validator` depends on:
  - `BaseDataLoader`
  - `BaseTokenizer`
  - `LossFunction`
  - `MetricsProcessor`
  - `ParallelDims`
- `MetricsProcessor` depends on:
  - `ParallelDims`
  - optional TensorBoard / WandB backends
- Quantization converters depend on:
  - `ParallelDims`
  - `torchtitan.models.common.linear.Linear`
  - `quantization/module_utils.py` protocol restoration

## Modification Checklist

When changing this module, verify the following:
- checkpoint resume still works when model parts > 1
- optimizer and LR scheduler state remain portable across topology changes
- dataloader state remains DP-rank-scoped
- validation still matches training-side attention-mask and CP shaping rules
- quantization converters still preserve `Linear` / `Module` protocol
- metrics rank selection still matches where the loss is actually visible
