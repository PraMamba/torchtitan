# Forge Reference

## File Inventory

- [`torchtitan/experiments/forge/README.md`](../../../torchtitan/experiments/forge/README.md)
  - Explains that forge is a lightweight subset of the full runtime intended for downstream workflows.
- [`torchtitan/experiments/forge/__init__.py`](../../../torchtitan/experiments/forge/__init__.py)
  - Re-exports `ForgeEngine` and keeps the public package surface narrow.
- [`torchtitan/experiments/forge/engine.py`](../../../torchtitan/experiments/forge/engine.py)
  - Defines `ForgeEngine` and `ForgeEngine.Config`.
- [`torchtitan/experiments/forge/example_train.py`](../../../torchtitan/experiments/forge/example_train.py)
  - Defines the example `Trainer` subclass and `main()`.

## `engine.py`

### `ForgeEngine.Config`

Key fields:
- `hf_assets_path: str = "./tests/assets/tokenizer"`
- `dump_folder: str = "./outputs"`
- `model_spec: ModelSpec`
- `optimizer: OptimizersContainer.Config`
- `lr_scheduler: LRSchedulersContainer.Config`
- `training: TrainingConfig`
- `parallelism: ParallelismConfig`
- `checkpoint: CheckpointManager.Config`
- `activation_checkpoint: ActivationCheckpointConfig`
- `compile: CompileConfig`
- `model_converters: ModelConvertersContainer.Config`
- `comm: CommConfig`
- `debug: DebugConfig`

Important behavior:
- `to_dict()` uses `dataclasses.asdict(...)`, so downstream loggers and metrics see a fully expanded nested config tree.

### `ForgeEngine.__init__(config)`

Responsibilities, in order:
1. API-usage logging through `torch._C._log_api_usage_once("torchtitan.train")`.
2. Resolve `self.device` from `LOCAL_RANK` and set the device via `utils.device_module.set_device(...)`.
3. Initialize distributed state with `dist_utils.init_distributed(...)`.
4. Build `ParallelDims` from `ParallelismConfig` and `WORLD_SIZE`.
5. Derive `dp_degree` / `dp_rank` from the `"batch"` mesh if DP is enabled.
6. Install `GarbageCollection` and deterministic seeding.
7. Pull `train_spec` and mutate `model_config` through `update_from_config(...)`.
8. Build the model on meta tensors under the training dtype.
9. Compute parameter count and FLOPs-per-token from `model_config.get_nparams_and_flops(...)`.
10. Choose `init_device` and `buffer_device` based on CPU offload.
11. Build the loss function with compile awareness.
12. Validate and derive `global_batch_size` and `gradient_accumulation_steps`.
13. Choose PP or non-PP model construction path.
14. Build optimizers, optional post-optimizer hooks, LR schedulers, checkpoint manager, train context, and AMP helper.

Notable invariants:
- `global_batch_size` must be positive.
- `global_batch_size` must be divisible by `local_batch_size * dp_degree`.
- If PP is enabled, `train_spec.pipelining_fn` must exist.

### PP branch

`train_spec.pipelining_fn(...)` receives:
- `model`
- `parallel_dims`
- `training`
- `model_converters`
- `parallelism`
- `compile_config`
- `ac_config`
- `dump_folder`
- `device`
- `model_config`
- `parallelize_fn`
- `loss_fn`

Returns:
- `pp_schedule`
- `model_parts`
- `pp_has_first_stage`
- `pp_has_last_stage`

Afterwards each part is:
- `to_empty(device=init_device)`
- `init_weights(buffer_device=buffer_device)` under `torch.no_grad()`
- `train()`

### Non-PP branch

`train_spec.parallelize_fn(...)` receives:
- `model`
- `parallel_dims`
- `training`
- `model_converters`
- `parallelism`
- `compile_config`
- `ac_config`
- `dump_folder`

Then the model is:
- `to_empty(device=init_device)`
- `init_weights(buffer_device=buffer_device)`
- `train()`
- stored as `self.model_parts = [model]`

### `ForgeEngine.close()`

Only closes `self.checkpointer` if present. Resource ownership above checkpointing belongs to subclasses or the caller.

## `example_train.py`

### `Trainer.__init__(config: TitanTrainer.Config)`

This is the canonical "superset config" pattern:
- Logs config when `config.debug.print_config` is enabled.
- Calls `super().__init__(config)` even though the type is the larger `TitanTrainer.Config`.
- Builds tokenizer if `config.tokenizer` is not `None`.
- Builds dataloader with DP world size/rank, tokenizer, seq len, and local batch size.
- Builds metrics, records `num_flops_per_token`, exposes optimizers to metrics, logs GPU peak FLOPs and reserved memory.
- Initializes `self.step = 0` before checkpoint loading.
- Builds validator if enabled, threading PP schedule metadata through when PP is active.

### `batch_generator(data_iterable)`

Behavior:
- Wraps iteration over the dataloader.
- Converts `StopIteration` into `DataloaderExhaustedError`.
- Updates `metrics_processor.ntokens_since_last_log` with `labels.numel()`.
- Appends data-loading time into `metrics_processor.data_loading_times`.
- Leaves tensors on CPU until `train_step()` moves them.

### `post_dataloading_process(input_dict, labels)`

Responsibilities:
- Splits `"input"` from auxiliary tensors.
- Tries to obtain attention masks from `self.model_parts[0].get_attention_masks(...)`.
- Stores attention masks in `extra_kwargs`, not `extra_inputs`, because PP forwards kwargs differently across stages.
- Applies `prepare_context_parallel_input(...)` when CP is enabled.

Important nuance:
- The `get_attention_masks(...)` call is guarded by `try/except TypeError`, so models that do not expose that API simply skip attention-mask injection.

### `forward_backward_step(...)`

PP path:
- Executes `self.pp_schedule.step(...)` under `self.train_context()`.
- Only first stage receives the input tensors.
- Only last stage receives `target` and collects `losses`.
- Aggregates PP losses as `sum(losses) / global_valid_tokens`.

Non-PP path:
- Runs single-model forward under `self.train_context()` and AMP.
- Computes `loss_sum = self.loss_fn(pred, labels)`.
- Normalizes to `loss = loss_sum / global_valid_tokens`.
- Deletes `pred` before backward to avoid memory spikes.

### `train_step(data_iterator)`

Responsibilities:
1. `optimizers.zero_grad()`
2. Pull `gradient_accumulation_steps` microbatches from the CPU iterator.
3. Count local valid tokens with `labels != IGNORE_INDEX`.
4. All-reduce valid tokens across the batch mesh if DP is enabled.
5. Move each microbatch to the active device.
6. Call `forward_backward_step(...)` and store detached losses.
7. Clip gradients with `dist_utils.clip_grad_norm_(...)`.
8. Wait for staged checkpoint writes if needed.
9. Step optimizers and schedulers.
10. Sum accumulated losses and log metrics when `metrics_processor.should_log(self.step)` is true.

Loss-reduction behavior:
- If `dp_cp_enabled`, the detached loss is reduced with `dist_sum` and `dist_max` over the `"loss"` mesh.
- Otherwise the local detached scalar is used directly.

### `train()`

Flow:
1. Load checkpoint state via `self.checkpointer.load(step=config.checkpoint.load_step)`.
2. Open profiling and memory snapshot contexts.
3. Iterate until `self.step < config.training.steps`.
4. Increment step, run GC, perform one train step.
5. Break cleanly if the dataloader is exhausted.
6. Optionally validate.
7. Save checkpoint every step, with `last_step=True` on the final configured step.
8. Advance profiler hooks.
9. After the first step, lower process-group timeout to `config.comm.train_timeout_seconds`.
10. Sleep two seconds on rank 0 before final completion logging.

### Checkpoint state hooks

- `state_dict()` returns `{"step": self.step}`
- `load_state_dict(state_dict)` restores `self.step`
- `close()` closes metrics first, then delegates to `ForgeEngine.close()`

### `main(custom_trainer_class=None)`

Responsibilities:
- Initialize logging.
- Log `torchtitan.__version__`.
- Parse CLI config.
- Short-circuit local tensor mode.
- Build the trainer either through the injected class or `config.build()`.
- Handle seed-checkpoint creation mode.
- Otherwise call `trainer.train()`.
- Ensure `trainer.close()` on both success and exception paths.
- Destroy the process group on the clean path if distributed is initialized.

## Cross-File Relationships

- `README.md` describes the intended boundary: forge owns constructor-time runtime assembly, while downstream users own specialized orchestration.
- `__init__.py` ensures downstream imports can use `from torchtitan.experiments.forge import ForgeEngine`.
- `engine.py` provides the reusable base class.
- `example_train.py` is both documentation-by-example and the primary modification template for downstream consumers.

## Modification Checklist

If changing engine bootstrap:
- Check `ForgeEngine.Config`
- Check `ForgeEngine.__init__()`
- Check how `example_train.Trainer` depends on the moved field

If changing data or validation flow:
- Check `Trainer.__init__()`
- Check `batch_generator()`
- Check `post_dataloading_process()`
- Check `train_step()` / `train()`

If changing checkpointed state:
- Check `ForgeEngine` state passed to `CheckpointManager`
- Check `Trainer.state_dict()`
- Check `Trainer.load_state_dict()`
- Ensure fields are initialized before checkpoint load
