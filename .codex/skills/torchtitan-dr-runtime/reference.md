# Runtime Reference

## File Inventory
- `torchtitan/__init__.py`
- `torchtitan/train.py`
- `torchtitan/trainer.py`

## Key Types And Data Structures

### `torchtitan.__version__`
- Defined in `torchtitan/__init__.py`.
- Value comes from `importlib.metadata.version("torchtitan")`.
- Fallback is the literal string `"0.0.0+unknown"` when package metadata lookup fails.

### `Trainer.Config`
Defined in `torchtitan/trainer.py:59`.

Fields:
- `model_spec: ModelSpec | None`
  Set programmatically before trainer construction; suppressed from Tyro CLI parsing.
- `hf_assets_path: str`
  Location of tokenizer/config/checkpoint assets used by tokenizer build and optional state-dict adapters.
- `dump_folder: str`
  Base output directory for metrics, checkpoints, profiling output, and saved config files.
- `profiling: ProfilingConfig`
- `metrics: MetricsProcessor.Config`
- `tokenizer: BaseTokenizer.Config`
- `dataloader: BaseDataLoader.Config`
- `model_converters: ModelConvertersContainer.Config`
- `optimizer: OptimizersContainer.Config`
- `lr_scheduler: LRSchedulersContainer.Config`
- `training: TrainingConfig`
- `parallelism: ParallelismConfig`
- `checkpoint: CheckpointManager.Config`
- `activation_checkpoint: ActivationCheckpointConfig`
- `compile: CompileConfig`
- `comm: CommConfig`
- `validator: Validator.Config`
- `debug: DebugConfig`

Important methods:
- `__post_init__()`
  Rejects `OptimizersInBackwardContainer.Config` when EP or PP is enabled.
- `to_dict() -> dict[str, Any]`
  Serializes nested config state; special-cases `model_spec` because callables inside `ModelSpec` are not JSON-serializable.
- `maybe_log() -> None`
  Logs pretty JSON config when `debug.print_config` is set and optionally saves it under `dump_folder/debug.save_config_file`, but only rank 0 writes when distributed is initialized.

### `Trainer`
Defined in `torchtitan/trainer.py:57`. Inherits:
- `torch.distributed.checkpoint.stateful.Stateful`
- `Configurable`

Persistent runtime attributes called out in the class body:
- Core config/state: `config`, `parallel_dims`
- Swappable components: `tokenizer`, `dataloader`, `model_config`, `model_parts`, `loss_fn`, `optimizers`, `lr_schedulers`, `validator`, `metrics_processor`, `checkpointer`
- Runtime utilities: `device`, `gc_handler`, `train_context`, `gradient_accumulation_steps`, `pp_has_first_stage`, `pp_has_last_stage`
- Checkpointed counters: `step`, `ntokens_seen`

## Function And Method Reference

### `main() -> None`
File: `torchtitan/train.py:16`

Responsibilities:
- Calls `init_logger()`.
- Imports `torchtitan` and logs `torchtitan.__version__`.
- Parses runtime config via `ConfigManager.parse_args()`.
- Skips execution for `config.comm.mode == "local_tensor"`.
- Builds the trainer with `config.build()`.
- If `config.checkpoint.create_seed_checkpoint` is set, enforces `WORLD_SIZE == 1` and `config.checkpoint.enable`, then writes step-0 checkpoint through `trainer.checkpointer.save(...)`.
- Otherwise delegates execution to `trainer.train()`.
- Guarantees `trainer.close()` on both success and exception when trainer construction succeeded.
- Destroys the distributed process group on the success path if it is still initialized.

### `Trainer.__init__(config: Config)`
File: `torchtitan/trainer.py:192`

Major phases:
1. Validate `config.model_spec`.
2. Bind current device from `LOCAL_RANK` and call `device_module.set_device(...)`.
3. Initialize distributed state through `self.init_distributed()`.
4. Log config through `config.maybe_log()`.
5. Derive batch-mesh degree/rank for dataloader construction.
6. Install garbage-collection control with `utils.GarbageCollection`.
7. Seed and determinism setup with `dist_utils.set_determinism(...)`.
8. Build tokenizer and dataloader.
9. Update `model_spec.model` from trainer config and build the model on `torch.device("meta")` under the selected default dtype.
10. Build model converters, call `model_converters.convert(model)`, and then `model.verify_module_protocol()`.
11. Build metrics, compute parameter count and FLOPs/token, and log memory/FLOP context.
12. Choose initialization pathway:
    - seed checkpoint: CPU init, no buffer device
    - CPU offload: CPU init, runtime buffer device on accelerator
    - default: accelerator init
13. Build loss function.
14. Validate `global_batch_size` and compute `gradient_accumulation_steps`.
15. Apply either:
    - `model_spec.pipelining_fn(...)` plus `Decoder.init_weights(...)` over each model part, or
    - `model_spec.parallelize_fn(...)` plus `BaseModel.init_weights(...)` on a single model.
16. Build optimizers, optional `post_optimizer_build_fn`, LR schedulers, and a post-step hook into `model_converters.post_optimizer_hook(...)`.
17. Initialize checkpointed counters and build `CheckpointManager`.
18. Build train context and AMP guard.
19. Optionally build validator.

### `init_distributed() -> ParallelDims`
File: `torchtitan/trainer.py:509`

Inputs:
- `self.config.comm`
- `self.config.training.enable_cpu_offload`
- `self.config.dump_folder`
- parallel degree fields from `self.config.parallelism`

Outputs:
- Calls `dist_utils.init_distributed(...)` to initialize backend/process groups and retrieve `world_size`.
- Returns `ParallelDims(...)` with shard, replicate, CP, TP, PP, EP, and ETP degrees.

### `batch_generator(data_iterable) -> Iterator[tuple[input_dict, labels]]`
File: `torchtitan/trainer.py:529`

Behavior:
- Iterates over the dataloader on CPU.
- Measures per-batch data-loading time.
- Increments `self.ntokens_seen` by `labels.numel()`.
- Increments `self.metrics_processor.ntokens_since_last_log`.
- Appends timing data to `self.metrics_processor.data_loading_times`.
- Raises `DataloaderExhaustedError` if the dataloader ends before a full training step can finish.

### `post_dataloading_process(input_dict, labels)`
File: `torchtitan/trainer.py:559`

Outputs:
- `inputs`: `input_dict["input"]`
- `labels`: unchanged or context-parallel transformed
- `extra_inputs`: all other input tensors, not forwarded across PP stages
- `extra_kwargs`: forwarded kwargs such as `attention_masks`

Important logic:
- Removes `positions` unless `attn_mask_type == "block_causal"`.
- Builds `attention_masks` by calling `Decoder.get_attention_masks(...)` when `attn_backend` is `"flex"` or `"varlen"`.
- Calls `prepare_context_parallel_input(...)` when CP is enabled.

### `forward_backward_step(...) -> torch.Tensor`
File: `torchtitan/trainer.py:634`

PP path:
- Runs `self.pp_schedule.step(...)` inside `self.train_context()`.
- Only last PP stages receive `targets`/`losses`.
- Returns normalized PP loss as `sum(losses) / global_valid_tokens` on last stages, otherwise returns a sentinel `torch.tensor([-1.0], device=self.device)`.

Non-PP path:
- Runs one module forward under `self.train_context()` and `self.maybe_enable_amp`.
- Computes summed loss via `self.loss_fn(pred, labels)`.
- Normalizes by `global_valid_tokens`.
- Deletes `pred` before `loss.backward()` to cap peak memory.

### `train_step(data_iterator)`
File: `torchtitan/trainer.py:700`

Inputs consumed:
- Exactly `self.gradient_accumulation_steps` microbatches from `data_iterator`.

Key steps:
- Zero grads and snapshot current LR from the first scheduler.
- Count valid tokens as `(labels != IGNORE_INDEX).sum()` across microbatches.
- All-reduce valid-token count on the batch mesh when DP is enabled.
- Move each input tensor and labels tensor onto `self.device`.
- Call `forward_backward_step(...)` per microbatch and collect detached losses.
- Clip grads through `dist_utils.clip_grad_norm_(...)`.
- Wait for checkpoint staging with `self.checkpointer.maybe_wait_for_staging()`.
- Step optimizers and LR schedulers.
- If logging is due, compute:
  - `global_avg_loss`
  - `global_max_loss`
  - `global_ntokens_seen`
  using `dist_sum` / `dist_max` on the loss mesh when `parallel_dims.dp_cp_enabled`.
- Emit metrics through `self.metrics_processor.log(...)`.

### `train()`
File: `torchtitan/trainer.py:804`

Behavior:
- Restores checkpoint state with `self.checkpointer.load(step=config.checkpoint.load_step)`.
- Opens profiling and memory-snapshot context managers.
- Builds the CPU-side `data_iterator` once and reuses it.
- Main loop order per step:
  1. increment `self.step`
  2. run GC
  3. execute `train_step(...)`
  4. save checkpoint
  5. run validation if enabled and scheduled
  6. advance profilers
  7. after step 1, shrink PG timeouts with `dist_utils.set_pg_timeouts(...)`
- If rank 0, sleeps for 2 seconds at the end so slower ranks can finish cleanly.

### `should_continue_training() -> bool`
File: `torchtitan/trainer.py:862`

Condition:
- `self.step < self.config.training.steps`

### `state_dict() -> dict[str, Any]`
File: `torchtitan/trainer.py:865`

Checkpoint payload:
- `{"step": self.step, "ntokens_seen": self.ntokens_seen}`

### `load_state_dict(state_dict)`
File: `torchtitan/trainer.py:868`

Restores:
- `self.step`
- `self.ntokens_seen`

### `close() -> None`
File: `torchtitan/trainer.py:872`

Cleanup behavior:
- Calls `self.checkpointer.close()` if present.
- Calls `self.metrics_processor.close()` if present.
- Does not destroy the global process group; that remains `train.py`'s responsibility.

## Cross-File Relationships
- `torchtitan/train.py` depends on `Trainer` for all non-trivial runtime work and on `torchtitan.__version__` for logging.
- `Trainer.Config` is the config shape that `ConfigManager.parse_args()` ultimately builds toward before `config.build()` materializes a `Trainer`.
- `Trainer.state_dict()` / `load_state_dict()` make the trainer itself a checkpoint-managed state object through the `states={"train_state": self}` registration in `CheckpointManager.build(...)`.

## Modification Pointers
- Startup/CLI/teardown behavior: `torchtitan/train.py`
- Runtime assembly order and component injection: `Trainer.__init__` in `torchtitan/trainer.py`
- Distributed topology derivation: `Trainer.init_distributed`
- Data preprocessing before forward: `Trainer.post_dataloading_process`
- Numerics and accumulation semantics: `Trainer.forward_backward_step` and `Trainer.train_step`
- Loop order, save/validate cadence, profiler interaction: `Trainer.train`
- Checkpointed trainer-local state: `Trainer.state_dict` and `Trainer.load_state_dict`
