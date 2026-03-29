# FT Reference

## File Map

- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/__init__.py`
  - Re-exports the manager surface used by the rest of the repo.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/config/__init__.py`
  - Re-exports `FaultTolerance` and `JobConfig`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/config/job_config.py`
  - FT dataclasses, especially Streaming DiLoCo-specific fields.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/diloco/utils.py`
  - Model fragmentation helpers.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/llama3/__init__.py`
  - `FaultTolerantModelSpec` registration for Llama3.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/llama3/config_registry.py`
  - Debug training preset for the FT Llama3 path.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/manager.py`
  - TorchFT runtime bootstrap and semi-sync context selection.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/optimizer.py`
  - Optimizer container wrapping for TorchFT.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/trainer.py`
  - FT-aware trainer implementation.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/tests/integration_tests.py`
  - FT integration test launcher.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/torchft.md`
  - User-facing launch guidance and environment model.

## Key Types And Fields

### `FaultTolerance`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/config/job_config.py`.

- Inherits all fields from `FTManager.Config`.
- `sync_steps: int = 5`
  - Semi-sync cadence for LocalSGD or DiLoCo.
- `should_quantize: bool = False`
  - Enables gradient quantization for DiLoCo sync.
- `fragment_sync_delay: int = 0`
  - Delay before blocking on fragment synchronization.
- `fragment_update_alpha: float = 0.0`
  - Mix ratio between local and global optimized parameters.
- `module_fqns_per_model_fragment: list[list[str]]`
  - Explicit FQN-based fragment definition.
- `num_fragments: int = 1`
  - Automatic fragment count when explicit FQNs are absent.

### `FTManager.Config`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/manager.py`.

- `enable: bool = False`
- `process_group: str = "gloo"`
- `process_group_timeout_ms: int = 10000`
- `replica_id: int = 0`
- `group_size: int = 0`
- `min_replica_size: int = 1`
- `semi_sync_method: str | None = None`

### `FaultTolerantTrainer.Config`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/trainer.py`.

- Inherits the entire base `Trainer.Config`.
- Adds `fault_tolerance: FaultTolerance`.

## Main Functions And Responsibilities

### `FTManager.__init__(config)`
- Builds no-op manager state if FT is disabled.
- Validates `torchft` availability.
- Creates `ft.ProcessGroupGloo` or `ft.ProcessGroupNCCL`.
- Sets `self.use_async_quorum = config.semi_sync_method is None`.
- Builds `ft.Manager(...)`.
- In async-quorum mode, creates `self.replicate_pg = ft.process_group.ManagedProcessGroup(...)` and registers `"dp_replicate"`.

### `FTManager.get_dp_info(dp_degree, dp_rank)`
- Returns adjusted `(dp_degree, dp_rank)` for dataloader partitioning.
- If FT is enabled, multiplies degree by `group_size` and offsets rank by `replica_id`.

### `FTManager.maybe_set_all_reduce_hook(model_parts)`
- Installs an FSDP all-reduce hook only when FT is enabled and async quorum is used.
- Hook does `dist.all_reduce(output, group=self.replicate_pg, op=ReduceOp.AVG)`.

### `FTManager.loss_sync_pg`
- Returns `self.replicate_pg` only in async-quorum mode.
- Returns `None` during semi-sync training, so extra replica-group loss sync is skipped.

### `maybe_semi_sync_training(ft_config, ft_manager, model, n_layers, optimizer, fragment_fn)`
- Returns `nullcontext()` when FT disabled or no semi-sync algorithm selected.
- `diloco` branch:
  - Optionally fragments the model via `fragment_fn(model, extend_ft_config, n_layers)`.
  - Builds one outer `torch.optim.SGD` per fragment.
  - Returns `local_sgd.DiLoCo(...)`.
- `local_sgd` branch:
  - Returns `local_sgd.LocalSGD(...)`.

### `module_split(model, module_fqns_per_model_fragment)`
- Reuses child modules from the original model to build fragment `nn.Module` containers.
- Handles `nn.ModuleDict` and `nn.ModuleList` specially.
- Uses dotted names like `layers.0`, `norm`, `output`.

### `fragment_llm(model, ft_config, n_layers)`
- Returns `[model]` when `num_fragments == 1` and no explicit fragment FQNs are supplied.
- Otherwise derives FQNs through `generate_llm_fqn_per_model_part(...)` and delegates to `module_split(...)`.

### `model_registry(flavor)`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/ft/llama3/__init__.py`.

- Returns `FaultTolerantModelSpec(...)` with:
  - `name="ft/llama3"`
  - `parallelize_fn=parallelize_llama`
  - `pipelining_fn=pipeline_llm`
  - `state_dict_adapter=Llama3StateDictAdapter`
  - `fragment_fn=fragment_llm`

### `llama3_ft_debugmodel()`
- Builds a full FT trainer preset.
- Key defaults:
  - `metrics.log_freq=1`
  - `training.local_batch_size=8`
  - `training.seq_len=2048`
  - `training.steps=100`
  - `checkpoint.interval=10`
  - `activation_checkpoint.mode="selective"`
  - `fault_tolerance.enable=True`
  - `fault_tolerance.semi_sync_method="diloco"`
  - `fault_tolerance.process_group="nccl"`
  - `fault_tolerance.sync_steps=10`
  - `fault_tolerance.num_fragments=2`

### `FTOptimizersContainer.__init__(config, model_parts, ft_manager)`
- Calls base optimizer construction.
- Forces optimizer-state initialization with `get_optimizer_state_dict(...)`.
- Creates `self._ft_optimizer = ft.Optimizer(ft_manager.manager, self)`.
- Stores cached optimizer state in `self.cache_state_dict`.

### `FTOptimizersContainer.state_dict()` / `load_state_dict()`
- `state_dict()` returns cached state instead of rebuilding it.
- `load_state_dict()` clears cache, calls base load, then refreshes cache with `init_cache_state_dict()`.

### `FTOptimizersContainer.step()` / `zero_grad()`
- Dispatches once through TorchFT optimizer wrapper when `_use_ft_optimizer` is true.
- Falls back to base container behavior for internal optimizer stepping paths.

### `FaultTolerantTrainer.init_distributed()`
- Builds `global_ranks` from `group_size` and `replica_id` when FT is enabled.
- Calls `dist_utils.init_distributed(..., ranks=global_ranks)`.
- Builds `self.ft_manager`.
- Returns `ParallelDims(...)` from standard parallelism config degrees.

### `FaultTolerantTrainer.__init__(config)`
- Mostly mirrors `Trainer.__init__()`.
- FT-specific differences:
  - device set before FT manager construction
  - dataloader rank/degree remapped via `ft_manager.get_dp_info(...)`
  - `build_loss_fn(..., ft_manager=self.ft_manager)`
  - metrics builder gets `ft_enable` and `ft_replica_id`
  - optimizer may be `FTOptimizersContainer`
  - checkpoint builder gets `ft_manager=self.ft_manager`
  - all-reduce hook installation via `ft_manager.maybe_set_all_reduce_hook(...)`

### `FaultTolerantTrainer.train_step(data_iterator)`
- Same accumulation structure as base trainer.
- When `parallel_dims.dp_cp_enabled`, uses `loss_sync_pg` to aggregate:
  - `global_avg_loss`
  - `global_max_loss`
  - `global_ntokens_seen`

### `FaultTolerantTrainer.train()`
- Loads checkpoint.
- Builds FT-specific profiling leaf folder `replica_<replica_id>` when enabled.
- Enters the context returned by `maybe_semi_sync_training(...)`.
- Uses the same main loop shape as the base trainer.

## Test Contract

`/home/scbjtfy/torchtitan/torchtitan/experiments/ft/tests/integration_tests.py` defines the operational test surface:

- `build_ft_test_list()` currently emits one flavor:
  - `--module ft.llama3 --config llama3_ft_debugmodel`
  - `--training.steps 10`
  - `--checkpoint.enable`
  - `ngpu=8`
- `run_single_test(...)` launches one command per replica group and currently hardcodes one `all_ranks` entry spanning GPUs `0..7`.
- `run_tests(...)` skips if available GPUs are fewer than the test flavor requires.

## Operational Notes From `torchft.md`

- FT expects multiple TorchTitan instances, one per replica group.
- `TORCHFT_LIGHTHOUSE` controls lighthouse discovery; default is `http://localhost:29510`.
- Async FT and semi-sync FT are described separately.
- The document states the lowest live replica ID is responsible for checkpoint saving.
- The semi-sync launch examples use `MODEL=llama3_ft`, but the code-path exposed by the registry is `--module ft.llama3 --config llama3_ft_debugmodel`; if updating docs, reconcile example naming with actual module/config resolution.
