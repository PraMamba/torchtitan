# Config Reference

## `TORCH_DTYPE_MAP`

Defined in [torchtitan/config/__init__.py](/home/scbjtfy/torchtitan/torchtitan/config/__init__.py).

- `"float16"` -> `torch.float16`
- `"float32"` -> `torch.float32`
- `"bfloat16"` -> `torch.bfloat16`

This is the shared string-to-dtype bridge used by training and distributed code.

## `TrainingConfig`

Defined in [torchtitan/config/configs.py](/home/scbjtfy/torchtitan/torchtitan/config/configs.py).

- `local_batch_size`: per-device batch size.
- `global_batch_size`: if left at `-1`, downstream code interprets it as local batch size times DP degree.
- `seq_len`: sequence length.
- `max_norm`: gradient clipping threshold.
- `steps`: total training steps.
- `enable_cpu_offload`: FSDP CPU offload switch.
- `dtype`: full training dtype (`"bfloat16"` or `"float32"`).
- `mixed_precision_param`: parameter dtype for FSDP/autocast mixed precision.
- `mixed_precision_reduce`: reduction dtype for FSDP; currently `"float32"`.
- `gc_freq`: periodic GC interval.
- `gc_debug`: per-step GC debugging mode.

## `ParallelismConfig`

Defined in [torchtitan/config/configs.py](/home/scbjtfy/torchtitan/torchtitan/config/configs.py).

Important clusters:

- Data parallel
  - `data_parallel_replicate_degree`
  - `data_parallel_shard_degree`
  - `fsdp_reshard_after_forward`
- Tensor / loss parallel
  - `tensor_parallel_degree`
  - `disable_loss_parallel`
  - `enable_async_tensor_parallel`
- Pipeline parallel
  - `pipeline_parallel_degree`
  - `module_fqns_per_model_part`
  - `pipeline_parallel_first_stage_less_layers`
  - `pipeline_parallel_last_stage_less_layers`
  - `pipeline_parallel_layers_per_stage`
  - `pipeline_parallel_schedule`
  - `pipeline_parallel_schedule_csv`
  - `pipeline_parallel_microbatch_size`
  - `pipeline_parallel_expert_parallel_overlap`
- Context parallel
  - `context_parallel_degree`
  - `context_parallel_load_balancer`
  - `context_parallel_rotate_method`
- Expert parallel
  - `expert_parallel_degree`
  - `expert_tensor_parallel_degree`
  - `expert_parallel_comm_backend`

Behavioral notes:

- `data_parallel_shard_degree = -1` means "use leftover ranks after other dimensions".
- `context_parallel_load_balancer` must be `None` or a named strategy such as `"headtail"` / `"ptrr"`; empty string is rejected in `__post_init__()`.
- `expert_parallel_comm_backend` selects `"standard"`, `"deepep"`, or `"hybridep"`, with comments documenting the hardware and install expectations.

## `Configurable.Config`

Defined in [torchtitan/config/configurable.py](/home/scbjtfy/torchtitan/torchtitan/config/configurable.py).

Key methods:

- `__repr__()`: safe repr for dataclasses containing unset `field(init=False)` slots.
- `to_dict()`: recursive serializer that skips unset build-time-only fields.
- `_replace(**overrides)`: clone config, fill `init=False` fields, and verify none remain unset.
- `build(**kwargs)`: instantiate the owner class.

`build()` modes:

- No kwargs: build directly from a cloned config.
- All kwargs match config fields: absorb them into the cloned config after conflict checks.
- All kwargs are non-config fields: forward to constructor as legacy runtime kwargs.
- Mixed kwargs: raise `TypeError`.

## `ConfigManager`

Defined in [torchtitan/config/manager.py](/home/scbjtfy/torchtitan/torchtitan/config/manager.py).

Key methods:

- `parse_args(args=sys.argv[1:])`
  - Load default config from registry.
  - Apply `tyro` parsing with custom registry.
  - Validate resulting config.
- `_load_config(args)`
  - Supports `--module X` and `--module=X`.
  - Supports `--config X` and `--config=X`.
  - Resolves shorthands against `_supported_models` and `_supported_experiments`.
  - Falls back to fully-qualified imports.
- `_validate_config()`
  - Handles `hf_assets_path` migration behavior.
- `register_tyro_rules(registry)`
  - Adds comma-separated `list[str]` parsing.

Deprecated helper:

- `_merge_configs(base, custom)`: legacy dataclass-merging path retained with a deprecation warning.
