---
name: torchtitan-dr-config
description: Use when changing how torchtitan loads config registries, parses CLI overrides, or defines shared training and parallelism configuration dataclasses.
---

# TorchTitan Config

## Overview

`torchtitan/config` is the configuration spine of TorchTitan. It provides the shared dataclasses that describe training and parallelism settings, the `Configurable` protocol that lets nested config objects build owning classes safely, and `ConfigManager`, which resolves `--module` and `--config` into a concrete runtime config with CLI overrides applied. Most of the repository assumes this module is the source of truth for both config shape and config-to-object construction semantics.

## Public Surface

- [torchtitan/config/__init__.py](/home/scbjtfy/torchtitan/torchtitan/config/__init__.py)
  - Exports `TORCH_DTYPE_MAP`, `Configurable`, `ConfigManager`, `TrainingConfig`, `ParallelismConfig`, `ActivationCheckpointConfig`, `CompileConfig`, `CommConfig`, `DebugConfig`.
- [torchtitan/config/configurable.py](/home/scbjtfy/torchtitan/torchtitan/config/configurable.py)
  - `Configurable`: base mixin for "nested Config dataclass + build()" components.
  - `Configurable.Config.build(**kwargs)`: builds owner class and mediates config-field vs runtime-argument overrides.
  - `Configurable.__init_subclass__()`: enforces `@dataclass(kw_only=True, slots=True)` and auto-wires config ownership.
- [torchtitan/config/configs.py](/home/scbjtfy/torchtitan/torchtitan/config/configs.py)
  - Shared dataclasses such as `TrainingConfig` and `ParallelismConfig`.
- [torchtitan/config/manager.py](/home/scbjtfy/torchtitan/torchtitan/config/manager.py)
  - `ConfigManager.parse_args()`: load registry config, apply CLI overrides through `tyro`, then validate.
  - `ConfigManager._load_config()`: resolves shorthands like `--module llama3`.
  - `ConfigManager.register_tyro_rules()`: custom parser support, currently including comma-separated `list[str]`.

## Design Logic

The core design is "configuration objects are first-class builders, not passive data bags." `Configurable.Config.build()` in [torchtitan/config/configurable.py](/home/scbjtfy/torchtitan/torchtitan/config/configurable.py) means each config instance knows its owning class and can instantiate it without handwritten factory code in every component. That keeps model, optimizer, metric, checkpoint, and converter configs consistent across the repository.

`Configurable` also enforces a strict split between config-time fields and runtime-only constructor arguments. If `build()` receives kwargs that all match config fields, it clones the config with those overrides. If kwargs are all non-config fields, it forwards them as legacy runtime constructor kwargs. Mixed usage is rejected with `TypeError` so caller mistakes fail early instead of constructing partially-mutated objects.

`ConfigManager` keeps module selection dynamic. Rather than hard-coding one config registry, `_load_config()` can resolve both supported shorthand names from `torchtitan.models` / `torchtitan.experiments` and fully qualified Python module paths. That lets experiments plug into the same CLI without modifying parser logic.

## State Flow

1. CLI parsing enters `ConfigManager.parse_args()` in [torchtitan/config/manager.py](/home/scbjtfy/torchtitan/torchtitan/config/manager.py).
2. `_load_config()` strips `--module` and `--config`, imports `<module>.config_registry`, finds the named config function, and calls it to get a default config object.
3. `tyro.cli()` re-parses the remaining args against the loaded config class, using the loaded config as defaults and `custom_registry` for custom primitive rules.
4. `_validate_config()` performs compatibility checks, especially around `hf_assets_path` migration behavior.
5. Downstream components call `.build()` on nested config objects. `Configurable.Config._replace()` clones configs, fills any `field(init=False)` slots, and verifies required build-time values were provided before the owner class is constructed.

## Core Data Structures

See [reference.md](/home/scbjtfy/torchtitan/.codex/skills/torchtitan-dr-config/reference.md) for field-level details. The most important types are:

- `TrainingConfig` in [torchtitan/config/configs.py](/home/scbjtfy/torchtitan/torchtitan/config/configs.py)
  - Batch sizing, sequence length, dtype, mixed precision, gradient clipping, GC behavior.
- `ParallelismConfig` in [torchtitan/config/configs.py](/home/scbjtfy/torchtitan/torchtitan/config/configs.py)
  - DP/HSDP/FSDP, TP, PP, CP, EP, and expert backend controls.
  - Includes pipeline schedule and microbatch configuration plus CP load-balancer settings.
- `Configurable.Config` in [torchtitan/config/configurable.py](/home/scbjtfy/torchtitan/torchtitan/config/configurable.py)
  - `_owner` class variable, `to_dict()`, `_replace()`, and `build()`.
- `ConfigManager` in [torchtitan/config/manager.py](/home/scbjtfy/torchtitan/torchtitan/config/manager.py)
  - Stateful parser wrapper that stores the resolved config on `self.config`.

## Error Handling And Side Effects

- `Configurable.__init_subclass__()` raises `TypeError` if a nested `Config` is not `kw_only` and `slots=True`.
- `Configurable.Config.build()` raises:
  - `NotImplementedError` if `_owner` was never wired.
  - `TypeError` for mixed config-field and runtime-arg kwargs.
  - `ValueError` when a caller supplies a config override that conflicts with an already-set value.
- `Configurable.Config._replace()` raises `TypeError` if any required `init=False` slot remains unset at build time.
- `ParallelismConfig.__post_init__()` rejects an empty string for `context_parallel_load_balancer`; callers must use `None` to disable it.
- `ConfigManager._load_config()` raises `ValueError` for missing `--module`/`--config`, `ImportError` for unresolved module shorthands, and `ValueError` when the named config function is absent.
- `ConfigManager._validate_config()` warns when `hf_assets_path` is missing, auto-falls back to an old tokenizer path in one migration case, and raises if the old `tokenizer.model` path is still being used.

## Modification Guide

- To add a new repository-wide config knob, first decide whether it belongs in a shared dataclass in [torchtitan/config/configs.py](/home/scbjtfy/torchtitan/torchtitan/config/configs.py) or near a single owning component. The file comment in `configs.py` documents that split explicitly.
- To add a new configurable component, subclass `Configurable` (or `Module` for `nn.Module` cases), define a nested `Config` dataclass with `kw_only=True, slots=True`, and let `__init_subclass__()` wire `build()` automatically.
- To add new CLI parsing behavior for primitive types, extend `ConfigManager.register_tyro_rules()` in [torchtitan/config/manager.py](/home/scbjtfy/torchtitan/torchtitan/config/manager.py); the existing `list[str]` comma-splitting rule is the pattern to copy.
- To support a new shorthand module namespace, change `_load_config()` in [torchtitan/config/manager.py](/home/scbjtfy/torchtitan/torchtitan/config/manager.py), where it currently searches `torchtitan.models` and `torchtitan.experiments`.
- To remove legacy constructor kwargs mode, the transition point is `Configurable.Config.build()` in [torchtitan/config/configurable.py](/home/scbjtfy/torchtitan/torchtitan/config/configurable.py), where the "old style" non-config kwargs path is still preserved behind a TODO.
