---
name: torchtitan-dr-tools
description: Use when working on torchtitan logging, profiling, device/runtime utility behavior, garbage collection control, or PyTorch environment compatibility checks
---

# TorchTitan Tools

## Overview

`torchtitan/tools` is the cross-cutting infrastructure layer for runtime concerns that do not belong to model code or trainer orchestration. It centralizes root logger setup, optional profiling and memory snapshot capture, device/runtime capability helpers, garbage collection control, simple terminal color palettes, and PyTorch-version guardrails.

The design bias is to keep the rest of the codebase thin: modules such as the trainer and distributed helpers import a few stable utilities rather than reimplementing logging, profiling, or device inspection. Most functionality is side-effectful, so the module favors module-level singletons and opt-in context managers over larger service objects.

## Module Purpose And Capabilities

Externally, the module provides three groups of capabilities:

- Logging setup and deduplicated warnings in [`torchtitan/tools/logging.py`](../../../../torchtitan/tools/logging.py)
- Profiling and memory snapshot context managers plus the `ProfilingConfig` dataclass in [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py)
- Device/runtime helpers, GC control, hardware FLOPs heuristics, color constants, PyTorch feature checks, and default-dtype scoping in [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py)

Primary public entry points:

- `init_logger() -> None`
- `warn_once(logger: logging.Logger, msg: str) -> None`
- `ProfilingConfig(...)`
- `maybe_enable_profiling(profiling_config, *, global_step=0, base_folder="", leaf_folder="")`
- `maybe_enable_memory_snapshot(profiling_config, *, global_step=0, base_folder="", leaf_folder="")`
- `has_cuda_capability(major, minor) -> bool`
- `has_rocm_capability(major, minor) -> bool`
- `get_device_info() -> tuple[str, ModuleType]`
- `GarbageCollection(gc_freq=1000, debug=False)`
- `get_peak_flops(device_name: str) -> float`
- `check_if_feature_in_pytorch(feature_name, pull_request, min_nightly_version=None) -> None`
- `set_default_dtype(dtype)`

## Core Design Logic

### Root logger, not per-module logger

[`logging.py`](../../../../torchtitan/tools/logging.py) binds `logger = logging.getLogger()` at module scope and `init_logger()` clears and rebuilds handlers. The intent is to make TorchTitan log formatting globally consistent and avoid handler duplication when different entrypoints initialize logging. `warn_once()` adds another layer of output control by deduplicating repeated warnings with the `_logged` global set.

### Profiling is opt-in and scoped

[`profiling.py`](../../../../torchtitan/tools/profiling.py) wraps expensive profiling behavior in context managers so callers can enable tracing or memory snapshots without changing the main training loop shape. `maybe_enable_profiling()` returns a configured `torch.profiler.profile(...)` only when `ProfilingConfig.enable_profiling` is true; otherwise it yields `None`. `maybe_enable_memory_snapshot()` uses the same pattern for memory history and snapshot dumps.

This design keeps the call sites simple and makes profiling overhead explicit. It also lets TorchTitan encode output naming conventions in one place: traces go under `iteration_<step>/<leaf_folder>/rank<rank>_trace.json`, while memory snapshots go under `iteration_<step>/<leaf_folder>/rank<rank>_memory_snapshot.pickle`.

### Runtime facts are discovered once, then reused

[`utils.py`](../../../../torchtitan/tools/utils.py) computes `device_type, device_module = get_device_info()` at import time. Other modules import these globals instead of repeatedly querying device backends. This reduces call-site clutter but means changes to device detection semantics affect all importers.

### Conservative fallbacks over hard failures

Several helpers prefer warnings and fallback behavior:

- `get_peak_flops()` falls back to A100 FLOPs when a device is unknown.
- `check_if_feature_in_pytorch()` warns rather than aborting when a source build or old nightly may be missing a required PR.
- `maybe_enable_memory_snapshot()` only emits an extra snapshot on `torch.OutOfMemoryError`; it does not mask the exception.

The tradeoff is clear: TorchTitan stays usable across heterogeneous environments, but accuracy of diagnostics like MFU estimation depends on the hardware table staying current.

## Core Data Structures

The main data structures are:

- `ProfilingConfig` in [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py): the configuration contract for trace profiling and memory snapshots
- `MemoryProfiler` nested inside `maybe_enable_memory_snapshot()` in [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py): a lightweight stateful helper with `step_num` and `freq`
- `GarbageCollection` in [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py): stateful GC controller for periodic or debug collections
- `Color` and `NoColor` in [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py): parallel dataclass-based color palettes whose fields are kept in sync by an assertion

See [reference.md](./reference.md) for the field-level index and function signatures.

## State Flow

### Logging flow

1. A caller imports `logger` from [`torchtitan/tools/logging.py`](../../../../torchtitan/tools/logging.py).
2. `init_logger()` sets root log level to `INFO`, clears existing handlers, installs a stdout `StreamHandler`, and applies the `[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s` formatter.
3. `init_logger()` also sets `KINETO_LOG_LEVEL=5`, suppressing verbose profiler-side logging.
4. Any caller can use `warn_once()` to emit a warning only if `msg` has not yet appeared in the `_logged` set.

### Trace profiling flow

1. A caller constructs or receives `ProfilingConfig`.
2. `maybe_enable_profiling()` checks `enable_profiling`.
3. If disabled, it yields `None`.
4. If enabled, it computes `trace_dir`, `wait`, and extra schedule parameters from optional config fields.
5. It determines the distributed `rank`, creates an `on_trace_ready` callback, and enters `torch.profiler.profile(...)`.
6. The callback writes a Chrome trace JSON file under an iteration- and rank-specific directory.
7. The returned profiler object is expected to have `.step()` invoked by the caller; `step_num` is seeded from `global_step` to support resumed runs.

### Memory snapshot flow

1. `maybe_enable_memory_snapshot()` checks `enable_memory_snapshot`.
2. If disabled, it yields `None`.
3. If enabled, it builds `snapshot_dir`, resolves `rank`, and constructs `MemoryProfiler(global_step, profile_freq)`.
4. `MemoryProfiler.__init__()` enables memory history recording through `device_module.memory._record_memory_history(...)`.
5. Callers invoke `profiler.step()` during training; only steps divisible by `freq` trigger a dump.
6. On `torch.OutOfMemoryError` escaping the context, the context manager calls `profiler.step(exit_ctx=True)` to emit an `_exit` snapshot before re-raising.

### Runtime utility flow

At import time, [`utils.py`](../../../../torchtitan/tools/utils.py) resolves `device_type` and `device_module`. Downstream code then uses:

- capability probes (`has_cuda_capability`, `has_rocm_capability`)
- environment checks (`check_if_feature_in_pytorch`)
- scoped dtype changes (`set_default_dtype`)
- housekeeping helpers (`GarbageCollection`)
- heuristic peak-FLOPs lookup (`get_peak_flops`)

## Error Handling And Side Effects

- `init_logger()` mutates global logging state and process environment.
- `warn_once()` mutates the module-global `_logged` set.
- `maybe_enable_profiling()` asserts `profile_freq >= profiler_warmup + profiler_active`.
- `maybe_enable_profiling()` and `maybe_enable_memory_snapshot()` create directories and write files.
- `MemoryProfiler` uses underscored memory APIs on `device_module.memory`, so it depends on backend support rather than a stable public interface.
- `GarbageCollection.__init__()` disables Python GC globally for the process.
- `get_peak_flops()` shells out to `lspci` when available.
- `check_if_feature_in_pytorch()` warns but never raises.
- `set_default_dtype()` always restores the previous dtype in `finally`.

## Common Modification Scenarios

### Add support for a new accelerator family

Edit [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py), primarily `get_peak_flops()`. Add a new device-name branch, decide whether `lspci`-based normalization is enough, and keep the fallback warning behavior intact. If the backend also needs capability probes, consider whether `has_cuda_capability()` or `has_rocm_capability()` should gain a sibling helper instead of overloading existing semantics.

### Change profiler output layout or metadata

Edit [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py). `trace_handler()` controls trace directory naming and `output_file`; `MemoryProfiler.step()` controls snapshot path conventions. Preserve the rank-qualified filenames unless you also update the downstream consumers that expect per-rank artifacts.

### Add a new profiling knob

Extend `ProfilingConfig` in [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py), then thread the new field into either `torch.profiler.schedule(...)`, `torch.profiler.profile(...)`, or memory snapshot behavior. The design expects `None` to mean "do not pass this optional parameter", so match the existing `additional_params` pattern when possible.

### Change log formatting or destination

Edit `init_logger()` in [`torchtitan/tools/logging.py`](../../../../torchtitan/tools/logging.py). Be careful that the function clears all handlers before rebuilding them; if you want multiple sinks, they must be re-added there. If you want warning deduplication keyed differently, update `_logged` or `warn_once()` in the same file.

### Tweak GC behavior during training

Edit `GarbageCollection` in [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py). The important behavioral choice is that GC is disabled in `__init__()` and only manually triggered afterward. If you change `run()`, review how trainer loops call it so you do not accidentally reintroduce pause-heavy automatic collections.

### Tighten or relax PyTorch feature gating

Edit `check_if_feature_in_pytorch()` in [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py). Today it is advisory only. If you convert warnings into hard failures, that changes startup semantics across the codebase.

## Modification Heuristics

- If the change writes files or touches trace cadence, start in [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py).
- If the change affects global process behavior, start in [`torchtitan/tools/logging.py`](../../../../torchtitan/tools/logging.py) or the `GarbageCollection` class in [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py).
- If the change is about hardware detection, MFU/throughput estimation, or backend capabilities, start in [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py).

