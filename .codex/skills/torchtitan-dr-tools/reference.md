# Tools Reference

## File Index

- [`torchtitan/tools/logging.py`](../../../../torchtitan/tools/logging.py)
- [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py)
- [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py)

## Public API Surface

### `torchtitan/tools/logging.py`

- `logger = logging.getLogger()`
- `init_logger() -> None`
- `warn_once(logger: logging.Logger, msg: str) -> None`

Key state:

- `_logged: set[str]`: tracks which warning messages have already been emitted through `warn_once()`

Behavior notes:

- `init_logger()` clears all root handlers before installing a single stdout handler.
- Formatter string: `[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Side effect: sets `os.environ["KINETO_LOG_LEVEL"] = "5"`

### `torchtitan/tools/profiling.py`

#### `ProfilingConfig`

Defined in [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py) as `@dataclass(kw_only=True, slots=True)`.

Fields:

- `enable_profiling: bool = False`
- `save_traces_folder: str = "profile_traces"`
- `profile_freq: int = 10`
- `profiler_active: int = 1`
- `profiler_warmup: int = 3`
- `profiler_repeat: int | None = None`
- `profiler_skip_first: int | None = None`
- `profiler_skip_first_wait: int | None = None`
- `enable_memory_snapshot: bool = False`
- `save_memory_snapshot_folder: str = "memory_snapshot"`

#### `maybe_enable_profiling(...)`

Signature:

```python
maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
)
```

Important internal logic:

- Computes `trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)`
- Computes `wait = profile_freq - (active + warmup)`
- Asserts `wait >= 0`
- Builds `additional_params` from non-`None` values of `profiler_repeat`, `profiler_skip_first`, and `profiler_skip_first_wait`
- Uses `torch.distributed.get_rank()` for rank-specific filenames
- Enables CPU activity plus CUDA or XPU activity when available
- Seeds `torch_profiler.step_num = global_step`

Trace output path:

- `iteration_<step>/<leaf_folder>/rank<rank>_trace.json`

#### `maybe_enable_memory_snapshot(...)`

Signature:

```python
maybe_enable_memory_snapshot(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
)
```

Nested helper:

- `MemoryProfiler(step_num: int, freq: int)`
  - `__init__()` calls `device_module.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)`
  - `step(exit_ctx: bool = False)` increments `step_num`, optionally skips non-frequency steps, and writes a pickle snapshot

Constants:

- `MEMORY_SNAPSHOT_MAX_ENTRIES = 100000`

Snapshot output paths:

- periodic: `iteration_<step>/<leaf_folder>/rank<rank>_memory_snapshot.pickle`
- OOM exit: `iteration_<step>_exit/<leaf_folder>/rank<rank>_memory_snapshot.pickle`

### `torchtitan/tools/utils.py`

#### Device and capability helpers

- `has_cuda_capability(major: int, minor: int) -> bool`
- `has_rocm_capability(major: int, minor: int) -> bool`
- `get_device_info() -> tuple[str, ModuleType]`

Module globals:

- `device_type`
- `device_module`

`get_device_info()` logic:

- uses `torch._utils._get_available_device_type() or "cuda"`
- resolves backend module with `torch._utils._get_device_module(device_type)`

#### GC control

- `class GarbageCollection`
  - `__init__(gc_freq: int = 1000, debug: bool = False)`
  - `run(step_count: int)`
  - `collect(reason: str, generation: int = 1)` as `@staticmethod`

Behavior:

- asserts `gc_freq > 0`
- disables Python GC in `__init__()`
- forces an initial collection
- in debug mode, imports `torch.utils.viz._cycles.warn_tensor_cycles` and only calls it on distributed rank 0
- in non-debug mode, only performs periodic collection when `step_count > 1` and divisible by `gc_freq`

#### Hardware heuristics

- `get_peak_flops(device_name: str) -> float`

Handled device-name branches:

- `A100`
- `H100` with special cases for `NVL` and `PCIe`
- `H200`
- `H20`
- `GB200` / `GB300`
- `B200`
- `MI355X`
- `MI300X` / `MI325X`
- `MI250X`
- `Data Center GPU Max 1550` (PVC)
- `l40s`
- `neuron` with sub-branches for `trn1`, `trn1n`, `inf2`, `trn2`, `trn2n`, `trn2u`, `trn3`, `trn3u`
- fallback to A100

Special behavior:

- attempts `subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)` to refine H100 detection
- logs warnings on missing `lspci`, unknown neuron device, or unknown accelerator

#### Color palettes

- `@dataclass(frozen=True) class Color`
- `@dataclass(frozen=True) class NoColor`

Shared field names:

- `black`
- `red`
- `green`
- `yellow`
- `blue`
- `magenta`
- `cyan`
- `white`
- `reset`
- `orange`
- `turquoise`

Invariant:

- `assert set(NoColor.__dataclass_fields__.keys()) == set(Color.__dataclass_fields__.keys())`

#### Compatibility and dtype helpers

- `check_if_feature_in_pytorch(feature_name: str, pull_request: str, min_nightly_version: str | None = None) -> None`
- `set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]`
- `_round_up(x: int, y: int) -> int`

Behavior notes:

- `check_if_feature_in_pytorch()` warns if `torch.__version__` contains `"git"` or if a non-`None` `min_nightly_version` is newer than the current `torch.__version__`
- `set_default_dtype()` always restores the previous default dtype in `finally`
- `_round_up()` returns the nearest multiple of `y` greater than or equal to `x`

## Cross-File Relationships

- [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py) imports `logger` from [`torchtitan/tools/logging.py`](../../../../torchtitan/tools/logging.py)
- [`torchtitan/tools/profiling.py`](../../../../torchtitan/tools/profiling.py) imports `device_module` from [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py)
- [`torchtitan/tools/utils.py`](../../../../torchtitan/tools/utils.py) imports `logger` from [`torchtitan/tools/logging.py`](../../../../torchtitan/tools/logging.py)

The dependency direction is intentionally simple:

- `logging.py` has no TorchTitan-internal dependency
- `utils.py` depends on `logging.py`
- `profiling.py` depends on both `logging.py` and `utils.py`
