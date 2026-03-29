---
name: torchtitan-dr-runtime
description: Use when tracing how torchtitan turns parsed CLI config into a live Trainer, runs the main training lifecycle, or changes top-level startup, loop, checkpoint, validation, or shutdown behavior
---

# Torchtitan Runtime

## Overview
The runtime module is the narrow top of TorchTitan's execution stack. [`torchtitan/train.py`](../../../torchtitan/train.py) is the process entrypoint: it initializes logging, parses CLI/config state through `ConfigManager`, materializes a `Trainer`, handles the seed-checkpoint shortcut, and guarantees cleanup of trainer-owned resources plus the global process group. [`torchtitan/trainer.py`](../../../torchtitan/trainer.py) is the real orchestrator: it wires together the config tree, distributed topology, tokenizer, dataloader, model construction, parallelization, optimizer/scheduler stacks, checkpointing, validation, profiling, and the step loop. [`torchtitan/__init__.py`](../../../torchtitan/__init__.py) only exposes `__version__`, but `train.py` relies on it for startup diagnostics.

## Public Surface
- `torchtitan.train.main()`: top-level executable entrypoint for a training job or seed-checkpoint creation.
- `torchtitan.trainer.Trainer`: stateful runtime object that can be built from config, loaded from checkpoint, trained, serialized, and closed.
- `torchtitan.trainer.Trainer.Config`: nested config container that aggregates all swappable runtime sub-configs.
- `Trainer.state_dict()` / `Trainer.load_state_dict()`: runtime checkpoint state for `step` and `ntokens_seen`.
- `torchtitan.__version__`: package version string resolved via `importlib.metadata.version()`, with `"0.0.0+unknown"` fallback.

## Design Logic
- `train.py` is intentionally thin. All operational branching beyond startup and guaranteed teardown is pushed into `Trainer`, which keeps the executable script stable while model/runtime behavior grows in config and component layers.
- `Trainer.Config` is a single aggregation point rather than a loose bundle of kwargs. That lets config parsing, config logging, serialization, and `config.build()` stay uniform across nested subsystems such as checkpointing, metrics, validation, and communication.
- Model construction uses meta-device initialization first, then applies converters/parallelization, then calls `to_empty()` plus `init_weights()` in the final placement path. This avoids allocating full dense weights before sharding or wrapping.
- The loop is normalized around "global valid tokens", not per-microbatch means. `train_step()` counts non-`IGNORE_INDEX` tokens across the whole gradient-accumulation window and uses that denominator in `forward_backward_step()` so gradients stay normalized across microbatches and distributed ranks.
- Data is deliberately kept on CPU until each microbatch is about to run. `batch_generator()` and `train_step()` work together to reduce GPU memory pressure during gradient accumulation.
- Pipeline parallelism is treated as a first-class alternate execution path rather than hidden behind the same single-model forward call. The runtime explicitly branches into `pp_schedule.step(...)`, tracks whether the rank owns the first/last stage, and handles PP loss visibility separately.
- Shutdown is defensive and two-tiered: `Trainer.close()` closes trainer-owned resources, while `train.py` destroys the global process group only after a successful `Trainer` close path.

## State Flow
1. Startup:
   `main()` in [`torchtitan/train.py`](../../../torchtitan/train.py) calls `init_logger()`, imports `torchtitan` for version logging, parses args via `ConfigManager.parse_args()`, and either returns early for `config.comm.mode == "local_tensor"` or builds a trainer with `config.build()`.
2. Trainer construction:
   `Trainer.__init__()` in [`torchtitan/trainer.py`](../../../torchtitan/trainer.py) sets the device from `LOCAL_RANK`, initializes distributed state through `init_distributed()`, logs config, sets determinism and GC policy, builds tokenizer/dataloader, updates the `ModelSpec` config from training config, builds the model on `torch.device("meta")`, applies model converters, verifies the module protocol, builds metrics, computes FLOPs/parameter counts, chooses initialization device/buffer policy, applies parallelization or pipelining, then builds optimizers, schedulers, checkpointing, AMP/train context, and optionally validation.
3. Training loop:
   `Trainer.train()` loads checkpoint state, enters profiling/memory-snapshot context managers, creates `data_iterator = self.batch_generator(self.dataloader)`, then loops while `should_continue_training()`. Each iteration increments `self.step`, runs GC, executes `train_step()`, saves checkpoints, optionally validates, advances profilers, and after the first successful step lowers process-group timeouts to `train_timeout_seconds`.
4. Batch path:
   `batch_generator()` pulls CPU batches, updates token counters and data-loading timing metrics, and raises `DataloaderExhaustedError` if the dataset ends mid-accumulation window.
5. Per-step path:
   `train_step()` zeroes grads, snapshots LR, collects `gradient_accumulation_steps` microbatches, counts valid tokens, all-reduces that count across the batch mesh when DP is active, moves each microbatch to device, and dispatches to `forward_backward_step()`.
6. Forward/backward path:
   `post_dataloading_process()` splits `input_dict["input"]` from auxiliary tensors, drops `positions` unless block-causal attention needs them, synthesizes `attention_masks` for `flex`/`varlen` backends, and may reshape inputs through `prepare_context_parallel_input()`. `forward_backward_step()` then either drives `self.pp_schedule.step(...)` for PP or runs the single-module forward, loss, and backward path under the train context and AMP guard.
7. Metrics and persistence:
   `train_step()` clips grads, waits for asynchronous checkpoint staging, steps optimizers/schedulers, reduces accumulated losses, computes global average/max loss plus `n_tokens_seen`, and hands them to `self.metrics_processor.log(...)`.
8. Shutdown and error handling:
   In `main()`, exceptions trigger `trainer.close()` if construction succeeded, then re-raise. On success, `trainer.close()` runs, the process group is destroyed if initialized, and `"Process group destroyed"` is logged. `Trainer.train()` also catches `DataloaderExhaustedError` to cancel the unfinished step cleanly rather than writing partial-step state.

## Error Handling And Side Effects
- `train.py` enforces two seed-checkpoint invariants before calling `trainer.checkpointer.save(curr_step=0, last_step=True)`: `WORLD_SIZE == 1` and `config.checkpoint.enable` must both hold.
- `Trainer.Config.__post_init__()` rejects optimizer-in-backward mode when expert parallelism or pipeline parallelism is enabled.
- `Trainer.__init__()` raises if `config.model_spec` is missing or if pipeline parallelism is enabled without a `model_spec.pipelining_fn`.
- `batch_generator()` converts `StopIteration` into `DataloaderExhaustedError`, and `Trainer.train()` turns that into a warning plus a clean loop exit.
- The runtime mutates persistent state in `self.step`, `self.ntokens_seen`, `metrics_processor` counters/timers, optimizer/scheduler state, checkpoint state, and optionally validation/profiling outputs under `config.dump_folder`.

## Common Modification Scenarios
- Add new top-level startup behavior before training begins:
  Put process-wide setup or preflight checks in [`torchtitan/train.py`](../../../torchtitan/train.py) near `main()`. Keep trainer construction and teardown invariants intact; if the new behavior owns resources, mirror the current `try/except/else` cleanup discipline.
- Change what gets built into a `Trainer`:
  Start in [`torchtitan/trainer.py`](../../../torchtitan/trainer.py) inside `Trainer.Config` and `Trainer.__init__()`. New runtime subsystems usually need a config field, a `build(...)` call during initialization, and optional wiring into checkpointing, metrics, or validation.
- Modify batch preprocessing or attention-mask behavior:
  Edit `Trainer.post_dataloading_process()` in [`torchtitan/trainer.py`](../../../torchtitan/trainer.py). That is the central hook for splitting `input_dict`, preserving/removing `positions`, building forwarded kwargs for PP, and handing data to context-parallel preprocessing.
- Change gradient accumulation, loss normalization, or token accounting:
  Edit `Trainer.batch_generator()`, `Trainer.forward_backward_step()`, and `Trainer.train_step()` together. These three methods jointly define when tokens are counted, how valid-token denominators are computed, and where device transfer happens.
- Change checkpoint cadence or validation timing:
  Edit the main loop in `Trainer.train()`. Checkpoint save order, validation order, profiler stepping, and timeout reduction are all controlled there.
- Extend what runtime state survives checkpoint restore:
  Update `Trainer.state_dict()` and `Trainer.load_state_dict()` in [`torchtitan/trainer.py`](../../../torchtitan/trainer.py), and make sure the new state is initialized before `self.checkpointer.load(...)` runs in `Trainer.train()`.

## File Map
- [`torchtitan/__init__.py`](../../../torchtitan/__init__.py): package version shim used for startup logging.
- [`torchtitan/train.py`](../../../torchtitan/train.py): executable bootstrap and global teardown.
- [`torchtitan/trainer.py`](../../../torchtitan/trainer.py): runtime config aggregation, distributed/model/component wiring, training loop, and checkpoint state hooks.

## See Also
- [`reference.md`](./reference.md): field inventory, function-by-function responsibilities, and exact file-path index for the runtime module.
