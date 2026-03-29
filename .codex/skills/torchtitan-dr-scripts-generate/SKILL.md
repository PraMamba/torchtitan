---
name: torchtitan-dr-scripts-generate
description: Use when working on TorchTitan's lightweight generation sanity-check scripts, especially when changing checkpoint smoke tests, prompt handling, tensor-parallel generation setup, or sampling/reporting behavior.
---

# TorchTitan Scripts Generate

## Overview

`scripts/generate` is TorchTitan's lightweight inference smoke-test module. It does not try to be a full serving or chat stack. Instead, it provides a minimal path for loading a model config, tokenizer, and distributed checkpoint, materializing the model on one or more devices, generating a short continuation from a prompt, and optionally emitting a JSON report. The module is split between a shell launcher in [`scripts/generate/run_llama_generate.sh`](../../../scripts/generate/run_llama_generate.sh), a Python entrypoint in [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py), and a tiny pure-PyTorch sampler in [`scripts/generate/_generation.py`](../../../scripts/generate/_generation.py). [`scripts/generate/README.md`](../../../scripts/generate/README.md) frames the whole package as a generation check rather than a production inference API.

## Public Surface

- [`scripts/generate/run_llama_generate.sh`](../../../scripts/generate/run_llama_generate.sh)
  - Shell wrapper around `torchrun -m scripts.generate.test_generate`
  - Reads `NGPU`, `LOG_RANK`, `MODULE`, `CONFIG`, `CHECKPOINT_DIR`, and `PROMPT` from environment variables
  - Special-cases `--prompt=...` so a file path can be expanded into prompt text
- [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py)
  - `apply_tp_minus_sp(model: nn.Module, tp_mesh: DeviceMesh)`: applies a hard-coded tensor-parallel plan without sequence parallelism
  - `test_generate(...)`: end-to-end loader/generator/reporter entrypoint
  - CLI arguments: `--module`, `--config`, `--checkpoint`, `--temperature`, `--max_new_tokens`, `--batch_size`, `--top_k`, `--seed`, `--deterministic`, `--prompt`, `--out`
- [`scripts/generate/_generation.py`](../../../scripts/generate/_generation.py)
  - `multinomial_sample_one(probs, rng=None) -> torch.Tensor`
  - `logits_to_probs(logits, temperature=1.0, top_k=None) -> torch.Tensor`
  - `generate_next_token(model, x, *, temperature=1.0, top_k=None, rng=None) -> torch.Tensor`
  - `generate(model, input_ids, *, max_new_tokens, temperature=1.0, top_k=None, seed=None) -> torch.Tensor`
- [`scripts/generate/README.md`](../../../scripts/generate/README.md)
  - Usage contract and examples for single-GPU and tensor-parallel smoke tests

## Design Logic

- The module is intentionally separate from the main trainer/runtime stack. It reuses TorchTitan config, tokenizer, checkpoint, distributed, metrics, and device helpers, but strips the flow down to "load once, generate once, report once." That keeps debugging of model assets and checkpoint compatibility cheap compared with a full training run.
- Sampling logic is isolated in [`_generation.py`](../../../scripts/generate/_generation.py) and stays framework-minimal. There is no tokenizer logic, checkpoint logic, or distributed state in that file, which makes the actual token loop easy to reason about and easy to replace.
- The Python entrypoint deliberately mirrors trainer setup only where necessary: config loading via `ConfigManager`, tokenizer building via `HuggingFaceTokenizer.Config().build(...)`, deterministic seeding via `dist_utils.set_determinism(...)`, and checkpoint restoration via `torch.distributed.checkpoint.load(...)`. That gives the smoke test high fidelity with the real runtime without requiring `Trainer`.
- Multi-GPU support is narrow by design. `test_generate.py` only applies a hand-authored TP plan in `apply_tp_minus_sp(...)`, and the plan assumes a Llama-style module layout (`tok_embeddings`, `output`, `attention.wq/wk/wv/wo`, `feed_forward.w1/w2/w3`). This script is therefore better understood as a Llama-family checkpoint check with limited TP support than as a model-agnostic generation interface.
- The script favors transparency over abstraction. It logs init device, checkpoint load duration, peak memory, colored prompt/response text, and JSON metadata from one place instead of hiding those steps behind helper classes.
- There is intentional shell/Python separation. The shell wrapper handles env-default ergonomics and prompt-file expansion; the Python module owns all actual model logic. That means CLI semantics are split across two files, and changes to prompt handling often require touching both.

## Core Data Structures

- `args` global in [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py)
  - Parsed only in `__main__`, but referenced inside `test_generate()` for the empty-prompt warning and `--out` JSON printing
  - This creates an implicit dependency between the function and CLI wrapper
- `ParallelDims` instance in [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py)
  - Built with only TP varying; DP/CP/PP/EP/ETP are forced to 1 or sentinel `-1`
  - Used only for mesh lookup and deterministic seeding
- `DeviceMemoryMonitor` product from `build_device_memory_monitor()`
  - Used before and after generation to capture peak reserved/active memory stats
- `output_data` dict in [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py)
  - Shape:
    - `metadata`: aggregate generation/memory/runtime fields
    - `responses`: per-sample `response_idx`, `input_text`, `output_text`
- Sampling tensors in [`scripts/generate/_generation.py`](../../../scripts/generate/_generation.py)
  - `input_ids`: `(B, T)` prompt tokens
  - `generated_tokens`: running decoded sequence
  - `next_token`: `(B, 1)` appended each loop
  - `rng`: optional `torch.Generator` bound to `input_ids.device`

See [`reference.md`](./reference.md) for a denser file-by-file inventory and function signature notes.

## State Flow

1. Shell entry:
   [`run_llama_generate.sh`](../../../scripts/generate/run_llama_generate.sh) reads environment defaults, strips `--prompt=...` from the raw argument list, expands the prompt from a file if the value points to an existing file, and forwards every other override to `torchrun`.
2. Python entry:
   In [`test_generate.py`](../../../scripts/generate/test_generate.py), `argparse` parses the CLI, stores the result in module-global `args`, and calls `test_generate(...)`.
3. Config and tokenizer setup:
   `test_generate(...)` initializes logging, resolves the requested model config through `ConfigManager.parse_args(["--module", model_name, "--config", config_name])`, then builds a `HuggingFaceTokenizer` from `config.hf_assets_path`.
4. Model construction:
   The script pulls `model_config = config.model_spec.model`, mutates it with `update_from_config(trainer_config=config)`, then builds the model on `"meta"` for multi-rank runs or directly on the device for single-rank runs.
5. Distributed and TP setup:
   If `WORLD_SIZE > 1`, `dist_utils.init_distributed(config.comm)` initializes distributed state, a TP-only `ParallelDims` is created, and `apply_tp_minus_sp(...)` applies rowwise/colwise TP wrappers to embeddings, output, attention projections, and MLP projections. Otherwise a degenerate `ParallelDims` is built for seed handling.
6. Materialization and checkpoint load:
   `dist_utils.set_determinism(...)` seeds the process, `model.to_empty(device=device_type)` materializes parameters, `model.init_weights()` populates tensor storage, and `dcp.load(state_dict, checkpoint_id=checkpoint_path)` overlays checkpoint values onto the model state dict.
7. Prompt preprocessing:
   The prompt is tokenized with `tokenizer.encode(prompt, add_bos=True, add_eos=False)`, reshaped to `(1, T)`, repeated `batch_size` times, and moved to `device_type`.
8. Generation loop:
   [`generate()`](../../../scripts/generate/_generation.py) clones the prompt tokens, optionally seeds a `torch.Generator`, repeatedly calls `generate_next_token(...)`, and concatenates one sampled token per step until `max_new_tokens` is reached.
9. Reporting:
   Rank 0 decodes each sample into input/output text, logs a colorized transcript, records runtime and peak memory stats, and prints JSON only when `args.out` is true.
10. Cleanup:
   After `test_generate(...)` returns, `__main__` destroys the process group if one was initialized.

## Error Handling And Side Effects

- `run_llama_generate.sh` uses `set -e`, so any failing `torchrun` or shell expansion aborts the script immediately.
- `test_generate(...)` is decorated with `@record`, so distributed launch errors are captured by TorchElastic.
- The module mutates global process state:
  - `sys.path` is extended at import time to support `from generate._generation import generate`
  - `init_logger()` resets root logging handlers
  - `device_module.set_device(device)` changes the current CUDA/XPU device
  - `dist_utils.init_distributed(...)` and `torch.distributed.destroy_process_group()` create and tear down global distributed state
- `test_generate(...)` has an implicit coupling to global `args` at lines that check `args.prompt` and `args.out`. Calling the function outside `__main__` without setting that global will fail.
- `apply_tp_minus_sp(...)` assumes `model.layers.items()` exists and that each block exposes `attention` and `feed_forward` submodules with Llama-style projection names. Using a different model family without adapting the plan will fail at attribute lookup time.
- `generate(...)` grows the full generated sequence with `torch.cat(...)` every step, so memory and latency scale linearly with the generated length and batch size.
- The `tokens_per_sec` report in [`test_generate.py`](../../../scripts/generate/test_generate.py) uses `(B * T) / elapsed_sec`, where `T` includes prompt tokens, so the throughput metric reflects total sequence length processed, not generated tokens alone.

## Common Modification Scenarios

- Add support for a non-Llama tensor-parallel generation smoke test:
  Start in `apply_tp_minus_sp(...)` in [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py). You will likely need model-family-specific sharding plans or a hook into each model's existing `parallelize.py` instead of the current hard-coded projection names.
- Change sampling behavior:
  Edit [`scripts/generate/_generation.py`](../../../scripts/generate/_generation.py). `logits_to_probs(...)` owns temperature and top-k filtering, `multinomial_sample_one(...)` owns the Gumbel-max-style sampling step, and `generate(...)` owns the autoregressive loop. This is where you would add top-p, repetition penalty, EOS stopping, or KV-cache-aware generation.
- Make `test_generate(...)` callable as a library function:
  Remove the hidden dependency on global `args` in [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py). The empty-prompt warning should use the `prompt` parameter directly, and JSON emission should be controlled by an explicit function argument instead of `args.out`.
- Change prompt ingestion semantics:
  Edit both [`run_llama_generate.sh`](../../../scripts/generate/run_llama_generate.sh) and [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py). The shell wrapper currently expands prompt files, while the Python script only sees the final string.
- Change metrics or report format:
  Edit the rank-0 reporting block in [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py). That block defines the JSON schema, colorized terminal output, and which memory/runtime fields are recorded.
- Switch checkpoint loading behavior:
  The central load path is `dcp.load(state_dict, checkpoint_id=checkpoint_path)` in [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py). If you need HF checkpoints, adapter-driven conversion, or partial load behavior, this is the handoff point.

## File Map

- [`scripts/generate/README.md`](../../../scripts/generate/README.md): framing, use cases, and example invocations
- [`scripts/generate/run_llama_generate.sh`](../../../scripts/generate/run_llama_generate.sh): env-driven `torchrun` launcher and prompt-file expansion
- [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py): config/model/tokenizer/checkpoint setup, optional TP wrapping, generation invocation, and reporting
- [`scripts/generate/_generation.py`](../../../scripts/generate/_generation.py): minimal autoregressive sampling loop

## See Also

- [`torchtitan-dr-config`](../torchtitan-dr-config/SKILL.md): how `--module` and `--config` resolve into a concrete config object
- [`torchtitan-dr-models-llama3`](../torchtitan-dr-models-llama3/SKILL.md): why the current TP sharding plan matches Llama-style module names
- [`torchtitan-dr-components`](../torchtitan-dr-components/SKILL.md): tokenizer and device-memory-monitor behaviors reused here
- [`torchtitan-dr-distributed`](../torchtitan-dr-distributed/SKILL.md): `ParallelDims`, distributed init, and determinism helpers reused here
- [`torchtitan-dr-tools`](../torchtitan-dr-tools/SKILL.md): logger and device utility behavior reused here
