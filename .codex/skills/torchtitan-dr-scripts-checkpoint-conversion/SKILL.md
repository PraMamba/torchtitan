---
name: torchtitan-dr-scripts-checkpoint-conversion
description: Use when converting TorchTitan checkpoints between Hugging Face safetensors and DCP, wiring a new model family's state-dict adapter into the offline conversion scripts, or validating that converted checkpoints preserve model behavior.
---

# TorchTitan Scripts Checkpoint Conversion

## Overview

This module is the offline bridge between Hugging Face checkpoint layouts and TorchTitan's distributed checkpoint (DCP) format. The scripts in [`scripts/checkpoint_conversion`](../../../../scripts/checkpoint_conversion) do not implement model-specific remapping themselves; instead they instantiate a TorchTitan model on CPU, ask the selected model's `state_dict_adapter` how to translate keys and tensors, and use `torch.distributed.checkpoint` readers/writers to move weights across formats.

The module exposes three practical capabilities:
- Convert Hugging Face checkpoints into TorchTitan DCP with [`convert_from_hf()`](../../../../scripts/checkpoint_conversion/convert_from_hf.py).
- Convert TorchTitan DCP checkpoints into Hugging Face safetensors with [`convert_to_hf()`](../../../../scripts/checkpoint_conversion/convert_to_hf.py).
- Numerically validate a conversion path by comparing Hugging Face and TorchTitan outputs with KL divergence in [`numerical_tests_example.py`](../../../../scripts/checkpoint_conversion/numerical_tests_example.py).

The design is intentionally thin. These scripts assume the model package already provides a correct `model_registry()` entry and `state_dict_adapter`; the conversion layer only handles model instantiation, state-dict allocation, storage I/O, and optional dtype/export concerns.

## File Map

- [`scripts/checkpoint_conversion/convert_from_hf.py`](../../../../scripts/checkpoint_conversion/convert_from_hf.py): HF -> DCP conversion entrypoint.
- [`scripts/checkpoint_conversion/convert_to_hf.py`](../../../../scripts/checkpoint_conversion/convert_to_hf.py): DCP -> HF conversion entrypoint.
- [`scripts/checkpoint_conversion/numerical_tests_example.py`](../../../../scripts/checkpoint_conversion/numerical_tests_example.py): example correctness harness using greedy-generation parity ideas and KL divergence.
- [`scripts/checkpoint_conversion/README.md`](../../../../scripts/checkpoint_conversion/README.md): testing rationale and expected validation workflow.
- [`reference.md`](./reference.md): function signatures, key types, and per-file implementation notes.

## Core Design Logic

### Why the scripts instantiate models on CPU first

Both conversion directions need a correctly shaped TorchTitan state dict before any checkpoint reader can fill tensors. `convert_from_hf.py` and `convert_to_hf.py` both import `torchtitan.models.{model_name}`, call `model_registry(model_flavor)`, build the model config, instantiate the model under `with torch.device("cpu")`, and wrap it with `ModelWrapper`. That gives the scripts an allocated-but-empty state dict via `ModelWrapper._get_state_dict()`, which is then used as the load target.

This avoids hardcoding tensor names or shapes in the scripts. The model package remains the source of truth for:
- which model flavors exist,
- how large each parameter tensor is,
- whether a conversion path is supported at all, and
- how keys/tensors are remapped between TorchTitan and Hugging Face.

### Why conversion is adapter-driven instead of script-driven

The scripts delegate semantic remapping to `model_spec.state_dict_adapter(...)`. That adapter is responsible for format differences such as renamed keys, permutation logic, or Hugging Face sharding metadata. This keeps conversion logic colocated with the relevant model family instead of growing a central script full of model-specific branches.

The main trade-off is strict dependency on adapter quality. If a model's adapter is incomplete or `None`, the script fails immediately with an assertion rather than attempting a partial conversion.

### Why the validation example compares logits instead of raw files

[`numerical_tests_example.py`](../../../../scripts/checkpoint_conversion/numerical_tests_example.py) tests behavioral equivalence, not just file-level shape compatibility. The README explicitly calls out subtle cases such as Llama RoPE permutations where a checkpoint may load successfully but still produce different outputs. The validation flow therefore computes KL divergence between Hugging Face outputs and TorchTitan outputs on random prompts, which catches semantic mismatches the file readers alone cannot.

## State Flow

### HF -> DCP

1. [`convert_from_hf()`](../../../../scripts/checkpoint_conversion/convert_from_hf.py) imports `torchtitan.models.{model_name}` and resolves `model_spec`.
2. It builds the model config, instantiates the model on CPU, and wraps it in `ModelWrapper`.
3. It constructs `sd_adapter = model_spec.state_dict_adapter(model_config, None)`.
4. It calls `model._get_state_dict()` to allocate a TorchTitan-format state dict.
5. It converts that empty TorchTitan dict to a Hugging Face-shaped dict with `sd_adapter.to_hf(state_dict)`.
6. `dcp.load(..., storage_reader=HuggingFaceStorageReader(path=input_dir))` fills those tensors from the Hugging Face checkpoint.
7. It converts the populated dict back to TorchTitan format with `sd_adapter.from_hf(hf_state_dict)`.
8. `dcp.save(..., checkpoint_id=output_dir)` writes a DCP checkpoint directory.

The key idea is that Hugging Face weights are loaded into a temporary HF-shaped view first, then normalized back into TorchTitan layout before saving.

### DCP -> HF

1. [`convert_to_hf()`](../../../../scripts/checkpoint_conversion/convert_to_hf.py) resolves the target model package and config the same way.
2. It instantiates the model on CPU and wraps it with `ModelWrapper`.
3. It constructs `sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)`.
4. It allocates an empty TorchTitan state dict and fills it with `dcp.load(state_dict, checkpoint_id=input_dir)`.
5. It converts that dict to Hugging Face format with `sd_adapter.to_hf(state_dict)`.
6. It builds `HuggingFaceStorageWriter` using `sd_adapter.fqn_to_index_mapping` so output shards/indexes line up with HF expectations.
7. If `export_dtype` is not `"float32"`, it maps the CLI string through `TORCH_DTYPE_MAP` and casts every tensor in the HF dict.
8. `dcp.save(..., storage_writer=storage_writer)` writes the Hugging Face checkpoint.

### Numerical validation path

1. [`forward_hf()`](../../../../scripts/checkpoint_conversion/numerical_tests_example.py) loads `AutoModelForCausalLM.from_pretrained(...)`, runs greedy generation, and collects per-step logits.
2. [`forward_tt()`](../../../../scripts/checkpoint_conversion/numerical_tests_example.py) uses `ConfigManager.parse_args(["--module", model_name, "--config", config_name])` to materialize the TorchTitan model, loads a DCP checkpoint with `dcp.load`, and returns final-token logits for each prompt.
3. [`loss_fn()`](../../../../scripts/checkpoint_conversion/numerical_tests_example.py) computes KL divergence between the two output distributions.
4. The `__main__` block compares a baseline converted checkpoint with a deliberately incorrect no-permutation variant and prints average losses.

## Error Handling And Side Effects

- Both conversion scripts fail fast if `state_dict_adapter` is missing. The assertion message makes it clear that the conversion path is unsupported.
- There is no custom recovery around `importlib.import_module`, `model_registry`, `dcp.load`, or `dcp.save`; bad model names, invalid checkpoint directories, or malformed assets surface directly as upstream exceptions.
- Both conversion functions are decorated with `@torch.inference_mode()` to avoid autograd overhead during offline conversion.
- The numerical test script deletes models and calls `torch.cuda.empty_cache()` after each forward path, so its main side effects are GPU memory churn and checkpoint reads.

## Common Modification Scenarios

### Add a new model family to the offline conversion scripts

You usually do not edit this module first. Instead:
- add a `model_registry()` entry under the new model package,
- provide a non-`None` `state_dict_adapter`,
- ensure `to_hf()` and `from_hf()` both work for plain CPU tensors,
- then verify the generic scripts can target it through `--model_name` and `--model_flavor`.

The only places in this module you may need to touch are the CLI defaults in [`convert_from_hf.py`](../../../../scripts/checkpoint_conversion/convert_from_hf.py) and [`convert_to_hf.py`](../../../../scripts/checkpoint_conversion/convert_to_hf.py), plus any model-specific guidance in [`README.md`](../../../../scripts/checkpoint_conversion/README.md).

### Change how exported Hugging Face weights are written

Focus on [`convert_to_hf.py`](../../../../scripts/checkpoint_conversion/convert_to_hf.py):
- `HuggingFaceStorageWriter(...)` controls distributed save behavior, consolidation, and thread count.
- `sd_adapter.fqn_to_index_mapping` controls how fully qualified names map to Hugging Face shard indices.
- `TORCH_DTYPE_MAP[export_dtype]` and the tensor cast comprehension control the final dtype.

If you change shard layout or dtype handling, re-run a validation loop because file-level compatibility can break even when tensor names still match.

### Strengthen or generalize correctness testing

Edit [`numerical_tests_example.py`](../../../../scripts/checkpoint_conversion/numerical_tests_example.py) and [`README.md`](../../../../scripts/checkpoint_conversion/README.md):
- replace hardcoded Llama 3 defaults with parameterized CLI inputs,
- increase prompt variety or sequence length,
- add additional metrics beyond KL divergence,
- or test both directions of conversion explicitly.

Keep the behavioral-comparison structure intact: one baseline from Hugging Face, one or more TorchTitan-loaded variants, and a scalar metric that exposes subtle mapping mistakes.

### Diagnose a conversion that loads but produces wrong outputs

The first files to inspect are:
- [`numerical_tests_example.py`](../../../../scripts/checkpoint_conversion/numerical_tests_example.py) for the comparison harness,
- the target model family's `state_dict_adapter`,
- and then [`convert_from_hf.py`](../../../../scripts/checkpoint_conversion/convert_from_hf.py) or [`convert_to_hf.py`](../../../../scripts/checkpoint_conversion/convert_to_hf.py) depending on direction.

The most likely failure mode is not DCP I/O itself but an adapter mismatch such as missing permutations, wrong key translation, or bad Hugging Face asset metadata.

## Common Pitfalls

- The README refers to `numerical_test_example.py` and `example.py`, but the checked-in script is [`numerical_tests_example.py`](../../../../scripts/checkpoint_conversion/numerical_tests_example.py). Use the actual filename from the tree.
- `forward_hf()` depends on the module-level `prompt_len` defined later in `__main__`; if you refactor it into a library helper, make that dependency explicit.
- `forward_tt()` uses `@torch.no_grad` without parentheses, while the conversion scripts use `@torch.inference_mode()`. If you standardize decorator behavior, check memory and autograd implications instead of making a cosmetic-only change.
