---
name: torchtitan-dr-models-qwen3
description: Use when working on torchtitan's Qwen3 model family, especially when changing model flavors, trainer presets, parallelization behavior, or Hugging Face checkpoint conversion
---

# TorchTitan Qwen3 Module

## Overview
`torchtitan/models/qwen3` packages the Qwen3 family as a `ModelSpec`-driven module that can be selected by the core trainer. It owns four things: the flavor catalog in [torchtitan/models/qwen3/__init__.py](torchtitan/models/qwen3/__init__.py), the actual decoder/block implementation in [torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py), the non-pipeline parallelization stack in [torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py), and HF checkpoint translation in [torchtitan/models/qwen3/state_dict_adapter.py](torchtitan/models/qwen3/state_dict_adapter.py). Runtime presets for a subset of flavors live in [torchtitan/models/qwen3/config_registry.py](torchtitan/models/qwen3/config_registry.py).

Externally, the public entry points are:
- `model_registry(flavor)` in [torchtitan/models/qwen3/__init__.py](torchtitan/models/qwen3/__init__.py): returns a `ModelSpec` wired to Qwen3 model build, pipelining, loss, parallelization, and state-dict conversion.
- `qwen3_configs` in [torchtitan/models/qwen3/__init__.py](torchtitan/models/qwen3/__init__.py): catalog of dense and MoE architecture flavors.
- `parallelize_qwen3(...)` in [torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py): applies TP, EP, CP, activation checkpointing, compile, and FSDP/HSDP/DDP-style wrapping.
- `Qwen3StateDictAdapter` in [torchtitan/models/qwen3/state_dict_adapter.py](torchtitan/models/qwen3/state_dict_adapter.py): converts checkpoints between TorchTitan and HF layouts, including grouped-expert MoE weights.
- Trainer presets such as `qwen3_debugmodel`, `qwen3_0_6b`, and `qwen3_moe_debug` in [torchtitan/models/qwen3/config_registry.py](torchtitan/models/qwen3/config_registry.py).

## Design Logic
The module is intentionally thin around shared infrastructure. `Qwen3Model` subclasses the common `Decoder`, and `Qwen3TransformerBlock` subclasses the common `TransformerBlock`, so the Qwen3-specific code only overrides behavior that differs from the shared decoder stack: depth-aware init, optional MoE blocks, weight-tying rules, and runtime constraints in `Config.update_from_config()` ([torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)).

The flavor catalog is data-first. Nearly all architecture differences are encoded as `Qwen3Model.Config` instances in `qwen3_configs` instead of branching code paths. That keeps the trainer API stable: `model_registry(flavor)` simply selects a config and plugs it into the same `ModelSpec` skeleton ([torchtitan/models/qwen3/__init__.py](torchtitan/models/qwen3/__init__.py)).

Parallelization is staged in dependency order. `parallelize_qwen3()` first checks sequence-length divisibility, then applies tensor-parallel sharding, then MoE EP/ETP, then context parallel, activation checkpointing, compile, and finally FSDP/HSDP or replica wrapping ([torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py)). That order matters:
- TP and EP need the unwrapped module structure.
- AC must wrap blocks before compile.
- Compile is turned on before FSDP so the block graph is finalized first.
- FSDP/HSDP is last because it changes parameter ownership and sharding.

Checkpoint conversion is similarly split between architecture-agnostic and Qwen3-specific logic. `Qwen3StateDictAdapter` inherits `MoEStateDictAdapter` so it can reuse grouped-expert split/concat helpers while only defining the Qwen3 name mapping and weight-tying special cases ([torchtitan/models/qwen3/state_dict_adapter.py](torchtitan/models/qwen3/state_dict_adapter.py)).

## State Flow
### Build and train
1. Core training selects a preset such as `qwen3_debugmodel()` or `qwen3_14b()` from [torchtitan/models/qwen3/config_registry.py](torchtitan/models/qwen3/config_registry.py).
2. That preset calls `model_registry(flavor)`, which returns a `ModelSpec` with:
   `model=qwen3_configs[flavor]`, `parallelize_fn=parallelize_qwen3`, `pipelining_fn=pipeline_llm`, `build_loss_fn=build_cross_entropy_loss`, and `state_dict_adapter=Qwen3StateDictAdapter` ([torchtitan/models/qwen3/__init__.py](torchtitan/models/qwen3/__init__.py)).
3. `Qwen3Model.Config.update_from_config()` mutates the selected config with trainer-derived settings:
   rope max length is synchronized to `training.seq_len`, MoE router debug load balancing is toggled from `trainer_config.debug`, and invalid PP/TP/CP combinations are rejected early ([torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)).
4. `Qwen3Model` builds a shared decoder stack. Each `Qwen3TransformerBlock` chooses either dense feed-forward or MoE execution based on `moe_enabled`, then applies residual attention followed by residual FFN/MoE in `forward()` ([torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)).
5. `parallelize_qwen3()` transforms the built model for distributed execution before training starts ([torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py)).

### Forward/inference path
Within a block, the input path is:
`attention_norm(x)` -> `attention(...)` -> residual add -> `ffn_norm(x)` -> either `feed_forward(...)` or `moe(...)` -> residual add ([torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)).

### Checkpoint import/export path
HF checkpoint names are mapped through `from_hf_map` in [torchtitan/models/qwen3/state_dict_adapter.py](torchtitan/models/qwen3/state_dict_adapter.py).
- `from_hf()` maps dense layers directly, injects `lm_head.weight` from embeddings when weight tying is enabled and HF omitted the head, and concatenates per-expert HF tensors into grouped expert tensors for MoE models.
- `to_hf()` does the reverse, splitting grouped expert weights into per-expert HF keys. When the source is a `DTensor`, it records placements, shapes, and meshes so a later `from_hf()` call can rebuild distributed grouped weights correctly.

## Error Handling And Constraints
- `Qwen3Model.Config.update_from_config()` logs a warning when `training.seq_len` exceeds the original rope max and then still updates `rope.max_seq_len` to the requested sequence length ([torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)).
- Context parallel with `attn_backend == "varlen"` raises `NotImplementedError`; only SDPA and FlexAttention are accepted ([torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)).
- Weight tying with pipeline parallel raises `NotImplementedError` ([torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)).
- Tensor parallel degree must divide both `n_heads` and `n_kv_heads`; in other words, TP must divide both attention head counts before training starts, or `ValueError` is raised ([torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)).
- `parallelize_qwen3()` asserts that sequence length is divisible by the TP/CP-derived `seq_len_divisor`, and explicitly raises `RuntimeError("Async TP requires torch.compile")` when async TP is requested without model compile ([torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py)).

## Modification Guide
### Add a new dense Qwen3 flavor
Edit [torchtitan/models/qwen3/__init__.py](torchtitan/models/qwen3/__init__.py):
- Add a new `Qwen3Model.Config` entry to `qwen3_configs`.
- Set `enable_weight_tying`, `dim`, `n_layers`, attention heads, FFN hidden size, and rope settings.
Then add a runnable trainer preset in [torchtitan/models/qwen3/config_registry.py](torchtitan/models/qwen3/config_registry.py) if you want CLI selection through `--config`.

### Add or change a MoE Qwen3 flavor
Start in [torchtitan/models/qwen3/__init__.py](torchtitan/models/qwen3/__init__.py):
- Set `Qwen3TransformerBlock.Config(moe_enabled=True)`.
- Populate `moe=MoE.Config(...)` and its `TokenChoiceTopKRouter.Config(...)`.
Then check three other places:
- [torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py) for router debug forcing and block execution.
- [torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py) for EP/ETP behavior.
- [torchtitan/models/qwen3/state_dict_adapter.py](torchtitan/models/qwen3/state_dict_adapter.py) for any HF naming differences if the MoE layout changes.

### Change parallelization behavior
Edit [torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py).
- Global ordering and feature compatibility live in `parallelize_qwen3()`.
- Dense-only TP sharding lives in `apply_non_moe_tp()`.
- MoE EP/ETP behavior is delegated through `apply_moe_ep_tp()` from the Llama4 parallelization utilities, so cross-module coordination may be required.
When changing layouts, re-check the divisibility constraints in `Qwen3Model.Config.update_from_config()` because those guardrails are the first line of failure reporting.

### Change weight tying or initialization
Edit [torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py).
- Constructor-time tying is in `Qwen3Model.__init__()`.
- Meta-device-safe re-tying and the reason for smaller output-layer init handling live in `Qwen3Model.init_weights()`.
- Block-level init standard deviation logic is in `Qwen3TransformerBlock.__init__()` and `init_weights()`.

### Change HF checkpoint compatibility
Edit [torchtitan/models/qwen3/state_dict_adapter.py](torchtitan/models/qwen3/state_dict_adapter.py).
- Rename mappings in `from_hf_map`.
- Dense conversions are handled in the generic `layers` and non-layer branches.
- MoE expert packing and unpacking are the sensitive paths in `to_hf()` and `from_hf()`.

## Quick Pointers
- Architecture and flavor catalog: [torchtitan/models/qwen3/__init__.py](torchtitan/models/qwen3/__init__.py)
- Model behavior and runtime validation: [torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)
- Distributed transformation entry point: [torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py)
- HF conversion logic: [torchtitan/models/qwen3/state_dict_adapter.py](torchtitan/models/qwen3/state_dict_adapter.py)
- Runnable trainer presets: [torchtitan/models/qwen3/config_registry.py](torchtitan/models/qwen3/config_registry.py)

See `reference.md` for the full flavor table, key type inventory, and API/index details.
