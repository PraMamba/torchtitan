---
name: torchtitan-dr-models-llama3
description: Use when adding a Llama 3 model flavor, changing Llama 3 training defaults, or debugging how TorchTitan parallelizes and converts its Llama 3 implementation.
---

# TorchTitan Llama3 Module

## Overview

`torchtitan/models/llama3` is the canonical dense decoder-only language-model package in TorchTitan. It does four things together: defines the Llama 3 block/model classes, publishes named architecture presets and a `ModelSpec`, sets training defaults through `config_registry.py`, and applies TorchTitan's non-pipeline distributed techniques in `parallelize.py`. It is also the reference checkpoint-interop path for Hugging Face Llama checkpoints via `Llama3StateDictAdapter`.

## Public Surface

- [torchtitan/models/llama3/__init__.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/__init__.py)
  - `llama3_configs`: preset `Llama3Model.Config` objects for `debugmodel`, `1B`, `3B`, `8B`, `70B`, `405B`, plus `flex` and `varlen` debug/8B variants.
  - `model_registry(flavor: str) -> ModelSpec`: returns the Llama 3 `ModelSpec`.
  - Re-exports `Llama3Model`, `parallelize_llama`, and `llama3_configs`.
- [torchtitan/models/llama3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/model.py)
  - `Llama3TransformerBlock`
  - `Llama3Model`
- [torchtitan/models/llama3/parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/parallelize.py)
  - `parallelize_llama(...)`
  - `apply_tp(...)`
  - `apply_fsdp(...)`
  - `apply_replicate(...)`
  - `disable_fsdp_gradient_division(...)`
- [torchtitan/models/llama3/state_dict_adapter.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/state_dict_adapter.py)
  - `Llama3StateDictAdapter.to_hf(...)`
  - `Llama3StateDictAdapter.from_hf(...)`
- [torchtitan/models/llama3/config_registry.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/config_registry.py)
  - Top-level trainer presets such as `llama3_debugmodel()`, `llama3_8b()`, `llama3_70b()`, `llama3_405b()`, and float8/debug variants.

## Design Logic

This module is intentionally split so architecture code stays close to the single-device model, while scaling logic stays outside the model class. `Llama3Model` and `Llama3TransformerBlock` in [model.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/model.py) only know about decoder structure, RoPE synchronization constraints, weight tying, and parameter-count/FLOP reporting. All tensor/data/context parallel sharding and FSDP wrapping lives in [parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/parallelize.py), which keeps the model definition readable and reusable.

The preset map in [__init__.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/__init__.py) is also part of that design. Instead of scattering architecture constants across trainer configs, each flavor is a ready-to-use `Llama3Model.Config`. The trainer-facing `config_registry.py` then selects one of those specs and adds runtime settings like optimizer, dataset, activation checkpointing, and metrics. The separation lets you tune training defaults without touching architectural presets, or vice versa.

Checkpoint conversion is model-specific because Hugging Face and TorchTitan arrange Llama attention weights differently around RoPE. `Llama3StateDictAdapter` owns those permutations in [state_dict_adapter.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/state_dict_adapter.py) so checkpoint code elsewhere can stay model-agnostic.

## Core Data Structures

See [reference.md](/home/scbjtfy/torchtitan/.codex/skills/torchtitan-dr-models-llama3/reference.md) for a field/index view. The main runtime types are:

- `Llama3TransformerBlock.Config`
  - Extends `TransformerBlock.Config` with `depth_init: bool`.
  - `depth_init=True` makes `weight_init_std` depend on the block depth (`layer_id + 1`) instead of total layer count.
- `Llama3TransformerBlock`
  - Children: `attention`, `feed_forward`, `attention_norm`, `ffn_norm`.
  - Forward path is standard residual attention then residual MLP.
- `Llama3Model.Config`
  - Key fields: `dim`, `n_layers`, `vocab_size`, `enable_weight_tying`, `layer`.
  - `update_from_config()` enforces runtime compatibility with sequence length, context parallel, tensor parallel, and pipeline parallel.
  - `get_nparams_and_flops()` delegates to `get_dense_model_nparams_and_flops`.
- `Llama3Model`
  - Inherits `Decoder`.
  - Re-ties embedding/output weights in `init_weights()` to survive meta-device initialization.
- `ModelSpec` returned by `model_registry()`
  - `parallelize_fn=parallelize_llama`
  - `pipelining_fn=pipeline_llm`
  - `build_loss_fn=build_cross_entropy_loss`
  - `state_dict_adapter=Llama3StateDictAdapter`

## State Flow

The standard path starts in `config_registry.py`, where a function like `llama3_8b()` builds `Trainer.Config` and injects `model_spec=model_registry("8B")`. The trainer later calls the spec's model config to build `Llama3Model`, then calls `Llama3Model.Config.update_from_config()` to synchronize runtime-dependent fields such as RoPE max sequence length.

Inside `Llama3Model`, `Decoder` builds the shared decoder scaffold while this module adds Llama-specific features: per-block `depth_init`, grouped-query attention defaults, and optional embedding/output weight tying. If weight tying is enabled, the constructor aliases `tok_embeddings.weight` to `output.weight`, and `init_weights()` repeats that alias because meta initialization can break the original reference.

`parallelize_llama()` then applies scaling features in a strict order:

1. Validate `training.seq_len` against `parallel_dims.seq_len_divisor`.
2. If TP is enabled, inspect float8 converter config and call `apply_tp()`.
3. If CP is enabled, call `apply_cp_to_attention_module()` on each block's `inner_attention`.
4. Apply activation checkpointing with `apply_ac()` if enabled.
5. Apply per-block `torch.compile` with `apply_compile_dense()` if model compilation is enabled.
6. Apply FSDP/HSDP with `apply_fsdp()` or replicated data parallel with `apply_replicate()`.

Checkpoint conversion goes through `Llama3StateDictAdapter`. `from_hf()` optionally synthesizes `lm_head.weight` from `model.embed_tokens.weight` for tied-weight checkpoints, reverse-permutes Q/K projections, and renames keys into TorchTitan's layer layout. `to_hf()` does the inverse and skips `output.weight` when weight tying means Hugging Face should only keep the embedding copy.

## Error Handling And Constraints

- `Llama3Model.Config.update_from_config()`:
  - warns if runtime `seq_len` exceeds the original RoPE max.
  - raises `NotImplementedError` if context parallel is requested with `attn_backend="varlen"`.
  - raises `ValueError` if tensor parallel degree does not divide `n_heads` or `n_kv_heads`.
  - raises `NotImplementedError` if weight tying is used with pipeline parallelism.
- `parallelize_llama()` asserts sequence length divisibility by the TP/CP-dependent divisor before any sharding is applied.
- `apply_fsdp()` special-cases tied weights so embeddings/norm/output are sharded together, avoiding duplicate all-gathers.
- `disable_fsdp_gradient_division()` forces gradient scaling to happen in the training loop instead of accepting FSDP/Replicate defaults.

## Modification Guide

- To add a new Llama 3 architecture flavor, add a new `Llama3Model.Config` entry to `llama3_configs` in [torchtitan/models/llama3/__init__.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/__init__.py), then expose a trainer preset in [torchtitan/models/llama3/config_registry.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/config_registry.py) if you want a CLI entrypoint.
- To change runtime validation rules, edit `Llama3Model.Config.update_from_config()` in [torchtitan/models/llama3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/model.py). That is where CP, TP, sequence length, and PP constraints are enforced.
- To change tensor-parallel sharding behavior, work in `apply_tp()` inside [torchtitan/models/llama3/parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/parallelize.py). The embedding/root/output plan and per-block QKV/MLP plans are separate and both matter.
- To change FSDP behavior for tied or untied weights, edit `apply_fsdp()` in [torchtitan/models/llama3/parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/parallelize.py). The module-grouping decisions there are performance-sensitive.
- To support a different external checkpoint layout, update `Llama3StateDictAdapter` in [torchtitan/models/llama3/state_dict_adapter.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/state_dict_adapter.py), especially `from_hf_map`, `_permute()`, `_reverse_permute()`, `to_hf()`, and `from_hf()`.
