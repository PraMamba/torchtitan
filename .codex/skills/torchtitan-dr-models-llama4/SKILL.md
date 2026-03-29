---
name: torchtitan-dr-models-llama4
description: Use when adding a Llama 4 flavor, changing Llama 4 MoE or iRoPE behavior, debugging its TP/EP/FSDP sharding rules, or converting checkpoints between Hugging Face and TorchTitan.
---

# TorchTitan Llama4 Module

## Overview

`torchtitan/models/llama4` packages TorchTitan's Llama 4 implementation as a MoE-first decoder family. The module defines the Llama 4 block/model classes in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py), publishes named architecture presets plus a `ModelSpec` in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py), provides trainer-facing presets in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py), applies dense/MoE distributed layouts in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py), and converts Hugging Face checkpoints in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/state_dict_adapter.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/state_dict_adapter.py).

## Public Surface

- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py)
  - `llama4_configs`: preset `Llama4Model.Config` objects for `debugmodel`, `17bx16e`, and `17bx128e`.
  - `model_registry(flavor: str) -> ModelSpec`: returns the Llama 4 `ModelSpec`.
  - Re-exports `Llama4Model`, `compute_moe_hidden_dim`, and `Llama4TransformerBlock`.
- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py)
  - `compute_moe_hidden_dim(...)`
  - `Llama4TransformerBlock`
  - `Llama4Model`
- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py)
  - `parallelize_llama(...)`
  - `apply_non_moe_tp(...)`
  - `apply_fsdp(...)`
  - `apply_moe_ep_tp(...)`
- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/state_dict_adapter.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/state_dict_adapter.py)
  - `Llama4StateDictAdapter.to_hf(...)`
  - `Llama4StateDictAdapter.from_hf(...)`
- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py)
  - `llama4_debugmodel()`
  - `llama4_17bx16e()`
  - `llama4_17bx128e()`

## Design Logic

This module keeps single-device architecture logic separate from scaling logic. [`model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py) owns the Llama 4-specific behaviors: interleaving dense and MoE FFN layers, disabling RoPE on every Nth layer for iRoPE-style attention, generating two FlexAttention masks (`rope` and `nope`), and synchronizing runtime-dependent fields such as RoPE length and MoE backend selection. [`parallelize.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py) then applies TP, optional borrowed TP for experts, AC, compile, and FSDP/HSDP/replicate wrapping without polluting the model definition.

The architecture presets are also intentionally split from trainer presets. [`__init__.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py) encodes model-shape facts like `dim`, `n_layers`, head counts, expert counts, RoPE scaling, and the MoE hidden-dimension formula. [`config_registry.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py) adds runtime choices such as datasets, batch size, checkpoint interval, and parallel degrees. That separation lets you add or compare new Llama 4 flavors without touching the trainer plumbing.

The checkpoint adapter is model-specific because TorchTitan stores expert weights in a more execution-friendly form than Hugging Face does. [`Llama4StateDictAdapter`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/state_dict_adapter.py) handles transposes for expert down-projection weights and splitting/combining HF's `gate_up_proj` tensors into TorchTitan's separate `w1` and `w3` expert parameters.

## Core Data Structures

See [`reference.md`](/home/scbjtfy/torchtitan/.codex/skills/torchtitan-dr-models-llama4/reference.md) for the full field/index view. The important runtime types are:

- `Llama4TransformerBlock.Config`
  - Extends `TransformerBlock.Config` with `depth_init`, `every_n_layers_nope`, `interleave_moe_layer_step`, and `fixed_attn_block_size`.
  - Controls whether a block uses RoPE, whether it uses `feed_forward` or `moe`, and how FlexAttention block masks are constructed.
- `Llama4TransformerBlock`
  - Creates attention first, then either `self.moe` or `self.feed_forward`.
  - Tracks `self.moe_enabled` and computes depth-scaled `weight_init_std`.
- `Llama4Model.Config`
  - Key fields: `dim`, `n_layers`, `vocab_size`, and nested `layer`.
  - `update_from_config()` mutates RoPE max length, MoE router debug flags, grouped-GEMM enablement, DeepEP config conversion, and TP divisibility checks.
  - `get_nparams_and_flops()` delegates to `get_moe_model_nparams_and_flops(...)`.
- `Llama4Model`
  - Inherits `Decoder`.
  - Overrides `get_attention_masks(...)` to build both `rope` and `nope` masks for FlexAttention.
- `ModelSpec` from `model_registry()`
  - `parallelize_fn=parallelize_llama`
  - `pipelining_fn=pipeline_llm`
  - `build_loss_fn=build_cross_entropy_loss`
  - `post_optimizer_build_fn=register_moe_load_balancing_hook`
  - `state_dict_adapter=Llama4StateDictAdapter`

## State Flow

The normal entrypoint starts in [`config_registry.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py), where a function such as `llama4_17bx16e()` builds `Trainer.Config` with `model_spec=model_registry("17bx16e")`. When the trainer materializes that config, the `ModelSpec` resolves to the `Llama4Model.Config` preset from [`__init__.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py).

During runtime synchronization, `Llama4Model.Config.update_from_config()` in [`model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py) updates `self.rope.max_seq_len` to the runtime sequence length, disables grouped GEMM if the GPU is below SM90, rejects context parallelism because Llama 4 requires FlexAttention, injects `_debug_force_load_balance` into the router, optionally replaces `self.layer.moe` with `DeepEPMoE.Config`, and validates that TP divides both `n_heads` and `n_kv_heads`.

Inside each block, `Llama4TransformerBlock.__init__()` decides whether that layer uses RoPE and whether it uses MoE. If `layer_id % every_n_layers_nope == 0`, the block clones the attention config with `use_rope=False`; otherwise it keeps RoPE. It also alternates between `self.moe` and `self.feed_forward` using `interleave_moe_layer_step`, so only selected layers contain routed experts.

At execution time, `Llama4Model.get_attention_masks(...)` emits a dictionary with `rope` and `nope` masks, both built from causal/document masks but only the RoPE path includes `get_fixed_block_mask_mod(...)`. `GQAttention.forward()` in the shared attention module then selects `attention_masks["rope"]` or `attention_masks["nope"]` depending on `self.use_rope`.

Scaling proceeds through `parallelize_llama(...)` in [`parallelize.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py):

1. Assert `training.seq_len` is divisible by `parallel_dims.seq_len_divisor`.
2. If TP is enabled, inspect float8 converters and call `apply_non_moe_tp(...)`.
3. If EP is enabled or TP is borrowed for MoE, call `apply_moe_ep_tp(...)`.
4. If CP were requested, route attention inner modules through `apply_cp_to_attention_module(...)`; the config layer prevents this for Llama 4.
5. Apply activation checkpointing with `apply_ac(...)`.
6. Optionally compile sparse/MoE blocks with `apply_compile_sparse(...)`.
7. Apply FSDP/HSDP with `apply_fsdp(...)` or replicated DP with `apply_replicate(...)`.

Checkpoint conversion flows through `Llama4StateDictAdapter`. `from_hf()` rewrites HF names into TorchTitan names, transposes expert `down_proj` tensors, and splits `gate_up_proj` into `w1` and `w3`. `to_hf()` performs the reverse mapping, re-transposes expert tensors, and concatenates `w1` plus `w3` back into HF's combined projection.

## Error Handling And Constraints

- `compute_moe_hidden_dim(...)` intentionally differs from the dense `compute_ffn_hidden_dim(...)` order of operations; if you swap them you change real expert sizes.
- `Llama4TransformerBlock.__init__()` raises `ValueError` if `every_n_layers_nope <= 1`.
- `Llama4Model.Config.update_from_config()`:
  - warns when runtime `seq_len` exceeds the preset RoPE maximum.
  - disables grouped GEMM when `has_cuda_capability(9, 0)` is false.
  - raises `NotImplementedError` for context parallelism.
  - raises `ValueError` when TP does not divide `n_heads` or `n_kv_heads`.
- `parallelize_llama(...)` raises if `deepep` or `hybridep` is requested without EP, and raises `NotImplementedError` when those backends are combined with ETP.
- `apply_fsdp(...)` changes expert shard placement from `Shard(0)` to `Shard(1)` when the effective FSDP degree exceeds `num_experts`; [`tests/unit_tests/test_fsdp_moe_sharding.py`](/home/scbjtfy/torchtitan/tests/unit_tests/test_fsdp_moe_sharding.py) explicitly covers that behavior.

## Modification Guide

- To add a new Llama 4 architecture flavor, edit `llama4_configs` in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py), then add a matching trainer preset in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py) if you want a CLI entrypoint.
- To change which layers use RoPE or MoE, edit `Llama4TransformerBlock.Config` and `Llama4TransformerBlock.__init__()` in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py). `every_n_layers_nope`, `interleave_moe_layer_step`, and `fixed_attn_block_size` are the core control points.
- To change FlexAttention mask behavior, edit `Llama4Model.get_attention_masks(...)` in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py). Keep the `rope`/`nope` dict contract intact because shared `GQAttention` relies on it.
- To change TP, EP, DeepEP, or FSDP sharding rules, work in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py). `apply_non_moe_tp(...)`, `apply_moe_ep_tp(...)`, and `apply_fsdp(...)` are independent but compose in a fixed order.
- To support a new external checkpoint layout, update `Llama4StateDictAdapter` in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/state_dict_adapter.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/state_dict_adapter.py), especially `from_hf_map`, `to_hf(...)`, and `from_hf(...)`.
- To change runtime guardrails for unsupported hardware or parallel modes, edit `Llama4Model.Config.update_from_config()` in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py).
