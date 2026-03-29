---
name: torchtitan-dr-models-deepseek-v3
description: Use when changing TorchTitan's DeepSeek-V3 model family, especially if work touches MLA attention shapes, DeepSeek flavor registration, distributed parallelization guards, or Hugging Face checkpoint conversion
---

# TorchTitan DeepSeek-V3 Module

## Overview
`torchtitan/models/deepseek_v3` is the self-contained DeepSeek-V3 integration layer inside TorchTitan. It defines the model-specific MLA attention and transformer block implementation in [torchtitan/models/deepseek_v3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/model.py), publishes concrete architecture flavors and the `ModelSpec` entrypoint in [torchtitan/models/deepseek_v3/__init__.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/__init__.py), supplies runnable `Trainer.Config` presets in [torchtitan/models/deepseek_v3/config_registry.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/config_registry.py), wires the model into TorchTitan distributed features in [torchtitan/models/deepseek_v3/parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/parallelize.py), and translates weights to and from Hugging Face layouts in [torchtitan/models/deepseek_v3/state_dict_adapter.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/state_dict_adapter.py).

## Public Surface
- `deepseekv3_configs` in [torchtitan/models/deepseek_v3/__init__.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/__init__.py) publishes architecture flavors: `debugmodel`, `debugmodel_flex_attn`, `16B`, `236B`, `671B`.
- `model_registry(flavor)` returns the DeepSeek-V3 `ModelSpec` with:
  `parallelize_fn=parallelize_deepseekv3`, `pipelining_fn=pipeline_llm`, `build_loss_fn=build_cross_entropy_loss`, `post_optimizer_build_fn=register_moe_load_balancing_hook`, and `state_dict_adapter=DeepSeekV3StateDictAdapter`.
- `deepseek_v3_debugmodel`, `deepseek_v3_debugmodel_flex_attn`, `deepseek_v3_16b`, and `deepseek_v3_671b` are the supported training presets.

## Design Logic
The module is split so model semantics stay separate from system concerns. `model.py` owns DeepSeek-specific math and shape logic, while `parallelize.py` owns layout decisions and feature gating. That keeps single-device model code readable and lets the distributed stack evolve without rewriting the forward path.

The central DeepSeek-specific choice is the custom `Attention` class rather than reuse of shared attention blocks. It implements Multi-head Latent Attention with two query modes: direct projection through `wq` when `q_lora_rank == 0`, or low-rank query projection through `wq_a -> q_norm -> wq_b` when `q_lora_rank > 0`. Keys and values always follow the latent path `wkv_a -> kv_norm -> wkv_b`, with RoPE applied only to the rotary slice before concatenation back into the full head representation.

Attention backend handling is deliberately narrow. `Attention.__init__()` in [torchtitan/models/deepseek_v3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/model.py) only accepts `"flex"` and `"sdpa"` and rejects `"varlen"` outright. `Attention.forward()` then enforces different mask contracts per backend: FlexAttention requires a `BlockMask`, while the SDPA path asserts that `attention_masks is None`. That split matters because later runtime config validation forbids context parallelism unless the backend stays on SDPA.

The other major design choice is that the same transformer block class supports a dense prefix and an MoE remainder. `DeepSeekV3TransformerBlock` checks `layer_id >= n_dense_layers` and instantiates either `feed_forward` or `moe`, so architecture flavors can vary between dense-first and mostly-MoE models without changing the training stack.

## State Flow
1. Runtime selection starts in `config_registry.py`, where a function returns a `Trainer.Config` containing a `model_spec=model_registry(...)`.
2. `model_registry` in `__init__.py` selects a `DeepSeekV3Model.Config` flavor from `deepseekv3_configs` and attaches the module's parallelization and checkpoint-conversion hooks.
3. During model construction, `DeepSeekV3Model.Config.update_from_config()` mutates config fields to match runtime conditions:
   it syncs `rope.max_seq_len` to `training.seq_len`, pushes rope fields into `layer.attention`, disables grouped GEMM on pre-SM90 GPUs, rejects CP with non-SDPA attention, propagates `debug.moe_force_load_balance`, and swaps in `DeepEPMoE.Config` when the EP comm backend is `deepep` or `hybridep`.
4. During forward execution, each `DeepSeekV3TransformerBlock` applies pre-norm residual attention and then either dense FFN or MoE.
5. After model creation, `parallelize_deepseekv3()` applies distributed transforms in order:
   non-MoE TP, MoE EP/ETP, CP, activation checkpointing, sparse compile, then FSDP/HSDP or pure replication.
6. For checkpoint conversion, `DeepSeekV3StateDictAdapter` maps TorchTitan parameter names to HF names, splitting grouped expert tensors on export and re-concatenating them on import.

The important mutation boundary is between static architecture presets in `deepseekv3_configs` and runtime legality checks in `DeepSeekV3Model.Config.update_from_config()`. Presets can request FlexAttention or grouped GEMM, but runtime code is allowed to downgrade or reject those requests once it sees the actual `trainer_config`, GPU capability, and parallelism choices.

## Important Constraints
- Tensor parallelism requires `training.seq_len % parallel_dims.seq_len_divisor == 0` in [parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/parallelize.py).
- CP is only allowed with `attn_backend == "sdpa"`; FlexAttention and varlen paths are rejected in `update_from_config()`.
- DeepEP and HybridEP require EP enabled and explicitly reject ETP.
- Float8 tensorwise TP is hard-disabled with `NotImplementedError`; only the tested rowwise path is allowed.
- The 671B preset uses float8 converters for `output`, `router.gate`, and grouped expert GEMMs.
- `Attention` changes its softmax scale when `rope_max_seq_len > rope_original_seq_len`, multiplying the base scale by `mscale^2` derived from `rope_factor`; shape or RoPE changes can therefore silently change attention numerics.

## Modification Guide
- Add a new DeepSeek flavor:
  update `deepseekv3_configs` in [torchtitan/models/deepseek_v3/__init__.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/__init__.py), then add a matching `Trainer.Config` function in [torchtitan/models/deepseek_v3/config_registry.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/config_registry.py). If the architecture changes query projection style, also verify `DeepSeekV3StateDictAdapter.__init__()` still selects the right HF query mapping.
- Change attention behavior or mask/backend support:
  start in `Attention.forward()` and `Attention.__init__()` in [torchtitan/models/deepseek_v3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/model.py), then revisit the CP and TP restrictions in `DeepSeekV3Model.Config.update_from_config()` and `parallelize_deepseekv3()`.
- Change MoE communication or sharding behavior:
  edit `parallelize_deepseekv3()` in [torchtitan/models/deepseek_v3/parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/parallelize.py). That function is where EP backend validation, `apply_moe_ep_tp()`, FSDP/EDP meshes, and dual-pipe-v integration are coordinated.
- Change HF checkpoint compatibility:
  edit `DeepSeekV3StateDictAdapter.from_hf_map`, `to_hf()`, and `from_hf()` in [torchtitan/models/deepseek_v3/state_dict_adapter.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/state_dict_adapter.py). Any rename in model parameter structure must be mirrored there, especially MoE expert tensor packing.
- Enable a currently blocked feature:
  check the explicit guards first. CP+TP is blocked in `parallelize_deepseekv3()`, CP+FlexAttention is blocked in `update_from_config()`, and float8 tensorwise TP is blocked in `parallelize_deepseekv3()`. Those are intentional safety barriers, not missing documentation.
- Change which layers are dense versus MoE:
  update `n_dense_layers` in the flavor configs in [torchtitan/models/deepseek_v3/__init__.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/__init__.py), then verify `DeepSeekV3TransformerBlock.__init__()` and any HF mapping assumptions in [torchtitan/models/deepseek_v3/state_dict_adapter.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/state_dict_adapter.py), because layers below the boundary use `feed_forward.*` names while layers at or above it use `moe.*`.

## File Map
- [torchtitan/models/deepseek_v3/README.md](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/README.md): tokenizer download, training commands, HF-to-DCP conversion command.
- [torchtitan/models/deepseek_v3/__init__.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/__init__.py): exported configs and `ModelSpec` registration.
- [torchtitan/models/deepseek_v3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/model.py): MLA attention, transformer block, runtime config mutation hooks.
- [torchtitan/models/deepseek_v3/config_registry.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/config_registry.py): runnable trainer presets.
- [torchtitan/models/deepseek_v3/parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/parallelize.py): TP/EP/CP/AC/compile/FSDP application order and restrictions.
- [torchtitan/models/deepseek_v3/state_dict_adapter.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/state_dict_adapter.py): HF import/export and quantized reader selection.

See [reference.md](/home/scbjtfy/torchtitan/.codex/skills/torchtitan-dr-models-deepseek-v3/reference.md) for the key types, functions, config flavor matrix, and conversion details.
