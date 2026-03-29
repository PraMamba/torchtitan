---
name: torchtitan-dr-models-gpt-oss
description: Use when changing TorchTitan's GPT-OSS model family, especially when touching its MoE experts, FlexAttention masking, parallelization rules, model flavors, or Hugging Face checkpoint conversion.
---

# TorchTitan GPT-OSS Module

## Overview

`torchtitan/models/gpt_oss` is the TorchTitan integration for the GPT-OSS model family. It defines the model registry and presets, implements a decoder-only transformer with FlexAttention and alternating sliding-window masks, swaps the default MoE expert implementation for GPT-OSS-specific grouped-matmul experts, applies TP/EP/ETP/FSDP parallelization, and converts checkpoints to and from Hugging Face naming/layouts.

## When To Use

- Use this module when adding or changing a GPT-OSS flavor such as `debugmodel`, `20b`, or `120b`.
- Use it when debugging GPT-OSS attention behavior, especially sink attention, sliding-window masking, or block-causal masking.
- Use it when changing MoE execution, grouped matmul, expert sharding, or tensor-parallel bias semantics.
- Use it when wiring GPT-OSS into training configs or checkpoint conversion flows.
- Do not use this module for generic decoder behavior shared across families; that logic lives in `torchtitan/models/common`.

## Public Surface

- `torchtitan/models/gpt_oss/__init__.py`
  - `gptoss_configs`: flavor-to-`GptOssModel.Config` registry.
  - `model_registry(flavor) -> ModelSpec`: exports GPT-OSS into TorchTitan's trainer/config system.
  - Re-exports `GptOssModel` and `parallelize_gptoss`.
- `torchtitan/models/gpt_oss/config_registry.py`
  - `gpt_oss_debugmodel() -> Trainer.Config`
  - `gpt_oss_20b() -> Trainer.Config`
  - `gpt_oss_120b() -> Trainer.Config`
- `torchtitan/models/gpt_oss/parallelize.py`
  - `parallelize_gptoss(...)`
  - `apply_non_moe_tp(...)`
  - `apply_moe_ep_tp(...)`
- `torchtitan/models/gpt_oss/state_dict_adapter.py`
  - `GptOssStateDictAdapter`

## Design Logic

- GPT-OSS is kept close to TorchTitan's common decoder stack, but it overrides the pieces that are architecture-specific: attention, MoE experts, parallelization, and checkpoint translation.
- Attention is hard-pinned to FlexAttention in `model.py`. The `Attention.Config.attn_backend` default is `"flex"`, and `Attention.__init__()` asserts that anything else is invalid. This prevents the family from silently running on an unsupported backend.
- Transformer blocks alternate between full basic masking and sliding-window masking via `layer_id % 2 == 0` in `GptOssTransformerBlock.__init__()`. The mask choice is resolved at runtime from a two-mask dictionary rather than rebuilding per-layer logic inside the attention kernel.
- The MoE path replaces the base grouped experts with `GptOssGroupedExperts` in `moe.py` because GPT-OSS needs custom SwiGLU behavior, grouped matmul support, and special bias scaling for tensor parallel output reduction.
- Parallelization is split into non-MoE TP and MoE EP/ETP logic in `parallelize.py`. That keeps the dense decoder path simple while letting the expert path choose among TP-only, EP-only, or combined expert-tensor sharding.
- Checkpoint conversion stays key-based in `state_dict_adapter.py`. The adapter maps between HF names and TorchTitan names without reshaping tensors, and it delegates quantized HF loading to `QuantizedHuggingFaceStorageReader` when `from_quantized=True`.

## State Flow

1. Training config entrypoints in `torchtitan/models/gpt_oss/config_registry.py` build `Trainer.Config` objects and point `model_spec` at `model_registry("<flavor>")`.
2. `model_registry()` in `torchtitan/models/gpt_oss/__init__.py` returns a `ModelSpec` containing:
   - the selected `GptOssModel.Config`
   - `parallelize_gptoss`
   - `build_cross_entropy_loss`
   - `register_moe_load_balancing_hook`
   - `GptOssStateDictAdapter`
3. `GptOssModel.Config.update_from_config()` in `model.py` synchronizes rope length to trainer `seq_len`, disables grouped matmul on pre-SM90 hardware, rejects CP, and validates TP divisibility of heads and KV heads.
4. `parallelize_gptoss()` in `parallelize.py` applies dense TP first, then MoE EP/ETP, then CP hooks if enabled, then activation checkpointing, then FSDP/HSDP or replication.
5. At runtime, `GptOssModel.get_attention_masks()` builds a `{"basic_mask", "sliding_window_mask"}` dictionary. Each `GptOssTransformerBlock.forward()` picks one mask based on `use_sliding_attention`, runs attention, then runs the MoE branch.
6. During save/load, `GptOssStateDictAdapter.from_hf()` and `to_hf()` translate keys, and `get_hf_storage_reader()` swaps in the quantized reader for MXFP4-based GPT-OSS checkpoints.

## Common Modification Scenarios

### Add a new GPT-OSS flavor

- Start in `torchtitan/models/gpt_oss/__init__.py` by adding a new `gptoss_configs[...]` entry.
- Add a matching trainer preset in `torchtitan/models/gpt_oss/config_registry.py`.
- If assets or usage instructions change, update `torchtitan/models/gpt_oss/README.md`.
- Check whether the new head counts still satisfy `tensor_parallel_degree` divisibility rules in `GptOssModel.Config.update_from_config()`.

### Change attention behavior or masking

- Edit `Attention.Config`, `Attention.forward()`, or sink handling in `torchtitan/models/gpt_oss/model.py`.
- If the mask policy changes, update `GptOssModel.get_attention_masks()` and the mask selection inside `GptOssTransformerBlock.forward()`.
- Preserve the contract that `attention_masks` is a dict containing `"basic_mask"` and `"sliding_window_mask"` entries, because block code assumes those keys exist.

### Change MoE compute kernels or TP bias behavior

- Edit `torchtitan/models/gpt_oss/moe.py`.
- `_run_experts_grouped_mm()` and `_run_experts_for_loop()` must stay semantically aligned; one is the optimized path and the other is the fallback path.
- Preserve `ScaleBiasForward`: GPT-OSS scales expert output bias in forward for TP, but deliberately leaves backward gradients unscaled.
- If sharding semantics change, also update `torchtitan/models/gpt_oss/expert_parallel.py` and the `apply_moe_ep_tp()` logic in `parallelize.py`.

### Extend checkpoint conversion

- Update `GptOssStateDictAdapter.from_hf_map` in `torchtitan/models/gpt_oss/state_dict_adapter.py`.
- Keep `to_hf()` and `from_hf()` symmetric for every new key pattern.
- If the new checkpoint source is quantized, update `get_hf_storage_reader()` rather than bolting quantization logic into the key-mapping methods.

## Common Mistakes

- Enabling context parallel for GPT-OSS. `GptOssModel.Config.update_from_config()` explicitly raises `NotImplementedError`, and `parallelize_gptoss()` also rejects TP+CP together.
- Changing grouped-matmul logic without preserving padding/permutation behavior. `indices_padding_wrapper()` exists to satisfy grouped-mm alignment without host synchronization.
- Treating the expert output bias like an ordinary parameter under TP. GPT-OSS intentionally uses `ScaleBiasForward` to correct the forward reduction only.
- Adding a new HF mapping in only one direction. The closed-book answer for checkpoint paths depends on `to_hf()` and `from_hf()` staying aligned.

## Reference

See `reference.md` for the key types, config defaults, function-level behavior, and HF state-dict mapping surface.
