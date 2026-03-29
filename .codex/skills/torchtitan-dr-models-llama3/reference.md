# Llama3 Reference

## File Index

- [torchtitan/models/llama3/__init__.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/__init__.py)
  - Flavor map and `model_registry()`.
- [torchtitan/models/llama3/config_registry.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/config_registry.py)
  - Trainer presets.
- [torchtitan/models/llama3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/model.py)
  - `Llama3TransformerBlock` and `Llama3Model`.
- [torchtitan/models/llama3/parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/parallelize.py)
  - TP/CP/AC/compile/FSDP/replicate orchestration.
- [torchtitan/models/llama3/state_dict_adapter.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/state_dict_adapter.py)
  - Hugging Face key mapping and RoPE-related Q/K permutation logic.

## Preset Flavors In `llama3_configs`

- Debug-oriented
  - `debugmodel`
  - `debugmodel_flex_attn`
  - `debugmodel_varlen_attn`
- Standard dense sizes
  - `1B`
  - `3B`
  - `8B`
  - `70B`
  - `405B`
- 8B attention backend variants
  - `8B_flex`
  - `8B_varlen`

Shared preset pattern:

- `tok_embeddings=Embedding.Config()`
- `norm=RMSNorm.Config()`
- `output=Linear.Config()`
- `layer=Llama3TransformerBlock.Config(...)`
- `rope=RoPE.Config(..., scaling="llama")`

## `Llama3TransformerBlock`

Defined in [torchtitan/models/llama3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/model.py).

Constructor inputs:

- `config`
- `layer_id`
- `dim`
- `n_layers`

Important behavior:

- Builds attention and feed-forward submodules from nested configs.
- Chooses `weight_init_std` with either depth-based scaling or total-layer-based scaling.
- `forward(x, freqs_cis, attention_masks, positions=None)`
  - `h = x + attention(...)`
  - `out = h + feed_forward(...)`
- `init_weights()`
  - Initializes norms directly.
  - Passes `weight_init_std` into attention and feed-forward initializers.

## `Llama3Model.Config.update_from_config()`

Defined in [torchtitan/models/llama3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/model.py).

Checks and mutations:

- Replaces `self.rope.max_seq_len` with runtime `training.seq_len`.
- Warns when runtime sequence length exceeds original RoPE max.
- Rejects `varlen` attention together with context parallel.
- Validates TP divisibility for `n_heads` and `n_kv_heads`.
- Rejects `enable_weight_tying=True` with pipeline parallelism.

## `parallelize_llama()` Order Of Operations

Defined in [torchtitan/models/llama3/parallelize.py](/home/scbjtfy/torchtitan/torchtitan/models/llama3/parallelize.py).

Inputs:

- `model`
- `parallel_dims`
- `training`
- `model_converters`
- `parallelism`
- `compile_config`
- `ac_config`
- `dump_folder`

Execution order:

1. Sequence-length divisibility assertion.
2. Optional TP setup, including float8-aware TP path selection.
3. Optional CP attachment on each block's attention inner module.
4. Optional activation checkpoint wrapping.
5. Optional model compilation.
6. Optional FSDP/HSDP or replicate wrapping.

## `apply_tp()` Sharding Plan

Root-module plan:

- `tok_embeddings`: `RowwiseParallel`, replicated input, sharded output.
- `norm`: `SequenceParallel`.
- `output`: `ColwiseParallel`, output sharded when loss parallel is enabled.

Per-transformer-block plan:

- `attention_norm`: `SequenceParallel`
- `attention`: `PrepareModuleInput` / `PrepareFloat8ModuleInput`
- `attention.wq`, `attention.wk`, `attention.wv`: colwise
- `attention.wo`: rowwise with output shard
- `ffn_norm`: `SequenceParallel`
- `feed_forward`: prepared replicated input from sharded hidden states
- `feed_forward.w1`, `feed_forward.w3`: colwise
- `feed_forward.w2`: rowwise with output shard

## `apply_fsdp()` Special Cases

- Builds `MixedPrecisionPolicy(param_dtype, reduce_dtype)`.
- Adds `CPUOffloadPolicy()` when `cpu_offload=True`.
- Uses `get_fsdp_reshard_after_forward_policy(...)`.
- If `enable_weight_tying` is on:
  - fully-shards `tok_embeddings`, `norm`, and `output` together.
- Otherwise:
  - sharded embedding separately.
  - `norm` and `output` grouped together.
- Every transformer block is then fully sharded.
- The root model is fully sharded last.
- Calls `disable_fsdp_gradient_division(model)` afterward.

## `Llama3StateDictAdapter`

Mapping source:

- `from_hf_map` maps Hugging Face keys like `model.layers.{}.self_attn.q_proj.weight` to TorchTitan keys like `layers.{}.attention.wq.weight`.
- `model.layers.{}.self_attn.rotary_emb.inv_freq` maps to `None`, so it is skipped.

Permutation rules:

- `q_proj.weight` and `k_proj.weight` are permuted/reverse-permuted because Hugging Face and native Llama RoPE layouts differ.
- `head_dim = dim // n_heads`
- KV permutation uses `key_value_dim = head_dim * n_kv_heads`

Weight-tying behavior:

- `from_hf()` synthesizes `lm_head.weight` from `model.embed_tokens.weight` when tied weights are enabled and HF omitted the head.
- `to_hf()` omits `output.weight` when tied weights are enabled so HF keeps a single embedding/head copy.
