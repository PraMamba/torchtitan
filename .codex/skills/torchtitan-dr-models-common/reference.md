# Models Common Reference

## File Inventory

### [`torchtitan/models/common/__init__.py`](../../../torchtitan/models/common/__init__.py)
- Re-exports the shared model-layer API used by model families and tests.
- The exported names are effectively the module's stable entry points; adding a new common building block usually requires updating `__all__` here.

### [`torchtitan/models/common/decoder.py`](../../../torchtitan/models/common/decoder.py)

Key types:
- `TransformerBlock.Config`
  - Required: `attention`, `attention_norm`, `ffn_norm`
  - Optional: `feed_forward`, `moe`
- `Decoder.Config`
  - Required: `dim`, `n_layers`, `vocab_size`, `output`, `tok_embeddings`, `norm`, `rope`, `layer`

Key methods:
- `Decoder.__init__(config)`
  - Builds token embedding, one shared `RoPE`, `layers` as a `ModuleDict`, final norm, and output projection.
- `Decoder.init_weights(buffer_device=..., **kwargs)`
  - Recomputes rope cache on the desired device, handles the PP case where `self.rope` was pruned, initializes every submodule, and trunc-normal initializes the output head with `dim**-0.5`.
- `Decoder.forward(tokens, attention_masks=None, positions=None)`
  - Embeds, iterates layers, applies final norm and output head.
- `Decoder._get_flex_attention_masks(input_batch, tokenizer, extra_inputs=None)`
  - Builds flex block masks for `causal` or `block_causal`.
- `Decoder.get_attention_masks(...)`
  - Dispatches to flex or varlen mask generation based on `self.attn_config.attn_backend`.
- `Decoder.attn_config`
  - Convenience property returning `self.config.layer.attention`.

Important design notes:
- The shared rope buffer is stored as non-persistent `freqs_cis`, so checkpoints do not serialize a huge per-sequence cache.
- Pipeline-parallel support depends on `forward()` tolerating missing `tok_embeddings`, `norm`, or `output`.

### [`torchtitan/models/common/attention.py`](../../../torchtitan/models/common/attention.py)

Key types:
- `VarlenMetadata(cu_seq_q, cu_seq_k, max_q, max_k)`
- `AttentionMasksType = dict[str, BlockMask] | BlockMask | VarlenMetadata`
- `BaseAttention.Config`
- `GQAttention.Config`

Backend wrappers:
- `LocalMapAttention`
  - Overrides `__call__()` to wrap attention modules with `torch.distributed.tensor.experimental.local_map` when q/k/v are DTensors sharded on head dimension.
- `VarlenAttentionWrapper`
  - Compiles `varlen_attn` once as `_compiled_varlen_attn`.
- `FlexAttentionWrapper`
  - Compiles `flex_attention` once with custom inductor settings.
- `ScaledDotProductAttentionWrapper`
  - Wraps `F.scaled_dot_product_attention` in a module so context-parallel hooks can target it.

Mask and metadata helpers:
- `get_causal_mask_mod()`
- `get_document_mask_mod(batch, eos_id)`
- `get_fixed_block_mask_mod(fixed_block_size)`
- `get_sliding_window_mask_mod(window_size)`
- `create_attention_mask(*args, **kwargs)`
- `create_varlen_metadata_for_document(input_batch, eos_id)`
- `annotate_flex_attention_for_regional_inductor()`

`GQAttention` responsibilities:
- Creates `wq`, `wk`, `wv`, `wo`.
- Handles `n_heads`, `n_kv_heads`, derived `head_dim`, `enable_gqa`, and optional Q/K RMSNorm.
- Applies RoPE only when `use_rope=True`.
- Supports flex dictionaries for iRoPE-style `{"rope": ..., "nope": ...}` masks.

### [`torchtitan/models/common/rope.py`](../../../torchtitan/models/common/rope.py)

Key type:
- `RoPE.Config`
  - Required: `dim`, `max_seq_len`
  - Optional scaling controls:
    - Llama scaling: `scaling_factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings`
    - YaRN scaling: `rope_factor`, `beta_fast`, `beta_slow`, `original_seq_len`, `mscale`

Key methods and helpers:
- `RoPE._precompute()`
  - Switches between complex and cos/sin cache formats.
- `RoPE._precompute_complex()`
  - Supports `none`, `llama`, and `yarn` scaling.
- `RoPE._precompute_cos_sin()`
  - Supports `none` and `yarn`; explicitly rejects llama scaling.
- `RoPE.init_weights(buffer_device=...)`
  - Recomputes the cache on the requested device.
- `_reshape_for_broadcast_complex(...)`
- `_reshape_for_broadcast_cos_sin(...)`
- `_maybe_wrap_positions(...)`
  - Converts plain `positions` tensors to DTensor when the attention tensors are DTensors.
- `apply_rotary_emb_complex(...)`
- `apply_rotary_emb_single_complex(...)`
- `apply_rotary_emb_cos_sin(...)`

Important invariants:
- Complex mode expects `freqs_cis` shaped `(max_seqlen, head_dim // 2)` as complex numbers.
- Cos/sin mode stores concatenated `(cos, sin)` values with shape `(max_seqlen, head_dim * 2)`.

### [`torchtitan/models/common/embedding.py`](../../../torchtitan/models/common/embedding.py)
- `Embedding.Config`
  - `num_embeddings` and `embedding_dim` are `field(init=False)` and must come through `build(...)`.
- `Embedding.init_weights()`
  - Uses `nn.init.normal_` with `init_mean` and `init_std`.

### [`torchtitan/models/common/linear.py`](../../../torchtitan/models/common/linear.py)
- `Linear.Config`
  - `in_features` and `out_features` are `field(init=False)`.
- `Linear.init_weights(init_std=None)`
  - Uses trunc-normal for weights and zeros for bias.

### [`torchtitan/models/common/rmsnorm.py`](../../../torchtitan/models/common/rmsnorm.py)
- `RMSNorm.Config`
  - `normalized_shape` is `field(init=False)`.
- `RMSNorm.init_weights()`
  - Delegates to `reset_parameters()`.

### [`torchtitan/models/common/feed_forward.py`](../../../torchtitan/models/common/feed_forward.py)
- `compute_ffn_hidden_dim(dim, multiple_of=1, ffn_dim_multiplier=None)`
  - Implements the Llama-style `2 * 4 * dim / 3` rule and rounds up to `multiple_of`.
- `FeedForward.Config`
  - `hidden_dim` is final hidden width, not a pre-scaled input.
  - `dim` is `field(init=False)` and must come via `build(dim=...)`.
- `FeedForward.forward(x)`
  - `w2(silu(w1(x)) * w3(x))`

### [`torchtitan/models/common/moe/moe.py`](../../../torchtitan/models/common/moe/moe.py)

Helper functions:
- `_run_experts_for_loop(...)`
- `_run_experts_grouped_mm(...)`

Key types:
- `GroupedExperts.Config`
- `TokenChoiceTopKRouter.Config`
- `TokenReorderer`
- `MoE.Config`

`GroupedExperts`:
- Stores expert weights as parameter tensors:
  - `w1`: `(num_experts, hidden_dim, dim)`
  - `w2`: `(num_experts, dim, hidden_dim)`
  - `w3`: `(num_experts, hidden_dim, dim)`
- Chooses between for-loop execution and `torch._grouped_mm`.
- Converts DTensor parameters to local tensors before dynamic-shape expert execution.

`TokenChoiceTopKRouter`:
- Computes gate scores in float32 under autocast.
- Supports `sigmoid` or `softmax`.
- Optionally applies node-limited routing through `_get_node_limited_routing_scores()`.
- Supports `_debug_force_load_balance_routing()`.
- Returns:
  - `top_scores`: `(tokens, top_k)`
  - `selected_experts_indices`: `(tokens, top_k)`
  - `num_tokens_per_expert`: `(num_experts,)`

`TokenReorderer`:
- Sorts flattened expert assignments so tokens are grouped by expert.

`MoE`:
- Creates grouped experts, router, reorderer, and optional shared FFN experts.
- Maintains:
  - `expert_bias` as persistent buffer when load balancing is enabled
  - `tokens_per_expert` as non-persistent running usage counter
- Supports two score application modes:
  - `score_before_experts=True`: scale routed input before expert execution
  - `score_before_experts=False`: score the unsorted expert outputs after execution

### [`torchtitan/models/common/moe/moe_deepep.py`](../../../torchtitan/models/common/moe/moe_deepep.py)
- `DeepEPMoE`
  - Reuses router and shared-expert logic from `MoE`.
  - Disables `reorderer` because DeepEP hooks handle dispatch ordering.
  - Calls `sync_combine()` after shared-expert compute to overlap combine communication with compute.

### [`torchtitan/models/common/moe/kernels.py`](../../../torchtitan/models/common/moe/kernels.py)

Key pieces:
- Triton `_fill_indices_kernel`
- `_fill_indices_impl(...)`
- `fill_indices_wrapper(...)`
  - Registered as custom op `torchtitan::fill_indices`
- `_fill_indices_fake(...)`
  - Fake registration for tracing/compile flows
- `fill_indices_cpu(...)`
  - CPU reference implementation
- `generate_permute_indices(...)`
  - Produces:
    - `permuted_indices`
    - `m_sizes`
    - `m_offsets`

Important role:
- This file is the bridge between expert token-count histograms and physically grouped token rows for grouped GEMM execution.

### [`torchtitan/models/common/moe/utils.py`](../../../torchtitan/models/common/moe/utils.py)

Key symbols:
- `TOKEN_GROUP_ALIGN_SIZE_M`
- `ValidTokenGroupAlignmentSize`
- `get_mxfp8_pad_multiple()`
- `set_token_group_alignment_size_m(alignment_size)`
- `maybe_align_num_tokens_for_mxfp8(num_tokens)`
- `_permute(...)`
- `_unpermute(...)`
- `indices_padding_wrapper(func)`

Important role:
- Aligns expert token groups to hardware- or quantization-driven boundaries before grouped GEMM.
- Couples MoE token reordering to `components.quantization.MXFP8_GROUP_ALIGNMENT_SIZE`.

## Cross-File Relationships

- `Decoder` consumes `BaseTokenizer` from `torchtitan.components.tokenizer` only for attention-mask generation; the actual token embedding path is independent of tokenizer logic.
- `GQAttention` depends on:
  - `Linear` and `RMSNorm` for projections/norms
  - `RoPE` helpers for position encoding
  - the backend wrappers defined in the same file
- `MoE` depends on:
  - `FeedForward` for shared experts
  - `Linear` indirectly through `TokenChoiceTopKRouter.gate`
  - `moe/utils.py` and `moe/kernels.py` for grouped token padding and permutation
- DTensor-aware logic in `LocalMapAttention`, `MoE`, and `DeepEPMoE` is coordinated with the distributed layer rather than hidden behind pure local tensors.

## Modification Entry Points

- New attention mask semantics:
  - `attention.py`: new mask-mod helper
  - `decoder.py`: hook it into `get_attention_masks()`
- New RoPE scaling mode:
  - `rope.py`: extend `RoPE.Config` and `_precompute_*()`
  - `attention.py`: update `GQAttention` only if application semantics change
- New MoE backend:
  - `moe/moe.py`: preserve router contracts
  - add backend-specific subclass similar to `DeepEPMoE` in `moe_deepep.py`
  - keep alignment/permutation semantics compatible with `moe/utils.py` and `moe/kernels.py`
