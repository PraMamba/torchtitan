---
name: torchtitan-dr-models-common
description: Use when changing TorchTitan's shared transformer building blocks, especially decoder flow, attention backends, rotary embeddings, grouped feed-forward layers, or MoE routing and expert execution behavior.
---

# TorchTitan Models Common

## Overview

`torchtitan/models/common` is the reusable transformer substrate that most TorchTitan model families build on. It defines the shared decoder stack in [`torchtitan/models/common/decoder.py`](../../../torchtitan/models/common/decoder.py), grouped-query attention and mask helpers in [`torchtitan/models/common/attention.py`](../../../torchtitan/models/common/attention.py), rotary-position caching and application in [`torchtitan/models/common/rope.py`](../../../torchtitan/models/common/rope.py), and the basic parameterized layers in [`embedding.py`](../../../torchtitan/models/common/embedding.py), [`linear.py`](../../../torchtitan/models/common/linear.py), [`rmsnorm.py`](../../../torchtitan/models/common/rmsnorm.py), and [`feed_forward.py`](../../../torchtitan/models/common/feed_forward.py). The `moe/` package adds the shared Mixture-of-Experts path: token-choice routing, grouped expert weights, permutation kernels, and the DeepEP-specific MoE variant.

## Public Surface

- [`torchtitan/models/common/__init__.py`](../../../torchtitan/models/common/__init__.py)
  - Re-exports the module's stable shared API: `Decoder`, `TransformerBlock`, `GQAttention`, `RoPE`, `FeedForward`, `MoE`, `Linear`, `Embedding`, `RMSNorm`, flex/varlen mask helpers, and rotary application functions.
- [`torchtitan/models/common/decoder.py`](../../../torchtitan/models/common/decoder.py)
  - `TransformerBlock.Config`: base config contract for model-specific transformer blocks.
  - `Decoder`: shared decoder-only model shell with embedding, rope cache management, layer construction, and attention-mask generation.
- [`torchtitan/models/common/attention.py`](../../../torchtitan/models/common/attention.py)
  - `GQAttention`: shared grouped-query attention implementation.
  - `FlexAttentionWrapper`, `VarlenAttentionWrapper`, `ScaledDotProductAttentionWrapper`: backend adapters.
  - `create_attention_mask()`, `create_varlen_metadata_for_document()`, `get_causal_mask_mod()`, `get_document_mask_mod()`, `get_fixed_block_mask_mod()`, `get_sliding_window_mask_mod()`.
- [`torchtitan/models/common/rope.py`](../../../torchtitan/models/common/rope.py)
  - `RoPE`: precomputes and serves complex or cos/sin rotary caches, with `none`, `llama`, and `yarn` scaling.
  - `apply_rotary_emb_complex()`, `apply_rotary_emb_single_complex()`, `apply_rotary_emb_cos_sin()`.
- [`torchtitan/models/common/feed_forward.py`](../../../torchtitan/models/common/feed_forward.py)
  - `compute_ffn_hidden_dim()` and `FeedForward`.
- [`torchtitan/models/common/moe/moe.py`](../../../torchtitan/models/common/moe/moe.py)
  - `GroupedExperts`, `TokenChoiceTopKRouter`, `TokenReorderer`, and `MoE`.
- [`torchtitan/models/common/moe/moe_deepep.py`](../../../torchtitan/models/common/moe/moe_deepep.py)
  - `DeepEPMoE`: alternate MoE forward path that cooperates with DeepEP communication hooks.

## Design Logic

- The module deliberately separates model-family policy from reusable mechanics. `Decoder` and `TransformerBlock.Config` define the shared shell, while Llama/Qwen/GPT-OSS/DeepSeek-specific modules decide exact block classes, norms, and flavor registries.
- Attention is backend-pluggable but API-stable. `GQAttention.forward()` always projects into Q/K/V, optionally normalizes and applies RoPE, then dispatches to `FlexAttentionWrapper`, `VarlenAttentionWrapper`, or `ScaledDotProductAttentionWrapper`. That lets distributed code and model code target one attention abstraction while switching kernels via config.
- RoPE is precomputed and cached once per decoder instead of per layer. `Decoder.__init__()` builds one `RoPE` and registers `freqs_cis` as a non-persistent buffer; every layer reuses that cache. This reduces redundant state and makes pipeline-parallel pruning manageable via the rebuild path in `Decoder.init_weights()`.
- The base layers use diamond inheritance (`nn.Linear`/`nn.Embedding`/`nn.RMSNorm` + `Module`) so TorchTitan keeps native PyTorch semantics without extra wrapper modules, while still gaining `Config.build(...)` and `init_weights()`.
- MoE is split into three stages: router (`TokenChoiceTopKRouter`), reorderer (`TokenReorderer`), and experts (`GroupedExperts`). This keeps load-balancing and expert assignment logic independent from the execution backend, which is why `DeepEPMoE` can override only the execution path and reuse the routing contracts.
- The shared MoE path is written around DTensor boundaries, not DTensor internals. `MoE.forward()` and `DeepEPMoE.forward()` convert TP-sharded DTensors to local tensors with `grad_placements=(Partial(),)` at the boundary, then do routing and expert execution on plain tensors. That keeps gradient-reduction semantics correct without forcing dynamic token routing into DTensor-friendly shapes.

## Core Data Structures

- `Decoder.Config` in [`decoder.py`](../../../torchtitan/models/common/decoder.py)
  - Defines `dim`, `n_layers`, `vocab_size`, `output`, `tok_embeddings`, `norm`, `rope`, and `layer`.
- `TransformerBlock.Config` in [`decoder.py`](../../../torchtitan/models/common/decoder.py)
  - Requires `attention`, `attention_norm`, `ffn_norm`, and optionally `feed_forward` or `moe`.
- `VarlenMetadata` in [`attention.py`](../../../torchtitan/models/common/attention.py)
  - Carries `cu_seq_q`, `cu_seq_k`, `max_q`, and `max_k` for varlen attention.
- `GQAttention.Config` in [`attention.py`](../../../torchtitan/models/common/attention.py)
  - Controls head counts, backend, mask type, optional QK norms, head dim override, and rope style.
- `RoPE.Config` in [`rope.py`](../../../torchtitan/models/common/rope.py)
  - Encodes backend and scaling mode, including Llama-style and YaRN parameters.
- `GroupedExperts.Config`, `TokenChoiceTopKRouter.Config`, and `MoE.Config` in [`moe/moe.py`](../../../torchtitan/models/common/moe/moe.py)
  - Define expert weight layout, routing strategy, shared experts, and load-balancing buffers.

See [`reference.md`](./reference.md) for the full file-by-file type inventory and function index.

## State Flow

1. Model construction starts in a family-specific module, but shared config objects flow into `Decoder.__init__()` in [`decoder.py`](../../../torchtitan/models/common/decoder.py). It builds token embeddings, precomputes a `RoPE` cache, instantiates each layer by calling `config.layer.build(...)`, and creates the final norm/output head.
2. During a forward pass, `Decoder.forward()` embeds tokens, passes the shared `freqs_cis` cache and optional `attention_masks` / `positions` through each layer, then applies the final norm and output projection.
3. Attention-mask creation is centralized in `Decoder.get_attention_masks()`. Flex attention uses `_get_flex_attention_masks()` and `create_attention_mask()`. Varlen attention uses `create_varlen_metadata_for_document()`. SDPA expects `attention_masks is None`.
4. `GQAttention.forward()` projects Q/K/V with `wq`, `wk`, `wv`, optionally applies `q_norm`/`k_norm`, applies either `apply_rotary_emb_complex()` or `apply_rotary_emb_cos_sin()`, transposes into backend-specific shape, then calls the chosen backend wrapper before projecting with `wo`.
5. In the dense FFN path, `FeedForward.forward()` computes SwiGLU as `w2(silu(w1(x)) * w3(x))`.
6. In the MoE path, `MoE.forward()` flattens `(bs, slen, dim)` to tokens, calls `router()` to get `top_scores`, `selected_experts_indices`, and token counts, updates `tokens_per_expert`, reorders token rows with `TokenReorderer`, runs `GroupedExperts.forward()`, optionally overlaps shared-expert compute, then unsorts and merges expert outputs.
7. `DeepEPMoE.forward()` keeps the same router contract but passes routing metadata into `experts(...)`, expecting distributed DeepEP hooks to dispatch/combine asynchronously and then synchronizing through `sync_combine()`.

## Error Handling And Invariants

- `BaseAttention.Config.__post_init__()` rejects unsupported `attn_backend` or `attn_mask_type` combinations; `sdpa + block_causal` is invalid.
- `GQAttention.Config.__post_init__()` enforces that `q_norm` and `k_norm` are both present or both absent.
- `Decoder.get_attention_masks()` raises if `varlen` is requested without `block_causal`, or if an unsupported backend reaches the mask-generation path.
- `Linear`, `Embedding`, and `RMSNorm` constructors raise `TypeError` when required `field(init=False)` values were not supplied via `build(...)`.
- `RoPE._precompute_cos_sin()` rejects Llama scaling and asserts YaRN correction ranges are valid.
- `LocalMapAttention.__call__()` asserts that DTensor q/k/v inputs are sharded on head dimension via `Shard(1)` and share placements.
- `MoE` and `DeepEPMoE` assert TP-only 1D meshes when converting DTensors to local tensors.

## Common Modification Scenarios

- Add a new shared attention behavior or backend:
  Start in [`attention.py`](../../../torchtitan/models/common/attention.py). If the change affects kernel selection, extend `BaseAttention.Config` and the `match self.attn_backend` branches in `GQAttention.__init__()` and `GQAttention.forward()`. If it needs a new mask type, also update `Decoder.get_attention_masks()` in [`decoder.py`](../../../torchtitan/models/common/decoder.py).
- Change rotary embedding format, scaling, or position handling:
  Edit [`rope.py`](../../../torchtitan/models/common/rope.py). The cache precompute path lives in `RoPE._precompute_*()`, while application-time shape/position logic lives in `_reshape_for_broadcast_*()`, `_maybe_wrap_positions()`, and the `apply_rotary_emb_*()` helpers.
- Change shared decoder-layer construction:
  Edit `Decoder.__init__()` and `Decoder.init_weights()` in [`decoder.py`](../../../torchtitan/models/common/decoder.py). If you add a new per-layer shared resource, make sure the pipeline-parallel rebuild case stays valid when `self.rope` or other modules are pruned.
- Change MoE routing or load balancing:
  Edit `TokenChoiceTopKRouter` and `MoE` in [`moe/moe.py`](../../../torchtitan/models/common/moe/moe.py). Routing score computation, node-limited routing, expert-bias handling, and score-before/after-expert semantics all live there.
- Change grouped expert execution or token padding:
  Edit `GroupedExperts` in [`moe/moe.py`](../../../torchtitan/models/common/moe/moe.py) and padding/permutation helpers in [`moe/utils.py`](../../../torchtitan/models/common/moe/utils.py) plus [`moe/kernels.py`](../../../torchtitan/models/common/moe/kernels.py). Those files jointly define how token groups are aligned, reordered, and fed into `_grouped_mm`.

## File Map

- [`torchtitan/models/common/__init__.py`](../../../torchtitan/models/common/__init__.py): public export surface.
- [`torchtitan/models/common/attention.py`](../../../torchtitan/models/common/attention.py): attention backends, masks, and GQA implementation.
- [`torchtitan/models/common/decoder.py`](../../../torchtitan/models/common/decoder.py): shared decoder shell and mask dispatch.
- [`torchtitan/models/common/embedding.py`](../../../torchtitan/models/common/embedding.py): configurable embedding layer.
- [`torchtitan/models/common/feed_forward.py`](../../../torchtitan/models/common/feed_forward.py): shared SwiGLU FFN.
- [`torchtitan/models/common/linear.py`](../../../torchtitan/models/common/linear.py): configurable linear layer.
- [`torchtitan/models/common/rmsnorm.py`](../../../torchtitan/models/common/rmsnorm.py): configurable RMSNorm layer.
- [`torchtitan/models/common/rope.py`](../../../torchtitan/models/common/rope.py): RoPE cache computation and application helpers.
- [`torchtitan/models/common/moe/`](../../../torchtitan/models/common/moe): MoE router, grouped experts, kernels, DeepEP variant, and alignment utilities.

## See Also

- [`reference.md`](./reference.md): complete API index, per-file responsibilities, and cross-file relationships.
