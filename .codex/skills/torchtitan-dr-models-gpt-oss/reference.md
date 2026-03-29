# GPT-OSS Reference

## File Index

- `torchtitan/models/gpt_oss/__init__.py`: model flavor registry and `ModelSpec` export.
- `torchtitan/models/gpt_oss/config_registry.py`: trainer presets for debug, 20B, and 120B runs.
- `torchtitan/models/gpt_oss/model.py`: GPT-OSS attention, transformer block, model config validation, and mask construction.
- `torchtitan/models/gpt_oss/moe.py`: expert kernels, grouped-mm expert module, and GPT-OSS MoE subclass.
- `torchtitan/models/gpt_oss/expert_parallel.py`: custom TP and ETP sharding plans for expert weights/biases.
- `torchtitan/models/gpt_oss/parallelize.py`: model-wide TP/EP/ETP/CP/FSDP orchestration.
- `torchtitan/models/gpt_oss/state_dict_adapter.py`: Hugging Face checkpoint reader selection and key translation.
- `torchtitan/models/gpt_oss/README.md`: quick-start and declared feature support.

## Core Types And Defaults

### `Attention.Config`

Defined in `torchtitan/models/gpt_oss/model.py`.

- `n_heads=64`
- `n_kv_heads=8`
- `head_dim=64`
- `linear_bias=False`
- `attn_backend="flex"`
- `attn_mask_type="causal"`
- `sliding_window_size=128`

Behavior:

- Builds `wq`, `wk`, `wv`, `wo`, and learnable `sinks`.
- Enables GQA when `n_heads > n_kv_heads`.
- Uses `FlexAttentionWrapper` only.

### `GptOssTransformerBlock`

Defined in `torchtitan/models/gpt_oss/model.py`.

- `use_sliding_attention = layer_id % 2 == 0`
- Always expects `config.moe` to be present.
- Sets `moe_enabled = True` for load-balancing/composability hooks.
- Derives `weight_init_std = 0.02 / sqrt(2 * (layer_id + 1))`.

### `GptOssModel.Config`

Defined in `torchtitan/models/gpt_oss/model.py`.

Defaults:

- `dim=2880`
- `n_layers=24`
- `vocab_size=201088`

Important methods:

- `update_from_config(trainer_config, **kwargs)`
  - warns when training `seq_len` exceeds rope max
  - replaces `self.rope.max_seq_len` with trainer `seq_len`
  - disables grouped mm if `use_grouped_mm=True` but hardware lacks CUDA capability 9.0
  - rejects `context_parallel_degree > 1`
  - validates `tensor_parallel_degree` divides both `n_heads` and `n_kv_heads`
- `get_nparams_and_flops(model, seq_len)`
  - delegates to `get_moe_model_nparams_and_flops()`
  - uses `n_heads` and `2 * head_dim`

### `GptOssGroupedExperts.Config`

Defined in `torchtitan/models/gpt_oss/moe.py`.

- `use_grouped_mm=True`
- `swiglu_limit=7.0`
- runtime-built fields: `dim`, `hidden_dim`, `num_experts`

### `GptOssMoE.Config`

Defined in `torchtitan/models/gpt_oss/moe.py`.

- extends `MoE.Config`
- adds `swiglu_limit=7.0`

## Flavor Registry

Defined in `torchtitan/models/gpt_oss/__init__.py`.

### `debugmodel`

- `vocab_size=2048`
- `dim=256`
- `n_layers=4`
- MoE: `hidden_dim=2880`, `num_experts=8`, `top_k=4`
- Rope: `dim=64`, `max_seq_len=131072`, `theta=150000.0`, `backend="cos_sin"`, `scaling="yarn"`, `rope_factor=32`

### `20b`

- `n_layers=24`
- `dim` falls back to config default `2880`
- `num_experts=32`

### `120b`

- `n_layers=36`
- `num_experts=128`

All flavors use:

- `TokenChoiceTopKRouter.Config(score_func="softmax", route_norm=True, gate=Linear.Config(bias=True), top_k=4)`
- `Attention.Config(linear_bias=True)`
- RMSNorm for attention and FFN norms

## Trainer Presets

Defined in `torchtitan/models/gpt_oss/config_registry.py`.

### `gpt_oss_debugmodel()`

- HF assets path: `./tests/assets/tokenizer`
- Dataset: `c4_test`
- LR: `8e-4`
- LR scheduler: linear, `warmup_steps=2`, `decay_ratio=0.8`, `min_lr_factor=0.0`
- Training: `local_batch_size=8`, `seq_len=2048`, `steps=10`
- Parallelism: `expert_parallel_degree=1`, `expert_tensor_parallel_degree=1`
- Checkpoint every 10 steps, `last_save_model_only=False`
- Activation checkpointing disabled
- Validator every 5 steps for 10 steps

### `gpt_oss_20b()` and `gpt_oss_120b()`

- HF assets paths: `./assets/hf/gpt-oss-20b` and `./assets/hf/gpt-oss-120b`
- Dataset: `c4`
- LR: `8e-4`
- Scheduler: cosine, `warmup_steps=2000`, `decay_ratio=0.8`, `min_lr_factor=0.1`
- Training: `local_batch_size=1`, `seq_len=8192`, `steps=10000`
- Checkpoint every 500 steps
- Activation checkpointing `mode="full"`

## Attention And Mask Flow

Defined in `torchtitan/models/gpt_oss/model.py`.

### `Attention.forward(x, freqs_cis, attention_masks, positions=None)`

Steps:

1. Project `x` through `wq`, `wk`, `wv`.
2. Reshape to `(batch, seq, heads_or_kv_heads, head_dim)`.
3. Apply RoPE with `apply_rotary_emb_cos_sin()`.
4. Transpose into attention layout.
5. Assert `attention_masks` is a `BlockMask`.
6. Run `self.inner_attention(..., return_lse=True, enable_gqa=self.enable_gqa)`.
7. Compute sink scaling as `sigmoid(lse - sinks)`.
8. Rescale attention output, transpose back, reshape, and project with `wo`.

### `GptOssModel.get_attention_masks(input_batch, tokenizer, extra_inputs=None)`

Produces a dict with:

- `basic_mask`
- `sliding_window_mask`

Rules:

- `attn_mask_type == "causal"`: uses `get_causal_mask_mod()` and `B=1`
- `attn_mask_type == "block_causal"`: uses `get_document_mask_mod(input_batch, tokenizer.eos_id)` and `B=input_batch.shape[0]`
- anything else raises `ValueError`

The sliding-window mask is built by composing the basic mask modifiers with `get_sliding_window_mask_mod(sliding_window_size)`.

### `GptOssTransformerBlock.forward(...)`

- selects `"sliding_window_mask"` on even-numbered layers
- selects `"basic_mask"` on odd-numbered layers
- applies pre-norm attention residual
- applies pre-norm MoE residual

## MoE Execution Details

Defined in `torchtitan/models/gpt_oss/moe.py`.

### `ScaleBiasForward`

- forward: divides bias by `tp_degree` if `tp_degree > 1`
- backward: returns `grad_output` unchanged

Purpose:

- corrects the extra forward reduction introduced by TP for expert output bias
- preserves normal gradient magnitude

### `indices_padding_wrapper(func)`

- permutes tokens by expert with `_permute()`
- keeps token counts aligned for grouped matmul
- calls the wrapped expert kernel
- restores token order with `_unpermute()`

### `_run_experts_for_loop(...)`

- fallback implementation
- converts `num_tokens_per_expert` to Python list
- splits token tensor per expert
- runs expert MLPs sequentially
- uses `ScaleBiasForward` on `mlp2_bias`
- pads output rows back to the original token count

### `_run_experts_grouped_mm(...)`

- optimized grouped-matmul implementation
- computes `offsets = cumsum(num_tokens_per_expert)`
- appends `tail_slack` so `repeat_interleave(..., output_size=x.shape[0])` stays static-shape
- runs grouped matmul for MLP1 and MLP2 in `bfloat16`
- materializes expert biases with `repeat_interleave`
- applies `ScaleBiasForward` to the second bias tensor

### `GptOssGroupedExperts.forward(x, num_tokens_per_expert)`

- converts DTensor params to local tensors when needed because EP dynamic shapes are awkward as DTensors
- infers `tp_degree` from `device_mesh.mesh_dim_names` when weights are sharded
- chooses:
  - `indices_padding_wrapper(_run_experts_grouped_mm)` for grouped-mm without EP DTensor sharding
  - `_run_experts_grouped_mm` for grouped-mm with EP-aware DTensor layout
  - `_run_experts_for_loop` when `use_grouped_mm=False`

### `GptOssMoE.__init__(config, dim)`

- calls `MoE.__init__`
- replaces `self.experts` with a `GptOssGroupedExperts` instance carrying `swiglu_limit` and `use_grouped_mm`

## Parallelization Surface

Defined in `torchtitan/models/gpt_oss/parallelize.py`.

### `parallelize_gptoss(...)`

Order of operations:

1. validate `training.seq_len % parallel_dims.seq_len_divisor == 0`
2. compute whether model compilation is enabled
3. if TP enabled:
   - reject async TP unless compile is enabled
   - inspect float8 converter config
   - call `apply_non_moe_tp(...)`
4. if TP or EP enabled:
   - compute `dual_pipe_v`
   - call `apply_moe_ep_tp(...)`
5. if CP enabled:
   - reject TP+CP for GPT-OSS
   - patch `block.attention.inner_attention` with `apply_cp_to_attention_module(...)`
6. if activation checkpointing enabled:
   - call `apply_ac(...)`
7. if FSDP or EP enabled:
   - derive DP/EDP meshes
   - call `apply_fsdp(...)`
8. else if only DP replicate enabled:
   - call `apply_replicate(...)`

### `apply_non_moe_tp(...)`

Sharding plan:

- `tok_embeddings`: `RowwiseParallel(input=Replicate, output=Shard(1))`
- `norm`: `SequenceParallel`
- `output`: `ColwiseParallel`, optionally loss-parallel on the vocab dimension
- per block:
  - `attention_norm`, `ffn_norm`: `SequenceParallel`
  - `attention`: `PrepareModuleInput` from sharded sequence layout to replicated layout
  - `attention.wq`, `attention.wk`, `attention.wv`: `ColwiseParallel`
  - `attention.wo`: `RowwiseParallel(output_layouts=Shard(1))`
  - `attention.sinks`: manually sharded across heads with `distribute_tensor(..., [Shard(0)])`

### `apply_moe_ep_tp(...)`

Behavior:

- wraps `transformer_block.moe` with `PrepareModuleInputOutput` when TP is present
- keeps router gate replicated via `NoParallel`
- uses `ReordererSequenceParallel` when TP is borrowed for EP without expert tensor parallel
- expert plan choice:
  - TP only: `GptossTensorParallel`
  - EP only or TP-borrowed-for-EP: `ExpertParallel`
  - EP + ETP: `GptossExpertTensorParallel`
- optionally wraps `BaseExpertParallel` plans in `DualPipeExpertParallel`

### `GptossTensorParallel` and `GptossExpertTensorParallel`

Defined in `torchtitan/models/gpt_oss/expert_parallel.py`.

- `GptossTensorParallel`
  - `mlp1_weight`, `mlp1_bias`: shard expert hidden/output dimension with `Shard(1)`
  - `mlp2_weight`: `Shard(2)`
  - `mlp2_bias`: `Replicate`
- `GptossExpertTensorParallel`
  - adds expert-axis sharding with `Shard(0)` and keeps TP sharding on hidden dims
  - `mlp2_bias`: `[Shard(0), Replicate()]`

## State Dict Adapter

Defined in `torchtitan/models/gpt_oss/state_dict_adapter.py`.

### HF reader selection

- `get_hf_storage_reader(path, from_quantized=False)`
  - returns `HuggingFaceStorageReader(path)` for normal checkpoints
  - returns `QuantizedHuggingFaceStorageReader(path=path, thread_count=4)` for quantized checkpoints

### Important key mappings

- `model.embed_tokens.weight` -> `tok_embeddings.weight`
- `model.layers.{i}.self_attn.q_proj.weight` -> `layers.{i}.attention.wq.weight`
- `model.layers.{i}.self_attn.q_proj.bias` -> `layers.{i}.attention.wq.bias`
- `model.layers.{i}.self_attn.k_proj.weight` -> `layers.{i}.attention.wk.weight`
- `model.layers.{i}.self_attn.v_proj.weight` -> `layers.{i}.attention.wv.weight`
- `model.layers.{i}.self_attn.o_proj.weight` -> `layers.{i}.attention.wo.weight`
- `model.layers.{i}.self_attn.sinks` -> `layers.{i}.attention.sinks`
- `model.layers.{i}.input_layernorm.weight` -> `layers.{i}.attention_norm.weight`
- `model.layers.{i}.post_attention_layernorm.weight` -> `layers.{i}.ffn_norm.weight`
- `model.layers.{i}.mlp.experts.gate_up_proj_blocks` -> `layers.{i}.moe.experts.mlp1_weight`
- `model.layers.{i}.mlp.experts.gate_up_proj_bias` -> `layers.{i}.moe.experts.mlp1_bias`
- `model.layers.{i}.mlp.experts.down_proj_blocks` -> `layers.{i}.moe.experts.mlp2_weight`
- `model.layers.{i}.mlp.experts.down_proj_bias` -> `layers.{i}.moe.experts.mlp2_bias`
- `model.layers.{i}.mlp.router.weight` -> `layers.{i}.moe.router.gate.weight`
- `model.layers.{i}.mlp.router.bias` -> `layers.{i}.moe.router.gate.bias`
- `model.norm.weight` -> `norm.weight`
- `lm_head.weight` -> `output.weight`

`to_hf()` inverts `from_hf_map` and only exports keys present in that inverse map. `from_hf()` assumes the incoming key exists in `from_hf_map`.
