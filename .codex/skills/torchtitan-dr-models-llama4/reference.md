# Llama4 Reference

## File Index

- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/README.md`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/README.md)
  - Only documents tokenizer download for `meta-llama/Llama-4-Scout-17B-16E`.
- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/__init__.py)
  - Flavor map and `model_registry()`.
- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/config_registry.py)
  - Trainer presets.
- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py)
  - `compute_moe_hidden_dim(...)`, `Llama4TransformerBlock`, and `Llama4Model`.
- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py)
  - TP/EP/AC/compile/FSDP orchestration.
- [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/state_dict_adapter.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/state_dict_adapter.py)
  - HF key mapping plus expert weight reshape logic.

## Preset Flavors In `llama4_configs`

- `debugmodel`
  - `dim=256`
  - `n_layers=6`
  - `vocab_size=2048`
  - attention: `n_heads=16`, FlexAttention, `attn_mask_type="block_causal"`, `rope_backend="complex"`
  - RoPE max length `1048576`
  - MoE hidden dim computed from `compute_moe_hidden_dim(256)`
- `17bx16e`
  - `dim=5120`
  - `n_layers=48`
  - `num_experts=16`
  - `n_heads=40`, `n_kv_heads=8`
  - `interleave_moe_layer_step=1` so every block uses MoE
  - RoPE max length `10485760`
- `17bx128e`
  - `dim=5120`
  - `n_layers=48`
  - `num_experts=128`
  - same head layout as `17bx16e`
  - `scaling="none"` in RoPE
  - RoPE max length `1048576`

Shared preset pattern:

- `tok_embeddings=Embedding.Config()`
- `norm=RMSNorm.Config()`
- `output=Linear.Config()`
- `layer=Llama4TransformerBlock.Config(...)`
- `rope=RoPE.Config(..., backend="complex")`

## `compute_moe_hidden_dim(...)`

Defined in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py).

Order of operations:

1. Start from `4 * dim`
2. Apply the Llama-style `2/3` reduction
3. Optionally multiply by `ffn_dim_multiplier`
4. Optionally divide by `top_k + num_shared_experts`
5. Round up to `multiple_of`

Key point:

- This is intentionally different from dense FFN sizing because the auto-scaling is applied before the final rounding.

## `Llama4TransformerBlock`

Defined in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py).

Constructor inputs:

- `config`
- `layer_id`
- `dim`
- `n_layers`

Important config fields:

- `depth_init: bool`
- `every_n_layers_nope: int | None`
- `interleave_moe_layer_step: int`
- `fixed_attn_block_size: int`

Important behavior:

- If `every_n_layers_nope` is set and `layer_id % every_n_layers_nope == 0`, the block clones `config.attention` with `use_rope=False`.
- `self.moe_enabled = (layer_id + 1) % interleave_moe_layer_step == 0`.
- Builds either:
  - `self.moe = config.moe.build(dim=dim)`, or
  - `self.feed_forward = config.feed_forward.build(dim=dim)`.
- `forward(x, freqs_cis, attention_masks, positions=None)`:
  - attention residual first
  - then MoE residual or dense FFN residual
- `init_weights(**kwargs)`:
  - initializes norms
  - passes `weight_init_std` to attention
  - passes `buffer_device` through to MoE init when applicable

## `Llama4Model.Config.update_from_config()`

Defined in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py).

Checks and mutations:

- Replaces `self.rope.max_seq_len` with runtime `training.seq_len`.
- Warns when runtime sequence length exceeds the preset maximum.
- Disables grouped GEMM by mutating `self.layer.moe.experts.use_grouped_mm` when GPU capability is below SM90.
- Rejects `parallelism.context_parallel_degree > 1`.
- Sets `self.layer.moe.router._debug_force_load_balance` from `debug.moe_force_load_balance`.
- Rebuilds the MoE config as `DeepEPMoE.Config` when `parallelism.expert_parallel_comm_backend == "deepep"`.
- Validates TP divisibility for `n_heads` and `n_kv_heads`.

## `Llama4Model.get_attention_masks(...)`

Defined in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/model.py).

Inputs:

- `input_batch`
- `tokenizer`
- `extra_inputs` is accepted but unused

Behavior:

- Starts with `get_causal_mask_mod()`.
- For `attn_mask_type="block_causal"`, also appends `get_document_mask_mod(input_batch, tokenizer.eos_id)`.
- Builds:
  - `rope_mask_mod = and_masks(..., get_fixed_block_mask_mod(fixed_attn_block_size))`
  - `nope_mask_mod = and_masks(...)`
- Returns:
  - `{"rope": create_attention_mask(...), "nope": create_attention_mask(...)}`

Integration note:

- Shared `GQAttention` checks `isinstance(attention_masks, dict)` and selects the `rope` or `nope` key based on `self.use_rope`.

## `config_registry.py` Presets

- `llama4_debugmodel()`
  - tokenizer path: `./tests/assets/tokenizer`
  - dataset: `c4_test`
  - `metrics.log_freq=1`
  - `training.local_batch_size=8`, `seq_len=2048`, `steps=10`
  - `expert_parallel_degree=1`, `expert_tensor_parallel_degree=1`
  - `checkpoint.interval=10`
  - `activation_checkpoint.mode="selective"`
- `llama4_17bx128e()`
  - HF assets: `./assets/hf/Llama-4-Maverick-17B-128E`
  - dataset: `c4`
  - `training.local_batch_size=1`, `seq_len=8192`, `steps=3000`
  - `tensor_parallel_degree=8`
  - `pipeline_parallel_degree=4`
  - `expert_parallel_degree=1`
  - `expert_tensor_parallel_degree=8`
  - `activation_checkpoint.mode="full"`
- `llama4_17bx16e()`
  - HF assets: `./assets/hf/Llama-4-Scout-17B-16E`
  - dataset: `c4`
  - `training.local_batch_size=8`, `seq_len=8192`, `steps=3000`
  - `tensor_parallel_degree=8`
  - `expert_parallel_degree=1`
  - `expert_tensor_parallel_degree=8`
  - `activation_checkpoint.mode="full"`

## `parallelize_llama()` Order Of Operations

Defined in [`/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py`](/home/scbjtfy/torchtitan/torchtitan/models/llama4/parallelize.py).

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
2. Optional TP setup via `apply_non_moe_tp(...)`, with float8 recipe inspection.
3. Validation for DeepEP/HybridEP constraints.
4. Optional MoE EP/ETP setup via `apply_moe_ep_tp(...)`.
5. Optional CP attachment to attention inner modules.
6. Optional activation checkpoint wrapping.
7. Optional sparse model compilation.
8. Optional FSDP/HSDP application or DP replication.

## `apply_non_moe_tp()` Sharding Plan

Root-module plan:

- `tok_embeddings`: `RowwiseParallel`, replicated input, sharded output on sequence dim
- `norm`: `SequenceParallel`
- `output`: `ColwiseParallel`, output sharded when loss parallel is enabled

Per-block dense plan:

- `attention_norm`: `SequenceParallel`
- `attention`: `PrepareModuleInput` or float8-specific equivalent
- `attention.wq`, `attention.wk`, `attention.wv`: colwise
- `attention.wo`: rowwise with sharded output
- `ffn_norm`: `SequenceParallel`
- if not `moe_enabled`:
  - `feed_forward`: prepared replicated input
  - `feed_forward.w1`, `feed_forward.w3`: colwise
  - `feed_forward.w2`: rowwise with sharded output

## `apply_moe_ep_tp()` Behavior

- Requires `ep_mesh` or `tp_mesh`.
- For TP-enabled MoE blocks, installs:
  - `PrepareModuleInputOutput` on `moe`
  - `NoParallel` on `moe.router.gate`
  - optional `ReordererSequenceParallel` when TP is borrowed for EP
  - shared expert sharding rules with `ColwiseParallelWithGradPlacement` and `RowwiseParallel(output_layouts=Partial())`
- Expert execution plan selection:
  - no `ep_mesh`: `TensorParallel()`
  - `ep_mesh` only: `ExpertParallel()` or `DeepEPExpertParallel(...)`
  - `ep_mesh + etp_mesh`: `ExpertTensorParallel()`
- If `dual_pipe_v` is active and the chosen experts plan is a `BaseExpertParallel`, it is wrapped in `DualPipeExpertParallel`.

## `apply_fsdp()` Special Cases

- Builds `MixedPrecisionPolicy(param_dtype, reduce_dtype)`.
- Adds `CPUOffloadPolicy()` when `cpu_offload=True`.
- Uses `get_fsdp_reshard_after_forward_policy(...)`.
- If `model.enable_weight_tying` is true:
  - sharded together: `tok_embeddings`, `norm`, `output`
- Otherwise:
  - embedding sharded separately
  - `norm` and `output` grouped together
- For MoE blocks:
  - routed experts are fully sharded separately from the block
  - if effective FSDP degree is greater than `num_experts`, `shard_placement_fn` switches to `Shard(1)`
- Calls `disable_fsdp_gradient_division(model)` after sharding.
- When `ep_degree > 1`, sets explicit forward/backward prefetch chains over transformer blocks, experts, and final layers.

## `Llama4StateDictAdapter`

Direct mappings include:

- `language_model.model.embed_tokens.weight` -> `tok_embeddings.weight`
- `language_model.model.norm.weight` -> `norm.weight`
- `language_model.lm_head.weight` -> `output.weight`
- attention projections -> `layers.{}.attention.{wq,wk,wv,wo}.weight`
- router gate -> `layers.{}.moe.router.gate.weight`
- shared expert projections -> `layers.{}.moe.shared_experts.{w1,w2,w3}.weight`
- post-attention norm -> `layers.{}.ffn_norm.weight`

Special cases:

- `feed_forward.experts.down_proj` maps to `layers.{}.moe.experts.w2` with transpose.
- HF `feed_forward.experts.gate_up_proj` is split into `layers.{}.moe.experts.w1` and `layers.{}.moe.experts.w3` in `from_hf()`.
- TorchTitan `w1` + `w3` are concatenated back into HF `gate_up_proj` in `to_hf()`.
- `from_hf_map` contains `None: "layers.{}.moe.expert_bias"`, but `to_hf()` skips any mapping whose HF key resolves to `None`.

## Relevant External Checks

- [`/home/scbjtfy/torchtitan/tests/unit_tests/test_fsdp_moe_sharding.py`](/home/scbjtfy/torchtitan/tests/unit_tests/test_fsdp_moe_sharding.py)
  - Confirms expert params shard on dim 1 when effective FSDP degree exceeds expert count.
- [`/home/scbjtfy/torchtitan/torchtitan/models/common/attention.py`](/home/scbjtfy/torchtitan/torchtitan/models/common/attention.py)
  - `GQAttention.forward()` contains the dict-mask branch used by Llama 4 iRoPE layers.
