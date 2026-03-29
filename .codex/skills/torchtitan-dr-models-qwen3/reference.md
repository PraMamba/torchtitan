# Qwen3 Reference

## File Index
- [torchtitan/models/qwen3/__init__.py](torchtitan/models/qwen3/__init__.py)
  Registers the module surface, defines `qwen3_configs`, and exposes `model_registry(flavor)`.
- [torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py)
  Implements `Qwen3TransformerBlock` and `Qwen3Model`.
- [torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py)
  Applies TP, EP/ETP, CP, AC, compile, FSDP/HSDP, or replica wrapping.
- [torchtitan/models/qwen3/state_dict_adapter.py](torchtitan/models/qwen3/state_dict_adapter.py)
  Maps HF parameter names and reshapes grouped MoE expert weights.
- [torchtitan/models/qwen3/config_registry.py](torchtitan/models/qwen3/config_registry.py)
  Defines runnable `Trainer.Config` presets.
- [torchtitan/models/qwen3/README.md](torchtitan/models/qwen3/README.md)
  Human-facing status note: Qwen3 is still under development; CP is called out as unsupported in the README even though `model.py` only explicitly rejects `varlen` CP.

## Key Types And Functions
### `Qwen3TransformerBlock.Config`
Defined in [torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py).
- Inherits `TransformerBlock.Config`.
- Adds `depth_init: bool = True`.
- Adds `moe_enabled: bool = False`.

### `Qwen3TransformerBlock.__init__(config, layer_id, dim, n_layers)`
Defined in [torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py).
- Builds `self.attention`.
- Chooses `self.moe` or `self.feed_forward` based on `config.moe_enabled`.
- Builds `attention_norm` and `ffn_norm`.
- Computes `weight_init_std` as:
  - `0.02 / sqrt(2 * (layer_id + 1))` when `depth_init` is enabled.
  - `0.02 / sqrt(2 * n_layers)` otherwise.

### `Qwen3TransformerBlock.forward(x, freqs_cis, attention_masks, positions=None)`
- Residual attention path first.
- Residual dense-FFN or residual MoE path second.
- `positions` is passed through to attention, which matters for CP-aware sharding and backend-specific attention handling.

### `Qwen3Model.Config.update_from_config(trainer_config, **kwargs)`
Defined in [torchtitan/models/qwen3/model.py](torchtitan/models/qwen3/model.py).
- Synchronizes `self.rope.max_seq_len` to `trainer_config.training.seq_len`.
- Copies `debug.moe_force_load_balance` into `self.layer.moe.router._debug_force_load_balance` when MoE is enabled.
- Rejects CP with `attn_backend == "varlen"`.
- Rejects PP with weight tying.
- Validates that TP divides both attention head counts.

### `Qwen3Model.Config.get_nparams_and_flops(model, seq_len)`
- Calls `get_moe_model_nparams_and_flops(...)`.
- Uses `n_heads` and `2 * head_dim` from the attention config.

### `Qwen3Model.__init__(config)`
- Calls shared `Decoder` init.
- Stores `enable_weight_tying`.
- Ties `tok_embeddings.weight` to `output.weight` immediately when enabled.

### `Qwen3Model.init_weights(buffer_device=None, **kwargs)`
- Re-ties token/output weights in case meta-device initialization broke the alias.
- Delegates the rest to the shared decoder initializer.

### `parallelize_qwen3(model, *, parallel_dims, training, model_converters, parallelism, compile_config, ac_config, dump_folder)`
Defined in [torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py).
- Requires `training.seq_len % parallel_dims.seq_len_divisor == 0`.
- Requires compile when `enable_async_tensor_parallel` is on.
- Detects float8 converter recipes and downgrades TP behavior for rowwise float8 recipes.
- Calls:
  - `apply_non_moe_tp(...)` for dense TP.
  - `apply_moe_ep_tp(...)` for MoE EP/ETP.
  - `apply_cp_to_attention_module(...)` for CP.
  - `apply_ac(...)` for activation checkpointing.
  - `apply_compile_sparse(...)` for per-block compile.
  - `apply_fsdp(...)` or `apply_replicate(...)` for data-parallel wrapping.

### `apply_non_moe_tp(model, tp_mesh, loss_parallel, enable_float8_tensorwise_tp, enable_async_tp, cp_enabled)`
Defined in [torchtitan/models/qwen3/parallelize.py](torchtitan/models/qwen3/parallelize.py).
- Root-module TP plan:
  - `tok_embeddings`: `RowwiseParallel(input=Replicate, output=Shard(1))`
  - `norm`: `SequenceParallel()`
  - `output`: `ColwiseParallel(...)`, optionally loss-parallel
- Per-block TP plan:
  - Attention input layout preparation
  - `wq/wk/wv`: colwise
  - `q_norm/k_norm`: sequence-parallel over dim 2
  - `wo`: rowwise to `Shard(1)`
  - Dense FFN `w1/w3`: colwise, `w2`: rowwise
- Uses float8-specific TP classes only for tensorwise recipes.
- Sets `torch._inductor.config._micro_pipeline_tp = True` when async TP is enabled.

### `Qwen3StateDictAdapter`
Defined in [torchtitan/models/qwen3/state_dict_adapter.py](torchtitan/models/qwen3/state_dict_adapter.py).
- `from_hf_map` includes:
  - embedding/output
  - attention projections and q/k norm weights
  - dense FFN projections
  - layer norms
  - MoE expert projections and router gate
  - skips HF rotary `inv_freq`
- `to_hf(state_dict)`:
  - in dense paths, renames keys directly
  - in MoE paths, splits grouped expert tensors into per-expert HF tensors
  - for `DTensor` grouped experts, records placements/shape/mesh metadata for future reconstruction
  - omits `output.weight` when weight tying is enabled so HF can share the embedding/head weights
- `from_hf(hf_state_dict)`:
  - injects `lm_head.weight` from embeddings if HF omitted it under weight tying
  - reconstructs grouped expert tensors from per-expert HF tensors
  - uses DTensor-aware concatenation if metadata from `to_hf()` is available, otherwise falls back to offline concatenation

## Flavor Catalog
### Dense flavors from `qwen3_configs`
- `debugmodel`
  - `dim=256`, `n_layers=8`, `vocab_size=2048`, `enable_weight_tying=True`
  - attention backend `sdpa`
  - rope max seq len `4096`, theta `1_000_000`
- `debugmodel_flex`
  - same scale as `debugmodel`
  - attention backend `flex`
- `0.6B`
  - `dim=1024`, `n_layers=28`, `n_heads=16`, `n_kv_heads=8`, FFN hidden dim `3072`
- `1.7B`
  - `dim=2048`, `n_layers=28`, FFN hidden dim `6144`
- `4B`
  - `dim=2560`, `n_layers=36`, `n_heads=32`, FFN hidden dim `9728`
- `8B`
  - `dim=4096`, `n_layers=36`, `n_heads=32`, FFN hidden dim `12288`
- `14B`
  - `dim=5120`, `n_layers=40`, `n_heads=40`, FFN hidden dim `17408`
- `32B`
  - `dim=5120`, `n_layers=64`, `n_heads=64`, FFN hidden dim `25600`

### MoE flavors from `qwen3_configs`
- `debugmodel_moe`
  - `dim=256`, `n_layers=8`
  - MoE hidden dim `768`
  - `num_experts=64`
  - router `TokenChoiceTopKRouter(top_k=8, score_func="softmax", route_norm=True)`
- `30B-A3B`
  - `dim=2048`, `n_layers=48`
  - `num_experts=128`
  - `n_heads=32`, `n_kv_heads=4`
  - rope max seq len `262144`
- `235B-A22B`
  - `dim=4096`, `n_layers=94`
  - `num_experts=128`
  - `n_heads=64`, `n_kv_heads=4`
  - rope theta `5_000_000`

## Trainer Presets
Defined in [torchtitan/models/qwen3/config_registry.py](torchtitan/models/qwen3/config_registry.py).

- `qwen3_debugmodel()`
  - `hf_assets_path="./tests/assets/tokenizer"`
  - dataset `c4_test`
  - `lr=8e-4`
  - `seq_len=2048`, `steps=10`
  - selective AC
- `qwen3_debugmodel_flex()`
  - same as debug preset but selects `debugmodel_flex`
- `qwen3_0_6b()`
  - `hf_assets_path="./assets/hf/Qwen3-0.6B"`
  - dataset `c4`
  - `lr=3e-4`
  - `seq_len=4096`, `steps=10`
  - selective AC
- `qwen3_1_7b()`
  - `hf_assets_path="./assets/hf/Qwen3-1.7B"`
  - `lr=8e-4`
  - `warmup_steps=20`
  - `steps=100`
- `qwen3_14b()`
  - `hf_assets_path="./assets/hf/Qwen3-14B"`
  - `steps=3000`
  - explicit parallelism: shard DP auto (`-1`), TP/CP/PP all `1`
  - full AC
- `qwen3_32b()`
  - same shape as 14B preset but `local_batch_size=2`
- `qwen3_moe_debug()`
  - debug tokenizer path
  - selects `debugmodel_moe`
  - enables `expert_parallel_degree=1` and `expert_tensor_parallel_degree=1`
  - selective AC

Notably absent from `config_registry.py`: runnable presets for `4B`, `8B`, `30B-A3B`, and `235B-A22B`, even though the architecture configs exist in `qwen3_configs`.

## Integration Notes
- `model_registry()` binds `pipelining_fn=pipeline_llm`, so pipeline parallel support is inherited from the shared pipeline stack even though `parallelize_qwen3()` itself only handles non-pipeline transforms.
- MoE parallel sharding relies on `apply_moe_ep_tp()` from the Llama4 parallelization module, not a Qwen3-local implementation.
- Dense/multi-head FLOP accounting is delegated to `get_moe_model_nparams_and_flops()` from `torchtitan.models.utils`.

## Questions To Re-Test Changes
- If I add a new attention backend, do I need to update both `Qwen3Model.Config.update_from_config()` and `parallelize_qwen3()` compatibility checks?
- If I add a new MoE flavor, does `from_hf_map` still match the HF naming scheme and grouped expert packing layout?
- If I change weight tying, did I update both constructor-time tying and `init_weights()` meta-device recovery?
