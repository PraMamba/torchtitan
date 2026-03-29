# DeepSeek-V3 Reference

## Key Types

### `Attention.Config`
Defined in [torchtitan/models/deepseek_v3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/model.py).

- Core architecture: `n_heads`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`
- Projection selection:
  `wq` is required when `q_lora_rank == 0`; `wq_a` and `wq_b` are required otherwise
- Runtime behavior:
  `attn_backend`, `attn_mask_type`, `mscale`, `rope_factor`, `rope_max_seq_len`, `rope_original_seq_len`

### `DeepSeekV3TransformerBlock.Config`
Defined in [torchtitan/models/deepseek_v3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/model.py).

- Extends `TransformerBlock.Config`
- Adds `n_dense_layers`, which decides the dense-prefix/MoE boundary by comparing against `layer_id`

### `DeepSeekV3Model.Config`
Defined in [torchtitan/models/deepseek_v3/model.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/model.py).

- Main fields: `dim`, `n_layers`, `vocab_size`, `layer`
- Important hooks:
  `update_from_config(trainer_config=...)`
  `get_nparams_and_flops(model, seq_len)`

### `DeepSeekV3StateDictAdapter`
Defined in [torchtitan/models/deepseek_v3/state_dict_adapter.py](/home/scbjtfy/torchtitan/torchtitan/models/deepseek_v3/state_dict_adapter.py).

- Inherits `MoEStateDictAdapter`
- Owns HF name mapping and grouped-expert split/concat logic
- Switches query mapping based on whether the model uses direct query projection or LoRA-style query projection

## Function Index

### Model construction and execution
- `Attention.__init__(config, dim)`
  builds query, KV, output projections and picks `FlexAttentionWrapper` or `ScaledDotProductAttentionWrapper`
- `Attention.forward(x, freqs_cis, attention_masks, positions=None)`
  computes MLA queries/keys/values, applies rotary embedding to the rotary slices only, runs the backend attention kernel, then projects back through `wo`
- `Attention.init_weights(init_std=...)`
  initializes all linear projections and RMSNorm layers
- `DeepSeekV3TransformerBlock.forward(x, freqs_cis, attention_masks, positions=None)`
  applies residual attention and then dense FFN or MoE depending on `moe_enabled`
- `DeepSeekV3Model.Config.update_from_config(trainer_config=...)`
  synchronizes rope settings, validates attention/backend combinations, toggles grouped GEMM support, forwards debug load-balancing flags, and swaps in `DeepEPMoE.Config` for DeepEP/HybridEP

### Registry and training presets
- `model_registry(flavor)`
  returns the DeepSeek-V3 `ModelSpec`
- `deepseek_v3_debugmodel()`
  tiny test config using `c4_test`, log frequency 1, selective AC, no EP/ETP
- `deepseek_v3_debugmodel_flex_attn()`
  same as debugmodel but swaps to the `debugmodel_flex_attn` architecture flavor
- `deepseek_v3_16b()`
  main 16B training preset with EP=8, selective AC, compile on `loss`
- `deepseek_v3_671b()`
  671B preset with float8 converters for output/router and grouped expert GEMMs

### Parallelization
- `parallelize_deepseekv3(model, parallel_dims, training, model_converters, parallelism, compile_config, ac_config, dump_folder)`
  master entrypoint for all distributed transforms
- `apply_non_moe_tp(model, tp_mesh, loss_parallel, enable_float8_tensorwise_tp, cp_enabled)`
  applies embedding/norm/output TP and then per-block attention/dense-FFN TP, leaving MoE handling to `apply_moe_ep_tp()`

### Checkpoint conversion
- `DeepSeekV3StateDictAdapter.get_hf_storage_reader(path, from_quantized=False)`
  returns `QuantizedHuggingFaceStorageReader` for quantized loads, otherwise standard `HuggingFaceStorageReader`
- `DeepSeekV3StateDictAdapter.to_hf(state_dict)`
  converts TorchTitan names to HF names and splits grouped expert tensors into per-expert tensors
- `DeepSeekV3StateDictAdapter.from_hf(hf_state_dict)`
  converts HF names to TorchTitan names and re-concatenates expert tensors into grouped tensors

## Flavor Matrix

### `debugmodel`
- `dim=256`, `n_layers=6`, `vocab_size=2048`
- `n_dense_layers=1`
- `q_lora_rank=0`, `attn_backend=sdpa`
- MoE: `num_experts=8`, `num_shared_experts=2`, router `top_k=3`, `score_func=softmax`

### `debugmodel_flex_attn`
- Same base shape as `debugmodel`
- Switches to `attn_backend=flex` and `attn_mask_type=block_causal`

### `16B`
- `dim=2048`, `n_layers=27`, `vocab_size=102400`
- `n_dense_layers=1`
- Still uses direct query projection (`q_lora_rank=0`)
- MoE: `64` experts, `2` shared experts, `top_k=6`

### `236B`
- `dim=5120`, `n_layers=60`
- Switches to low-rank query projection: `q_lora_rank=1536`, `wq_a`, `wq_b`
- Router adds grouped routing controls: `num_expert_groups=8`, `num_limited_groups=3`, `route_scale=16.0`

### `671B`
- `dim=7168`, `n_layers=61`, `vocab_size=129280`
- `n_dense_layers=3`
- Low-rank query projection with grouped router
- Router uses `score_func=sigmoid`, `route_norm=True`, `top_k=8`, `num_shared_experts=1`

## Conversion Details
- `from_hf_map` maps:
  embedding, per-layer attention/KV projections, dense MLP weights, norms, MoE expert weights, shared expert weights, router gate, expert bias, final norm, and `lm_head`
- Query mapping is conditional:
  HF `q_proj` maps to `attention.wq.weight` for direct-query models
  HF `q_a_proj`/`q_a_layernorm`/`q_b_proj` map to `wq_a`/`q_norm`/`wq_b` for low-rank-query models
- MoE export path:
  grouped expert tensors stay grouped in TorchTitan but are split into individual HF expert tensors
- MoE import path:
  individual HF expert tensors are buffered per layer and concatenated back into grouped tensors

## Operational Commands from the Module README
- Download tokenizer for 671B:
  `python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer`
- Download tokenizer for 16B-compatible testing:
  `python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer`
- Debug run:
  `MODEL=deepseek_v3 CONFIG=deepseek_v3_debugmodel ./run_train.sh`
- 16B run:
  `MODEL=deepseek_v3 CONFIG=deepseek_v3_16b ./run_train.sh`
- 671B run:
  `MODEL=deepseek_v3 CONFIG=deepseek_v3_671b ./run_train.sh`
- Offline HF to DCP conversion:
  `python scripts/checkpoint_conversion/convert_from_hf.py <hf_checkpoints_dir> <dcp_output_dir> --model_name deepseek_v3 --model_flavor 671B`
