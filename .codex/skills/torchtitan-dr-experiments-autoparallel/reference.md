# Reference

## File Index

- `torchtitan/experiments/autoparallel/configs.py`
  - `AutoParallelCompileConfig(CompileConfig)`
  - `AutoParallelConfig(Trainer.Config)`
- `torchtitan/experiments/autoparallel/llama3/__init__.py`
  - `model_registry(flavor: str) -> ModelSpec`
- `torchtitan/experiments/autoparallel/llama3/config_registry.py`
  - `autoparallel_llama3_debugmodel() -> AutoParallelConfig`
- `torchtitan/experiments/autoparallel/llama3/parallelize_llama.py`
  - `parallelize_llama(...)`
- `torchtitan/experiments/autoparallel/deepseek_v3/__init__.py`
  - `model_registry(flavor: str) -> ModelSpec`
- `torchtitan/experiments/autoparallel/deepseek_v3/config_registry.py`
  - `autoparallel_deepseek_v3_debugmodel() -> AutoParallelConfig`
- `torchtitan/experiments/autoparallel/deepseek_v3/parallelize_deepseekv3.py`
  - `create_functional_router_forward(self) -> Callable`
  - `_moe_forward(...)`
  - `moe_forward(self, x: torch.Tensor) -> torch.Tensor`
  - `monkey_patch_checks(moe)`
  - `monkey_patch_local_map_moe(model, sparse_mesh)`
  - `set_torchtitan_fields(orig, new)`
  - `parallelize_deepseekv3(...)`
  - `_preserve_moe_attributes(original_model, parallel_model)`
- `torchtitan/experiments/autoparallel/local_map_deepseek_v3/__init__.py`
  - `get_model_args()`
  - `model_registry(flavor: str) -> ModelSpec`
- `torchtitan/experiments/autoparallel/local_map_deepseek_v3/args.py`
  - `DeepSeekV3ModelArgs`
  - `get_sample_config() -> DeepSeekV3ModelArgs`
- `torchtitan/experiments/autoparallel/local_map_deepseek_v3/model.py`
  - `DeepSeekV3Model`
- `torchtitan/experiments/autoparallel/local_map_deepseek_v3/config_registry.py`
  - `autoparallel_local_map_deepseek_v3_debugmodel() -> AutoParallelConfig`
- `torchtitan/experiments/autoparallel/local_map_deepseek_v3/parallelize_deepseekv3.py`
  - `set_torchtitan_fields(orig, new)`
  - `parallelize_deepseekv3(...)`
  - `_preserve_moe_attributes(original_model, parallel_model)`
- `torchtitan/experiments/autoparallel/tests/integration_tests.py`
  - `build_autoparallel_test_list() -> list[OverrideDefinitions]`
  - `main()`

## Public API Surface

### Config layer

`AutoParallelCompileConfig` in `configs.py`
- `comms_bucket_reorder_strategy: str = "aten"`
- `autop_force_bf16: bool = False`

`AutoParallelConfig` in `configs.py`
- subclasses `Trainer.Config`
- overrides only the `compile` field so the rest of TorchTitan still sees a normal trainer config

### Model registration layer

`llama3.model_registry()`
- returns a `ModelSpec` with:
  - `name="autoparallel/llama3"`
  - `model=llama3_configs[flavor]`
  - `parallelize_fn=parallelize_llama`
  - `pipelining_fn=pipeline_llm`
  - `build_loss_fn=build_cross_entropy_loss`
  - `state_dict_adapter=Llama3StateDictAdapter`

`deepseek_v3.model_registry()`
- deep-copies `deepseekv3_configs`
- replaces each non-`flex_attn` config's `layer.attention` with `DeepSeekV3Model.Config().layer.attention`
- returns a `ModelSpec` named `"autoparallel/deepseek_v3"`
- adds `post_optimizer_build_fn=register_moe_load_balancing_hook`

`local_map_deepseek_v3.model_registry()`
- builds model args through `get_model_args()`
- returns a `ModelSpec` named `"autoparallel/local_map_deepseek_v3"`
- also uses `register_moe_load_balancing_hook`

### Debug config builders

All three `config_registry.py` files build a debug run with the same broad pattern:
- `hf_assets_path="./tests/assets/tokenizer"`
- `optimizer=OptimizersContainer.Config(lr=8e-4)`
- `lr_scheduler` warmup/linear decay config
- `training.local_batch_size=8`, `training.seq_len=2048`, `training.steps=10`
- `dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test")`
- `metrics.log_freq=1`
- `checkpoint.interval=10`, `last_save_model_only=False`
- `activation_checkpoint.mode="selective"`

Family-specific differences:
- Llama sets `parallelism.pipeline_parallel_schedule="Interleaved1F1B"`
- both DeepSeek configs set `expert_parallel_degree=1` and `expert_tensor_parallel_degree=1`

## Parallelization Functions

### `llama3/parallelize_llama.py`

Inputs:
- `model`
- `parallel_dims: ParallelDims`
- `training: TrainingConfig`
- `model_converters: ModelConvertersContainer.Config`
- `parallelism: ParallelismConfig`
- `compile_config: AutoParallelCompileConfig`
- `ac_config: ActivationCheckpointConfig`
- `dump_folder: str`

Key behavior:
- sets Inductor global flags before AutoParallel runs
- computes `dense_mesh` from enabled `dp_replicate`, `fsdp`, `tp`
- defines synthetic integer token inputs in `input_fn()`
- rejects DDP, CP, and PP
- optionally casts model to BF16 if `compile_config.autop_force_bf16`
- derives `MixedPrecisionPolicy` from `training.mixed_precision_param` and `training.mixed_precision_reduce`
- enters `AutoParallel(...) as autop`
- registers parameter-memory constraints plus input/output sharding constraints
- runs `sharding_placement = autop.optimize_placement(verbose=False)`
- returns `autop.apply_placement(sharding_placement)`
- if tensor-parallel loss parallel is active, adds a forward hook that repacks the output as a TP-only `DTensor`

Important sharding maps:
- input:
  - `"dp_replicate" -> Shard(0)`
  - `"fsdp" -> Shard(0)`
  - `"tp" -> Replicate()`
- output for loss parallel:
  - `"fsdp" -> Shard(0)`
  - `"tp" -> Shard(2)`

### `deepseek_v3/parallelize_deepseekv3.py`

This file contains both the wrapper entrypoint and the MoE adaptation logic.

`create_functional_router_forward(self)`
- returns a pure function that computes router scores, top-k experts, optional bias-adjusted routing, optional debug forced load balancing, route normalization, route scaling, and `num_tokens_per_expert`

`_moe_forward(...)`
- flattens `[bs, slen, dim]` into token rows
- calls the functional router
- uses `reorderer()` to regroup tokens per expert
- dispatches grouped computation through `_run_experts_grouped_mm(...)`
- runs shared experts through raw `F.linear(...)`
- unsorts routed expert outputs
- combines them with router scores and adds shared expert output
- returns both the output tensor and token counts for bias/load-balance tracking

`moe_forward(self, x)`
- calls `_moe_forward(...)`
- mutates `self.tokens_per_expert` under `torch.no_grad()`
- exists because HOPs/local-map support cannot preserve the original mutation pattern inside the graph

`monkey_patch_checks(moe)`
- enforces assumptions required by the monkey patch:
  - `score_before_experts` must be false
  - router gate bias must be absent
  - grouped GEMM must be enabled
  - shared experts must exist
  - shared expert linear layers must be bias-free
  - reorderer must not carry parameters or buffers

`monkey_patch_local_map_moe(model, sparse_mesh)`
- wraps `_moe_forward` with `torch.distributed._tensor.experimental.local_map`
- fixes in/out placements to replicated tensors plus non-distributed callable/reorderer inputs
- replaces `block.moe.forward` with `types.MethodType(moe_forward, block.moe)` for every MoE block

`parallelize_deepseekv3(...)`
- builds a sparse mesh from `dp_replicate`, `efsdp`, `ep`, `etp`
- calls `monkey_patch_local_map_moe(model, sparse_mesh)` before entering AutoParallel
- rejects DDP, CP, and PP
- uses `AutoParallel(..., compile=compile_config)` with `mp_policy=None`
- constrains inputs/outputs according to sparse mesh names
- applies a TP-style loss hook using `sparse_mesh["etp"]` when loss parallel is enabled
- calls `set_torchtitan_fields(model, parallel_mod)`
- calls `_preserve_moe_attributes(model, parallel_mod)`

`set_torchtitan_fields(orig, new)`
- expects `new.layers` to be a `torch.nn.ModuleDict`
- marks each block with `block.moe_enabled = hasattr(block, "moe")`

`_preserve_moe_attributes(original_model, parallel_model)`
- walks `.layers` on both models
- handles both `ModuleDict` and fallback child iteration
- copies `load_balance_coeff` to the parallel MoE modules so optimizer hooks still work

### `local_map_deepseek_v3/parallelize_deepseekv3.py`

This variant is simpler because the underlying AutoParallel test model is already shaped for its local-map use case.

Key differences from the main DeepSeek wrapper:
- explicitly requires `sparse_mesh.ndim == 2`
- injects `layer.moe.mesh = sparse_mesh` and `layer.moe.axis_name = "ep"` into every MoE layer before AutoParallel
- compile handling is boolean `should_compile = compile_config.enable`, with assertions that only `components == ["model"]` and `backend == "inductor"` are supported
- uses `AutoParallel(..., compile=should_compile, dynamic=True)`
- hardcodes `x_sharding = (Shard(0), Shard(0))`
- asserts `not loss_parallel_enabled`

The same metadata-restoration helpers exist here because AutoParallel can still rewrite module structure.

## Local-Map DeepSeek Compatibility Types

`local_map_deepseek_v3/args.py`
- `DeepSeekV3ModelArgs` combines AutoParallel test-model args with TorchTitan `BaseModel.Config`
- `get_sample_config()` returns a small debug configuration:
  - `vocab_size=2048`
  - `max_seq_len=2048`
  - `dim=256`
  - `inter_dim=1024`
  - `moe_inter_dim=256`
  - `n_layers=4`
  - `n_dense_layers=0`
  - `n_heads=16`
  - `top_k=2`
  - `num_experts=4`
  - `num_shared_experts=2`

`local_map_deepseek_v3/model.py`
- wraps `autoparallel._testing.models.dsv3.DeepSeekV3Model`
- inherits `BaseModel`
- deliberately does not call `BaseModel.__init__()` after the AutoParallel test model constructor because that would clear `nn.Module` state

`local_map_deepseek_v3/__init__.py`
- `get_model_args()` deep-copies TorchTitan `deepseekv3_configs`
- replaces each config object with the sample config from AutoParallel's test model
- preserves the original `update_from_config` and `get_nparams_and_flops` callables from the copied TorchTitan config

## Test Harness

`tests/integration_tests.py`
- `build_autoparallel_test_list()` returns one active 4-GPU Llama test:
  - module: `autoparallel.llama3`
  - config: `autoparallel_llama3_debugmodel`
  - overrides: shard degree 2, tensor-parallel degree 2
- includes a commented DeepSeek test block with a TODO to re-enable it later
- `main()` enforces an empty output directory before delegating to `tests.integration_tests.run_tests`

## Dependency Relationships

Inbound dependencies on this module:
- CLI/module loading through TorchTitan experiment discovery elsewhere in the repo

Outbound dependencies used heavily here:
- `torchtitan.components.*` for loss, optimizer hooks, checkpoint/LR/metrics config
- `torchtitan.models.llama3` and `torchtitan.models.deepseek_v3`
- `torchtitan.distributed.pipeline_parallel.pipeline_llm`
- `torchtitan.hf_datasets.text_datasets.HuggingFaceTextDataLoader`
- `torchtitan.protocols.model_spec.ModelSpec`
- `torchtitan.protocols.model_converter.ModelConvertersContainer`
- external `autoparallel.api.AutoParallel`
- external `autoparallel.auto_bucketing.configure_inductor_for_autobucketing`
- external `autoparallel._testing.models.dsv3` in the local-map DeepSeek path

## High-Risk Areas

- `deepseek_v3/parallelize_deepseekv3.py`
  - most custom logic lives here, especially nonstandard MoE execution and metadata copying
- `local_map_deepseek_v3/__init__.py`
  - silently replaces the normal DeepSeek config objects with AutoParallel sample configs
- `tests/integration_tests.py`
  - only partial coverage; DeepSeek support is effectively not integration-tested here
