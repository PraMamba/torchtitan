# Reference

## Public Entry Points

### `scripts/checkpoint_conversion/convert_from_hf.py`

#### `convert_from_hf(input_dir, output_dir, model_name, model_flavor)`
- Decorator: `@torch.inference_mode()`
- Inputs:
  - `input_dir`: path-like Hugging Face checkpoint directory read by `HuggingFaceStorageReader`
  - `output_dir`: path-like DCP checkpoint destination
  - `model_name`: TorchTitan model package suffix, used in `importlib.import_module(f"torchtitan.models.{model_name}")`
  - `model_flavor`: flavor string forwarded to `model_registry(model_flavor)`
- Output:
  - no return value; writes a DCP checkpoint to `output_dir`
- Internal sequence:
  - resolve `model_spec`
  - instantiate `model_config.build()` on CPU
  - wrap with `ModelWrapper`
  - build `sd_adapter = model_spec.state_dict_adapter(model_config, None)`
  - allocate TorchTitan state dict with `model._get_state_dict()`
  - transform empty TT dict to HF layout with `sd_adapter.to_hf(...)`
  - load tensors with `dcp.load(..., storage_reader=HuggingFaceStorageReader(path=input_dir))`
  - remap back with `sd_adapter.from_hf(...)`
  - persist with `dcp.save(..., checkpoint_id=output_dir)`

#### CLI defaults
- positional `input_dir: Path`
- positional `output_dir: Path`
- optional `--model_name`, default `"llama3"`
- optional `--model_flavor`, default `"8B"`

### `scripts/checkpoint_conversion/convert_to_hf.py`

#### `convert_to_hf(input_dir, output_dir, model_name, model_flavor, hf_assets_path, export_dtype)`
- Decorator: `@torch.inference_mode()`
- Inputs:
  - `input_dir`: DCP checkpoint directory
  - `output_dir`: Hugging Face export destination
  - `model_name`, `model_flavor`: same registry lookup pattern as `convert_from_hf`
  - `hf_assets_path`: passed into the adapter so it can access Hugging Face asset metadata such as index mapping
  - `export_dtype`: string key into `torchtitan.config.TORCH_DTYPE_MAP`
- Output:
  - no return value; writes Hugging Face checkpoint files via `HuggingFaceStorageWriter`
- Internal sequence:
  - resolve `model_spec`
  - instantiate CPU model and `ModelWrapper`
  - build adapter with `model_spec.state_dict_adapter(model_config, hf_assets_path)`
  - allocate TorchTitan state dict and load DCP tensors with `dcp.load`
  - remap TT -> HF with `sd_adapter.to_hf(...)`
  - build `HuggingFaceStorageWriter(path=output_dir, save_distributed=True, fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping, enable_consolidation=True, thread_count_consolidation=5)`
  - optionally cast tensors to `float16` or `bfloat16`
  - save through `dcp.save(..., storage_writer=storage_writer)`

#### CLI defaults
- positional `input_dir: Path`
- positional `output_dir: Path`
- optional `--hf_assets_path`, default `./assets/hf/Llama-3.1-8B`
- optional `--model_name`, default `"llama3"`
- optional `--model_flavor`, default `"8B"`
- optional `--export_dtype`, choices `float16|bfloat16|float32`, default `"float32"`

### `scripts/checkpoint_conversion/numerical_tests_example.py`

#### `loss_fn(logits1, logits2)`
- Converts `logits1` with `F.log_softmax`
- Converts `logits2` with `F.softmax`
- Returns `F.kl_div(..., reduction="mean")`

#### `forward_hf(model_name, model_path: str | None, input_ids)`
- Decorator: `@torch.no_grad`
- Loads `AutoModelForCausalLM.from_pretrained(model_path or model_name)`
- Moves model to `device_type`
- For each prompt tensor:
  - moves prompt to device
  - runs `model.generate(..., max_length=prompt_len + 1, do_sample=False, output_logits=True, return_dict_in_generate=True)`
  - stacks `outputs.logits`
- Returns `list[Tensor]` of generated logits per prompt

#### `forward_tt(model_name, config_name, checkpoint_path, test_set)`
- Decorator: `@torch.no_grad`
- Uses `ConfigManager.parse_args(["--module", model_name, "--config", config_name])`
- Pulls `config.model_spec.model`
- Calls `model_config.update_from_config(trainer_config=config)`
- Builds model, materializes it with `to_empty(device=device)` and `init_weights(buffer_device=device)`
- Wraps model in `ModelWrapper`, allocates `state_dict`, and loads DCP weights with `dcp.load`
- For each prompt:
  - ensures batch dimension
  - runs `model(input_ids)[:, -1, :].unsqueeze(1)` to capture final-token logits
- Returns `list[Tensor]`

## Key External Types And Dependencies

- `torchtitan.components.checkpoint.ModelWrapper`
  - Used as the adapter-friendly wrapper that exposes `_get_state_dict()`.
- `torch.distributed.checkpoint.HuggingFaceStorageReader`
  - Reads Hugging Face checkpoint directories into a prepared state dict.
- `torch.distributed.checkpoint.HuggingFaceStorageWriter`
  - Writes Hugging Face-format output with shard/index metadata.
- `torchtitan.config.TORCH_DTYPE_MAP`
  - Maps CLI strings to real `torch.dtype` values during export.
- `ConfigManager`
  - Used only by the numerical example to build a live TorchTitan config and model spec.
- `AutoModelForCausalLM`
  - Hugging Face baseline model for behavioral comparison.

## Data Structure Relationships

- `model_spec`
  - Produced by `model_registry(model_flavor)`
  - Supplies:
    - `model_spec.model`: the model config object used to build the TorchTitan model
    - `model_spec.state_dict_adapter`: adapter factory used for remapping
- `state_dict`
  - TorchTitan-native mapping produced by `ModelWrapper._get_state_dict()`
  - Read/written by DCP
- `hf_state_dict`
  - Hugging Face-shaped mapping produced by `sd_adapter.to_hf(state_dict)`
  - Read by `HuggingFaceStorageReader` and written by `HuggingFaceStorageWriter`
- `fqn_to_index_mapping`
  - Adapter-provided metadata needed when exporting sharded Hugging Face checkpoints

## Behavior Notes Worth Remembering

- `convert_from_hf.py` passes `None` as the adapter's second argument; `convert_to_hf.py` passes `hf_assets_path`. That difference reflects the export path's need for Hugging Face asset metadata.
- The validation example is only an example harness; it hardcodes `hf_model_name = "meta-llama/Meta-Llama-3-8B"`, `model_name = "llama3"`, and specific checkpoint paths under `outputs/test_checkpoint`.
- The README's “sanity check” recommends greedy decoding through generation scripts, while the code example performs a stronger KL-divergence comparison.
