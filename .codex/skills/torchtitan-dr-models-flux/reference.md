# FLUX Reference

## File Index
- `torchtitan/models/flux/__init__.py`: public exports, `flux_configs`, and model-spec registration surface.
- `torchtitan/models/flux/config_registry.py`: recipe presets returning `FluxTrainer.Config`.
- `torchtitan/models/flux/configs.py`: `FluxEncoderConfig`, `SamplingConfig`, `Inference`.
- `torchtitan/models/flux/flux_datasets.py`: streaming datasets, image preprocessing, CFG prompt dropout, validation timestep injection, and `FluxDataLoader`.
- `torchtitan/models/flux/tokenizer.py`: dual-tokenizer container plus real/test tokenizer implementations.
- `torchtitan/models/flux/utils.py`: preprocessing, latent noise generation, position ids, pack/unpack helpers.
- `torchtitan/models/flux/trainer.py`: `FluxTrainer`, training forward/backward path, latent-target construction, logging.
- `torchtitan/models/flux/validate.py`: `FluxValidator`, sample-image generation during validation, validation-loss computation.
- `torchtitan/models/flux/parallelize.py`: activation checkpointing, context parallel, FSDP/HSDP for model and T5 encoder.
- `torchtitan/models/flux/inference/infer.py`: distributed offline inference entrypoint.
- `torchtitan/models/flux/inference/sampling.py`: denoising schedule, CFG sampling, latent decode, image save.
- `torchtitan/models/flux/model/model.py`: `FluxModel` and top-level transformer config.
- `torchtitan/models/flux/model/layers.py`: FLUX attention/modulation blocks and output layer.
- `torchtitan/models/flux/model/autoencoder.py`: convolutional VAE-style encoder/decoder and checkpoint loader.
- `torchtitan/models/flux/model/hf_embedder.py`: frozen Hugging Face CLIP/T5 wrappers.
- `torchtitan/models/flux/model/state_dict_adapter.py`: Hugging Face <-> Torchtitan weight mapping for the transformer.

## Core Types And Relationships

### Recipe and config types
- `FluxTrainer.Config` in `trainer.py`
  - Extends `Trainer.Config`
  - Replaces `tokenizer` with `FluxTokenizerContainer.Config`
  - Adds `encoder: FluxEncoderConfig`
  - Adds `inference: Inference`
- `FluxValidator.Config` in `validate.py`
  - Extends `Validator.Config`
  - Adds `dataloader`, `all_timesteps`, `save_img_count`, `save_img_folder`, `sampling`
- `FluxEncoderConfig` in `configs.py`
  - Holds T5 path, CLIP path, autoencoder checkpoint path, and `random_init`
- `SamplingConfig` in `configs.py`
  - Governs CFG enablement, CFG scale, and number of denoising steps
- `Inference` in `configs.py`
  - Output dir, prompt file, inference batch size, image size, and nested `SamplingConfig`

### Dataset and tokenizer types
- `FluxDataset` in `flux_datasets.py`
  - Streaming iterable dataset with checkpointable progress
  - Splits data by node via `split_dataset_by_node(...)`
  - Stores empty T5/CLIP token tensors for CFG dropout
- `FluxValidationDataset`
  - Extends `FluxDataset`
  - Injects round-robin timesteps from `[1/8 * (i + 0.5) for i in range(8)]`
- `FluxDataLoader`
  - Requires `FluxTokenizerContainer`
  - Chooses `FluxValidationDataset` only when `generate_timesteps=True`
- `FluxTokenizerContainer`
  - Wraps both T5 and CLIP tokenizers and returns `{"clip": ..., "t5": ...}`
- `FluxTokenizer`
  - Uses `CLIPTokenizer.from_pretrained(...)` or `T5Tokenizer.from_pretrained(...)`
- `FluxTestTokenizer`
  - Uses Torchtitan's `HuggingFaceTokenizer` for deterministic test-mode padding/chunking

### Model types
- `FluxModel.Config` in `model/model.py`
  - Contains top-level dimensions plus nested configs for `EmbedND`, `MLPEmbedder`, `DoubleStreamBlock`, `SingleStreamBlock`, and `LastLayer`
- `FluxModel`
  - Main flow transformer over packed latents and text sequences
  - Members: `img_in`, `time_in`, `vector_in`, `txt_in`, `double_blocks`, `single_blocks`, `final_layer`
- `DoubleStreamBlock` in `model/layers.py`
  - Separate image/text modulation and attention prep
  - Performs joint attention over concatenated Q/K/V, then splits outputs back into image/text paths
- `SingleStreamBlock`
  - Processes concatenated text+image sequence after fusion
- `LastLayer`
  - AdaLN-style conditioning from timestep+vector embedding into final latent-patch output
- `AutoEncoder`
  - `encode(...)` scales and shifts the diagonal-Gaussian latent sample
  - `decode(...)` inverts the same affine transform before the decoder
- `FluxEmbedder`
  - Frozen CLIP text or T5 encoder wrapper, selecting `pooler_output` or `last_hidden_state`
- `FluxStateDictAdapter`
  - Direct maps plus concat/split plans for combined QKV/MLP linear layers

## Important Function Signatures
- `preprocess_data(device, dtype, *, autoencoder, clip_encoder, t5_encoder, batch) -> dict[str, Tensor]`
  - Adds `clip_encodings`, `t5_encodings`, and optionally `img_encodings`
- `create_position_encoding_for_latents(bsz, latent_height, latent_width, position_dim=3) -> Tensor`
- `pack_latents(x: Tensor) -> Tensor`
- `unpack_latents(x: Tensor, latent_height: int, latent_width: int) -> Tensor`
- `FluxTrainer.forward_backward_step(*, input_dict, labels, global_valid_tokens=None) -> torch.Tensor`
- `FluxValidator.validate(model_parts, step) -> None`
- `parallelize_flux(model, *, parallel_dims, training, model_converters, parallelism, compile_config, ac_config, dump_folder)`
- `parallelize_encoders(t5_model, clip_model, parallel_dims, *, training)`
- `generate_image(device, dtype, img_height, img_width, enable_classifier_free_guidance, denoising_steps, classifier_free_guidance_scale, model, prompt, autoencoder, tokenizer, t5_encoder, clip_encoder) -> torch.Tensor`
- `denoise(device, dtype, model, img_width, img_height, denoising_steps, clip_encodings, t5_encodings, enable_classifier_free_guidance=False, empty_t5_encodings=None, empty_clip_encodings=None, classifier_free_guidance_scale=None) -> torch.Tensor`
- `load_ae(ckpt_path, autoencoder_params, device='cuda', dtype=torch.bfloat16, random_init=False) -> AutoEncoder`

## Data And Tensor Conventions
- Input images are normalized to `[-1, 1]` in `_process_cc12m_image(...)`.
- Autoencoder latents are `16` channels and spatially downsampled by `8x`.
- Packed latent sequences use `2x2` patches, so sequence length is `(latent_h / 2) * (latent_w / 2)`.
- Position ids are 3D vectors where channel `1` is row index and channel `2` is column index; channel `0` stays zero in `create_position_encoding_for_latents(...)`.
- `FluxModel.forward(...)` expects:
  - `img`: `[batch, packed_latent_tokens, in_channels]`
  - `txt`: `[batch, text_tokens, context_in_dim]`
  - `y`: pooled CLIP conditioning vector
  - `img_ids` / `txt_ids`: per-token position ids used by `EmbedND`

## Key Behavioral Details
- CFG training is implemented by replacing token tensors with precomputed empty-prompt tensors independently for T5 and CLIP in `FluxDataset.__iter__(...)`.
- Validation can either use one stratified timestep per example (`generate_timesteps=True`) or expand every sample across all eight validation timesteps (`all_timesteps=True` in `FluxValidator`).
- `FluxTrainer` explicitly rejects gradient accumulation.
- `parallelize_flux(...)` ignores `compile_config` and `model_converters` today; the function signature matches Torchtitan's broader model-spec contract but FLUX does not yet implement compile support here.
- `FluxModel.Config.update_from_config(...)` is a no-op.
- `FluxModel.Config.get_nparams_and_flops(...)` logs a warning and returns `(nparams, 1)`.
- `save_image(...)` writes EXIF metadata including software tag, make/model, and optionally prompt text.

## State-Dict Adapter Notes
- `from_hf_map_direct` handles direct parameter renames such as `x_embedder.* -> img_in.*` and `context_embedder.* -> txt_in.*`.
- `combination_plan` joins separate HF Q/K/V and MLP projections into Torchtitan's fused linear weights for:
  - `single_blocks.{i}.linear1`
  - `double_blocks.{i}.txt_attn.qkv`
  - `double_blocks.{i}.img_attn.qkv`
- `_swap_scale_shift(...)` compensates for the scale/shift ordering difference in the final AdaLN layer.
- If `hf_assets_path` points at a multimodal FLUX repo root containing `model_index.json`, the adapter automatically descends into the `transformers/` subdirectory before looking for safetensor index files.
