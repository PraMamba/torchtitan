---
name: torchtitan-dr-models-flux
description: Use when working on Torchtitan's FLUX text-to-image module, especially when changing latent preprocessing, dual-text-encoder setup, flow-model training, validation sampling, or checkpoint/inference wiring
---

# Torchtitan FLUX

## Overview
`torchtitan/models/flux` is Torchtitan's text-to-image stack for training and sampling the FLUX transformer over autoencoder latents. It wires together three distinct subsystems: a frozen image/text encoding layer (`model/autoencoder.py`, `model/hf_embedder.py`, `tokenizer.py`), a latent-space flow transformer (`model/model.py`, `model/layers.py`), and recipe glue for training, validation, checkpoint conversion, and inference (`trainer.py`, `validate.py`, `parallelize.py`, `inference/`, `config_registry.py`).

Externally, the module exposes:
- `FluxModel` and `flux_configs` in [torchtitan/models/flux/__init__.py](../torchtitan/models/flux/__init__.py)
- `model_registry(...)` consumers through `config_registry.py`
- `FluxDataLoader` for paired image/text batches in [torchtitan/models/flux/flux_datasets.py](../torchtitan/models/flux/flux_datasets.py)
- `FluxTrainer` and `FluxValidator` for the Torchtitan training loop
- `parallelize_flux(...)` and `parallelize_encoders(...)` for FSDP/CP/activation-checkpointing
- `generate_image(...)` and `save_image(...)` for validation or offline inference
- `FluxStateDictAdapter` for Hugging Face <-> Torchtitan transformer conversion

## Design Logic

### Why the module is split this way
- The FLUX transformer itself stays focused on latent-sequence prediction. `FluxModel.forward(...)` in [torchtitan/models/flux/model/model.py](../torchtitan/models/flux/model/model.py) only sees packed image latents, text embeddings, positional ids, pooled CLIP conditioning, and timesteps.
- Expensive or frozen preprocessing is pushed out of the model and into `preprocess_data(...)` in [torchtitan/models/flux/utils.py](../torchtitan/models/flux/utils.py). That keeps the transformer recipe independent from tokenizer/encoder implementations and lets the trainer decide dtype/device handling.
- The training recipe keeps the autoencoder and text encoders outside the main `model_parts` path. `FluxTrainer.__init__(...)` loads them separately, optionally shards only T5 via `parallelize_encoders(...)`, and uses them as preprocessing utilities rather than trainable model heads.
- Flux uses two attention streams first, then one merged stream. `DoubleStreamBlock` keeps image and text tokens separate until cross-modal attention is done; `SingleStreamBlock` then processes the concatenated sequence. That matches the intended architecture and makes it obvious where to modify text/image coupling behavior.
- Validation and inference reuse the same sampling primitives from [torchtitan/models/flux/inference/sampling.py](../torchtitan/models/flux/inference/sampling.py), so image generation behavior stays aligned between offline inference and periodic validation.

### Main architectural tradeoffs
- The module favors explicit preprocessing and patch packing over a monolithic end-to-end forward path. That makes training logic easier to inspect, but means behavior is spread across `trainer.py`, `utils.py`, and `inference/sampling.py`.
- The CLIP encoder is intentionally not FSDP-sharded in `parallelize_encoders(...)`; only T5 is sharded because CLIP is treated as too communication-heavy relative to compute.
- Context parallelism is applied only to the latent/text sequence tensors, not to an attention-mask-heavy language-model path. `cp_shard(...)` is called directly in training and validation.
- The model-side `get_nparams_and_flops(...)` is intentionally incomplete and returns a dummy FLOP count of `1`, so throughput accounting is not yet faithful for FLUX.

## State Flow

### Training path
1. `flux_debugmodel()`, `flux_dev()`, or `flux_schnell()` in [torchtitan/models/flux/config_registry.py](../torchtitan/models/flux/config_registry.py) builds a `FluxTrainer.Config` with tokenizer, encoders, dataloader, validator, optimizer, and model spec.
2. `FluxDataLoader` builds either `FluxDataset` or `FluxValidationDataset` in [torchtitan/models/flux/flux_datasets.py](../torchtitan/models/flux/flux_datasets.py). Dataset processors resize/crop images, tokenize prompts with both T5 and CLIP, and optionally drop prompts for classifier-free guidance training.
3. `FluxTrainer.__init__(...)` in [torchtitan/models/flux/trainer.py](../torchtitan/models/flux/trainer.py) loads the autoencoder, CLIP encoder, and T5 encoder, chooses encoder dtype, and optionally FSDP-shards T5.
4. `forward_backward_step(...)` calls `preprocess_data(...)` to turn batch tokens and images into:
   - `clip_encodings`
   - `t5_encodings`
   - `img_encodings` from the autoencoder
5. The trainer samples random noise and timesteps, interpolates between clean latents and noise, creates latent/text position ids, and patch-packs both latents and denoising targets.
6. If context parallel is enabled, `cp_shard(...)` shards `latents`, `latent_pos_enc`, `t5_encodings`, `text_pos_enc`, and `target`.
7. `FluxModel.forward(...)` predicts latent noise residuals, and MSE loss is normalized by the total number of latent elements rather than token count from the original image tensor.

### Validation and inference path
- `FluxValidator.validate(...)` in [torchtitan/models/flux/validate.py](../torchtitan/models/flux/validate.py) first generates and saves sample images through `generate_image(...)`, then computes denoising loss using either one stratified timestep per sample or all eight validation timesteps.
- `inference/infer.py` reuses `FluxTrainer` as a loader for distributed inference, partitions prompts round-robin across ranks, restores a checkpoint, and calls `generate_image(...)` in batches.
- `generate_image(...)` tokenizes prompts, encodes T5/CLIP features, optionally builds unconditional encodings for classifier-free guidance, calls `denoise(...)`, and decodes latents back to pixels with the autoencoder.

### Error handling and side effects
- `_validate_dataset(...)` raises `ValueError` for unsupported datasets.
- `FluxDataLoader.__init__(...)` raises if the tokenizer is not a `FluxTokenizerContainer`.
- `FluxDataLoader.Config.__post_init__(...)` enforces `prompt_dropout_prob == 0.0` when validation timesteps are auto-generated.
- `FluxModel.__init__(...)` raises if `hidden_size` is not divisible by `num_heads` or if `axes_dim` does not sum to per-head positional width.
- `load_ae(...)` raises if the autoencoder checkpoint path does not exist.
- `inference(...)` raises if there are fewer prompts than ranks, because distributed FSDP loading would otherwise hang.
- Side effects are concentrated in checkpoint loading, image saving, prompt file reading, Hugging Face model downloads/loading, and EXIF metadata writes in `save_image(...)`.

## Modification Guide

### Add a new FLUX training preset
- Start in [torchtitan/models/flux/config_registry.py](../torchtitan/models/flux/config_registry.py).
- Add a new `FluxTrainer.Config` factory and point `model_spec=model_registry("<flavor>")` at a matching entry in `flux_configs`.
- If the architecture changes, also add a new `FluxModel.Config` entry in [torchtitan/models/flux/__init__.py](../torchtitan/models/flux/__init__.py).

### Change tokenizer or text-encoder behavior
- Edit [torchtitan/models/flux/tokenizer.py](../torchtitan/models/flux/tokenizer.py) for tokenization shape/padding semantics.
- Edit [torchtitan/models/flux/model/hf_embedder.py](../torchtitan/models/flux/model/hf_embedder.py) for encoder loading, output selection, or freezing behavior.
- Recheck `preprocess_data(...)` in [torchtitan/models/flux/utils.py](../torchtitan/models/flux/utils.py), because trainer, validator, and inference all depend on its output keys and dtypes.

### Change latent packing, positional ids, or denoising targets
- The packing helpers live in [torchtitan/models/flux/utils.py](../torchtitan/models/flux/utils.py): `pack_latents(...)`, `unpack_latents(...)`, and `create_position_encoding_for_latents(...)`.
- Training target construction happens in `FluxTrainer.forward_backward_step(...)`.
- Sampling-side symmetry lives in `denoise(...)` in [torchtitan/models/flux/inference/sampling.py](../torchtitan/models/flux/inference/sampling.py). If you change packing or timestep math, update both places.

### Change multimodal attention structure inside the FLUX transformer
- Modify [torchtitan/models/flux/model/layers.py](../torchtitan/models/flux/model/layers.py), especially `DoubleStreamBlock`, `SingleStreamBlock`, `SelfAttention`, `Modulation`, and `LastLayer`.
- If tensor shapes or block config fields change, also update `FluxModel.Config` and the concrete config builders in [torchtitan/models/flux/__init__.py](../torchtitan/models/flux/__init__.py).
- If Hugging Face checkpoint conversion must still work, update `FluxStateDictAdapter` accordingly.

### Add more distributed features
- FSDP, activation checkpointing, and CP entry points live in [torchtitan/models/flux/parallelize.py](../torchtitan/models/flux/parallelize.py).
- Encoder sharding is separate from model sharding; `parallelize_encoders(...)` only handles T5 today.
- For PP/TP support, expect changes in both `parallelize.py` and the training recipe assumptions in [torchtitan/models/flux/trainer.py](../torchtitan/models/flux/trainer.py).

### Change validation or image export behavior
- Validation loop and image-save quota logic are in [torchtitan/models/flux/validate.py](../torchtitan/models/flux/validate.py).
- Prompt batching and distributed prompt partitioning are in [torchtitan/models/flux/inference/infer.py](../torchtitan/models/flux/inference/infer.py).
- Sampling metadata and output-file behavior are in `save_image(...)` in [torchtitan/models/flux/inference/sampling.py](../torchtitan/models/flux/inference/sampling.py).

## Quick Pointers
- Read [reference.md](./reference.md) for the file index, major types, and important function signatures.
- If a change crosses training and inference, verify the same latent packing and timestep conventions are preserved on both sides.
- If a change touches model weights or layer names, check `FluxStateDictAdapter` before assuming checkpoints still round-trip.
