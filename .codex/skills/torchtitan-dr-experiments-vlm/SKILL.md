---
name: torchtitan-dr-experiments-vlm
description: Use when working on TorchTitan's VLM experiment, especially when changing multimodal dataset processing, image patch collation, SigLIP2 plus Llama3 model wiring, or VLM-specific parallelization and test coverage.
---

# TorchTitan Experiments VLM

## Overview

`/home/scbjtfy/torchtitan/torchtitan/experiments/vlm` is TorchTitan's experimental vision-language stack. It extends the core Llama 3 runtime with a SigLIP2-style vision encoder, a projector that injects visual embeddings into text token positions, a Hugging Face multimodal dataloader that handles native-resolution images and interleaved text/image samples, and a VLM-specific non-PP parallelization path. The design goal is to keep the LLM side compatible with TorchTitan's existing `Trainer`, `ModelSpec`, loss builder, checkpointing, and FSDP/CP machinery while isolating multimodal concerns in dataset, encoder, and scatter logic.

## Public Surface

- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/__init__.py`
  - `llama3_siglip2_configs`: preset model config map, currently only `debugmodel`.
  - `model_registry(flavor: str) -> ModelSpec`: exports the VLM experiment as module name `vlm`.
  - Re-exports `HuggingFaceMultiModalDataLoader`, `parallelize_vlm`, `Llama3Siglip2Transformer`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/config_registry.py`
  - `vlm_debugmodel() -> MultiModalTrainerConfig`: canonical training preset.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/configs.py`
  - `MultiModalTrainerConfig`: `Trainer.Config` subclass that narrows `dataloader` to the multimodal loader config.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_datasets.py`
  - `HuggingFaceMultiModalDataset`: iterable/stateful dataset with optional sample packing.
  - `HuggingFaceMultiModalDataLoader`: configurable `ParallelAwareDataloader` wrapper.
  - `MM_DATASETS`: dataset registry for `obelics`, `cc12m`, and `cc12m-test`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_collator_nld.py`
  - `MultiModalCollatorNLD`: batches text plus image patches into LLM and encoder inputs.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/model.py`
  - `Projector`: MLP bridge from encoder embedding space to Llama embedding space.
  - `Llama3Siglip2Transformer`: combined multimodal model.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/siglip2.py`
  - `VisionTransformer`, `VisionEmbeddings`, `TransformerLayer`, `Attention`, `FeedForward`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/infra/parallelize.py`
  - `parallelize_vlm(...)`: applies AC, compile, FSDP/HSDP/replication, and CP validation rules.

## Design Logic

- The experiment reuses the existing Llama 3 decoder instead of building a fresh multimodal decoder. `Llama3Siglip2Transformer` subclasses `torchtitan.models.llama3.Llama3Model`, then adds only `encoder` and `projector`. That keeps tokenizer, decoder blocks, attention mask plumbing, and output heads aligned with core TorchTitan.
- Visual data is normalized into fixed `N x L x D` tensors before it reaches the model. The README and `MultiModalCollatorNLD` enforce two invariants: a fixed encoder batch size `max_images_per_batch` and a fixed per-image patch length `max_patches_per_image`. This trades wasted padding for distributed-training compatibility.
- Image-to-token insertion is handled as placeholder replacement rather than cross-attention. `_process_mm_sample()` converts image positions to repeated `<|image|>` placeholder tokens bracketed by BOI/EOI markers, and `_scatter_img_tokens()` overwrites the corresponding text embeddings with projected vision features in-place.
- Dataset processing is deliberately split into sample normalization, optional packing, and final collate. `mm_datasets.py` interprets source dataset schemas and token/image structure, `packing.py` concatenates multimodal samples up to `seq_len`, and `mm_collator_nld.py` performs the final tensorization and shape padding.
- Parallelization is intentionally conservative. `parallelize_vlm()` rejects TP outright and only supports CP with SDPA attention, because the encoder path and legacy local tensor outputs have not been generalized yet.

## Core Data Structures

- `MultiModalTrainerConfig` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/configs.py`
  - Extends `Trainer.Config` and replaces the dataloader config type with `HuggingFaceMultiModalDataLoader.Config`.
- `SpecialTokens` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/args.py`
  - Holds string/token-id pairs for image, BOI, EOI, pad, and ignore values.
  - `from_tokenizer()` assumes those special tokens already exist in the HF tokenizer's added-token decoder.
- `Siglip2Config` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/args.py`
  - Defines encoder width, depth, heads, positional embedding grid, patch size, merge size, and attention backend settings.
- `Llama3Siglip2Transformer.Config` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/model.py`
  - Inherits the full `Llama3.Config` and adds `encoder: Siglip2Config`.
- `HuggingFaceMultiModalDataset` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_datasets.py`
  - Persistent state is `_sample_idx` plus optional `SamplePacker` buffers.
- `HuggingFaceMultiModalDataLoader.Config` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_datasets.py`
  - Key knobs: `dataset`, `infinite`, `max_images_per_batch`, `max_patches_per_image`, `patch_size`, `spatial_merge_size`, `packing_buffer_size`.
- `MultiModalCollatorNLD` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_collator_nld.py`
  - The batching contract between dataset samples and model inputs.
- `SamplePacker` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/utils/packing.py`
  - Uses `sample_buffer` and `packed_samples` deques to greedily concatenate samples by descending token length.

## State Flow

1. Registration and config selection:
   `model_registry()` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/__init__.py` packages the VLM model as a `ModelSpec`. `vlm_debugmodel()` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/config_registry.py` binds that model spec to optimizer, scheduler, checkpoint, activation-checkpoint, and multimodal dataloader settings.
2. Dataset selection and per-sample normalization:
   `HuggingFaceMultiModalDataset.__init__()` validates the configured dataset through `_validate_mm_dataset()`, loads it via `datasets.load_dataset`, shards it with `split_dataset_by_node()`, and stores tokenizer, patching, and packing parameters. `_process_obelics_sample()` and `_process_cc12_wd_sample()` both funnel into `_process_mm_sample()`.
3. Text and image encoding:
   `_process_mm_sample()` normalizes scalar inputs to aligned text/image lists, processes every image through `process_image()`, computes placeholder token counts with `calculate_image_tokens()`, then renders interleaved text with `process_text_with_images()`. It tokenizes that string, clones it into `labels`, and masks BOI/EOI/IMG token ids to `ignore_id`.
4. Packing and iteration:
   `HuggingFaceMultiModalDataset.__iter__()` resumes from `_sample_idx`, skips malformed or too-long samples, and either yields each processed sample directly or feeds it into `SamplePacker`. `SamplePacker.get_next_batch()` concatenates `input_ids`, `labels`, and `pixel_values` across multiple samples until batch-sized output is ready.
5. Final collation:
   `MultiModalCollatorNLD.__call__()` may drop trailing samples if their combined image count would exceed `max_images_per_batch`. It flattens all images, converts each one to patches plus coordinates via `convert_to_patches()` and `pad_patches()`, then pads the encoder batch with `pad_empty_images_to_target_batch_size()`. In parallel it pads/truncates text to `seq_len + 1`, pads batch size, and returns shifted `(input_ids[:, :-1], labels[:, 1:])`.
6. Model forward:
   `Llama3Siglip2Transformer.forward()` embeds text with `tok_embeddings`. If the encoder is present, it derives `pixel_masks` from `grid_thw`, runs `VisionTransformer.forward()` over patch tensors, projects the output with `Projector`, and injects the resulting vectors into text-token embedding positions selected by `special_tokens.img_id`. The combined sequence then runs through the inherited Llama 3 layers and output head.
7. Attention mask flow:
   `Llama3Siglip2Transformer.get_attention_masks()` gets Llama masks from the superclass and encoder masks from `VisionTransformer.get_attention_masks()`, returning a dict with separate `llama3_masks` and `encoder_masks`.
8. Parallelization and training:
   `parallelize_vlm()` applies AC to both decoder and encoder, optionally compiles both, then wraps them in FSDP/HSDP or replication. It enforces `seq_len` divisibility and rejects unsupported TP/CP configurations before training reaches runtime execution.

## Error Handling And Side Effects

- `_process_mm_sample()` returns `None` for bad alignment between text/image lists, partial image-processing failure, or any exception during sample processing.
- `process_image()` swallows fetch/parse/resize failures and logs a warning, so malformed inputs quietly drop samples upstream.
- `pad_patches()` returns `(None, None)` if truncation would be required, and logs an error stating that truncation should not happen.
- `MultiModalCollatorNLD.__call__()` mutates the incoming `batch` list by popping samples from the end until total image count fits `max_images_per_batch`.
- `parallelize_vlm()` raises `NotImplementedError` for TP and for CP with non-SDPA attention.
- `SpecialTokens.from_tokenizer()` assumes all required special tokens are present; missing added-token entries will raise a `KeyError` when building the mapping.

## Common Modification Scenarios

- Add a new multimodal dataset format:
  Start in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_datasets.py`. Add a dataset entry to `MM_DATASETS`, write a schema-specific sample processor, and funnel it into `_process_mm_sample()` if the dataset can be represented as interleaved text/image slots.
- Change image patch sizing or visual-token compression:
  Update `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/utils/image.py` for resize and patch-count rules, then keep `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/utils/text.py` and `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/model.py` consistent so placeholder counts still match scattered encoder tokens.
- Swap the vision encoder or projector architecture:
  Edit `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/args.py`, `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/siglip2.py`, and `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/model.py` together. The critical contract is that the encoder output shape must remain compatible with `_scatter_img_tokens()` after projection.
- Add TP or PP support:
  The main blocker is `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/infra/parallelize.py`, which currently rejects TP and assumes no pipelining. You would also need to inspect mask generation and image-token scatter in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/model.py`.
- Make sample packing smarter:
  Modify `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/utils/packing.py` and review resume semantics in `HuggingFaceMultiModalDataset.state_dict()` and `load_state_dict()`, because packer buffers are part of the checkpointed dataloader state.
- Expand test coverage:
  Add new integration variants in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/tests/integration_tests.py`. Existing coverage is FSDP-only and sets higher `max_patches_per_image` / `max_images_per_batch` overrides explicitly.

## File Map

- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/__init__.py`: module export surface and model registry.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/config_registry.py`: train preset(s).
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/configs.py`: trainer config specialization.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/`: schema processing, packing, collate, and image/text helpers.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/infra/parallelize.py`: multimodal model wrapping for distributed training.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/`: special-token config, encoder, projector, and combined model.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/tests/integration_tests.py`: experiment integration matrix.

## See Also

- `reference.md`: denser API and file-by-file responsibilities for `experiments-vlm`.
