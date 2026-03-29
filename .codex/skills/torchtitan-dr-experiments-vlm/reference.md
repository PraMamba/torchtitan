# VLM Reference

## File Index

- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/README.md`
  - Declares the core experiment constraints: native resolution, interleaved data, fixed `N` image batch size, fixed `L` patch length, and planned future TP/PP/FlexAttention work.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/__init__.py`
  - Builds `llama3_siglip2_configs` from the core Llama 3 debug preset.
  - `model_registry(flavor)` returns `ModelSpec(name="vlm", parallelize_fn=parallelize_vlm, build_loss_fn=build_cross_entropy_loss, state_dict_adapter=None)`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/config_registry.py`
  - `vlm_debugmodel()` wires the experiment into `Trainer`-style config construction.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/configs.py`
  - `MultiModalTrainerConfig(Trainer.Config)` only changes the dataloader type.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_collator_nld.py`
  - Final multimodal batch assembly into decoder tokens plus encoder patches.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_datasets.py`
  - Dataset registry, schema normalization, iterable state machine, state dict save/restore, and configurable dataloader wrapper.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/utils/image.py`
  - Image decoding, resize policy, patchification, coordinate-grid generation, patch padding, and encoder-batch padding.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/utils/packing.py`
  - Greedy sample packer.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/utils/text.py`
  - Sequence padding and placeholder-text expansion.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/infra/parallelize.py`
  - AC, compile, FSDP/HSDP/replication logic; TP unsupported.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/args.py`
  - `SpecialTokens` and `Siglip2Config`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/model.py`
  - Projector, image-token scatter helper, combined Llama 3 plus Siglip2 model.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/siglip2.py`
  - Vision transformer internals and resized positional embeddings.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/tests/integration_tests.py`
  - Current integration matrix, currently one VLM FSDP flavor.

## Key Types And Responsibilities

### Config And Registration

- `MultiModalTrainerConfig` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/configs.py`
  - Inherits all `Trainer.Config` fields.
  - Overrides `dataloader` to `HuggingFaceMultiModalDataLoader.Config`.
- `llama3_siglip2_configs` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/__init__.py`
  - Uses `_get_dict()` to preserve nested dataclasses when copying the base Llama config.
  - The `debugmodel` config replaces only the encoder settings, not the decoder defaults.
- `vlm_debugmodel()` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/config_registry.py`
  - Important defaults:
    - `hf_assets_path="./tests/assets/tokenizer"`
    - `training.local_batch_size=8`
    - `training.seq_len=2048`
    - `training.steps=10`
    - `dataloader.dataset="cc12m-test"`
    - selective activation checkpointing enabled.

### Token And Encoder Config

- `SpecialTokens` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/args.py`
  - Fields:
    - `img_token`, `img_id`
    - `boi_token`, `boi_id`
    - `eoi_token`, `eoi_id`
    - `pad_token`, `pad_id`
    - `ignore_id=-100`
  - `from_tokenizer()` reads `tokenizer.tokenizer.get_added_tokens_decoder()` rather than relying on base tokenizer vocab ids.
- `Siglip2Config` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/args.py`
  - Encoder dimensions: `dim`, `ffn_dim`, `n_layers`, `n_heads`
  - Positional/patch parameters: `n_pos_embs`, `n_channels`, `patch_size`, `spatial_merge_size`
  - Attention settings: `attn_backend="flex"`, `attn_mask_type="causal"`

## Dataset Processing Contract

### Schema Normalization

- `_process_mm_sample(...)` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_datasets.py`
  - Input contract: `texts` and `images` must be aligned lists where image positions contain `None` in `texts`.
  - For each non-`None` image:
    - `process_image(...)`
    - `calculate_image_tokens(...)`
    - replace matching text slot with `special_tokens.img_token`
  - Calls `process_text_with_images(...)` to expand BOI/IMG/EOI spans.
  - Masks BOI/EOI/IMG ids out of labels with `ignore_id`.
- `_process_obelics_sample(...)`
  - Pass-through wrapper over `_process_mm_sample`.
- `_process_cc12_wd_sample(...)`
  - Converts image-caption pairs into the experiment's interleaved format:
    - `texts = [None, text]`
    - `images = [image, None]`

### Dataset Registry

- `MM_DATASETS` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/datasets/mm_datasets.py`
  - `obelics`: `HuggingFaceM4/OBELICS`, streaming.
  - `cc12m`: `pixparse/cc12m-wds`, streaming.
  - `cc12m-test`: local tar-backed test fixture under `tests/assets/cc12m_test`.
- `_validate_mm_dataset(...)`
  - Raises `ValueError` for unsupported names.
  - Returns `(path, loader, sample_processor)`.

### Iterable Dataset State

- `HuggingFaceMultiModalDataset.__iter__()`
  - Increments `_sample_idx` before sample processing.
  - Skips `None` results and overlength samples.
  - If packing is enabled:
    - `self.packer.add_sample(processed)`
    - yield packed samples only when `has_batch_ready()`
  - Flushes residual packer contents at dataset end.
  - Resets `_sample_idx=0` only when `infinite=True`.
- `HuggingFaceMultiModalDataset._get_data_iter()`
  - Replays the iterator up to `_sample_idx` for resume support.
  - Returns `iter([])` if the dataset is already exhausted or if errors occur.
- `state_dict()` / `load_state_dict()`
  - Save and restore `_sample_idx`.
  - Also persist both `SamplePacker` deques if packing is enabled.

## Collation And Padding Contract

- `MultiModalCollatorNLD.collate_images(all_images)`
  - Converts each image to patches and grids.
  - Pads each image to `max_patches_per_image`.
  - Stacks images and then pads the image batch to `max_images_per_batch`.
- `MultiModalCollatorNLD.collate_text(batch)`
  - Pads sample sequences with `pad_sequence`.
  - Uses `seq_len + 1` before shifting, so final `input_ids` and `labels` each have length `seq_len`.
  - Pads batch size up to the target `batch_size`.
- `MultiModalCollatorNLD.__call__(batch)`
  - Drops samples from the end while total images exceed `max_images_per_batch`.
  - Returns:
    - `input_dict["input"]`: `B x S`
    - `input_dict["pixel_values"]`: `N x L x D` or `None`
    - `input_dict["grid_thw"]`: `N x L x 3` or `None`
    - `input_dict["special_tokens"]`
    - labels: shifted `B x S`

## Helper Semantics

### `utils/image.py`

- `process_image(...)`
  - Accepts URL string, path string, raw bytes, or `PIL.Image`.
  - Converts to RGB, resizes to a patch-compatible shape, normalizes with CLIP mean/std, and returns `(1, H, W, 3)`.
- `_smart_resize(...)`
  - Enforces:
    - both dimensions >= `factor`
    - aspect ratio <= 200
  - Snaps dimensions to multiples of `factor = patch_size * merge_size`.
- `_resize_image_by_patch_count(...)`
  - Scales up tiny images to meet `min_patch_per_image`.
  - Scales down large images to fit `max_patch_per_image`.
- `calculate_image_tokens(...)`
  - Returns `(total_tokens, tokens_per_row, num_rows)`.
- `convert_to_patches(...)`
  - Rearranges `(T, H, W, C)` to `(L, D)` and emits `(L, 3)` coordinate grid.
- `pad_patches(...)`
  - Pads with zero patches and `-1` coordinates.
  - Returns `(None, None)` instead of truncating.
- `pad_empty_images_to_target_batch_size(...)`
  - Adds all-zero images plus all-`-1` grids until image batch size reaches the target.

### `utils/text.py`

- `pad_text_batch(...)`
  - Pads or truncates sequence length, then rewrites padding labels to `ignore_idx`.
- `pad_input_ids_and_labels_to_target_batch_size(...)`
  - Pads batch dimension, then rewrites label padding to `ignore_idx`.
- `process_text_with_images(...)`
  - Replaces each `img_token` placeholder with:
    - `boi_token`
    - repeated `img_token` placeholders `num_image_tokens` times
    - `eoi_token`
  - Uses `"".join(parts)`, so token strings must already contain intended spacing or tokenizer-friendly delimiters.

### `utils/packing.py`

- `SamplePacker._pack_buffered_samples()`
  - Sorts by descending input length.
  - Greedily concatenates samples until length overflow would occur.
  - Concatenates `pixel_values` as a flat list, preserving multimodal ordering.
- `has_batch_ready()`
  - Requires `len(self.packed_samples) >= batch_size`.
- `get_next_batch()`
  - Opportunistically flushes `sample_buffer` if no ready batch exists.

## Model And Mask Semantics

- `_scatter_img_tokens(h_BSD, tokens_BS, i_NLD, i_mask_NL, img_id)` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/model.py`
  - Selects valid encoder outputs with `pixel_masks`.
  - Asserts equality between number of valid visual embeddings and count of image placeholder token positions.
  - Uses `masked_scatter_`, so it mutates text embeddings in place.
- `Projector`
  - Two-layer MLP: `w1 -> SiLU -> w2`.
- `Llama3Siglip2Transformer.get_attention_masks(...)`
  - Returns dict:
    - `llama3_masks`
    - `encoder_masks`
  - Expects both to be `BlockMask`.
- `Llama3Siglip2Transformer.forward(...)`
  - The encoder path requires `attention_masks` to be present.
  - Uses `grid_thw[:, :, 1:]` because Siglip2 handles only spatial `h,w`, not temporal layout.
- `VisionEmbeddings.forward(...)` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/model/siglip2.py`
  - Adds resized learned 2D positional embeddings to already-patchified pixel tokens.
- `VisionTransformer.get_attention_masks(...)`
  - Builds causal masks by default.
  - For `block_causal`, adds document-aware masking derived from `pixel_masks`.
- `VisionTransformer.forward(...)`
  - Ignores `pixel_masks` after mask construction; the real padding signal is already embedded into the block mask and the padded coordinates.

## Parallelization Invariants

- `parallelize_vlm(...)` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/infra/parallelize.py`
  - Requires `training.seq_len % parallel_dims.seq_len_divisor == 0`.
  - CP is allowed only when decoder attention backend is `sdpa`.
  - TP currently always raises `NotImplementedError`.
  - Applies:
    - activation checkpointing to decoder and encoder
    - dense compile to decoder and encoder
    - FSDP/HSDP via `apply_fsdp(...)`
    - replication if only replicated DP is enabled
- `apply_fsdp(...)`
  - Shards:
    - `tok_embeddings` when present
    - every encoder layer in `model.encoder.layers`
    - every decoder layer in `model.layers`
    - `[model.norm, model.output]` together when present
    - the root model last
  - Disables FSDP automatic gradient division through `disable_fsdp_gradient_division(model)`.

## Test Coverage

- `build_vlm_test_list()` in `/home/scbjtfy/torchtitan/torchtitan/experiments/vlm/tests/integration_tests.py`
  - Only one test flavor:
    - `--module vlm`
    - `--config vlm_debugmodel`
    - `--parallelism.data_parallel_shard_degree 4`
    - `--dataloader.max_patches_per_image 1024`
    - `--dataloader.max_images_per_batch 64`
- `main()`
  - Requires an empty output directory before dispatching `run_tests(...)`.

## Modification Checklist

- If a change touches image token counts:
  - update `calculate_image_tokens(...)`
  - verify `process_text_with_images(...)`
  - verify `_scatter_img_tokens(...)` assertion still holds
- If a change touches resume or packing behavior:
  - update `HuggingFaceMultiModalDataset.state_dict()` / `load_state_dict()`
  - verify `SamplePacker` state shape is still serializable
- If a change touches distributed support:
  - update `parallelize_vlm(...)`
  - verify mask generation and encoder path still work under the new mesh assumptions
