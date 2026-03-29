---
name: torchtitan-dr-hf-datasets
description: Use when working on TorchTitan Hugging Face text dataset loading, dataset registration, streaming or resumable iteration, or document-to-token packing behavior.
---

# TorchTitan HF Datasets

## Overview

The `torchtitan/hf_datasets` module is TorchTitan's adapter layer between Hugging Face datasets and the trainer-facing dataloader stack. It defines a small registry format in [`torchtitan/hf_datasets/__init__.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/__init__.py) and one concrete implementation in [`torchtitan/hf_datasets/text_datasets.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/text_datasets.py) that turns text documents into `(input, positions) -> label` training batches, splits work by DP rank, and preserves enough state to resume after checkpointing.

The module is intentionally narrow. It does not try to be a general-purpose data framework. Instead, it standardizes three things TorchTitan needs from a text dataset: how to load it, how to turn a raw sample into text, and how to serialize iterator progress plus token buffers so training can restart cleanly.

## Public Surface

- `DatasetConfig` in [`torchtitan/hf_datasets/__init__.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/__init__.py)
  - Fields:
    - `path: str`
    - `loader: Callable`
    - `sample_processor: Callable`
- `DATASETS` in [`torchtitan/hf_datasets/text_datasets.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/text_datasets.py)
  - Built-in entries: `c4`, `c4_test`, `c4_validation`
- `_validate_dataset(dataset_name, dataset_path) -> tuple[str, Callable, Callable]`
  - Validates registry membership, resolves override path, returns `(path, loader, sample_processor)`.
- `HuggingFaceTextDataset`
  - Iterable + checkpointable dataset that tokenizes text and emits training samples.
- `HuggingFaceTextDataLoader`
  - Thin `ParallelAwareDataloader` wrapper that instantiates `HuggingFaceTextDataset` and applies worker/batch settings.
- `HuggingFaceTextDataLoader.Config`
  - Extends `ParallelAwareDataloader.Config` with:
    - `dataset: str = "c4_test"`
    - `infinite: bool = True`

## Capabilities

- Registers named datasets behind a stable `DatasetConfig` schema.
- Supports both streaming and map-style Hugging Face datasets.
- Splits one logical dataset by data-parallel rank via `split_dataset_by_node(...)`.
- Tokenizes whole documents with BOS/EOS and packs them into fixed-length next-token training chunks.
- Tracks per-token positions separately from token IDs so document packing works with RoPE-aware attention.
- Supports checkpoint/resume for both in-memory token buffers and underlying iterable dataset state.
- Exposes a trainer-compatible dataloader config instead of forcing callers to wire `StatefulDataLoader` directly.

## Core Design Logic

### Registry-driven dataset onboarding

The central design choice is the `DatasetConfig` dataclass plus the `DATASETS` registry. The module treats "dataset support" as metadata plus two callables:

- how to load the dataset
- how to extract text from one sample

That keeps adding a new dataset localized to a small diff in [`torchtitan/hf_datasets/text_datasets.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/text_datasets.py) instead of requiring a new dataset subclass per source.

### One concrete dataset class for text pretraining

`HuggingFaceTextDataset` is specialized for autoregressive text training. It assumes every sample can become one text string, tokenizes with `add_bos=True` and `add_eos=True`, and emits shifted `(input, label)` pairs. The class is not abstract because TorchTitan wants a single known-good path for numerics and checkpointability rather than many pluggable iterator implementations.

### Checkpointability is a first-class requirement

The class inherits `Stateful` and explicitly saves:

- `_token_buffer`
- `_position_buffer`
- either `_sample_idx` for map-style datasets or nested dataset state for iterable datasets

This is the main reason the implementation owns buffering itself instead of delegating packing to a generic collate function. The buffered partial document tail must survive checkpoints.

### Position tracking is separate from token tracking

The module stores `_position_buffer` alongside `_token_buffer` and emits `{"input": input, "positions": positions}, label`. That is a deliberate design for document packing with RoPE-based models. Positions reset at document boundaries and wrap with `i % self.seq_len`, which lets the trainer know where each token sits within its originating document segment even when multiple documents are concatenated in the same token buffer.

## Core Data Structures

### `DatasetConfig`

Defined in [`torchtitan/hf_datasets/__init__.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/__init__.py).

- `path`
  - Default dataset identifier or local path.
- `loader`
  - Callable that receives the resolved path and returns a Hugging Face dataset object.
- `sample_processor`
  - Callable that extracts the text string to tokenize from one raw sample.

Relationship: each `DATASETS[...]` entry holds one `DatasetConfig`, and `_validate_dataset(...)` resolves that config for runtime use.

### `DATASETS`

Defined in [`torchtitan/hf_datasets/text_datasets.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/text_datasets.py).

- `c4`
  - Uses `_load_c4_dataset(..., split="train")`
- `c4_test`
  - Uses `load_dataset(path, split="train")` against the local test asset path
- `c4_validation`
  - Uses `_load_c4_dataset(..., split="validation")`

All three use `_process_c4_text(sample) -> sample["text"]`.

### `HuggingFaceTextDataset`

Defined in [`torchtitan/hf_datasets/text_datasets.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/text_datasets.py).

Important fields:

- `dataset_name`
  - Lowercased name used for registry lookup and logging.
- `_data`
  - Result of `split_dataset_by_node(ds, dp_rank, dp_world_size)`.
  - Can behave as either a Hugging Face `Dataset` or an iterable streaming dataset.
- `_tokenizer: BaseTokenizer`
  - Must provide `encode(...) -> list[int]`.
- `seq_len`
  - Chunk size basis; emitted token window length is `seq_len`.
- `infinite`
  - Controls whether exhaustion stops iteration or rewinds.
- `_text_processor`
  - Dataset-specific sample-to-text callable from the registry.
- `_sample_idx`
  - Resume cursor for map-style datasets only.
- `_token_buffer`
  - Accumulates token IDs across documents until enough tokens exist to emit a training window.
- `_position_buffer`
  - Mirrors `_token_buffer` with per-token position IDs.

### `HuggingFaceTextDataLoader.Config`

Defined in [`torchtitan/hf_datasets/text_datasets.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/text_datasets.py), extending [`ParallelAwareDataloader.Config`](/home/scbjtfy/torchtitan/torchtitan/components/dataloader.py).

Inherited knobs matter here:

- `dataset_path`
- `num_workers`
- `persistent_workers`
- `pin_memory`
- `prefetch_factor`

Module-specific knobs:

- `dataset`
- `infinite`

## State Flow

### 1. Dataset resolution

`HuggingFaceTextDataset.__init__` lowercases `dataset_name`, then calls `_validate_dataset(...)`.

`_validate_dataset(...)`:

- rejects unsupported names with a `ValueError`
- chooses `dataset_path` override if present, otherwise registry default
- logs the chosen dataset and path
- returns the loader and sample processor tied to that registry entry

### 2. Dataset loading and rank partitioning

Still in `__init__`:

- `dataset_loader(path)` materializes or streams the HF dataset
- `split_dataset_by_node(ds, dp_rank, dp_world_size)` partitions data by DP rank
- tokenizer, sequence length, infinity mode, and checkpoint buffers are stored on the instance

### 3. Iterator construction

`_get_data_iter()` chooses resume behavior based on dataset type:

- if `_data` is a Hugging Face `Dataset` (map-style), resume is implemented by `self._data.skip(self._sample_idx)` unless the cursor already equals dataset length
- otherwise it returns `iter(self._data)` and assumes the iterable dataset object carries its own progress

This split is the key resume policy in the module.

### 4. Sample processing and packing

Inside `__iter__()`:

1. Pull one raw sample from `_get_data_iter()`
2. Convert it to text with `_text_processor`
3. Tokenize with `self._tokenizer.encode(sample_text, add_bos=True, add_eos=True)`
4. Append tokens into `_token_buffer`
5. Append positions `i % self.seq_len` into `_position_buffer`
6. Increment `_sample_idx`

Whenever the buffers reach `1 + self.seq_len` tokens, the module emits one training example:

- `x = token_buffer[:seq_len + 1]`
- `pos = position_buffer[:seq_len + 1]`
- `input = x[:-1]`
- `label = x[1:]`
- `positions = pos[:-1]`
- yield `({"input": input, "positions": positions}, label)`

The `+1` is necessary because next-token prediction needs an input window and its one-token-shifted label window.

### 5. Exhaustion and looping

After the dataset iterator finishes:

- `infinite == False`
  - logs that the dataset ran out of data
  - breaks the outer loop
- `infinite == True`
  - resets `_sample_idx = 0`
  - logs re-looping
  - if `_data` is iterable and supports `set_epoch` plus `epoch`, increments the epoch to make resumed streaming iteration work correctly

### 6. Checkpoint save/load

`state_dict()` always saves:

- `token_buffer`
- `position_buffer`

Then it chooses one resume strategy:

- map-style dataset: save `sample_idx`
- iterable dataset: save `_data.state_dict()` under `data`

`load_state_dict(...)` restores those pieces with backward compatibility:

- if `position_buffer` is missing, it logs a warning and falls back to `[]`
- map-style dataset restores `sample_idx`
- iterable dataset requires `"data"` and delegates to `_data.load_state_dict(...)`

That warning matters because older checkpoints without position tracking can resume, but may produce wrong RoPE positions when block-causal packing is in play.

## Error Handling And Edge Cases

- Unsupported dataset name:
  - `_validate_dataset(...)` raises `ValueError` with the supported dataset list.
- Map-style dataset fully consumed:
  - `_get_data_iter()` returns `iter([])` if `_sample_idx == len(self._data)`.
- Missing `position_buffer` in checkpoint:
  - `load_state_dict(...)` logs a compatibility warning and uses an empty buffer.
- Iterable dataset checkpoint without `"data"`:
  - `load_state_dict(...)` asserts the key exists.
- Finite dataset exhaustion:
  - iteration ends after a warning, without throwing a custom exhaustion exception from this module.

## Integration Points

- [`torchtitan/components/tokenizer.py`](/home/scbjtfy/torchtitan/torchtitan/components/tokenizer.py)
  - `BaseTokenizer.encode(...)` is the only tokenizer contract this module depends on directly.
- [`torchtitan/components/dataloader.py`](/home/scbjtfy/torchtitan/torchtitan/components/dataloader.py)
  - `HuggingFaceTextDataLoader.Config` inherits worker and dataset-path fields from `ParallelAwareDataloader.Config`.
  - `HuggingFaceTextDataLoader` delegates batching and worker-process behavior to `ParallelAwareDataloader`.
- Hugging Face Datasets runtime
  - `load_dataset(...)`
  - `split_dataset_by_node(...)`
  - `Dataset.skip(...)`
  - iterable dataset `state_dict()/load_state_dict()`

## Modification Guide

### Add a new text dataset

Edit [`torchtitan/hf_datasets/text_datasets.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/text_datasets.py):

1. Add a loader function like `_load_c4_dataset(...)`
2. Add a sample processor that returns a single string
3. Register a new `DatasetConfig` entry in `DATASETS`

If the dataset needs a custom path or split, encode that in the loader callable, often via `functools.partial(...)`.

### Change how documents are packed into sequences

Edit `HuggingFaceTextDataset.__iter__` in [`torchtitan/hf_datasets/text_datasets.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/text_datasets.py).

Key logic lives in:

- `max_buffer_token_len = 1 + self.seq_len`
- `_token_buffer.extend(...)`
- `_position_buffer.extend(...)`
- the slice-and-shift block that builds `input`, `label`, and `positions`

This is where to implement truncation, dropping, alternate overflow policy, or different position semantics. The inline TODO already points to making overflow policy configurable.

### Change checkpoint compatibility or resume semantics

Edit `load_state_dict`, `state_dict`, and `_get_data_iter` in [`torchtitan/hf_datasets/text_datasets.py`](/home/scbjtfy/torchtitan/torchtitan/hf_datasets/text_datasets.py).

Use this path if you need:

- a new checkpoint field
- stricter backward-compat behavior
- map-style resharding support
- different resume rules for iterable datasets

Be careful to preserve compatibility with older checkpoints that lack `position_buffer`.

### Add validation-only or local test datasets

Most of the work is again in `DATASETS`. Compare:

- `c4` / `c4_validation` for remote streaming splits
- `c4_test` for a local asset-backed dataset path

This is the fastest path when you need a tiny deterministic dataset for tests or CI.

## Mental Model

Think of this module as a three-stage pipeline:

1. resolve a named dataset entry
2. turn documents into one continuous rank-local token stream with resettable positions
3. cut that stream into fixed-length autoregressive training windows that can survive checkpoint/restart

If you keep those three stages separate, the code is easy to modify. Most changes belong cleanly to either the registry, the token/position buffering loop, or the checkpoint/resume methods.
