---
name: torchtitan-dr-protocols
description: Use when adding or debugging torchtitan model contracts, model specs, module wrappers, converter hooks, or checkpoint adapter interfaces.
---

# TorchTitan Protocols

## Overview

`torchtitan/protocols` defines the contracts that let the rest of TorchTitan treat models, modules, checkpoint adapters, and post-build converters uniformly. The module is intentionally thin: it does not run training itself, but it fixes the shape of the objects that `trainer.py`, model packages, checkpoint code, and parallelization helpers exchange. If you are changing how a model is constructed, parallelized, converted, or serialized, this is the compatibility layer that determines what the rest of the system expects.

## Public Surface

- `torchtitan/protocols/__init__.py`
  - Re-exports `BaseModel`, `ModelConverter`, `ModelConvertersContainer`, `ModelSpec`, `FaultTolerantModelSpec`, `Module`, `StateDictAdapter`, and `BaseStateDictAdapter`.
- `torchtitan/protocols/module.py`
  - `Module`: combines `torch.nn.Module` with `torchtitan.config.Configurable`.
  - `Module.from_nn_module(nn_module_cls)`: wraps plain `nn.Module` subclasses so they satisfy TorchTitan's `Module` protocol.
  - `ModuleList`, `ModuleDict`, `Sequential`: container variants with recursive `init_weights()`.
- `torchtitan/protocols/model.py`
  - `BaseModel(Module)`: base class for model families.
  - `BaseModel.verify_module_protocol()`: rejects non-`Module` children early.
  - `BaseModel.Config.update_from_config(...)` and `get_nparams_and_flops(...)`: required abstract hooks for model configs.
- `torchtitan/protocols/model_spec.py`
  - `ModelSpec`: bundle of selected architecture config plus build/parallelize/pipeline/loss/state-dict hooks.
  - `FaultTolerantModelSpec`: extends `ModelSpec` with `fragment_fn` for FT flows.
- `torchtitan/protocols/model_converter.py`
  - `ModelConverter` protocol: `convert()` and `post_optimizer_hook()`.
  - `ModelConvertersContainer`: builds configured converters and applies them sequentially.
- `torchtitan/protocols/state_dict_adapter.py`
  - `BaseStateDictAdapter`: abstract conversion interface between native and external checkpoint formats.
  - `StateDictAdapter`: default implementation for Hugging Face sharded safetensor index handling.

## Design Logic

TorchTitan keeps these interfaces separate from concrete model code so model packages can stay mostly single-device and architecture-focused while the trainer still plugs in parallelization, quantization, checkpoint conversion, and alternative runtimes. The central design choice is "bundle behavior as callables, not inheritance trees": `ModelSpec` in [torchtitan/protocols/model_spec.py](/home/scbjtfy/torchtitan/torchtitan/protocols/model_spec.py) stores a model config together with builder callables instead of forcing one monolithic base class for every training concern.

Another important choice is the `Module` protocol in [torchtitan/protocols/module.py](/home/scbjtfy/torchtitan/torchtitan/protocols/module.py). TorchTitan wants every trainable submodule to have both configuration semantics and an `init_weights()` entrypoint. `Module.from_nn_module()` exists so common PyTorch layers can be wrapped without re-implementing them, while `BaseModel.verify_module_protocol()` in [torchtitan/protocols/model.py](/home/scbjtfy/torchtitan/torchtitan/protocols/model.py) catches violations before later systems fail in harder-to-debug ways.

Checkpoint conversion is similarly abstracted. `BaseStateDictAdapter` defines the contract, while `StateDictAdapter` only implements the common Hugging Face index-file behavior. Model families supply the real tensor-name mapping logic in their own adapter subclasses, which keeps model-specific serialization out of core checkpoint code.

## Core Data Structures

- `Module` in [torchtitan/protocols/module.py](/home/scbjtfy/torchtitan/torchtitan/protocols/module.py)
  - Inherits `nn.Module` and `Configurable`.
  - Default `init_weights()` is a no-op, so layers loaded from checkpoints do not need fake initialization paths.
  - Uses `_created_classes` cache so repeated `from_nn_module()` calls reuse the same wrapper class.
- `BaseModel.Config` in [torchtitan/protocols/model.py](/home/scbjtfy/torchtitan/torchtitan/protocols/model.py)
  - Extends `Module.Config`.
  - Requires `update_from_config()` to sync model hyperparameters with trainer/runtime config.
  - Requires `get_nparams_and_flops()` so the trainer can report model scale and throughput.
- `ModelSpec` in [torchtitan/protocols/model_spec.py](/home/scbjtfy/torchtitan/torchtitan/protocols/model_spec.py)
  - Fields: `name`, `flavor`, `model`, `build_loss_fn`, `parallelize_fn`, `pipelining_fn`, `post_optimizer_build_fn`, `state_dict_adapter`.
  - This is the trainer's handoff object for "everything needed to run one model flavor".
- `ModelConvertersContainer.Config` in [torchtitan/protocols/model_converter.py](/home/scbjtfy/torchtitan/torchtitan/protocols/model_converter.py)
  - `converters`: ordered config objects for converters.
  - `print_after_conversion`: optional debugging switch.
- `StateDictAdapter`
  - Stores `hf_assets_path` and optional `fqn_to_index_mapping`.
  - Parses `model.safetensors.index.json` into HF key -> shard index metadata.

## State Flow

The typical flow starts in a model package, which returns a `ModelSpec`. The trainer consumes that spec, builds the selected `BaseModel.Config`, constructs the model, and invokes the spec's loss/parallelization/pipelining hooks. During model construction, `BaseModel.verify_module_protocol()` may be called to ensure all nested layers conform to `Module`.

If model converters are configured, `ModelConvertersContainer.__init__()` builds each converter from its config and runtime-only inputs (`parallel_dims`, `model_compile_enabled`), then `convert()` applies them in order. `_validate_quantization()` prevents mixed quantization families from being composed in one container.

For checkpoint interoperability, the checkpoint path flows through a model-specific adapter subclass of `BaseStateDictAdapter`. `StateDictAdapter.__init__()` optionally reads the HF safetensor index file and builds `fqn_to_index_mapping`; `get_hf_storage_reader()` returns a `HuggingFaceStorageReader`, warning if quantized-load mode is requested on an unsupported adapter.

## Error Handling

- `BaseModel.verify_module_protocol()` raises `RuntimeError` with fully qualified child names if any submodule is not a `Module`.
- `Module.from_nn_module()` avoids dynamic-class duplication by using `_created_classes`; repeated wrapping is deterministic.
- `_validate_quantization()` asserts that every `QuantizationConverter.Config` in a converter list shares the same `_quantization_type`.
- `StateDictAdapter.__init__()` treats missing `model.safetensors.index.json` as a warning, not a hard failure, and falls back to single-file save behavior.
- `get_hf_storage_reader(..., from_quantized=True)` warns instead of failing when quantized checkpoint loading is unsupported.

## Modification Guide

- To add a new model family, start by returning a `ModelSpec` from that model package and ensure its config class inherits `BaseModel.Config`. The contract lives in [torchtitan/protocols/model_spec.py](/home/scbjtfy/torchtitan/torchtitan/protocols/model_spec.py) and [torchtitan/protocols/model.py](/home/scbjtfy/torchtitan/torchtitan/protocols/model.py).
- To make a plain PyTorch layer compatible with TorchTitan initialization/config assumptions, wrap it through `Module.from_nn_module()` in [torchtitan/protocols/module.py](/home/scbjtfy/torchtitan/torchtitan/protocols/module.py) instead of creating an ad hoc subclass.
- To add a new model conversion pass, implement the `ModelConverter` protocol and make its config buildable through `Configurable`; then insert that config into `ModelConvertersContainer.Config.converters` in [torchtitan/protocols/model_converter.py](/home/scbjtfy/torchtitan/torchtitan/protocols/model_converter.py).
- To support loading/saving another checkpoint format, subclass `BaseStateDictAdapter` or `StateDictAdapter` in the model package and wire the subclass into `ModelSpec.state_dict_adapter`.
- To change fault-tolerant model fragmentation behavior, extend the `fragment_fn` carried by `FaultTolerantModelSpec` in [torchtitan/protocols/model_spec.py](/home/scbjtfy/torchtitan/torchtitan/protocols/model_spec.py) instead of embedding FT assumptions into core trainer code.
