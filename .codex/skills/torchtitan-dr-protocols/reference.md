# Protocols Reference

## File Index

### `torchtitan/protocols/__init__.py`

- Imports `Configurable` from `torchtitan.config`.
- Re-exports:
  - `BaseModel`
  - `ModelConverter`
  - `ModelConvertersContainer`
  - `ModelSpec`
  - `FaultTolerantModelSpec`
  - `Module`
  - `BaseStateDictAdapter`
  - `StateDictAdapter`

### `torchtitan/protocols/module.py`

- `_created_classes: dict[type, type]`
  - cache from original `nn.Module` subclasses to the dynamically created TorchTitan-compatible wrapper class.
- `class Module(nn.Module, Configurable)`
  - `init_weights(self, **kwargs) -> None`
    - default no-op hook for subclasses without learnable parameters or externally loaded weights.
  - `@classmethod from_nn_module(cls, nn_module_cls: type[nn.Module]) -> type[Module]`
    - returns a cached wrapper subclass inheriting `(nn_module_cls, Module)`.
    - injects `init_weights()` that calls `reset_parameters()` when that method exists on `nn_module_cls`.
    - rewrites `__module__` and `__qualname__` to keep the generated class attached to `torchtitan.protocols.module`.
- `_container_init_weights(self: Module, **kwargs) -> None`
  - iterates over `self.children()`.
  - asserts each child is a `Module`.
  - delegates `child.init_weights(**kwargs)`.
- `class ModuleList(nn.ModuleList, Module)`
  - sets `init_weights = _container_init_weights`.
- `class ModuleDict(nn.ModuleDict, Module)`
  - sets `init_weights = _container_init_weights`.
- `class Sequential(nn.Sequential, Module)`
  - sets `init_weights = _container_init_weights`.

### `torchtitan/protocols/model.py`

- `class BaseModel(Module)`
  - `verify_module_protocol(self) -> None`
    - walks `self.named_modules()`.
    - collects `(fqn, type_name)` for any submodule that is not an instance of `Module`.
    - raises `RuntimeError` with a joined list of offending FQNs if any are found.
- `class BaseModel.Config(Module.Config)`
  - dataclass options: `kw_only=True`, `slots=True`.
  - `update_from_config(self, *, trainer_config, **kwargs) -> None`
    - abstract; used to reconcile model config with trainer/runtime config.
  - `get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]`
    - abstract; returns parameter count and FLOP estimate.

### `torchtitan/protocols/model_converter.py`

- `class ModelConverter(Protocol)`
  - `convert(self, model: nn.Module)`
    - in-place model rewrite.
  - `post_optimizer_hook(self, model: nn.Module | list[nn.Module])`
    - optional second-phase hook after optimizer construction.
- `class ModelConvertersContainer(Configurable, ModelConverter)`
  - `class Config(Configurable.Config)`
    - dataclass options: `kw_only=True`, `slots=True`.
    - `converters: list = field(default_factory=list)`
    - `print_after_conversion: bool = False`
  - `__init__(self, config: Config, *, parallel_dims: ParallelDims, model_compile_enabled: bool)`
    - validates quantization config compatibility.
    - builds each configured converter with `parallel_dims` and `model_compile_enabled`.
  - `convert(self, model: nn.Module)`
    - applies every converter in `self.converters`.
    - logs the final model when `print_after_conversion` is `True`.
  - `post_optimizer_hook(self, model: nn.Module | list[nn.Module])`
    - forwards the hook to each contained converter.
- `_validate_quantization(converters: list[Configurable.Config])`
  - inspects only entries that are instances of `QuantizationConverter.Config`.
  - enforces a single `_quantization_type` across all such configs.

### `torchtitan/protocols/model_spec.py`

- Type aliases:
  - `ParallelizeFunction = Callable[..., nn.Module]`
  - `PipeliningFunction = Callable[..., tuple[_PipelineSchedule, list[nn.Module], bool, bool]]`
  - `LossFunctionBuilder = Callable[..., LossFunction]`
  - `FragmentFunction = Callable[..., list[nn.Module]]`
  - `PostOptimizerBuildFn = Callable[..., None]`
- `@dataclass class ModelSpec`
  - `name: str`
  - `flavor: str`
  - `model: BaseModel.Config`
  - `build_loss_fn: Callable`
  - `parallelize_fn: Callable`
  - `pipelining_fn: Callable | None`
  - `post_optimizer_build_fn: Callable | None`
  - `state_dict_adapter: type[BaseStateDictAdapter] | None`
- `@dataclass class FaultTolerantModelSpec(ModelSpec)`
  - `fragment_fn: Callable | None`

### `torchtitan/protocols/state_dict_adapter.py`

- `class BaseStateDictAdapter(ABC)`
  - attributes declared for implementations:
    - `fqn_to_index_mapping: dict[Any, int] | None`
    - `hf_assets_path: str | None`
  - abstract methods:
    - `__init__(self, model_config: BaseModel.Config, hf_assets_path: str | None)`
    - `to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]`
    - `from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]`
    - `get_hf_storage_reader(self, path: str, from_quantized: bool = False) -> HuggingFaceStorageReader`
- `class StateDictAdapter(BaseStateDictAdapter)`
  - `__init__(self, model_config: BaseModel.Config, hf_assets_path: str | None)`
    - stores `hf_assets_path`.
    - tries to open `<hf_assets_path>/model.safetensors.index.json`.
    - if present, loads `weight_map` and maps each HF key to an integer shard id parsed from the filename.
    - if absent, logs a warning and sets `fqn_to_index_mapping = None`.
  - `get_hf_storage_reader(self, path: str, from_quantized: bool = False) -> HuggingFaceStorageReader`
    - warns if `from_quantized` is `True`.
    - returns `HuggingFaceStorageReader(path)`.

## Cross-File Relationships

- `BaseModel` depends on `Module`, so every model is also a configurable module.
- `BaseModel.Config` depends on `Module.Config`, so model configs inherit the same build/config protocol as lower-level modules.
- `ModelSpec.model` stores a `BaseModel.Config`, not a constructed model instance.
- `ModelSpec.state_dict_adapter` stores an adapter class, not an adapter instance.
- `ModelConvertersContainer` depends on `Configurable` so converter configs can build concrete converter objects.
- `StateDictAdapter` depends on `BaseModel.Config` only for constructor shape consistency; the base implementation itself does not read fields from `model_config`.

## Practical Lookup

- Need to make an `nn.Module` acceptable to TorchTitan init flows:
  - `torchtitan/protocols/module.py`
- Need to enforce or debug protocol violations inside a model tree:
  - `torchtitan/protocols/model.py`
- Need to add registry metadata or hooks returned by a model package:
  - `torchtitan/protocols/model_spec.py`
- Need to add model rewrites such as quantization:
  - `torchtitan/protocols/model_converter.py`
- Need to convert checkpoints to or from HF naming/layout:
  - `torchtitan/protocols/state_dict_adapter.py`
