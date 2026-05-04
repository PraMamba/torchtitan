---
name: add-model
description: Guide for adding a new model to torchtitan. Use when user wants to add support for a new model architecture (LLM or other).
---

# Add Model to TorchTitan

Add support for a new model architecture in the torchtitan training framework.

## When to Use

- User asks "how do I add a model to torchtitan?"
- User wants to support a new model family (e.g., Mistral, Gemma, Phi)
- User mentions adding a new `ModelSpec` or model type

## Prerequisites

- Understand the target model's architecture (from HuggingFace or paper)
- Know the model type: decoder LLM, diffusion model, or other

## Step-by-Step Guide

### Step 1: Analyze the Target Model

Read the model's HuggingFace configuration and modeling source to identify:

```
Target model: <name>
Type: [Dense LLM / MoE LLM / Diffusion / Other]
Attention: [GQA / MHA / MLA / with QK norm / with bias / sliding window]
FFN: [SwiGLU / GeGLU / standard MLP]
MoE: [no / yes - num_experts, top_k, shared_experts]
RoPE: [standard / YaRN / NTK-aware / other]
Norm: [RMSNorm / LayerNorm]
Weight tying: [yes / no]
```

### Step 2: Select Reference Model

Choose the closest existing implementation:

| Target Characteristics | Reference | Why |
|----------------------|-----------|-----|
| Dense, standard GQA | `llama3` | Simplest baseline |
| Dense with QK norm | `qwen3` | QK norm support |
| MoE with standard routing | `llama4` | MoE + dense layers |
| MoE with complex routing | `deepseek_v3` | Multi-latent attention, custom EP |
| Non-decoder (diffusion) | `flux` | Different architecture entirely |

### Step 3: Create Model Directory

```
torchtitan/models/<model_name>/
├── __init__.py              # model_registry() -> ModelSpec
├── model.py                 # Model architecture classes
├── config_registry.py       # Training configs (debugmodel, sizes)
├── parallelize.py           # apply_tp, apply_fsdp, apply_ac, apply_compile
├── state_dict_adapter.py    # HF <-> torchtitan state dict mapping
└── README.md                # Model-specific instructions
```

### Step 4: Implement Model Architecture (`model.py`)

Model must follow the Module protocol:

```python
from torchtitan.protocols.module import Module
from torchtitan.models.common.decoder import Decoder

class NewModel(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # Model-specific hyperparameters
        num_layers: int = 32
        num_heads: int = 32
        # ... etc

    def __init__(self, config: Config):
        super().__init__(config)
        # Build layers using common components

    def init_weights(self, **kwargs):
        # Weight initialization
        pass

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        # Return parameter count and FLOPs per token
        pass

    def update_from_config(self, config) -> None:
        # Sync training config into model config
        pass
```

**Reuse common components** from `torchtitan/models/common/`:
- `GQAttention` from `common/attention.py`
- `FeedForward` from `common/feed_forward.py`
- `TransformerBlock` from `common/decoder.py`
- `RoPE` from `common/rope.py`
- `RMSNorm` from `common/rmsnorm.py`
- `MoE` from `common/moe/` (if applicable)

### Step 5: Implement Config Registry (`config_registry.py`)

```python
from torchtitan.trainer import Trainer

def newmodel_debugmodel() -> Trainer.Config:
    """Small debug config for testing."""
    return Trainer.Config(
        model_spec=ModelSpec(
            name="newmodel",
            flavor="debugmodel",
            model=NewModel.Config(num_layers=4, ...),
            build_loss_fn=build_cross_entropy_loss,
            parallelize_fn=parallelize_newmodel,
            state_dict_adapter=NewModelStateDictAdapter,
        ),
        training=TrainingConfig(steps=10, seq_len=256, ...),
        ...
    )
```

### Step 6: Implement Parallelization (`parallelize.py`)

Follow the standard parallelization order:

```python
def parallelize_newmodel(model, parallel_dims, ...):
    # 1. Apply TP
    if parallel_dims.tp_enabled:
        apply_tp(model, parallel_dims, ...)

    # 2. Apply CP (if decoder)
    if parallel_dims.cp_enabled:
        apply_cp(model, parallel_dims)

    # 3. Apply AC
    if ac_config:
        apply_ac(model, ac_config)

    # 4. Apply compile
    if compile_config:
        apply_compile_dense(model)  # or apply_compile_sparse for MoE

    # 5. Apply FSDP
    apply_fsdp(model, parallel_dims, ...)

    return model
```

**Reuse existing functions** where possible — don't duplicate `apply_fsdp`, `apply_ac`, `apply_compile` if they already work for your architecture.

### Step 7: Implement State Dict Adapter (`state_dict_adapter.py`)

```python
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

class NewModelStateDictAdapter(BaseStateDictAdapter):
    # Map HF key patterns to torchtitan key patterns
    ...
```

**Verification**: `from_hf(to_hf(state_dict))` must preserve all keys.

### Step 8: Register the Model (`__init__.py`)

```python
from torchtitan.protocols.model_spec import ModelSpec

def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="newmodel",
        flavor=flavor,
        model=NewModel.Config(...),
        build_loss_fn=build_cross_entropy_loss,
        parallelize_fn=parallelize_newmodel,
        state_dict_adapter=NewModelStateDictAdapter,
    )
```

Add to `torchtitan/models/__init__.py`:
```python
_supported_models = frozenset({..., "newmodel"})
```

### Step 9: Add Tests

**Required tests:**

1. **Unit test** (`tests/unit_tests/test_newmodel.py`):
   - Config creation and validation
   - Model construction on meta device
   - Forward pass shape correctness
   - State dict adapter roundtrip

2. **Integration test** (add to `tests/integration_tests/models.py`):
   - Training with debug config
   - At least TP and FSDP parallelism
   - Loss comparison with reference

### Step 10: Validate

1. `pre-commit run --all-files` passes
2. `pytest tests/unit_tests/ -x` passes
3. Debug model trains with `torchrun --nproc_per_rank=4`
4. Checkpoint save/load works
5. HF checkpoint conversion works (if state_dict_adapter implemented)

## Common Mistakes

- Not registering model in `_supported_models` frozenset
- Putting model-agnostic functionality in model-specific files
- Not reusing common components (reimplementing attention, FFN, etc.)
- Wrong TP plan for attention with/without QK norm
- Missing `init_weights()` implementation
- Not implementing `get_nparams_and_flops()` correctly
- State dict adapter missing keys (causes silent weight drops)
- Not auditing all model variants when changing shared components

## File Checklist

- [ ] `torchtitan/models/<name>/__init__.py` — model_registry()
- [ ] `torchtitan/models/<name>/model.py` — Model architecture
- [ ] `torchtitan/models/<name>/config_registry.py` — Training configs
- [ ] `torchtitan/models/<name>/parallelize.py` — Parallelization
- [ ] `torchtitan/models/<name>/state_dict_adapter.py` — HF interop
- [ ] `torchtitan/models/__init__.py` — Registered in _supported_models
- [ ] `tests/unit_tests/test_<name>.py` — Unit tests
- [ ] `tests/integration_tests/models.py` — Integration test entry
