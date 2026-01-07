# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TorchTitan is a PyTorch-native platform for training large generative AI models with multi-dimensional parallelism. It's designed for rapid experimentation and production-ready training of LLMs and other generative models.

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Download tokenizer (required before training)
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=YOUR_HF_TOKEN
```

### Training
```bash
# Basic training run (8 GPUs)
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh

# Custom GPU count
NGPU=4 CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh

# Single-GPU dry-run mode (config validation without actual GPU execution)
NGPU=32 COMM_MODE="fake_backend" ./run_train.sh

# Single-GPU debug mode (simulates multi-GPU on one GPU)
NGPU=8 COMM_MODE="local_tensor" ./run_train.sh

# With custom config overrides
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh \
  --training.steps=1000 --optimizer.lr=3e-4
```

### Testing
```bash
# Run all unit tests
pytest -s tests/unit_tests/

# Run specific test file
pytest -s tests/unit_tests/test_job_config.py

# Run specific test function
pytest -s tests/unit_tests/test_job_config.py::TestJobConfig::test_command_line_args

# Run integration tests (requires GPUs)
python -m tests.integration_tests.run_tests test_output --test_suite features --ngpu 8

# Run specific integration test
python -m tests.integration_tests.run_tests test_output --test_suite features --test_name gradient_accumulation --ngpu 2
```

### Linting and Pre-commit
```bash
# Run all pre-commit checks
pre-commit run --all-files

# Individual linters
flake8 --config=.flake8
ufmt format .
pyrefly-check  # Type checking
```

## Architecture

### Core Structure
```
torchtitan/
├── train.py                 # Main training loop and Trainer class
├── components/              # Reusable components (checkpoint, dataloader, optimizer, etc.)
├── config/                  # Configuration system (TOML + CLI parsing)
│   ├── job_config.py       # Complete JobConfig dataclass hierarchy
│   └── manager.py          # Config parser with precedence: CLI > TOML > defaults
├── distributed/             # Parallelism implementations
│   ├── parallel_dims.py    # ParallelDims - manages multi-dimensional device meshes
│   ├── pipeline_parallel.py # PP schedules (1F1B, Interleaved, ZBV)
│   └── tensor_parallel.py  # TP utilities
├── models/                  # Model implementations
│   └── {model_name}/
│       ├── model/          # Model definition
│       ├── infra/          # Parallelization logic (parallelize.py)
│       └── train_configs/  # TOML configuration files
├── protocols/              # Abstract interfaces and extension points
│   └── train_spec.py       # TrainSpec dataclass (defines swappable components)
├── experiments/            # Experimental features (one-way dependency: experiments → core)
└── tools/                  # Utilities (logging, profiling)
```

### Training Flow

**Initialization** (train.py):
1. `ConfigManager.parse_args()` - Load TOML file + apply CLI overrides
2. `Trainer.__init__()`:
   - Initialize distributed (create ParallelDims from world mesh)
   - Get model's TrainSpec via `get_train_spec(model_name)`
   - Build tokenizer, dataloader
   - Create model on meta device
   - Apply parallelism: `pipelining_fn()` (if PP > 1) or `parallelize_fn()`
   - Build optimizers, lr_schedulers, checkpointer
3. `Trainer.train()` - Main training loop

**Training Loop**:
```python
while step < max_steps:
    for microbatch in gradient_accumulation_steps:
        forward_backward_step()  # Loss computation + backward

    clip_grad_norm_()
    optimizers.step()
    lr_schedulers.step()

    # Periodic tasks: logging, checkpointing, validation
```

### Parallelism System

The system uses `ParallelDims` (torchtitan/distributed/parallel_dims.py) to create multi-dimensional device meshes:
- `dp_replicate` - DDP/HSDP replicate dimension
- `dp_shard` - FSDP shard dimension
- `tp` - Tensor Parallel
- `pp` - Pipeline Parallel
- `cp` - Context Parallel (for long sequences)
- `ep` - Expert Parallel (MoE models)

**Parallelism Application**:
- If `pp > 1`: Call `pipelining_fn()` which splits model into stages, then applies `parallelize_fn()` to each stage
- If `pp == 1`: Call `parallelize_fn()` directly, which applies TP, activation checkpointing, compile, and data parallel

### Configuration System

Configurations use TOML files with CLI overrides:
```bash
# TOML file specifies base config
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh \
  # CLI args override TOML values
  --training.steps=5000 \
  --parallelism.tensor_parallel_degree=2
```

**Key config sections** (all in JobConfig dataclass in torchtitan/config/job_config.py):
- `model` - Model selection, HF assets path
- `training` - Batch sizes, steps, dtype, dataset
- `parallelism` - DP/TP/PP/CP/EP degrees
- `optimizer` - AdamW configuration
- `lr_scheduler` - Warmup-stable-decay schedule
- `checkpoint` - DCP options, async checkpointing, HF format conversion
- `activation_checkpoint` - AC modes (selective, full, memory_budget)
- `compile` - torch.compile settings
- `quantize` - Float8/MXFP8 quantization

### Extension Points (TrainSpec Protocol)

Models integrate via TrainSpec (torchtitan/protocols/train_spec.py):
```python
@dataclass
class TrainSpec:
    model_cls: type[ModelProtocol]
    model_args: Mapping[str, BaseModelArgs]
    parallelize_fn: ParallelizeFunction
    pipelining_fn: PipeliningFunction | None
    build_optimizers_fn: OptimizersBuilder
    build_lr_schedulers_fn: LRSchedulersBuilder
    build_dataloader_fn: DataLoaderBuilder
    build_tokenizer_fn: TokenizerBuilder | None
    build_loss_fn: LossFunctionBuilder
    build_validator_fn: ValidatorBuilder | None
    build_metrics_processor_fn: MetricsProcessorBuilder | None
    state_dict_adapter: type[BaseStateDictAdapter] | None
```

To add a new model, create:
```
torchtitan/models/{model_name}/
├── __init__.py              # Export get_train_spec()
├── model/
│   ├── model.py            # Model class (implements ModelProtocol)
│   └── args.py             # ModelArgs dataclass
├── infra/
│   └── parallelize.py      # parallelize_fn and optional pipelining_fn
└── train_configs/
    └── {flavor}.toml       # TOML configs for different sizes
```

### Experiments Folder

The `torchtitan/experiments/` folder is for new ideas and contributions:
- Each experiment in separate subfolder with clear theme
- Must reuse core components via TrainSpec
- Independent dependencies and tests
- One-way dependency: experiments → core (never core → experiments)
- Can have custom train.py if needed
- See torchtitan/experiments/README.md for full guidelines

## Key Files for Common Tasks

### Understanding the Training Loop
- `torchtitan/train.py` - Main Trainer class and training loop (lines 74-709)
- `torchtitan/protocols/train_spec.py` - TrainSpec protocol definition

### Working with Models
- `torchtitan/models/llama3/model/model.py` - Reference model implementation
- `torchtitan/models/llama3/infra/parallelize.py` - How to apply parallelism

### Modifying Parallelism
- `torchtitan/distributed/parallel_dims.py` - Device mesh creation
- `torchtitan/distributed/pipeline_parallel.py` - PP schedule implementations
- Model's `infra/parallelize.py` - Parallelism application for specific model

### Checkpointing
- `torchtitan/components/checkpoint.py` - DCP save/load with async support
- `docs/checkpoint.md` - Checkpoint interoperability with torchtune

### Quantization
- `torchtitan/components/quantization/float8.py` - Float8 implementation
- `docs/float8.md` - Float8 usage guide
- `docs/mxfp8.md` - MXFP8 for Blackwell GPUs

### Configuration
- `torchtitan/config/job_config.py` - All config options (1024 lines)
- `torchtitan/models/llama3/train_configs/` - Example TOML configs

## Important Conventions

### Code Style
- Minimal model code changes for parallelism (use `parallelize_fn` and `pipelining_fn`)
- Modular components that can be swapped via TrainSpec builders
- PyTorch-native techniques preferred over external libraries

### Testing Requirements
- Changes that don't impact computation: verify identical loss with deterministic mode
- Changes that impact computation: verify loss convergence on C4 dataset
- Memory and throughput metrics must meet expectations
- Add integration tests in tests/integration_tests/ for new features

### Pre-commit Checks
- ufmt (black + usort) for formatting
- flake8 for linting
- pyrefly for type checking
- License header insertion
- No commits directly to main branch
