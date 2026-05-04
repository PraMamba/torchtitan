# TorchTitan Codex Project Guide

This guide preserves the project guidance originally written for Claude Code and
makes it discoverable for Codex users. The source material below remains
informational unless a future repo-root `AGENTS.md` explicitly adopts it.

## Migrated scoped rules

- `.codex/rules/code-style.md` — migrated from `.claude/rules/code-style.md`
- `.codex/rules/config.md` — migrated from `.claude/rules/config.md`
- `.codex/rules/distributed.md` — migrated from `.claude/rules/distributed.md`
- `.codex/rules/experiments.md` — migrated from `.claude/rules/experiments.md`
- `.codex/rules/models.md` — migrated from `.claude/rules/models.md`
- `.codex/rules/testing.md` — migrated from `.claude/rules/testing.md`

## Source: root `CLAUDE.md`

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


---

## Source: `.claude/CLAUDE.md`

# TorchTitan Development Guide

## Build & Test

```bash
# Install dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Lint and format (required before any PR)
pre-commit run --all-files

# Run unit tests
pytest tests/ -x
```

### Run GPU integration tests (requires GPUs)
Integration tests override default config for Llama 3 debug model.
See tests/integration_tests/ for `OverrideDefinitions`.

### Validating Numerics
Non-computation changes (e.g. activation checkpointing, refactoring) must produce
**identical loss** before vs. after with `--debug.seed=42` and `--debug.deterministic`.
Computation changes require loss convergence on representative datasets (e.g. C4).

With the same parallelisms, GPU settings, and the debug options, two runs should produce
bit-wise identical loss and grad_norm. Note that stdout only prints the most
significant five digits, which may not be enough. Follow `scripts/loss_compare.py` to
enable profiling and check loss and grad_norm from the TensorBoard results.

You should NEVER use `--debug.deterministic_warn_only`.

## Core Principles

1. **PyTorch-native training techniques.** Core torchtitan's training infrastructure
   and parallelism code must not depend on non-PyTorch libraries. Techniques with
   moderate-to-large complexity belong in their proper upstream repo (pytorch/pytorch
   for parallelisms, pytorch/data for data loaders, etc.).

2. **Investigate root cause before patching.** Don't land band-aid fixes. Understand
   *why* something fails before proposing a solution. If a change seems to help but
   you can't explain why, dig deeper.

3. **Reuse over duplication.** Before writing new code, check if existing implementations
   already handle the case. Unify similar code paths across models rather than creating
   per-model wrappers. If upstream (torchao, PyTorch) already provides functionality,
   use it.

4. **Don't leak experiments into core.** The `torchtitan/experiments/` folder exists for
   a reason. Don't modify core torchtitan code to accommodate experiment-specific needs
   (e.g. don't add `if experiment_x:` branches to core files). Deprecated files should
   be removed, not updated.

5. **Protect battle-tested code paths.** Be cautious changing converged behavior. Flag
   potential silent breakage of existing user code or checkpoints. When in doubt, ask.

6. **Audit all callsites.** When changing shared code (common model components, config
   fields, distributed utilities), check and update every callsite. This includes all
   model variants: llama3, llama4, qwen3, deepseek_v3, gpt_oss, flux, etc.

## Code Style

### Naming
- Names must be **accurate, descriptive, and reflect actual scope**. Don't use
  "toy/test/temp" in production names — put that context in docstrings instead.
- Follow upstream conventions: match torchao and PyTorch naming where applicable.
  E.g. if torchao calls it `Float8Linear`, use `Float8Linear` not `Float8Config`.
- Use `num_` prefix for counts (e.g. `num_expert_groups` not `n_expert_groups`)
  when not directly matching an upstream API.

### Code Placement
- Put code in the **most general applicable location**:
  - Model-agnostic parallelism helpers → `torchtitan/distributed/`
  - Shared model components (attention, MoE, embeddings) → `torchtitan/models/common/`
  - Model-specific code → the specific model folder
- Don't put model-agnostic functionality in model-specific files just because
  that's where you first needed it.

### Assertions and Error Handling
- **`ValueError`** for user-facing errors (bad config, invalid input).
- **`assert`** only for internal invariants that indicate programmer error.
- Always validate mesh dimensions, tensor placements, and config values explicitly
  in distributed code — don't assume 1D mesh or specific placements.
- When a code path silently skips user configuration, **emit a warning**.

### Parameters and Config
- Important parameters first; less important ones later.
- Prefer keyword-only arguments after the first positional arg.
- No `None` defaults for required config fields.
- `dataclasses.replace()` is a shallow copy: nested dataclasses and list/dict
  fields are shared by reference. Be explicit when deep copies are needed.

### Comments and Documentation
- Add comments only for genuinely non-obvious things: dimension semantics,
  parallelism gradient placements, why a workaround exists.
- Use TODO comments for known limitations with a brief explanation.
- Put descriptions in docstrings, not in names.

## PR Expectations

1. **Lint first.** Run `pre-commit run --all-files` and fix all issues before
   requesting review. CI linting failures waste everyone's time.
2. **Show numerical proof.** Include loss comparison for any non-trivial change.
3. **Explain "why" not just "what"** in the PR description.
4. **Add tests.** New features need CPU unit tests at minimum; GPU integration
   tests when involving parallelism. Verify CI actually runs the intended test
   config (check `--model.name` and other flags).
5. **Keep model code minimal.** After model changes, ensure original checkpoints
   still load correctly. Document reasons for model changes.
