# AGENTS.md

This file provides repository-level guidance for Codex when working in this
TorchTitan checkout. It is adapted from `CLAUDE.md` and the Codex compatibility
assets under `.codex/`, but it is written for Codex instruction discovery and
Codex workflows.

## Codex Instruction Scope

- This file is the repository-root Codex guidance file. It applies to the entire
  repository unless a deeper `AGENTS.md` or `AGENTS.override.md` provides more
  specific guidance.
- Keep this file concise enough for Codex project-instruction loading. Put long
  reference material in `.codex/` and link to it here.
- Treat `.codex/PROJECT_GUIDE.md`, `.codex/rules/`, `.codex/skills/`,
  `.codex/workflows/`, and `.codex/agents/` as supporting Codex assets.
- `.codex/AGENTS.md` is scoped only to `.codex/` maintenance; this root file is
  the authoritative repository-wide Codex instruction file.
- Do not translate legacy Claude hooks into executable Codex hooks unless a
  separate, explicit task requests and plans that migration.

## Overview

TorchTitan is a PyTorch-native platform for training large generative AI models
with multi-dimensional parallelism. It is designed for rapid experimentation and
production-ready training of LLMs and other generative models.

Core themes:
- PyTorch-native training infrastructure and parallelism.
- Composable model, data, optimizer, checkpoint, metric, and validation
  components.
- Multi-dimensional parallelism: DP/FSDP, TP, PP, CP, EP, and related MoE
  strategies.
- Careful numerical validation for training changes.

## Common Commands

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Download tokenizer assets before training
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

# Run a specific test file
pytest -s tests/unit_tests/test_job_config.py

# Run a specific test function
pytest -s tests/unit_tests/test_job_config.py::TestJobConfig::test_command_line_args

# Run integration tests (requires GPUs)
python -m tests.integration_tests.run_tests test_output --test_suite features --ngpu 8

# Run a specific integration test
python -m tests.integration_tests.run_tests test_output --test_suite features --test_name gradient_accumulation --ngpu 2
```

### Linting and Pre-commit

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Individual linters/checkers
flake8 --config=.flake8
ufmt format .
pyrefly-check
```

Run the narrowest meaningful verification during iteration, then run the broader
checks required by the change before declaring completion.

## Architecture

### Core Structure

```text
torchtitan/
├── train.py                 # Main training loop and Trainer class
├── components/              # Reusable components: checkpoint, dataloader, optimizer, etc.
├── config/                  # Configuration system (TOML + CLI parsing)
│   ├── job_config.py        # Complete JobConfig dataclass hierarchy
│   └── manager.py           # Config parser with precedence: CLI > TOML > defaults
├── distributed/             # Parallelism implementations
│   ├── parallel_dims.py     # ParallelDims: multi-dimensional device meshes
│   ├── pipeline_parallel.py # PP schedules (1F1B, Interleaved, ZBV)
│   └── tensor_parallel.py   # TP utilities
├── models/                  # Model implementations
│   └── {model_name}/
│       ├── model/           # Model definition
│       ├── infra/           # Parallelization logic
│       └── train_configs/   # TOML configuration files
├── protocols/               # Abstract interfaces and extension points
│   └── train_spec.py        # TrainSpec protocol
├── experiments/             # Experimental features (experiments -> core only)
└── tools/                   # Utilities: logging, profiling, etc.
```

### Training Flow

1. `ConfigManager.parse_args()` loads config defaults and CLI overrides.
2. `Trainer.__init__()` initializes distributed state, resolves the model
   `TrainSpec`, builds tokenizer/dataloader/model/loss/optimizer/scheduler,
   applies parallelism, initializes weights, and configures checkpointing and
   metrics.
3. `Trainer.train()` runs the forward/backward/optimizer loop with periodic
   logging, checkpointing, and validation.

### Parallelism System

`ParallelDims` in `torchtitan/distributed/parallel_dims.py` creates and validates
multi-dimensional device meshes:

- `dp_replicate` — DDP/HSDP replicate dimension
- `dp_shard` — FSDP shard dimension
- `tp` — Tensor Parallel
- `pp` — Pipeline Parallel
- `cp` — Context Parallel
- `ep` — Expert Parallel for MoE models

Parallelism application order is generally TP, CP, activation checkpointing,
`torch.compile`, FSDP, then PP stage scheduling. Consider all parallelism
combinations when changing distributed behavior.

### Configuration System

Configurations use TOML files plus CLI overrides. CLI values override config
registry/default values.

Key config areas include:
- `model` — model selection and HF assets
- `training` — batch sizes, sequence length, dtype, steps, dataset
- `parallelism` — DP/TP/PP/CP/EP degrees and policies
- `optimizer` — AdamW and related options
- `lr_scheduler` — warmup/stable/decay schedule
- `checkpoint` — DCP, async checkpointing, HF conversion
- `activation_checkpoint` — selective/full/memory-budget modes
- `compile` — `torch.compile` settings
- `quantize` — Float8/MXFP8 quantization

### Extension Points

Models integrate through `TrainSpec` in `torchtitan/protocols/train_spec.py`.
To add a model, follow the established model directory pattern:

```text
torchtitan/models/{model_name}/
├── __init__.py              # Export registry/get_train_spec entrypoint
├── model/
│   ├── model.py             # Model class implementing the module protocol
│   └── args.py              # Model args/config dataclasses
├── infra/
│   └── parallelize.py       # parallelize_fn and optional pipelining_fn
└── train_configs/
    └── {flavor}.toml        # TOML configs for sizes/flavors
```

Experiments belong under `torchtitan/experiments/`, must reuse core components
where practical, and must not force experiment-specific branches into core code.

## Key Files for Common Tasks

### Training Loop
- `torchtitan/train.py` — main `Trainer` and training loop.
- `torchtitan/protocols/train_spec.py` — TrainSpec protocol.

### Models
- `torchtitan/models/llama3/` — reference dense LLM implementation.
- `torchtitan/models/common/` — shared attention, feed-forward, decoder, MoE,
  RoPE, normalization, and related components.
- Model-specific `infra/parallelize.py` files — per-model parallelization.

### Distributed / Parallelism
- `torchtitan/distributed/parallel_dims.py` — mesh creation and validation.
- `torchtitan/distributed/tensor_parallel.py` — TP utilities.
- `torchtitan/distributed/pipeline_parallel.py` — PP splitting and schedules.
- `torchtitan/distributed/context_parallel.py` — CP hooks.
- `torchtitan/distributed/expert_parallel.py` — EP/MoE parallelism.
- `torchtitan/distributed/utils.py` — shared distributed utilities.

### Checkpointing
- `torchtitan/components/checkpoint.py` — DCP save/load and async support.
- `scripts/checkpoint_conversion/` — HF/TorchTitan conversion utilities.
- `docs/checkpoint.md` — checkpoint interoperability guidance.

### Quantization
- `torchtitan/components/quantization/float8.py` — Float8 implementation.
- `docs/float8.md` — Float8 usage.
- `docs/mxfp8.md` — MXFP8 usage.

### Configuration
- `torchtitan/config/` — config dataclasses, manager, configurable base.
- `torchtitan/models/*/train_configs/` — model config examples.

## Important Conventions

### General Engineering

- Prefer PyTorch-native techniques and upstream PyTorch/torchao patterns.
- Reuse existing utilities and shared components before adding new abstractions.
- Keep model code minimal; training infrastructure belongs outside model
  architecture files.
- Put model-agnostic parallelism helpers in `torchtitan/distributed/`.
- Put shared model components in `torchtitan/models/common/`.
- Put model-specific behavior in the specific model directory.
- Do not modify core code solely to support one experiment.
- Remove deprecated code when appropriate rather than updating dead paths.

### Naming and Style

- Use accurate, descriptive names that reflect actual scope.
- Follow upstream PyTorch and torchao naming when applicable.
- Use `num_` prefixes for counts, unless matching an upstream API.
- Use tensor shape conventions such as `[batch, seq_len, hidden]`.
- Group imports as stdlib, third-party, then `torchtitan`; avoid wildcard imports.
- Add comments only for genuinely non-obvious dimension semantics, placement
  behavior, or workaround rationale.

### Config and Error Handling

- Avoid `None` defaults for required config fields.
- Put important and commonly used config fields first.
- Prefer keyword-only arguments after the first positional argument.
- Emit warnings when a user config option is silently ignored in a code path.
- Use `ValueError` for user-facing bad config/input errors.
- Use `assert` only for internal invariants that indicate programmer error.
- Remember that `dataclasses.replace()` is shallow; deep-copy nested mutable
  fields when needed.

### Distributed Code

- Never assume a 1D mesh; validate mesh dimensions explicitly.
- Validate tensor placements (`Replicate`, `Shard`, `Partial`) explicitly.
- Document `DTensor.to_local` gradient placements when relevant.
- Include parallelism configuration in bug reports and test descriptions.
- Be conservative with converged distributed behavior; watch for silent
  correctness issues.

### Models

- Audit all model variants when changing shared components: llama3, llama4,
  qwen3, deepseek_v3, gpt_oss, flux, and any additional present variants.
- Do not create per-model wrappers for functionality that should be shared.
- Ensure original checkpoints still load after model changes.
- Keep important control flow visible in `forward()` rather than hiding key
  routing decisions in helpers.

### Experiments

- Experiments may move faster than core but must still pass lint and relevant
  tests.
- Keep distinct experiments in separate folders.
- Use TorchTitan's config/job-config infrastructure; do not introduce parallel
  argument/config systems unless explicitly justified.
- Experiments may depend on core; core must not depend on experiments.

## Testing and Numerical Validation

- Unit tests live in `tests/unit_tests/` and must run without GPUs.
- Integration tests live in `tests/integration_tests/` and often require GPUs.
- Use `pytest` with `-x` for focused local debugging.
- Use `torch.testing.assert_close()` with explicit `rtol`/`atol` for tensor
  comparisons; do not compare floating point values with `==`.
- Skip GPU tests gracefully when CUDA is unavailable.

Numerical expectations:
- Non-computation changes, refactors, and activation-checkpointing changes must
  produce identical loss and grad-norm under deterministic settings.
- Computation changes must show convergence on representative datasets.
- Use `--debug.seed=42` and `--debug.deterministic` for deterministic checks.
- Never use `--debug.deterministic_warn_only` as proof of correctness.
- Use `scripts/loss_compare.py` and TensorBoard/profiling outputs for precise
  loss/grad-norm comparison; stdout precision is not enough.

## Codex Project Assets

This repository includes a `.codex/` compatibility layer migrated from prior
Claude-oriented workflow assets.

### Custom Agents

Project-scoped Codex custom agents are in `.codex/agents/*.toml`:

- `checkpoint-expert` — checkpointing, DCP, state dict adapters, HF conversion.
- `code-verifier` — formatting, lint, tests, and readiness checks.
- `config-expert` — Configurable pattern, config manager, dataclass configs.
- `distributed-expert` — FSDP, TP, PP, CP, EP, DTensor, activation checkpointing.
- `model-expert` — model architecture, ModelSpec, adapters, model parallelism.
- `planner` — implementation planning for complex multi-file work.
- `trainer-expert` — trainer loop and training components.

Use these agents when the user explicitly asks for subagents or when the active
runtime permits delegated specialist review. Keep each delegated task bounded and
read-only unless the agent is explicitly assigned implementation ownership.

### Skills and Workflows

- `.codex/skills/add-model/SKILL.md` — adding model architectures.
- `.codex/skills/add-experiment/SKILL.md` — adding experiments.
- `.codex/skills/debug-distributed/SKILL.md` — distributed debugging.
- `.codex/skills/torch_bisect/SKILL.md` — PyTorch regression bisection.
- `.codex/skills/validate-numerics/SKILL.md` — numerical validation.
- `.codex/workflows/create-pr.md` — PR creation workflow.
- `.codex/workflows/gen-commit-msg.md` — commit-message workflow.
- `.codex/workflows/review-pr.md` — PR review workflow.

Treat `.codex/workflows/*.md` as Codex-invokable workflow references, not native
Claude slash commands. Ask Codex to follow the workflow by file path.

### Scoped Rule References

Detailed source-preserving rules are in `.codex/rules/`:

- `code-style.md` — general code style and placement.
- `config.md` — config-system rules.
- `distributed.md` — distributed training rules.
- `experiments.md` — experiments-folder rules.
- `models.md` — model implementation rules.
- `testing.md` — testing rules.

These files document intended source scopes. This root `AGENTS.md` adopts their
substantive guidance for repository work; consult them for detailed checklists.

## Pull Request and Commit Expectations

- Run relevant lint/tests before requesting review.
- Explain why a change is needed, not just what changed.
- Include numerical proof for non-trivial training or computation-path changes.
- Add CPU unit tests for new features when feasible.
- Add GPU integration tests for parallelism/distributed features when applicable.
- Verify CI actually exercises the intended model/config/test path.
- Do not commit directly to `main`.

## Verification Before Completion

Before reporting a task complete:

1. Confirm changed files are within the intended scope.
2. Run the relevant formatter/linter/test commands or explain any environment
   limitation clearly.
3. For config/documentation-only changes, run syntax/static checks appropriate to
   the edited files.
4. For `.codex` asset changes, run the verification from `.codex/README.md` or
   `.omx/plans/test-spec-claude-to-codex-compat.md`.
5. Report exact commands and outcomes, including skipped checks and why.
