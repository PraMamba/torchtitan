---
name: add-experiment
description: Guide for adding a new experiment to torchtitan. Use when user wants to create a new experimental feature in torchtitan/experiments/.
---


## Codex Compatibility Note

Migrated from `.claude/skills/add-experiment/SKILL.md`. This skill preserves the original Claude-oriented workflow
for information fidelity, with the following Codex adaptation rules:

- Treat Claude-specific tool names or UI mechanisms as historical references.
- Use available Codex shell/MCP/subagent tools that satisfy the same workflow step.
- Keep destructive or external side-effect steps gated by the current Codex
  permissions and project policy.
- If a step references Claude-only cache paths or invocation syntax, treat that
  as source context unless the user explicitly asks to reproduce it.

---


# Add Experiment to TorchTitan

Create a new experimental feature in `torchtitan/experiments/`.

## When to Use

- User wants to add a new training paradigm (RL, fine-tuning, etc.)
- User wants to experiment with alternative infrastructure (graph training, auto-parallel)
- User has a feature that's not ready for core but needs a home

## Key Principles

1. **Don't modify core** — Never add `if experiment_x:` branches to core files
2. **Use torchtitan's config system** — Don't introduce custom argument parsing
3. **Still lint** — Must pass `pre-commit run --all-files`
4. **Separate concerns** — Keep distinct features in separate folders

## Step-by-Step Guide

### Step 1: Create Experiment Directory

```
torchtitan/experiments/<experiment_name>/
├── __init__.py              # Required for module discovery
├── config_registry.py       # Experiment-specific configs
├── trainer.py               # Custom Trainer subclass (if needed)
├── README.md                # Experiment documentation
└── ...                      # Other experiment-specific files
```

### Step 2: Define Config Registry

```python
# torchtitan/experiments/<name>/config_registry.py
from torchtitan.trainer import Trainer

def experiment_debugmodel() -> Trainer.Config:
    """Debug config for experiment."""
    # Can extend Trainer.Config with custom fields if needed
    return Trainer.Config(...)
```

### Step 3: Register the Experiment

Add to `torchtitan/experiments/__init__.py`:

```python
# Note: uses frozenset([...]) with list syntax, not set syntax
# Use dotted names for model-specific variants (e.g., "myexp.llama3")
_supported_experiments = frozenset([
    ...,
    "<experiment_name>",          # For model-agnostic experiments
    "<experiment_name>.llama3",   # For model-specific variants
])
```

### Step 4: Custom Trainer (If Needed)

If the experiment needs custom training logic, subclass Trainer:

```python
from torchtitan.trainer import Trainer

class ExperimentTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        # Add experiment-specific config fields
        custom_field: str = "default"

    def train(self):
        # Custom training loop
        ...
```

### Step 5: Custom Components (If Needed)

Keep experiment-specific components within the experiment folder:
- Custom datasets -> `<experiment>/datasets/`
- Custom models -> `<experiment>/models/`
- Custom loss functions -> `<experiment>/loss.py`

### Step 6: Add Tests

```
tests/integration_tests/<experiment_name>.py
```

Follow existing experiment test patterns (e.g., `graph_trainer`, `rl`).

### Step 7: Add CI (If Needed)

Create `.github/workflows/integration_test_<experiment>.yaml` following
existing workflow patterns.

## Existing Experiments Reference

| Experiment | Registered Names | Purpose | Directory |
|-----------|-----------------|---------|-----------|
| graph_trainer | `graph_trainer.llama3`, `graph_trainer.deepseek_v3` | Graph-based training via torch.fx | `experiments/graph_trainer/` |
| autoparallel | `autoparallel.llama3`, `autoparallel.deepseek_v3` | Automatic parallelism discovery | `experiments/autoparallel/` |
| vlm | `vlm` | Vision Language Models | `experiments/vlm/` |
| ft | `ft.llama3` | Fine-tuning (TorchFT, Diloco) | `experiments/ft/` |
| rl | `rl` | Reinforcement Learning | `experiments/rl/` |
| transformers_modeling_backend | `transformers_modeling_backend` | HF Transformers integration | `experiments/transformers_modeling_backend/` |
| forge | (not yet registered) | Advanced infrastructure | `experiments/forge/` |

## Common Mistakes

- Modifying core torchtitan code for experiment needs
- Introducing custom argument parsing instead of using Config system
- Bundling unrelated features in the same experiment folder
- Not adding to `_supported_experiments` frozenset
- Forgetting to lint (`pre-commit run --all-files`)
