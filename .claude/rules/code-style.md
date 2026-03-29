---
description: Code style rules for torchtitan
globs: torchtitan/**
---

# Code Style Rules

## Design Patterns
- All components use nested Config dataclasses with `@dataclass(kw_only=True, slots=True)`
- Composition over inheritance; shallow class hierarchies
- All trainable components inherit from `Module` (nn.Module + Configurable)
- `init_weights()` for weight initialization, not scattered in `__init__`

## Naming Conventions
- Names must be accurate, descriptive, and reflect actual scope
- Follow PyTorch and torchao naming conventions upstream
- Use `num_` prefix for count fields (e.g., `num_expert_groups`)
- Config classes: `XxxConfig`; components: descriptive names
- Tensor shape convention: `[batch, seq_len, hidden]`

## Import Organization
- Group: stdlib -> third-party -> torchtitan
- No wildcard imports
- Explicit imports for large modules
- Use relative imports within the same package

## Code Placement
- Model-agnostic parallelism helpers -> `torchtitan/distributed/`
- Shared model components -> `torchtitan/models/common/`
- Model-specific code -> the specific model folder
- Don't put model-agnostic functionality in model-specific files

## Performance Patterns
- Avoid GPU-CPU sync (`.item()`, `.tolist()`) in training hot paths
- Prefer batch operations over loops
- Use `torch.no_grad()` for inference-only code paths
- Be cautious with in-place operations in autograd context

## Assertions and Error Handling
- `ValueError` for user-facing errors (bad config, invalid input)
- `assert` only for internal invariants (programmer error)
- Validate mesh dimensions, tensor placements, config values explicitly
- Emit warnings when config silently doesn't take effect

## Comments and Documentation
- Comments only for genuinely non-obvious things
- Dimension semantics, parallelism gradient placements, workaround rationale
- Use TODO comments for known limitations with brief explanation
- Descriptions in docstrings, not in names
