---
description: Testing rules for torchtitan
globs: tests/**
---

# Testing Rules

## Test Structure
- Unit tests in `tests/unit_tests/`
- Integration tests in `tests/integration_tests/`
- Test naming: `test_<what>_<condition>_<expected>`
- Use Arrange-Act-Assert pattern

## Unit Tests
- Must run without GPUs
- Use `pytest` with `-x` flag for early stopping
- Test files mirror source structure: `test_<module>.py`
- Use `torch.testing.assert_close()` with explicit `rtol`/`atol`
- Use `expecttest` for snapshot testing where appropriate

## Integration Tests
- Require GPUs; organized in `tests/integration_tests/`
- Use `OverrideDefinitions` to specify test configs as CLI argument overrides
- Test suites: `features.py`, `models.py`, `flux.py`, `h100.py`
- Run via `run_tests.py` with torchrun-style process launching
- Always include parallelism configuration in test descriptions

## Distributed Mocking
- Use `torch.distributed.fake_pg` for process group mocking
- Mock `get_rank`/`get_world_size` for CPU-only distributed tests
- Skip GPU tests gracefully with conditional skips

## GPU Test Patterns
```python
import unittest
import torch

@unittest.skipIf(not torch.cuda.is_available(), "Requires CUDA")
class TestGPUFeature(unittest.TestCase):
    def setUp(self):
        torch.cuda.empty_cache()

    def tearDown(self):
        torch.cuda.empty_cache()
```

## Numerical Validation
- Non-computation changes must produce **identical loss** with `--debug.seed=42` and `--debug.deterministic`
- Computation changes require loss convergence on representative datasets
- Use `scripts/loss_compare.py` for comparing TensorBoard results
- Never use `--debug.deterministic_warn_only`
- Compare both loss and grad_norm (stdout shows only 5 significant digits)

## Test Assertions
- `torch.testing.assert_close()` for tensor comparisons — specify tolerances
- `assertEqual` for exact matches (ints, strings, configs)
- Never compare floating point with `==`

## CI Integration
- Unit tests run on every PR via GitHub Actions (`lint.yaml`)
- Integration tests on 8-GPU runners (`integration_test_8gpu_*.yaml`)
- Verify CI actually runs the intended test config
