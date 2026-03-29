---
name: validate-numerics
description: Guide for validating numerical correctness of torchtitan changes. Use when verifying loss convergence, comparing runs, or checking computation changes.
---

# Validate Numerics

Workflow for validating numerical correctness of changes to torchtitan.

## When to Use

- After any non-trivial code change before submitting PR
- After refactoring that touches computation paths
- After changing activation checkpointing, parallelism, or compile settings
- When adding new model architectures

## Decision Tree

```
Is this a computation change?
├── NO (refactoring, AC, config) → Must produce IDENTICAL loss
│   └── Use: Deterministic comparison (Phase 1)
└── YES (new model, optimizer, loss) → Must CONVERGE
    └── Use: Convergence validation (Phase 2)
```

## Phase 1: Deterministic Comparison (Non-Computation Changes)

### Step 1: Run Baseline

```bash
# Run with deterministic mode
torchrun --nproc_per_node=<N> -m torchtitan.train \
  --module <model> --config <config> \
  --debug.seed=42 --debug.deterministic \
  --training.steps=20 \
  --metrics.log_freq=1 \
  --profiling.enable_profiling
```

**Important:** Never use `--debug.deterministic_warn_only`.

### Step 2: Apply Changes and Re-run

Run the same command after applying your changes.

### Step 3: Compare Results

```bash
# Use loss_compare.py for detailed comparison
python scripts/loss_compare.py \
  --baseline <baseline_tb_dir> \
  --comparison <comparison_tb_dir>
```

**What to compare:**
- Loss values (must be bit-wise identical)
- grad_norm values (must be bit-wise identical)
- stdout only prints 5 significant digits — **not sufficient** for validation
- Use TensorBoard results from profiling for full precision

### Step 4: Report

```markdown
## Numerical Validation

**Type:** Non-computation change (identical loss required)
**Config:** llama3_debugmodel, 4 GPUs, TP=2, FSDP=2
**Seed:** 42, deterministic mode

| Step | Baseline Loss | After Loss | Match |
|------|---------------|------------|-------|
| 1    | 10.1234567    | 10.1234567 | YES   |
| 5    | 9.8765432     | 9.8765432  | YES   |
| 10   | 9.5432109     | 9.5432109  | YES   |
| 20   | 9.1234567     | 9.1234567  | YES   |

**Result:** Bit-wise identical loss and grad_norm confirmed.
```

## Phase 2: Convergence Validation (Computation Changes)

### Step 1: Run on Representative Dataset

```bash
# Use C4 or similar representative dataset
torchrun --nproc_per_node=<N> -m torchtitan.train \
  --module <model> --config <config> \
  --training.steps=1000 \
  --training.dataset=c4 \
  --metrics.log_freq=10 \
  --profiling.enable_profiling
```

### Step 2: Compare Loss Curves

```bash
python scripts/loss_compare.py \
  --baseline <baseline_tb_dir> \
  --comparison <comparison_tb_dir> \
  --convergence  # Flag for convergence comparison
```

### Step 3: Report

```markdown
## Numerical Validation

**Type:** Computation change (convergence required)
**Config:** llama3_8b, 8 GPUs, TP=2, FSDP=4
**Dataset:** C4, 1000 steps

| Metric | Baseline | After | Delta |
|--------|----------|-------|-------|
| Final loss | 3.456 | 3.461 | +0.005 |
| Min loss | 3.234 | 3.238 | +0.004 |
| Loss std | 0.12 | 0.13 | +0.01 |

**Result:** Loss converges within acceptable range.
```

## Multiple Parallelism Configs

For distributed code changes, test across multiple configs:

```bash
# Config 1: Pure FSDP
--parallelism.dp_shard=4 --parallelism.tp=1 --parallelism.pp=1

# Config 2: FSDP + TP
--parallelism.dp_shard=2 --parallelism.tp=2 --parallelism.pp=1

# Config 3: FSDP + PP
--parallelism.dp_shard=2 --parallelism.tp=1 --parallelism.pp=2

# Config 4: Full 3D
--parallelism.dp_shard=1 --parallelism.tp=2 --parallelism.pp=2
```

## Key Rules

1. **Never use `--debug.deterministic_warn_only`**
2. stdout's 5 significant digits are NOT sufficient — use TensorBoard
3. Same parallelisms + GPU settings + debug options = bit-wise identical
4. Always specify `--debug.seed=42 --debug.deterministic`
5. Include validation results in PR description
