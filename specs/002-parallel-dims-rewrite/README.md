---
status: complete
created: '2026-01-03'
tags:
  - architecture
  - parallelism
  - device-mesh
  - api-redesign
priority: high
created_at: '2026-01-03T15:20:26.454Z'
updated_at: '2026-01-03T15:22:44.075Z'
completed_at: '2026-01-03T15:22:44.075Z'
completed: '2026-01-03'
transitions:
  - status: complete
    at: '2026-01-03T15:22:44.075Z'
---

# ParallelDims API Rewrite with DeviceMesh Unflatten

> **Status**: ✅ Complete · **Priority**: High · **Created**: 2026-01-03 · **Tags**: architecture, parallelism, device-mesh, api-redesign

## Overview

**Problem**: ParallelDims mesh creation logic is complex and hard to maintain. Multiple code paths for building multi-dimensional device meshes. No unified API for accessing submeshes.

**Solution**: Leverage PyTorch DeviceMesh's `unflatten()` API to simplify mesh creation. Single world mesh as source, unflatten to create all submeshes. Unified access through `get_mesh()` and `get_optional_mesh()`.

**Impact**: 32 files changed, +1200/-515 lines. Clearer code, better testability, easier to add new parallelism strategies.

**Why now**: DeviceMesh unflatten API recently stabilized. Current complexity blocking new parallelism features.

## Design

### Core Philosophy

**Three-step approach**:
1. Create single world mesh: `[world_size]`
2. Unflatten to global meshes: dataloading, dense, sparse
3. Extract 1-D submeshes for individual access

**Example** (world_size=8, dp_replicate=2, dp_shard=2, tp=2):
```
world_mesh: [8]
    ↓ unflatten
dense_mesh: [pp=1, dp_replicate=2, fsdp=2, tp=2]
    ↓ extract
submeshes: {"dp_replicate": mesh[2], "fsdp": mesh[2], "tp": mesh[2]}
```

### Three Global Meshes

**dataloading_mesh**: `["pp", "batch", "cp", "tp"]`
- Used for data loading, determines global batch size
- `batch = dp_replicate * dp_shard`

**dense_mesh**: `["pp", "dp_replicate", "fsdp", "tp"]`
- Used for dense (non-MoE) layer parallelization
- `fsdp = dp_shard * cp`

**sparse_mesh**: `["pp", "dp_replicate", "efsdp", "ep", "etp"]`
- Used for MoE layer parallelization
- `efsdp = fsdp * tp / (etp * ep)`

### New API

**get_optional_mesh(dims: str | list[str]) -> DeviceMesh | None**
- Returns `None` if parallelism not enabled (degree=1)
- Use when parallelism is optional

**get_mesh(dims: str | list[str]) -> DeviceMesh**
- Raises `ValueError` if not enabled
- Use when parallelism is required

**Examples**:
```python
# Single dimension
tp_mesh = parallel_dims.get_mesh("tp")

# Combined dimensions (creates n-D mesh)
dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])

# Optional (may be None)
tp_mesh = parallel_dims.get_optional_mesh("tp")
if tp_mesh is not None:
    apply_tp(model, tp_mesh)
```

### Key Constraints

**Dimension formulas**:
```
dp_replicate * dp_shard * cp * tp * pp = world_size
batch = dp_replicate * dp_shard
fsdp = dp_shard * cp
efsdp = fsdp * tp / (etp * ep)
```

**ETP constraint**: When `ep > 1`, must have `etp = tp` or `etp = 1`

**efsdp special case**: Exists whenever `ep > 1` (even if size=1) because MoE experts need FSDP wrapper for mixed precision.

### Backend Override (Optimization)

**fake backend** for dimensions where degree=1 or no communication needed:
```python
backend_override = {}
for name, degree in zip(dim_names, dim_degrees):
    if degree == 1 or name == "batch":
        backend_override[name] = "fake"
```

Avoids creating unnecessary process groups.

## Plan

- [x] Rewrite `ParallelDims.build_mesh()` using unflatten
  - [x] Create world mesh
  - [x] Unflatten to three global meshes
  - [x] Extract 1-D submeshes
  - [x] Add mesh size validation
- [x] Implement `get_mesh()` and `get_optional_mesh()` APIs
  - [x] Single dimension access
  - [x] Multi-dimension combination (from same global mesh)
  - [x] Error handling for invalid combinations
- [x] Update all callsites (32 files)
  - [x] Replace `world_mesh["tp"]` with `get_mesh("tp")`
  - [x] Replace `world_mesh[("dp", "tp")]` with `get_mesh(["dp", "tp"])`
  - [x] Update function signatures (remove `world_mesh` param)
  - [x] Handle optional meshes with `get_optional_mesh()`
- [x] Update distributed utils
  - [x] Accept `DeviceMesh | None` in reduction functions
  - [x] Update `set_determinism()` to use ParallelDims
- [x] Write comprehensive tests (569 lines)
  - [x] Validation tests (no distribution required)
  - [x] Mesh building tests (with distribution)
  - [x] API tests (get_mesh, get_optional_mesh)

## Test

### Correctness
- [x] Mesh sizes match expected formulas
- [x] All 1-D submeshes correctly extracted
- [x] Multi-dimensional mesh combinations work
- [x] Invalid combinations raise clear errors

### API Behavior
- [x] `get_mesh()` raises when degree=1
- [x] `get_optional_mesh()` returns None when degree=1
- [x] efsdp exists when ep>1 regardless of size

### Integration
- [x] All models parallelize correctly (llama3, llama4, deepseek_v3, qwen3, flux)
- [x] Data loading uses batch mesh correctly
- [x] Loss reduction uses loss mesh correctly
- [x] Seed determinism works with new API

### Edge Cases
- [x] Single GPU (world_size=1, all degrees=1)
- [x] Auto dp_shard calculation (dp_shard=-1)
- [x] Invalid world_size raises AssertionError
- [x] Invalid etp (not tp or 1) raises AssertionError

## Notes

### Why Single World Mesh?

**Considered**: Create separate meshes for each parallelism type

**Rejected because**:
- Duplication: same devices appear in multiple meshes
- Inconsistency: hard to ensure meshes are aligned
- Resource waste: redundant process group creation

Single world mesh ensures all submeshes are consistent derivatives.

### Why Three Global Meshes?

Different parallelism strategies need different dimension groupings:
- **dataloading**: Groups all data-related dims (dp_replicate, dp_shard, cp)
- **dense**: Groups dims for non-MoE layers (fsdp includes cp for FSDP all-gather)
- **sparse**: Groups dims for MoE layers (ep, etp separate from dense)

Can't flatten all into one mesh because different layers need different combinations.

### Why Unflatten vs Manual Slicing?

**Unflatten approach**:
```python
mesh = world_mesh._unflatten(0, (2, 2, 2), ("pp", "dp", "tp"))
# Automatically maps devices: [0,1,2,3,4,5,6,7] -> [[[0,1],[2,3]],[[4,5],[6,7]]]
```

**Manual slicing approach**:
```python
mesh_3d = init_device_mesh("cuda", (2, 2, 2))
pp_mesh = mesh_3d[:, 0, 0]  # Requires understanding indexing
```

Unflatten is clearer about intent and avoids indexing errors.

### Migration Pattern

**Before**:
```python
def parallelize(model, world_mesh, parallel_dims, ...):
    if parallel_dims.tp_enabled:
        apply_tp(model, world_mesh["tp"])
```

**After**:
```python
def parallelize(model, parallel_dims, ...):
    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp(model, tp_mesh)
```

**Benefits**:
- Fewer parameters (remove world_mesh)
- Clearer ownership (parallel_dims owns meshes)
- Better error messages (from get_mesh)

### Dimension Naming Changes

| Old Name | New Name | Meaning |
|----------|----------|---------|
| dp_shard_cp | fsdp | FSDP dimension (dp_shard * cp) |
| - | batch | Data loading dimension (dp_replicate * dp_shard) |
| - | loss | Loss reduction dimension (batch * cp) |

More descriptive names aligned with their usage.
