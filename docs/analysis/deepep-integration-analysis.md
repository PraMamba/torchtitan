# TorchTitan DeepEP 集成详细分析

**文档版本**: 1.0
**创建日期**: 2026-01-03
**Commit**: 36a4b69 (2025-12-17)
**作者**: 基于源码分析

---

## 目录

1. [概述](#概述)
2. [DeepEP 是什么](#deepep-是什么)
3. [集成架构](#集成架构)
4. [核心组件详解](#核心组件详解)
5. [集成流程](#集成流程)
6. [性能优化原理](#性能优化原理)
7. [使用指南](#使用指南)
8. [限制与注意事项](#限制与注意事项)

---

## 概述

### 集成背景

**问题**:
- 传统的 Expert Parallelism (EP) 使用 PyTorch 标准的 all-to-all 通信
- 在超大规模 MoE 模型（如 DeepSeek-V3 671B）训练中，all-to-all 通信成为性能瓶颈
- MFU (Model FLOPS Utilization) 仅为 9.83%，远低于理想值

**解决方案**:
- 集成 DeepEP (Deep Expert Parallelism)
- DeepEP 是 DeepSeek 团队开发的高效 MoE 通信后端
- 通过自定义 CUDA 内核优化 all-to-all 通信

**效果**:
```
性能提升（DeepSeek-V3 671B on 512 H100 GPUs）:
- TPS:     346 → 579 (+67%)
- TFLOPS:  97.24 → 162.82 (+67%)
- MFU:     9.83% → 16.46% (+67%)
- Memory:  60.18 GiB → 56.75 GiB (-5.7%)
```

### 集成原则

1. **最小侵入性**: 不破坏现有架构，通过配置开关控制
2. **兼容性优先**: 确保与 torch.compile、SAC (Selective Activation Checkpointing) 等功能兼容
3. **可选性**: 用户可以选择使用标准后端或 DeepEP
4. **正确性保证**: Loss 曲线验证，确保数值正确性

---

## DeepEP 是什么

### 官方来源

- **GitHub**: https://github.com/deepseek-ai/deepep
- **开发者**: DeepSeek AI 团队
- **许可证**: 开源（需查看具体 license）

### 核心功能

DeepEP 提供两个核心操作：

1. **Dispatch (分发)**:
   - 将 tokens 从本地 rank 分发到负责处理的 expert ranks
   - 相当于 all-to-all 通信的发送阶段

2. **Combine (聚合)**:
   - 将处理后的 tokens 从 expert ranks 聚合回原始 ranks
   - 相当于 all-to-all 通信的接收阶段

### 优化技术

DeepEP 通过以下技术优化通信：

1. **Buffer 管理**:
   - 预分配通信 buffer（NVL 和 RDMA）
   - 避免运行时内存分配开销

2. **异步执行**:
   - 支持通信与计算重叠
   - 通过 CUDA events 管理流同步

3. **自定义 CUDA 内核**:
   - 优化的 dispatch/combine 内核
   - 针对 MoE 通信模式特化

---

## 集成架构

### 整体架构图

```
配置层 (job_config.py)
    ↓
    expert_parallel_comm_backend: "deepep" | "standard"
    ↓
模型定义层 (model/model.py)
    ↓
    MoE 或 DeepEPMoE
    ↓
并行化层 (infra/parallelize.py)
    ↓
    检测配置 → 选择 EP 实现
    ↓
通信层 (distributed/)
    ├── expert_parallel.py
    │   ├── ExpertParallel (标准)
    │   └── DeepEPExpertParallel (DeepEP)
    └── deepep/
        ├── __init__.py
        └── deepep.py (核心实现)
```

### 文件组织

```
torchtitan/
├── config/
│   └── job_config.py                    # 配置定义 (+12行)
├── distributed/
│   ├── __init__.py                      # 导出 DeepEPExpertParallel (+6行)
│   ├── expert_parallel.py               # DeepEPExpertParallel 类 (+67行)
│   └── deepep/
│       ├── __init__.py                  # 模块入口 (15行)
│       └── deepep.py                    # 核心实现 (462行)
└── models/
    ├── moe/
    │   ├── __init__.py                  # 导出 DeepEPMoE (+4行)
    │   ├── moe.py                       # MoE 基类修改 (+19行)
    │   └── moe_deepep.py                # DeepEPMoE 类 (58行)
    ├── deepseek_v3/
    │   ├── model/
    │   │   ├── args.py                  # 添加 expert_parallel_comm_backend (+7行)
    │   │   └── model.py                 # 根据配置选择 MoE 实现 (+10行)
    │   └── infra/
    │       └── parallelize.py           # DeepEP 检测与设置 (+27行)
    └── llama4/
        └── infra/
            └── parallelize.py           # 同样的 DeepEP 集成逻辑 (+47行)
```

---

## 核心组件详解

### 1. 配置系统 (`config/job_config.py`)

#### 新增配置项

```python
@dataclass
class Parallelism:
    expert_parallel_comm_backend: Literal["standard", "deepep"] = "standard"
    """
    Expert-parallel communication backend for MoE models.

    - "standard": PyTorch all-to-all collectives (default)
    - "deepep": DeepEP custom kernels (requires ep_degree > 1)

    No effect for non-MoE models or when ep_degree = 1.
    """
```

**关键点**:
- 默认值为 `"standard"`，保持向后兼容
- 只对 MoE 模型且 EP > 1 时有效
- 通过 TOML 或命令行设置

---

### 2. DeepEP 通信层 (`distributed/deepep/deepep.py`)

#### 2.1 依赖管理

```python
try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap
except ImportError as e:
    raise ImportError(
        "DeepEP is required for this module. "
        "Install from: https://github.com/deepseek-ai/deepep"
    ) from e
```

**说明**:
- DeepEP 是外部依赖，需要单独安装
- 如果未安装但配置了 DeepEP，会明确报错

#### 2.2 全局状态管理

```python
# 全局 buffer（每个进程一个，组变化时重建）
_buffer: Buffer = None

# 全局缓存，用于 dispatch handles，按 cache_id 索引
_handle_cache: dict = {}
_cache_counter: int = 0
```

**Buffer 管理**:
- `get_buffer(group, hidden_bytes)`: 获取或创建 buffer
- Buffer 包含两种内存：
  - `num_nvl_bytes`: NVLink 通信 buffer
  - `num_rdma_bytes`: RDMA 通信 buffer

**Handle 缓存机制**:
- Dispatch 返回一个 handle，存储在 `_handle_cache` 中
- Combine 时使用相同的 handle 进行反向操作
- 使用 CPU tensor 作为 cache_id，避免 GPU-CPU 同步
- 这对 SAC (Selective Activation Checkpointing) 至关重要

#### 2.3 自定义 Op 注册

```python
_lib = torch.library.Library("deepep", "DEF")

# dispatch 签名
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "Tensor num_tokens_per_rank, Tensor num_tokens_per_rdma_rank, "
    "Tensor is_token_in_rank, Tensor num_tokens_per_expert) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)

# combine 签名
_lib.define("combine(Tensor x, Tensor cache_id) -> Tensor")
```

**为什么需要自定义 Op**:
1. **torch.compile 兼容性**: 自定义 Op 可以被 torch.compile 识别和优化
2. **Autograd 集成**: 通过 `torch.library.register_autograd` 注册反向传播
3. **SAC 支持**: 自定义 Op 可以添加到 `op_sac_save_list`

#### 2.4 Dispatch 实现

```python
@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_rdma_rank: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> Tuple[...]:
    global _buffer
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"

    # 创建异步事件（用于流同步）
    previous_event = _create_event_if_async(True)

    # 调用 DeepEP buffer.dispatch
    (
        recv_x,                # 接收到的 tokens
        recv_indices,          # token 的 expert 索引
        recv_scores,           # routing scores
        num_recv_list,         # 每个 expert 接收的 token 数量
        handle,                # dispatch handle（combine 时需要）
        after_event,           # 完成事件
    ) = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights.to(torch.float32),  # DeepEP 要求 float32
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=True,                           # 异步执行
        allocate_on_comm_stream=True,                # 在通信流上分配
    )

    # 同步流
    _sync_stream_if_async(True, after_event)

    # 缓存 handle 用于 combine
    cache_id = _get_next_cache_id()  # CPU tensor
    _handle_cache[cache_id.item()] = handle

    # 返回结果
    num_recv_tensor = torch.tensor(num_recv_list, dtype=torch.int32, device="cpu")
    return recv_x, recv_indices, recv_scores, num_recv_tensor, cache_id
```

**关键点**:
1. **异步执行**: `async_finish=True` 允许通信与计算重叠
2. **流管理**: 在通信流上分配，避免阻塞默认流
3. **Handle 缓存**: 保存 dispatch handle 用于 combine
4. **类型转换**: 确保 topk_weights 为 float32

#### 2.5 Combine 实现

```python
@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(x: torch.Tensor, cache_id: torch.Tensor) -> torch.Tensor:
    global _buffer
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    # 从缓存中获取 dispatch handle
    handle = _handle_cache.get(cache_id.item())
    assert handle is not None, f"Handle not found for cache_id={cache_id.item()}"

    previous_event = _create_event_if_async(True)

    # 调用 DeepEP buffer.combine
    combined, _, after_event = buffer.combine(
        x=x,
        handle=handle,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    _sync_stream_if_async(True, after_event)

    return combined
```

**关键点**:
- 使用 dispatch 时保存的 handle
- Handle 包含了 routing 信息，知道如何反向路由 tokens

#### 2.6 Autograd 集成

```python
def _dispatch_backward(ctx, grad_recv_x, ...):
    """Dispatch 的反向传播：在梯度上执行 combine"""
    handle = _handle_cache.get(ctx.cache_id_int)

    # 使用 buffer.combine 处理梯度
    grad_x, grad_scores, after_event = _buffer.combine(
        x=grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_scores.float() if grad_recv_scores is not None else None,
        ...
    )

    # 清理 handle 缓存
    _handle_cache.pop(ctx.cache_id_int, None)

    return grad_x, None, grad_topk_weights, ...

def _combine_backward(ctx, grad_combined):
    """Combine 的反向传播：在梯度上执行 dispatch"""
    handle = ctx.saved_handle

    # 使用 buffer.dispatch 处理梯度
    grad_x, _, _, _, _, after_event = _buffer.dispatch(
        x=grad_combined,
        handle=handle,
        ...
    )

    return grad_x, None

# 注册反向传播
torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)
```

**对称性**:
- Forward: dispatch → combine
- Backward: combine (on grads) → dispatch (on grads)

#### 2.7 高层 API

```python
@dataclass
class DispatchState:
    """Dispatch 返回的状态，combine 时需要"""
    cache_id: torch.Tensor              # CPU tensor，用于检索 handle
    sorted_indices: torch.Tensor        # token 排序索引
    num_recv_tokens: int                # 接收到的 token 数量
    permuted_scores: Optional[torch.Tensor] = None  # 可选的 routing scores

def dispatch_tokens(
    hidden_states: torch.Tensor,            # [num_tokens, hidden_dim]
    selected_experts_indices: torch.Tensor, # [num_tokens, top_k]
    top_scores: torch.Tensor,               # [num_tokens, top_k]
    num_local_experts: int,
    num_experts: int,
    group: ProcessGroup,
    score_before_experts: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """
    通过 DeepEP 将 tokens 分发到 experts。

    返回:
        permuted_tokens:      # 按 expert 排序的 tokens
        tokens_per_expert:    # 每个 expert 的 token 数量
        state_for_combine:    # combine 需要的状态
    """
    # 1. 确保输入格式正确
    # 2. 获取/创建 buffer
    buffer = get_buffer(group, get_hidden_bytes(hidden_states))

    # 3. 获取 dispatch layout
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert_dispatch,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(
        topk_idx=selected_experts_indices,
        num_experts=num_experts
    )

    # 4. 调用 dispatch op
    (
        hidden_states,
        dispatched_indices,
        dispatched_expert_scores,
        tokens_per_expert,
        cache_id,
    ) = torch.ops.deepep.dispatch(...)

    # 5. 将 indices 转换为 multihot routing map
    dispatched_routing_map, dispatched_expert_scores_multihot = _indices_to_multihot(
        dispatched_indices, dispatched_expert_scores, num_local_experts
    )

    # 6. 按 expert 排序 tokens（为 grouped_mm 准备）
    hidden_states, permuted_scores, sorted_indices = _permute_tokens(
        hidden_states, dispatched_routing_map, scores=dispatched_expert_scores_multihot
    )

    # 7. 应用 routing scores（可选）
    if score_before_experts and permuted_scores is not None:
        hidden_states = hidden_states * permuted_scores.to(hidden_states.dtype).reshape(-1, 1)
        permuted_scores_for_state = None
    else:
        permuted_scores_for_state = permuted_scores

    # 8. 创建状态
    state = DispatchState(
        cache_id=cache_id,
        sorted_indices=sorted_indices,
        num_recv_tokens=num_recv_tokens,
        permuted_scores=permuted_scores_for_state,
    )

    return hidden_states, tokens_per_expert, state

def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
) -> torch.Tensor:
    """通过 DeepEP 聚合 expert 输出"""
    # 1. 应用 routing scores（如果尚未应用）
    if state.permuted_scores is not None:
        hidden_states = hidden_states * state.permuted_scores.to(hidden_states.dtype).reshape(-1, 1)

    # 2. 反向排序（unpermute）
    hidden_states = _unpermute_tokens(
        hidden_states, state.sorted_indices, state.num_recv_tokens
    )

    # 3. 调用 combine op
    hidden_states = torch.ops.deepep.combine(hidden_states, state.cache_id)

    return hidden_states
```

**数据流**:
```
input tokens [N, D]
    ↓ dispatch_tokens()
    ├─ buffer.get_dispatch_layout()      # 计算通信布局
    ├─ torch.ops.deepep.dispatch()        # All-to-all 发送
    ├─ _indices_to_multihot()             # 转换 routing 格式
    ├─ _permute_tokens()                  # 按 expert 排序
    └─ apply scores (optional)            # 应用 routing weights
    ↓
expert computation [M, D]  (M 可能 != N)
    ↓ combine_tokens()
    ├─ apply scores (if not before)       # 应用 routing weights
    ├─ _unpermute_tokens()                # 反向排序
    └─ torch.ops.deepep.combine()         # All-to-all 接收
    ↓
output tokens [N, D]
```

---

### 3. Expert Parallel 层 (`distributed/expert_parallel.py`)

#### 3.1 DeepEPExpertParallel 类

```python
class DeepEPExpertParallel(BaseExpertParallel):
    """使用 DeepEP 的 Expert Parallel 实现

    期望输入格式:
        (hidden_states, num_tokens_per_expert, selected_experts_indices,
         top_scores, num_experts)

    Args:
        score_before_experts: 是否在 expert 计算前应用 routing scores
    """

    def __init__(self, score_before_experts: bool = True):
        super().__init__()
        self._state = None  # 保存 dispatch 和 combine 之间的状态
        self.score_before_experts = score_before_experts

    def _token_dispatch(self, mod, inputs, device_mesh):
        """通过 DeepEP dispatch tokens"""
        from torchtitan.distributed.deepep import dispatch_tokens

        hidden_states, _, selected_experts_indices, top_scores, num_experts = inputs

        # 获取本地 expert 数量
        if isinstance(mod.w1, DTensor):
            num_local_experts = mod.w1.to_local().shape[0]
        else:
            num_local_experts = mod.w1.shape[0]

        ep_group = device_mesh.get_group()

        # 调用 dispatch
        hidden_states, tokens_per_expert, self._state = dispatch_tokens(
            hidden_states,
            selected_experts_indices,
            top_scores,
            num_local_experts,
            num_experts,
            ep_group,
            score_before_experts=self.score_before_experts,
        )

        return hidden_states, tokens_per_expert

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        """在 expert 维度上 shard 权重"""
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])),
            )

    def _token_combine(self, mod, routed_output, device_mesh):
        """通过 DeepEP combine tokens"""
        from torchtitan.distributed.deepep import combine_tokens

        routed_output = combine_tokens(routed_output, self._state)
        self._state = None  # 清理状态
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """应用 DeepEP 并行化"""
        return distribute_module(
            module,
            device_mesh,
            partition_fn=DeepEPExpertParallel._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
```

**与标准 ExpertParallel 的对比**:

| 特性 | ExpertParallel (标准) | DeepEPExpertParallel |
|------|----------------------|----------------------|
| 通信方式 | PyTorch all-to-all | DeepEP custom kernels |
| Dispatch | `all_to_all_single_autograd()` | `dispatch_tokens()` |
| Combine | `all_to_all_single_autograd()` | `combine_tokens()` |
| Token 重排 | `_permute()` / `_unpermute()` | `_permute_tokens()` / `_unpermute_tokens()` |
| Handle 管理 | 无需 | 需要缓存 dispatch handle |
| 异步支持 | 标准 autograd | 自定义 event management |
| 性能 | 基准 | **+67% MFU** |

---

### 4. MoE 模型层 (`models/moe/moe_deepep.py`)

#### 4.1 DeepEPMoE 类

```python
class DeepEPMoE(MoE):
    """使用 DeepEP 通信的 MoE

    继承自 MoE，但重写 forward() 以传递 routing 信息给 experts，
    让 DeepEPExpertParallel hooks 处理 dispatch/combine。
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args, dim, hidden_dim)
        # DeepEP 不使用 reorderer - routing 由 DeepEPExpertParallel 处理
        self.reorderer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 DeepEP 通信的 forward pass。

        DeepEPExpertParallel hooks 拦截 experts() 调用，
        并通过 deepep 函数处理 dispatch/combine。
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # Router 计算
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x, self.expert_bias
        )

        # Load balance tracking
        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # 调用 experts，传递 routing 信息
        # hooks 会处理 DeepEP dispatch/combine
        routed_output = self.experts(
            x,                          # hidden_states
            num_tokens_per_expert,      # 不使用（由 DeepEP 重新计算）
            selected_experts_indices,   # expert indices
            top_scores,                 # routing scores
            self.experts.num_experts,   # total experts
        )

        # Shared experts（可选）
        out = self.shared_experts(x) if self.shared_experts is not None else None

        if out is None:
            return routed_output.reshape(bs, slen, dim)
        return (out + routed_output).reshape(bs, slen, dim)
```

**关键差异**:

| MoE (标准) | DeepEPMoE |
|-----------|-----------|
| 使用 `self.reorderer` 进行 token 重排 | `reorderer = None` |
| `experts()` 接收 2 个参数 | `experts()` 接收 5 个参数 |
| EP 在 `reorderer` 中处理 | EP 在 `DeepEPExpertParallel` hooks 中处理 |

**为什么传递更多参数**:
- `selected_experts_indices`: DeepEP 需要知道每个 token 去哪个 expert
- `top_scores`: Routing scores，用于加权 expert 输出
- `num_experts`: 全局 expert 数量，用于计算通信布局

---

### 5. 模型集成层 (`models/deepseek_v3/`)

#### 5.1 Model Args (`model/args.py`)

```python
@dataclass
class DeepSeekV3ModelArgs(BaseModelArgs):
    # ... 其他参数 ...

    # Expert parallel communication backend (从 config 设置)
    expert_parallel_comm_backend: str = "standard"  # "standard" or "deepep"

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        # ... 其他更新 ...

        # 从 config 读取 DeepEP 设置
        self.moe_impl = job_config.parallelism.expert_parallel_comm_backend
```

#### 5.2 Model (`model/model.py`)

```python
from torchtitan.models.moe import build_moe

class DeepSeekV3Transformer(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        # ... 初始化 ...

        # 根据配置选择 MoE 实现
        if layer_args.moe_impl == "deepep":
            from torchtitan.models.moe import DeepEPMoE
            moe = DeepEPMoE(...)
        else:
            from torchtitan.models.moe import MoE
            moe = MoE(...)
```

**动态选择**:
- 根据 `moe_impl` 配置选择 `MoE` 或 `DeepEPMoE`
- 保持接口统一，外部代码无需修改

#### 5.3 Parallelize (`infra/parallelize.py`)

```python
def parallelize_deepseekv3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    # ... TP setup ...

    # 检查是否使用 DeepEP
    if job_config.parallelism.expert_parallel_comm_backend == "deepep":
        # 验证配置
        if not parallel_dims.ep_enabled:
            raise ValueError(
                "DeepEP requires expert parallelism (ep_degree > 1). "
                "The DeepEP MoE model code does not support EP=1. "
                "Please set expert_parallel_degree > 1 or use standard communication backend."
            )
        if parallel_dims.etp_enabled:
            raise NotImplementedError(
                "DeepEP with Expert Tensor Parallelism (ETP) is not supported yet. "
                "Please set expert_tensor_parallel_degree=1 or use standard communication backend."
            )

        use_deepep = True

        # 导入 deepep 模块以注册自定义 ops
        import torchtitan.distributed.deepep  # noqa: F401 - registers torch.ops.deepep

        # 将 DeepEP ops 添加到 SAC save list
        _op_sac_save_list.add(torch.ops.deepep.dispatch.default)
        _op_sac_save_list.add(torch.ops.deepep.combine.default)
    else:
        use_deepep = False

    # 应用 MoE EP/TP
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        dual_pipe_v = get_dual_pipe_v_flag(job_config, parallel_dims)

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            dual_pipe_v=dual_pipe_v,
            use_deepep=use_deepep,  # ← 传递给 apply_moe_ep_tp
        )

    # ... AC, Compile, FSDP ...
```

**关键步骤**:
1. **配置检查**: 验证 DeepEP 与其他并行策略的兼容性
2. **Op 注册**: 导入 `torchtitan.distributed.deepep` 注册自定义 ops
3. **SAC 集成**: 将 `torch.ops.deepep.dispatch/combine` 添加到 save list
4. **传递标志**: 通过 `use_deepep=True` 通知 `apply_moe_ep_tp`

---

## 集成流程

### 完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 配置阶段 (Job Config)                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
    parallelism.expert_parallel_comm_backend = "deepep"
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 模型初始化阶段 (Model __init__)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
    args.moe_impl = "deepep"  ← 从 config 读取
                            ↓
    if moe_impl == "deepep":
        layer.moe = DeepEPMoE(...)      # 使用 DeepEPMoE
    else:
        layer.moe = MoE(...)            # 使用标准 MoE
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 并行化阶段 (parallelize_deepseekv3)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
    if job_config.parallelism.expert_parallel_comm_backend == "deepep":
        # 验证配置
        assert ep_degree > 1
        assert etp_degree == 1

        # 注册 DeepEP ops
        import torchtitan.distributed.deepep

        # SAC 集成
        _op_sac_save_list.add(torch.ops.deepep.dispatch.default)
        _op_sac_save_list.add(torch.ops.deepep.combine.default)

        use_deepep = True
                            ↓
    apply_moe_ep_tp(..., use_deepep=True)
        ↓
        for each MoE layer:
            if use_deepep:
                ep_parallel = DeepEPExpertParallel(score_before_experts=True)
            else:
                ep_parallel = ExpertParallel()

            ep_parallel._apply(layer.experts, ep_mesh)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 前向传播阶段 (Forward Pass)                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
    DeepEPMoE.forward(x):
        # Router
        top_scores, selected_experts_indices, num_tokens_per_expert = router(x)

        # Expert computation with DeepEP
        routed_output = experts(
            x,                          # hidden_states
            num_tokens_per_expert,      # (不使用，由 DeepEP 重新计算)
            selected_experts_indices,   # routing indices
            top_scores,                 # routing scores
            num_experts,                # total experts
        )
                            ↓
        ↓ (DeepEPExpertParallel hook 拦截)
                            ↓
        DeepEPExpertParallel._token_dispatch():
            # 调用 DeepEP dispatch
            hidden_states, tokens_per_expert, state = dispatch_tokens(
                hidden_states=x,
                selected_experts_indices=selected_experts_indices,
                top_scores=top_scores,
                num_local_experts=...,
                num_experts=num_experts,
                ep_group=ep_mesh.get_group(),
                score_before_experts=True,
            )
                            ↓
            dispatch_tokens 内部:
                # 1. 获取 buffer
                buffer = get_buffer(group, hidden_bytes)

                # 2. 计算通信布局
                layout = buffer.get_dispatch_layout(topk_idx, num_experts)

                # 3. Dispatch (All-to-All 发送)
                recv_x, recv_indices, recv_scores, tokens_per_expert, cache_id =
                    torch.ops.deepep.dispatch(...)

                # 4. 转换为 multihot routing map
                routing_map, scores_multihot = _indices_to_multihot(...)

                # 5. 按 expert 排序 tokens
                sorted_x, sorted_scores, sorted_indices = _permute_tokens(...)

                # 6. 应用 routing scores (optional)
                if score_before_experts:
                    sorted_x = sorted_x * sorted_scores

                # 7. 返回状态
                state = DispatchState(cache_id, sorted_indices, ...)
                return sorted_x, tokens_per_expert, state
                            ↓
        # Expert 权重计算 (on sorted tokens)
        expert_output = GroupedExperts(sorted_x, tokens_per_expert)
                            ↓
        DeepEPExpertParallel._token_combine():
            # 调用 DeepEP combine
            output = combine_tokens(expert_output, state)
                            ↓
            combine_tokens 内部:
                # 1. 应用 routing scores (if not before)
                if state.permuted_scores is not None:
                    x = x * state.permuted_scores

                # 2. 反向排序
                x = _unpermute_tokens(x, state.sorted_indices, ...)

                # 3. Combine (All-to-All 接收)
                x = torch.ops.deepep.combine(x, state.cache_id)

                return x
                            ↓
        return routed_output
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 反向传播阶段 (Backward Pass)                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
    torch.ops.deepep.combine.backward:
        # 在梯度上执行 dispatch
        grad_x = _buffer.dispatch(grad_output, handle=saved_handle, ...)
                            ↓
    torch.ops.deepep.dispatch.backward:
        # 在梯度上执行 combine
        grad_x = _buffer.combine(grad_output, handle=cached_handle, ...)

        # 清理 handle 缓存
        _handle_cache.pop(cache_id)
```

---

## 性能优化原理

### 为什么 DeepEP 更快

#### 1. 优化的通信内核

**标准 all-to-all**:
```python
# PyTorch 标准实现
output = all_to_all_single_autograd(
    input,
    output_split_sizes,
    input_split_sizes,
    group,
)
```

**DeepEP**:
```python
# 自定义 CUDA 内核
recv_x, ..., handle = buffer.dispatch(
    x=input,
    topk_idx=routing_indices,
    topk_weights=routing_scores,
    ...
)
```

**优化点**:
- 针对 MoE 的 routing 模式特化
- 减少不必要的内存拷贝
- 优化的 buffer 管理

#### 2. Buffer 预分配

**标准方式**:
- 每次通信时动态分配 buffer
- 开销：`malloc() + memcpy() + free()`

**DeepEP**:
```python
_buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
# Buffer 在整个训练过程中重用
```

**好处**:
- 避免重复分配/释放
- 更好的内存局部性
- 减少碎片化

#### 3. 异步执行与流重叠

```python
# DeepEP 异步执行
previous_event = EventOverlap(EventHandle())
recv_x, ..., after_event = buffer.dispatch(
    ...,
    previous_event=previous_event,
    async_finish=True,
    allocate_on_comm_stream=True,  # 在通信流上分配
)
after_event.current_stream_wait()  # 同步回计算流
```

**效果**:
- 通信在专用流上执行
- 可以与其他计算重叠
- 减少流同步开销

#### 4. 融合操作

**标准方式**:
```python
# 分离的操作
tokens = all_to_all(...)          # 通信
tokens = permute(tokens, ...)     # 重排
tokens = tokens * scores          # 应用 scores
```

**DeepEP**:
```python
# 在 dispatch 内部融合
recv_x, recv_scores, ... = dispatch(...)
# 已经完成了通信、重排和部分 routing
```

**好处**:
- 减少 kernel 启动次数
- 更好的内存访问模式
- 降低中间结果的内存占用

#### 5. 针对 GPU 架构优化

- **NVLink**: 优化的 NVLink 通信模式
- **RDMA**: 支持 RDMA 通信（跨节点）
- **SM90+**: 针对 H100 架构优化

### 性能数据解析

**Before (标准 EP)**:
```
TPS: 346      # 每秒处理 346 个 tokens
MFU: 9.83%    # 仅利用了 GPU 理论算力的 9.83%
Memory: 60.18 GiB
```

**After (DeepEP)**:
```
TPS: 579      # +67%
MFU: 16.46%   # +67%
Memory: 56.75 GiB  # -5.7%
```

**提升原因**:
1. **通信时间减少**: 优化的 all-to-all 实现
2. **通信与计算重叠**: 异步执行提高 GPU 利用率
3. **内存效率**: Buffer 重用减少分配开销

**为什么 MFU 仍然只有 16.46%**:
- MoE 模型固有的低 MFU 特性（稀疏激活）
- 通信开销仍然存在（虽然已优化）
- 还有其他瓶颈（如 router 计算、内存带宽）

---

## 使用指南

### 1. 安装 DeepEP

```bash
# Clone DeepEP repository
git clone https://github.com/deepseek-ai/deepep
cd deepep

# Install
pip install -e .
```

**注意**: 检查 DeepEP 的具体安装要求和依赖

### 2. 配置训练

#### TOML 配置文件

```toml
# train_configs/deepseek_v3_671b_deepep.toml

[parallelism]
# 启用 DeepEP
expert_parallel_comm_backend = "deepep"

# Expert Parallelism 必须 > 1
expert_parallel_degree = 32

# Expert Tensor Parallelism 必须 = 1 (DeepEP 不支持 ETP)
expert_tensor_parallel_degree = 1

# 其他并行度
data_parallel_shard_degree = 64
tensor_parallel_degree = 1
pipeline_parallel_degree = 8
```

#### 命令行参数

```bash
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b.toml" \
./run_train.sh \
--parallelism.expert_parallel_comm_backend=deepep \
--parallelism.expert_parallel_degree=32 \
--parallelism.expert_tensor_parallel_degree=1
```

### 3. 验证配置

训练启动时，检查日志:

```
[INFO] Using DeepEP for Expert Parallel communication
[INFO] DeepEP ops registered: torch.ops.deepep.dispatch, torch.ops.deepep.combine
[INFO] DeepEP Buffer created: NVL=..., RDMA=...
```

### 4. 监控性能

关注以下指标:
- **TPS (Tokens Per Second)**: 吞吐量
- **MFU (Model FLOPS Utilization)**: GPU 利用率
- **Memory**: GPU 内存使用

```bash
# TensorBoard
tensorboard --logdir=/tmp/torchtitan/outputs

# 或 WandB
wandb login
# metrics 会自动上传
```

---

## 限制与注意事项

### 当前限制

#### 1. Expert Parallelism 必需

```python
if not parallel_dims.ep_enabled:
    raise ValueError(
        "DeepEP requires expert parallelism (ep_degree > 1). "
        "The DeepEP MoE model code does not support EP=1."
    )
```

**原因**:
- DeepEPMoE 的 forward 签名与标准 MoE 不同
- 需要传递额外的 routing 信息
- EP=1 时不需要通信，使用标准 MoE 更简单

**解决方案**: 设置 `expert_parallel_degree > 1`

#### 2. 不支持 Expert Tensor Parallelism

```python
if parallel_dims.etp_enabled:
    raise NotImplementedError(
        "DeepEP with Expert Tensor Parallelism (ETP) is not supported yet."
    )
```

**原因**:
- DeepEP dispatch/combine 还未适配 ETP 的 DTensor 输入
- ETP 需要额外的 TP 维度处理

**解决方案**: 设置 `expert_tensor_parallel_degree = 1`

#### 3. 依赖外部库

```python
try:
    from deep_ep import Buffer
except ImportError:
    raise ImportError("DeepEP is required. Install from: ...")
```

**影响**:
- 需要单独安装 DeepEP
- 增加了部署复杂度

**解决方案**:
- 在文档中明确依赖
- 提供安装脚本

### 兼容性矩阵

| 功能 | DeepEP 支持 | 备注 |
|------|------------|------|
| torch.compile | ✅ 支持 | 通过自定义 Op 集成 |
| SAC (Selective AC) | ✅ 支持 | DeepEP ops 在 save list 中 |
| Full AC | ✅ 支持 | - |
| FSDP/HSDP | ✅ 支持 | - |
| Tensor Parallel (TP) | ✅ 支持 | 非 MoE 层 |
| Pipeline Parallel (PP) | ✅ 支持 | - |
| Context Parallel (CP) | ✅ 支持 | - |
| Expert Parallel (EP) | ✅ **必需** | ep_degree > 1 |
| Expert TP (ETP) | ❌ 不支持 | 未来可能支持 |
| Float8 | ✅ 支持 | - |
| MXFP8 | ✅ 支持 | Blackwell GPUs |
| DDP | ✅ 支持 | - |

### 最佳实践

#### 1. 选择合适的 EP 度

```toml
# 对于 DeepSeek-V3 671B:
expert_parallel_degree = 32  # 512 GPUs / 16 experts per GPU

# 一般规则:
# ep_degree * num_local_experts = num_total_experts
```

#### 2. Buffer 大小

DeepEP 会自动计算 buffer 大小，但可以通过 hidden_bytes 影响:

```python
hidden_bytes = hidden_dim * max(dtype_size, 2)
# 更大的 hidden_dim → 更大的 buffer
```

#### 3. 监控通信开销

```python
# 使用 profiler
with torch.profiler.profile(...) as prof:
    model(input)

prof.export_chrome_trace("trace.json")
# 在 Chrome trace 中查看 deepep::dispatch 和 deepep::combine 时间
```

#### 4. 调试模式

```toml
[debug]
moe_force_load_balance = true  # 强制均衡 load，便于调试
```

### 故障排查

#### 问题 1: ImportError: No module named 'deep_ep'

**原因**: DeepEP 未安装

**解决**:
```bash
pip install git+https://github.com/deepseek-ai/deepep
```

#### 问题 2: ValueError: DeepEP requires ep_degree > 1

**原因**: 配置了 DeepEP 但 EP=1

**解决**:
```toml
[parallelism]
expert_parallel_degree = 8  # 或其他 > 1 的值
```

#### 问题 3: NotImplementedError: DeepEP with ETP is not supported

**原因**: 同时启用了 DeepEP 和 ETP

**解决**:
```toml
[parallelism]
expert_tensor_parallel_degree = 1  # 禁用 ETP
```

#### 问题 4: CUDA out of memory

**原因**: DeepEP buffer 分配过大

**解决**:
- 减小 batch size
- 减小 sequence length
- 减小 expert 数量

#### 问题 5: Loss 不收敛

**原因**: 可能的数值问题

**解决**:
1. 验证 loss 曲线与标准 EP 对比
2. 检查 router scores 是否正常
3. 尝试调整 learning rate

---

## 总结

### 集成特点

1. **模块化设计**: DeepEP 作为可选后端，不影响现有代码
2. **配置驱动**: 通过配置开关控制，易于启用/禁用
3. **分层集成**: 从配置→模型→并行化→通信，层次清晰
4. **兼容性强**: 与 torch.compile、SAC、各种并行策略兼容

### 性能收益

- **吞吐量**: +67% TPS
- **效率**: +67% MFU
- **内存**: -5.7% GPU 内存

### 适用场景

**适合**:
- 超大规模 MoE 模型训练
- 需要高 GPU 利用率的场景
- 多节点训练（RDMA 支持）

**不适合**:
- 小规模 MoE 模型（EP=1 不支持）
- 需要 ETP 的场景
- 不支持 DeepEP 的硬件

### 未来方向

1. **支持 ETP**: 集成 Expert Tensor Parallelism
2. **自动调优**: 自动选择最优 buffer 大小
3. **更多模型**: 扩展到 Llama4 和其他 MoE 模型
4. **性能优化**: 进一步降低通信开销

---

**参考资源**:
- DeepEP GitHub: https://github.com/deepseek-ai/deepep
- TorchTitan Docs: https://github.com/pytorch/torchtitan/tree/main/docs
- Commit: https://github.com/pytorch/torchtitan/commit/36a4b69

