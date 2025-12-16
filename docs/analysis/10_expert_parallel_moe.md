# TorchTitan Expert Parallel (MoE) 实现详解

## 目录
1. [什么是 MoE 和 Expert Parallel](#1-什么是-moe-和-expert-parallel)
2. [搬桌子比喻：专业工人分工](#2-搬桌子比喻专业工人分工)
3. [MoE 的核心组件](#3-moe-的核心组件)
4. [Expert Parallel 的实现](#4-expert-parallel-的实现)
5. [Expert Tensor Parallel (2D)](#5-expert-tensor-parallel-2d)
6. [Load Balancing 负载均衡](#6-load-balancing-负载均衡)
7. [Grouped GEMM 优化](#7-grouped-gemm-优化)
8. [源码实现详解](#8-源码实现详解)
9. [配置和使用](#9-配置和使用)
10. [性能数据](#10-性能数据)
11. [最佳实践](#11-最佳实践)
12. [总结](#12-总结)
13. [参考资料](#13-参考资料)

---

## 1. 什么是 MoE 和 Expert Parallel

### 1.1 MoE (Mixture of Experts) 的核心思想

**MoE** 是一种**稀疏激活**（Sparse Activation）的模型架构，核心思想是：

```
传统 Dense 模型：
  所有 tokens → 通过全部参数 → 输出

  问题：参数量大时，计算量线性增长

MoE 模型：
  所有 tokens → Router 选择专家 → 只通过部分专家 → 输出

  优势：参数量可以很大，但每个 token 只激活少量参数（稀疏）
```

**关键概念**：

1. **Router（路由器）**：决定每个 token 去哪些 experts
2. **Experts（专家）**：多个独立的 FFN 层
3. **Top-K Routing**：每个 token 选择 top-K 个 experts
4. **Sparse Activation**：每个 token 只激活 K/N 的参数（K << N）

**示例**：Llama4 17Bx16E
- **17B**: Dense 参数（attention 等）= 17B
- **16E**: 16 个 experts，每个约 3.5B
- **Top-2**: 每个 token 选择 2 个 experts
- **总参数**: 17B + 16 × 3.5B ≈ 73B
- **激活参数**: 17B + 2 × 3.5B ≈ 24B（约 33% 激活）

### 1.2 Expert Parallel (EP) 的必要性

**问题**：16 个 experts 的参数太大，单 GPU 放不下！

```
Llama4 17Bx16E 参数分布：
  - Dense 参数（attention 等）: 17B
  - 16 experts × 3.5B each = 56B

  → 总共 73B 参数，单 GPU 装不下！
```

**解决方案**：Expert Parallel (EP) —— 第 5 个并行维度

```
Expert Parallel:
  将 16 个 experts 分片到多个 GPU

  EP = 4:
    GPU 0: Expert 0,  1,  2,  3   (4 experts)
    GPU 1: Expert 4,  5,  6,  7   (4 experts)
    GPU 2: Expert 8,  9,  10, 11  (4 experts)
    GPU 3: Expert 12, 13, 14, 15  (4 experts)
```

**EP vs 其他并行**：

| 并行方式 | 切分维度 | 适用场景 |
|---------|---------|---------|
| DP (FSDP) | 参数切分，所有 GPU 有相同结构 | Dense 模型 |
| TP | 单层权重切分 | 单层太大 |
| PP | 层切分 | 模型层数多 |
| CP | 序列切分 | 序列太长 |
| **EP** | **专家切分** | **MoE 模型，experts 太多** |

---

## 2. 搬桌子比喻：专业工人分工

延续我们的"搬桌子"比喻系列，MoE + Expert Parallel 就像**专业工人分工**。

### 2.1 Dense 模型：所有工人处理所有桌子

```
┌─────────────────────────────────────────────────────────────┐
│          传统 Dense FFN（所有工人处理所有桌子）               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  桌子 1 → 所有工人处理 → 输出                                │
│  桌子 2 → 所有工人处理 → 输出                                │
│  桌子 3 → 所有工人处理 → 输出                                │
│  ...                                                        │
│                                                             │
│  问题：                                                     │
│   - 工人数量多时，每张桌子都要经过所有工人（慢）              │
│   - 所有工人都要上班（参数激活率 100%）                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 MoE 模型：专业工人，按需分配

```
┌─────────────────────────────────────────────────────────────┐
│      MoE 模型（专业工人，每张桌子只找对应的专家）              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  16 个专业工人（Experts）:                                   │
│    Expert 0: 擅长搬运红木桌                                  │
│    Expert 1: 擅长搬运玻璃桌                                  │
│    Expert 2: 擅长搬运大理石桌                                │
│    ...                                                      │
│    Expert 15: 擅长搬运折叠桌                                 │
│                                                             │
│  Router（调度员）：判断桌子类型，分配给 Top-2 专家           │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  搬运流程：                                                  │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  桌子 1（红木桌）:                                           │
│    Router: "这是红木桌，找 Expert 0 和 Expert 3"             │
│    → 只有 Expert 0 和 Expert 3 处理                         │
│    → 其他 14 个专家休息（稀疏激活）                          │
│                                                             │
│  桌子 2（玻璃桌）:                                           │
│    Router: "这是玻璃桌，找 Expert 1 和 Expert 5"             │
│    → 只有 Expert 1 和 Expert 5 处理                         │
│                                                             │
│  优势：                                                     │
│   ✅ 每张桌子只需 2 个专家（2/16 = 12.5% 激活）              │
│   ✅ 总共可以有很多专家（参数量大）                          │
│   ✅ 实际激活的参数少（计算量小）                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Expert Parallel：工人分散到不同工地

```
┌─────────────────────────────────────────────────────────────┐
│       Expert Parallel（工人分散到 4 个工地）                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  工地 0 (GPU 0):                                            │
│    Expert 0, 1, 2, 3                                       │
│                                                             │
│  工地 1 (GPU 1):                                            │
│    Expert 4, 5, 6, 7                                       │
│                                                             │
│  工地 2 (GPU 2):                                            │
│    Expert 8, 9, 10, 11                                     │
│                                                             │
│  工地 3 (GPU 3):                                            │
│    Expert 12, 13, 14, 15                                   │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  搬运流程（Token Dispatch + Combine）:                      │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  1. Router 决定：                                           │
│     桌子 1 → Expert 0 (工地 0) + Expert 5 (工地 1)          │
│     桌子 2 → Expert 1 (工地 0) + Expert 8 (工地 2)          │
│     桌子 3 → Expert 3 (工地 0) + Expert 12 (工地 3)         │
│     桌子 4 → Expert 7 (工地 1) + Expert 10 (工地 2)         │
│                                                             │
│  2. Token Dispatch（All-to-All 分发桌子）:                 │
│     按照 expert 分配，把桌子送到对应工地                     │
│                                                             │
│     工地 0 收到: 桌子1, 桌子2, 桌子3（要给 Expert 0,1,3）   │
│     工地 1 收到: 桌子1, 桌子4（要给 Expert 5,7）            │
│     工地 2 收到: 桌子2, 桌子4（要给 Expert 8,10）           │
│     工地 3 收到: 桌子3（要给 Expert 12）                    │
│                                                             │
│  3. Local Expert Processing（本地专家处理）:               │
│     每个工地的专家处理分配给他们的桌子                       │
│                                                             │
│  4. Token Combine（All-to-All 收集结果）:                  │
│     把处理好的桌子送回原来的位置                            │
│                                                             │
│     桌子 1 结果 = Expert 0 结果 + Expert 5 结果             │
│     桌子 2 结果 = Expert 1 结果 + Expert 8 结果             │
│     ...                                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键点**：
- **Router**: 调度员（决定每张桌子找哪些专家）
- **Token Dispatch**: 桌子分发（All-to-All 通信）
- **Local Experts**: 本地专家处理
- **Token Combine**: 结果收集（All-to-All 通信）

---

## 3. MoE 的核心组件

### 3.1 整体架构

一个 MoE 层包含以下组件：

```python
class MoE(nn.Module):
    def __init__(self, moe_args, dim, hidden_dim):
        # 1. Router: 路由器（调度员）
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=16,      # 16 个专家
            top_k=2,             # 每个 token 选 2 个专家
        )

        # 2. Experts: 专家组
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=16,
            use_grouped_mm=True  # 使用 grouped GEMM 加速
        )

        # 3. Reorderer: Token 重排序
        self.reorderer = TokenReorderer(
            num_experts=16,
            top_k=2
        )

        # 4. Shared Experts: 共享专家（可选）
        self.shared_experts = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim * num_shared_experts
        )

        # 5. Load Balancing: 负载均衡
        self.tokens_per_expert = torch.zeros(16)  # 追踪每个专家处理的 token 数
        self.expert_bias = torch.zeros(16)        # 用于负载均衡的 bias
```

### 3.2 Forward 流程

```python
def forward(self, x):
    # x shape: (batch_size, seq_len, dim)
    bs, slen, dim = x.shape
    x = x.view(-1, dim)  # shape: (bs*slen, dim)

    # ─────────────────────────────────────────────────────────
    # Step 1: Router 选择专家
    # ─────────────────────────────────────────────────────────
    # top_scores: (bs*slen, top_k) - 每个 token 选中的 experts 的分数
    # selected_experts_indices: (bs*slen, top_k) - 选中的 expert 索引
    # num_tokens_per_expert: (num_experts,) - 每个 expert 分配到的 token 数
    top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x)

    # ─────────────────────────────────────────────────────────
    # Step 2: Token Reorder（重排序）
    # ─────────────────────────────────────────────────────────
    # 将 tokens 按照 expert 分组重排序
    (
        top_scores_experts_sorted,         # 重排序后的分数
        token_indices_experts_sorted,      # 重排序后的 token 索引
        num_tokens_per_expert,
    ) = self.reorderer(top_scores, selected_experts_indices)

    # ─────────────────────────────────────────────────────────
    # Step 3: Route tokens to experts
    # ─────────────────────────────────────────────────────────
    # 根据 token_indices_experts_sorted 提取 tokens
    routed_input = x[token_indices_experts_sorted // self.router.top_k]

    # 可选：在 experts 之前乘以分数
    if self.score_before_experts:
        routed_input = routed_input * top_scores_experts_sorted.reshape(-1, 1)

    # ─────────────────────────────────────────────────────────
    # Step 4: Expert Processing（关键：这里发生 EP 并行）
    # ─────────────────────────────────────────────────────────
    routed_output = self.experts(routed_input, num_tokens_per_expert)

    # ─────────────────────────────────────────────────────────
    # Step 5: Shared Experts（可选）
    # ─────────────────────────────────────────────────────────
    out_shared = self.shared_experts(x) if self.shared_experts else None

    # ─────────────────────────────────────────────────────────
    # Step 6: Unsort and Combine
    # ─────────────────────────────────────────────────────────
    # 将 routed_output 恢复到原来的 token 顺序
    routed_output_unsorted = torch.zeros(
        (bs * slen * self.router.top_k, dim),
        device=routed_output.device
    )
    routed_output_unsorted[token_indices_experts_sorted] = routed_output
    routed_output_unsorted = routed_output_unsorted.reshape(-1, self.router.top_k, dim)

    # 如果没有在 experts 之前乘分数，则在这里乘
    if not self.score_before_experts:
        out_experts = torch.bmm(
            top_scores.reshape(-1, 1, self.router.top_k),
            routed_output_unsorted
        ).squeeze(1)
    else:
        out_experts = routed_output_unsorted.sum(dim=1)

    # Combine shared experts + routed experts
    if out_shared is None:
        return out_experts.reshape(bs, slen, dim)
    return (out_shared + out_experts).reshape(bs, slen, dim)
```

### 3.3 Router（路由器）

**作用**：决定每个 token 应该去哪些 experts。

**实现**：

```python
class TokenChoiceTopKRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k, score_func="sigmoid"):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)  # 路由层
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func  # "sigmoid" or "softmax"

    def forward(self, x):
        # x: (bs*slen, dim)

        # 1. 计算路由分数
        scores = self.gate(x)  # (bs*slen, num_experts)

        # 2. 应用激活函数
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=1)

        # 3. 选择 Top-K experts
        top_scores, selected_experts_indices = torch.topk(scores, k=self.top_k, dim=1)
        # top_scores: (bs*slen, top_k)
        # selected_experts_indices: (bs*slen, top_k)

        # 4. 统计每个 expert 分配到的 token 数
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts
        )  # (num_experts,)

        return top_scores, selected_experts_indices, num_tokens_per_expert
```

**示例**：

```
假设 num_experts=4, top_k=2, batch=2, seq_len=4
输入：8 个 tokens

Router 输出：
  Token 0 → Expert 0 (score=0.8), Expert 2 (score=0.6)
  Token 1 → Expert 1 (score=0.9), Expert 3 (score=0.5)
  Token 2 → Expert 0 (score=0.7), Expert 1 (score=0.6)
  Token 3 → Expert 2 (score=0.9), Expert 3 (score=0.7)
  Token 4 → Expert 0 (score=0.6), Expert 1 (score=0.5)
  Token 5 → Expert 2 (score=0.8), Expert 0 (score=0.7)
  Token 6 → Expert 1 (score=0.9), Expert 2 (score=0.6)
  Token 7 → Expert 3 (score=0.8), Expert 0 (score=0.5)

num_tokens_per_expert:
  Expert 0: 5 tokens (from Token 0, 2, 4, 5, 7)
  Expert 1: 4 tokens (from Token 1, 2, 4, 6)
  Expert 2: 4 tokens (from Token 0, 3, 5, 6)
  Expert 3: 3 tokens (from Token 1, 3, 7)
```

### 3.4 GroupedExperts（分组专家）

**作用**：实际执行专家计算的模块。

**实现**：

```python
class GroupedExperts(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, use_grouped_mm=True):
        super().__init__()
        # 所有 experts 的权重存储为 3D tensor
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm

    def forward(self, x, num_tokens_per_expert):
        # x: routed tokens, shape (total_routed_tokens, dim)
        # num_tokens_per_expert: (num_experts,) or (num_local_experts,)

        if self.use_grouped_mm:
            # 使用 torch._grouped_mm（高效）
            offsets = torch.cumsum(num_tokens_per_expert, dim=0)
            h = F.silu(torch._grouped_mm(x, self.w1.transpose(-2, -1), offs=offsets))
            h = h * torch._grouped_mm(x, self.w3.transpose(-2, -1), offs=offsets)
            out = torch._grouped_mm(h, self.w2.transpose(-2, -1), offs=offsets)
        else:
            # For-loop 实现（慢，但可读性好）
            num_tokens_per_expert = num_tokens_per_expert.tolist()
            x_splits = torch.split(x, num_tokens_per_expert, dim=0)
            out_splits = []
            for expert_idx, x_expert in enumerate(x_splits):
                h = F.silu(torch.matmul(x_expert, self.w1[expert_idx].transpose(-2, -1)))
                h = h * torch.matmul(x_expert, self.w3[expert_idx].transpose(-2, -1))
                out = torch.matmul(h, self.w2[expert_idx].transpose(-2, -1))
                out_splits.append(out)
            out = torch.cat(out_splits, dim=0)

        return out
```

**Grouped GEMM 的优势**：

```
For-loop 方式：
  for expert_idx in range(num_experts):
      out[expert_idx] = matmul(x[expert_idx], w[expert_idx])

  → 每个 expert 单独调用一次 matmul kernel（慢）

Grouped GEMM 方式：
  out = torch._grouped_mm(x, w, offsets=offsets)

  → 一次调用处理所有 experts（快）
  → GPU 可以并行处理多个 experts
```

---

## 4. Expert Parallel 的实现

### 4.1 EP 的核心：All-to-All 通信

**问题**：每个 GPU 上的 tokens 需要去不同 GPU 上的 experts。

```
示例：EP=4, 16 experts → 每个 GPU 有 4 个 local experts

GPU 0 有: Expert 0, 1, 2, 3
GPU 1 有: Expert 4, 5, 6, 7
GPU 2 有: Expert 8, 9, 10, 11
GPU 3 有: Expert 12, 13, 14, 15

GPU 0 上的 tokens:
  Token 0 → Expert 0 (local),  Expert 5 (GPU 1)
  Token 1 → Expert 2 (local),  Expert 10 (GPU 2)
  Token 2 → Expert 7 (GPU 1),  Expert 12 (GPU 3)
  ...

→ 需要 All-to-All 通信，把 tokens 送到对应的 GPU！
```

**All-to-All 通信模式**：

```
┌─────────────────────────────────────────────────────────────┐
│               All-to-All Token Dispatch                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Before All-to-All (每个 GPU 有全部 tokens 的一部分):      │
│                                                             │
│    GPU 0: [Token 0, 1, 2, 3]  (需要去 Expert 0,2,5,7,10,12)│
│    GPU 1: [Token 4, 5, 6, 7]  (需要去 Expert 1,3,8,11,13,15)│
│    GPU 2: [Token 8, 9, 10, 11](需要去 Expert 0,4,6,9,12,14)│
│    GPU 3: [Token 12,13,14,15] (需要去 Expert 2,5,8,10,13,15)│
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  All-to-All Dispatch（按照 expert 分组重新分发）:          │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  After All-to-All (每个 GPU 收到去 local experts 的 tokens):│
│                                                             │
│    GPU 0: [tokens for Expert 0, 1, 2, 3]                   │
│    GPU 1: [tokens for Expert 4, 5, 6, 7]                   │
│    GPU 2: [tokens for Expert 8, 9, 10, 11]                 │
│    GPU 3: [tokens for Expert 12, 13, 14, 15]               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 ExpertParallel 类实现

文件：`torchtitan/distributed/expert_parallel.py:67-169`

```python
class ExpertParallel(ParallelStyle):
    def _token_dispatch(self, mod, inputs, device_mesh):
        """Token Dispatch: All-to-All 分发 tokens"""
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # ─────────────────────────────────────────────────────────
        # Step 1: 计算 All-to-All 的 split sizes
        # ─────────────────────────────────────────────────────────
        with torch.no_grad():
            # 先做一次 All-to-All 交换 num_tokens_per_expert
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )

            # 计算 input_splits 和 output_splits
            # input_splits: 从当前 GPU 发送到各个 GPU 的 token 数
            # output_splits: 从各个 GPU 接收到当前 GPU 的 token 数
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .tolist()
            )
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .tolist()
            )

        # ─────────────────────────────────────────────────────────
        # Step 2: All-to-All dispatch tokens
        # ─────────────────────────────────────────────────────────
        routed_input = all_to_all_single_autograd(
            routed_input,
            output_splits,
            input_splits,
            device_mesh.get_group(),
        )

        # ─────────────────────────────────────────────────────────
        # Step 3: Permute tokens（重新排列）
        # ─────────────────────────────────────────────────────────
        # 将 tokens 按照 local expert 顺序排列，并做 padding
        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(
            routed_input,
            num_tokens_per_expert_group,
            ep_degree,
            num_local_experts
        )

        return routed_input, num_tokens_per_expert_group

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        """切分 experts 的权重"""
        # 在 expert 维度（dim=0）切分
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            mod.register_parameter(name, dist_param)

    def _token_combine(self, mod, routed_output, device_mesh):
        """Token Combine: All-to-All 收集结果"""
        # Unpermute
        routed_output = _unpermute(
            routed_output,
            self.input_shape,
            self.permuted_indices
        )

        # All-to-All combine
        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,   # 注意：这里 input/output 交换了
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output
```

### 4.3 Permute 和 Unpermute

**为什么需要 Permute？**

All-to-All 后，tokens 的顺序是：

```
[tokens for Expert 0 from GPU 0,
 tokens for Expert 1 from GPU 0,
 tokens for Expert 0 from GPU 1,
 tokens for Expert 1 from GPU 1,
 ...]
```

但我们需要的顺序是：

```
[all tokens for Expert 0,
 all tokens for Expert 1,
 ...]
```

**Permute 的作用**：

```python
def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    # 1. 生成 permute indices（使用 Triton kernel）
    permuted_indices, num_tokens_per_expert, offsets = generate_permute_indices(
        num_tokens_per_expert,
        num_local_experts,
        ep_degree,
        padded_max_len,
        TOKEN_GROUP_ALIGN_SIZE_M,  # Padding 对齐（8/16/32）
    )

    # 2. 根据 indices 重新排列 tokens
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert
```

**Unpermute**：恢复原来的顺序

```python
def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    return out_unpermuted[:-1]  # 去掉 padding
```

---

## 5. Expert Tensor Parallel (2D)

### 5.1 为什么需要 Expert TP？

**问题**：即使用了 EP，单个 expert 的权重仍然可能太大！

```
Llama4 17Bx128E:
  - 128 experts，每个约 3.5B
  - EP=16 → 每个 GPU 有 8 个 experts
  - 8 × 3.5B = 28B 参数，单 GPU 仍然放不下！

解决方案：Expert Tensor Parallel (ETP)
  → 在 expert 维度切分（EP）
  → 同时在 expert 权重维度切分（TP）
```

### 5.2 2D 并行：EP × TP

```
┌─────────────────────────────────────────────────────────────┐
│          Expert Tensor Parallel (2D Mesh: EP × TP)          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  假设：16 experts, EP=4, TP=2                               │
│                                                             │
│         TP Rank 0              TP Rank 1                    │
│  EP  ┌─────────────────┬─────────────────┐                 │
│  R   │  GPU 0          │  GPU 1          │                 │
│  a   │  Expert 0,1,2,3 │  Expert 0,1,2,3 │                 │
│  n   │  (w1/w3的左半)  │  (w1/w3的右半)  │                 │
│  k   │  (w2的上半)     │  (w2的下半)     │                 │
│  0   └─────────────────┴─────────────────┘                 │
│                                                             │
│  EP  ┌─────────────────┬─────────────────┐                 │
│  R   │  GPU 2          │  GPU 3          │                 │
│  a   │  Expert 4,5,6,7 │  Expert 4,5,6,7 │                 │
│  n   │  (w1/w3的左半)  │  (w1/w3的右半)  │                 │
│  k   │  (w2的上半)     │  (w2的下半)     │                 │
│  1   └─────────────────┴─────────────────┘                 │
│                                                             │
│  ... (EP Rank 2 和 3)                                       │
│                                                             │
│  权重切分方式：                                             │
│    w1: [experts, hidden_dim, dim]                          │
│       → Shard on dim=0 (experts) for EP                    │
│       → Shard on dim=1 (hidden_dim) for TP (Colwise)       │
│                                                             │
│    w2: [experts, dim, hidden_dim]                          │
│       → Shard on dim=0 (experts) for EP                    │
│       → Shard on dim=2 (hidden_dim) for TP (Rowwise)       │
│                                                             │
│    w3: [experts, hidden_dim, dim]                          │
│       → Shard on dim=0 (experts) for EP                    │
│       → Shard on dim=1 (hidden_dim) for TP (Colwise)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 ExpertTensorParallel 实现

文件：`torchtitan/distributed/expert_parallel.py:172-219`

```python
class ExpertTensorParallel(ExpertParallel):
    def _partition_fn_2d(self, name, mod, ep_tp_mesh):
        """2D 切分：EP × TP"""
        # w1: (experts, hidden_dim, dim)
        # Shard(0): 在 expert 维度切分（EP）
        # Shard(1): 在 hidden_dim 维度切分（TP，Colwise）
        mod.register_parameter(
            "w1",
            nn.Parameter(
                distribute_tensor(mod.w1, ep_tp_mesh, [Shard(0), Shard(1)])
            ),
        )

        # w2: (experts, dim, hidden_dim)
        # Shard(0): 在 expert 维度切分（EP）
        # Shard(2): 在 hidden_dim 维度切分（TP，Rowwise）
        mod.register_parameter(
            "w2",
            nn.Parameter(
                distribute_tensor(mod.w2, ep_tp_mesh, [Shard(0), Shard(2)])
            ),
        )

        # w3: (experts, hidden_dim, dim)
        # Shard(0): 在 expert 维度切分（EP）
        # Shard(1): 在 hidden_dim 维度切分（TP，Colwise）
        mod.register_parameter(
            "w3",
            nn.Parameter(
                distribute_tensor(mod.w3, ep_tp_mesh, [Shard(0), Shard(1)])
            ),
        )
```

**通信模式**：

```
EP 通信（All-to-All）：
  - Token Dispatch: 把 tokens 分发到对应的 EP rank
  - Token Combine: 收集结果

TP 通信（All-Reduce）：
  - Colwise (w1, w3): 输出需要 All-Reduce
  - Rowwise (w2): 输入已经 Replicate，输出直接合并
```

---

## 6. Load Balancing 负载均衡

### 6.1 负载不均的问题

**问题**：Router 可能导致 experts 负载不均。

```
理想情况（16 experts, 1000 tokens, top_k=2）:
  每个 expert 应该处理 1000 × 2 / 16 = 125 tokens

实际情况（负载不均）:
  Expert 0: 200 tokens  ← 过载
  Expert 1: 180 tokens  ← 过载
  Expert 2: 120 tokens  ✓
  Expert 3: 50 tokens   ← 闲置
  Expert 4: 60 tokens   ← 闲置
  ...

问题：
  - 某些 experts 过载（计算瓶颈）
  - 某些 experts 闲置（浪费资源）
  - 训练不稳定
```

### 6.2 Auxiliary-Loss-Free Load Balancing

**传统方法**：添加 auxiliary loss 来惩罚负载不均

```python
load_balance_loss = coeff * torch.var(num_tokens_per_expert)
total_loss = task_loss + load_balance_loss
```

**TorchTitan 的方法**：Auxiliary-Loss-Free（[论文](https://arxiv.org/abs/2408.15664)）

**核心思想**：通过调整 expert bias 来平衡负载，无需 auxiliary loss。

**实现**：

```python
class MoE(nn.Module):
    def __init__(self, moe_args, ...):
        # 1. tokens_per_expert: 追踪每个 expert 处理的 token 数
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,  # 不保存到 checkpoint
        )

        # 2. expert_bias: 用于 routing 的 bias
        self.load_balance_coeff = moe_args.load_balance_coeff  # 例如 1e-3
        if self.load_balance_coeff is not None:
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,  # 保存到 checkpoint
            )

    def forward(self, x):
        # 3. Router 使用 expert_bias
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x,
            self.expert_bias  # ← 传入 bias
        )

        # 4. 累积 tokens_per_expert（在 forward 中）
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
```

**Router 中使用 expert_bias**：

```python
class TokenChoiceTopKRouter(nn.Module):
    def forward(self, x, expert_bias=None):
        scores = self.gate(x)  # (bs*slen, num_experts)
        scores = torch.sigmoid(scores)

        # 使用 bias 影响 routing（但不影响 gating scores）
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias,  # ← 加 bias
                k=self.top_k,
                dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(scores, k=self.top_k, dim=1)

        return top_scores, selected_experts_indices, num_tokens_per_expert
```

**更新 expert_bias**（在 optimizer step 之前）：

```python
# 在 optimizer pre-hook 中
def update_expert_bias(moe_module):
    # 计算每个 expert 的负载（使用 sign 函数）
    expert_usage = torch.sign(moe_module.tokens_per_expert)  # 0 或 1

    # 更新 bias（增加闲置 experts 的吸引力，降低过载 experts 的吸引力）
    moe_module.expert_bias.add_(
        moe_module.load_balance_coeff * (
            expert_usage.mean() - expert_usage
        )
    )

    # 重置 tokens_per_expert
    moe_module.tokens_per_expert.zero_()
```

**效果**：

```
Step 0:
  Expert 0: 200 tokens → expert_bias[0] 降低
  Expert 3: 50 tokens  → expert_bias[3] 提高
  Expert 4: 60 tokens  → expert_bias[4] 提高

Step 1:
  Router 倾向于选择 Expert 3 和 4（bias 更高）
  → 负载逐渐平衡

经过几个 steps:
  所有 experts 负载接近 125 tokens
```

---

## 7. Grouped GEMM 优化

### 7.1 为什么需要 Grouped GEMM？

**问题**：不同 expert 处理的 token 数不同，矩阵大小不同。

```
Expert 0: 处理 200 tokens → matmul(200 × dim, hidden_dim × dim)
Expert 1: 处理 180 tokens → matmul(180 × dim, hidden_dim × dim)
Expert 2: 处理 120 tokens → matmul(120 × dim, hidden_dim × dim)
...

For-loop 方式：
  for i in range(num_experts):
      out[i] = matmul(x[i], w[i])  ← 16 次 kernel 调用（慢）

Grouped GEMM 方式：
  out = torch._grouped_mm(x, w, offsets=offsets)  ← 1 次 kernel 调用（快）
```

### 7.2 torch._grouped_mm 的使用

```python
def _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert):
    # num_tokens_per_expert: (num_experts,) 例如 [200, 180, 120, ...]

    # 1. 计算 offsets（累积和）
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    # offsets: [200, 380, 500, ...]

    # 2. Grouped GEMM
    # x shape: (total_tokens, dim)
    # w1 shape: (num_experts, hidden_dim, dim)
    # offsets 指示每个 expert 的起始位置

    h = F.silu(
        torch._grouped_mm(x, w1.transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(x, w3.transpose(-2, -1), offs=offsets)
    out = torch._grouped_mm(h, w2.transpose(-2, -1), offs=offsets)

    return out
```

**工作原理**：

```
┌─────────────────────────────────────────────────────────────┐
│                torch._grouped_mm 工作原理                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input x: [token_0, token_1, ..., token_499]               │
│           (500 tokens total)                               │
│                                                             │
│  Weights w: [w_0, w_1, w_2, ..., w_15]                     │
│            (16 experts)                                    │
│                                                             │
│  Offsets: [200, 380, 500, ...]                            │
│           → Expert 0: tokens [0:200]                       │
│           → Expert 1: tokens [200:380]                     │
│           → Expert 2: tokens [380:500]                     │
│           → ...                                            │
│                                                             │
│  Grouped GEMM:                                             │
│    out[0:200]   = matmul(x[0:200],   w_0)                 │
│    out[200:380] = matmul(x[200:380], w_1)                 │
│    out[380:500] = matmul(x[380:500], w_2)                 │
│    ...                                                     │
│                                                             │
│  → GPU 可以并行执行这些 matmul（快）                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Padding 和对齐

**问题**：Grouped GEMM 需要每个 expert 的 token 数是某个值的倍数。

```
TOKEN_GROUP_ALIGN_SIZE_M:
  - bf16: 8 (16 byte / 2 byte per elem = 8)
  - fp8: 16 (16 byte / 1 byte per elem = 16)
  - mxfp8: 32 (scaling block size = 32)

示例：
  Expert 0: 203 tokens → padding到 208 tokens (203 → 208 = 26×8)
  Expert 1: 177 tokens → padding到 184 tokens (177 → 184 = 23×8)
```

**实现**：`_permute` 函数

```python
def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    # 使用 Triton kernel 生成 permuted_indices
    # 同时处理 padding（对齐到 TOKEN_GROUP_ALIGN_SIZE_M）
    permuted_indices, num_tokens_per_expert, offsets = generate_permute_indices(
        num_tokens_per_expert,
        num_local_experts,
        ep_degree,
        padded_max_len,
        TOKEN_GROUP_ALIGN_SIZE_M,  # 8/16/32
    )

    # 添加一个 zero token 用于 padding
    x = torch.vstack((x, x.new_zeros((1, x.shape[-1]))))

    # 根据 indices 重新排列（包括 padding）
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert
```

---

## 8. 源码实现详解

### 8.1 MoE Layer 的应用

文件：`torchtitan/models/llama4/model/llama4.py`（假设）

```python
class TransformerBlock(nn.Module):
    def __init__(self, model_args, layer_id):
        self.attention = Attention(...)
        self.attention_norm = RMSNorm(...)

        # MoE 或 Dense FFN
        if layer_id in moe_layer_ids:
            self.feed_forward = MoE(
                moe_args=model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.intermediate_size,
            )
            self.moe_enabled = True
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim,
                hidden_dim=model_args.intermediate_size,
            )
            self.moe_enabled = False

        self.ffn_norm = RMSNorm(...)

    def forward(self, x):
        # Attention
        h = x + self.attention(self.attention_norm(x))

        # FFN (MoE or Dense)
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
```

### 8.2 EP 的应用

文件：`torchtitan/models/llama4/infra/parallelize.py`（假设）

```python
def apply_expert_parallel(model, ep_mesh):
    """应用 Expert Parallel"""
    from torchtitan.distributed.expert_parallel import ExpertParallel

    ep_style = ExpertParallel()

    for layer in model.layers:
        if layer.moe_enabled:
            # 对 GroupedExperts 应用 EP
            layer.feed_forward.experts = ep_style._apply(
                layer.feed_forward.experts,
                ep_mesh
            )
```

### 8.3 ETP（2D）的应用

```python
def apply_expert_tensor_parallel(model, ep_tp_mesh):
    """应用 Expert Tensor Parallel (2D: EP × TP)"""
    from torchtitan.distributed.expert_parallel import ExpertTensorParallel

    etp_style = ExpertTensorParallel()

    for layer in model.layers:
        if layer.moe_enabled:
            # 对 GroupedExperts 应用 ETP
            layer.feed_forward.experts = etp_style._apply(
                layer.feed_forward.experts,
                ep_tp_mesh  # 2D mesh: [ep, tp]
            )
```

### 8.4 Load Balancing 的集成

文件：`torchtitan/components/optimizer.py`

```python
def build_optimizers_with_moe_load_balancing(model_parts, optimizer_config, ...):
    """构建 optimizer 并注册 load balancing hook"""

    # 1. 构建 optimizer
    optimizers = build_optimizers(model_parts, optimizer_config, ...)

    # 2. 为每个 MoE layer 注册 pre-step hook
    for model in model_parts:
        for module in model.modules():
            if isinstance(module, MoE) and module.load_balance_coeff is not None:
                # 注册 hook：在 optimizer step 之前更新 expert_bias
                register_expert_bias_update_hook(optimizers, module)

    return optimizers

def register_expert_bias_update_hook(optimizers, moe_module):
    def update_expert_bias_hook():
        # 计算 expert usage
        expert_usage = torch.sign(moe_module.tokens_per_expert)

        # 更新 bias
        moe_module.expert_bias.add_(
            moe_module.load_balance_coeff * (
                expert_usage.mean() - expert_usage
            )
        )

        # 重置计数器
        moe_module.tokens_per_expert.zero_()

    # 注册 hook
    optimizers.register_step_pre_hook(update_expert_bias_hook)
```

---

## 9. 配置和使用

### 9.1 基础配置

**Llama4 17Bx16E (64 H100s)**：

```toml
[model]
name = "llama4"
flavor = "17bx16e"  # 17B dense + 16 experts

[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 8
expert_parallel_degree = 1        # EP degree
expert_tensor_parallel_degree = 8 # ETP degree (使用 TP ranks 作为 EP)

[activation_checkpoint]
mode = "full"

[compile]
enable = false  # MoE + compile 需要特殊处理
```

**说明**：
- `expert_parallel_degree = 1` + `expert_tensor_parallel_degree = 8`
  - 表示：不使用独立的 EP维度，而是借用 TP 维度
  - 16 experts 在 8 个 TP ranks 上切分 → 每个 rank 有 2 个 experts
  - 同时每个 expert 的权重在 TP 维度切分

### 9.2 独立 EP 配置

**更大的模型（例如 DeepSeek V3 671B）**：

```toml
[parallelism]
data_parallel_shard_degree = 16
tensor_parallel_degree = 8
pipeline_parallel_degree = 1
expert_parallel_degree = 4   # 独立的 EP 维度
expert_tensor_parallel_degree = 1  # 不使用 ETP
```

**说明**：
- 总共 16 × 8 × 4 = 512 GPUs
- 256 experts / 4 (EP) = 64 experts per GPU

### 9.3 Load Balancing 配置

```toml
[model.moe_args]
num_experts = 16
num_shared_experts = 1
top_k = 2
load_balance_coeff = 1e-3  # 启用 auxiliary-loss-free load balancing
```

### 9.4 Grouped GEMM 配置

```python
# 在代码中设置（根据量化类型）
from torchtitan.models.moe.utils import set_token_group_alignment_size_m

# BF16
set_token_group_alignment_size_m(8)

# FP8
set_token_group_alignment_size_m(16)

# MXFP8
set_token_group_alignment_size_m(32)
```

---

## 10. 性能数据

### 10.1 MoE vs Dense 对比

**Llama3 8B vs Llama4 17Bx16E (64 H100s)**：

| 模型 | 总参数 | 激活参数 | TPS/GPU | 显存 (GiB) |
|-----|--------|---------|---------|-----------|
| Llama3 8B | 8B | 8B (100%) | 6,500 | 62 |
| Llama4 17Bx16E | 73B | ~24B (33%) | 5,800 | 68 |

**观察**：
- MoE 模型参数多 9x，但激活参数只多 3x
- TPS/GPU 略低（因为 routing 和通信开销）
- 显存占用相近（稀疏激活的优势）

### 10.2 EP Scaling

**Llama4 17Bx16E (固定 TP=8)**：

| 配置 | EP | ETP | GPUs | TPS/GPU | 相对 baseline |
|-----|----|----|------|---------|--------------|
| Baseline | 1 | 8 | 64 | 5,800 | 1.00x |
| Scale EP | 2 | 4 | 128 | 5,600 | 0.97x |
| Scale EP | 4 | 2 | 256 | 5,400 | 0.93x |

**观察**：
- EP scaling 有一定的效率损失（All-to-All 通信开销）
- 但允许训练更大的 MoE 模型

### 10.3 Load Balancing 效果

**无 Load Balancing**：

```
Expert usage (step 100):
  Expert 0: 1200 tokens
  Expert 1: 980 tokens
  Expert 2: 650 tokens  ← 利用率低
  Expert 3: 450 tokens  ← 利用率低
  ...
  Variance: 高
```

**有 Load Balancing (coeff=1e-3)**：

```
Expert usage (step 100):
  Expert 0: 1050 tokens
  Expert 1: 1020 tokens
  Expert 2: 980 tokens
  Expert 3: 990 tokens
  ...
  Variance: 低（更均衡）
```

---

## 11. 最佳实践

### 11.1 何时使用 MoE + EP？

✅ **推荐使用**：
1. **需要超大模型**：想要 100B+ 参数但计算资源有限
2. **任务多样性高**：不同 tokens 需要不同的处理逻辑
3. **有足够的 GPUs**：至少 64+ GPUs（需要足够的并行度）
4. **训练数据丰富**：MoE 需要更多数据来训练所有 experts

❌ **不推荐使用**：
1. **小规模训练**：<32 GPUs，通信开销抵消收益
2. **任务单一**：所有 tokens 都类似，experts 难以专业化
3. **推理为主**：MoE 推理相对复杂，Dense 模型更简单
4. **调试阶段**：MoE + EP 调试困难，先用 Dense 验证

### 11.2 EP vs ETP 选择

**使用 EP（独立的 EP 维度）**：
```toml
expert_parallel_degree = 4
expert_tensor_parallel_degree = 1
```
- ✅ 更灵活的并行配置
- ✅ 适合 experts 数量多（>64）
- ⚠️ 额外的 All-to-All 通信开销

**使用 ETP（借用 TP 维度）**：
```toml
expert_parallel_degree = 1
expert_tensor_parallel_degree = 8
```
- ✅ 节省一个并行维度
- ✅ 适合 experts 数量适中（16-32）
- ✅ TP 和 EP 通信可以部分重叠

### 11.3 调优 Checklist

1. **启用 Grouped GEMM**：
```toml
[model.moe_args]
use_grouped_mm = true  # 比 for-loop 快很多
```

2. **配置正确的对齐**：
```python
# 根据量化类型设置
set_token_group_alignment_size_m(8)   # BF16
set_token_group_alignment_size_m(16)  # FP8
set_token_group_alignment_size_m(32)  # MXFP8
```

3. **启用 Load Balancing**：
```toml
[model.moe_args]
load_balance_coeff = 1e-3  # 推荐值：1e-4 到 1e-2
```

4. **选择合适的 top_k**：
```toml
[model.moe_args]
top_k = 2  # 推荐：2 或 3（平衡效果和计算量）
```

5. **过滤 Router 的量化**：
```toml
[quantize.linear.float8]
filter_fqns = ["router.gate"]  # Router 需要高精度
```

6. **监控 Expert Usage**：
```python
# 训练中打印 expert usage
print("Expert usage:", moe_module.tokens_per_expert)
```

### 11.4 常见问题

**Q1: MoE 训练不稳定怎么办？**

A: 可能的原因和解决方案：
1. **负载不均**：启用 load balancing
2. **梯度爆炸**：降低学习率，启用 gradient clipping
3. **Router 不收敛**：使用 sigmoid 而非 softmax，添加 route_scale
4. **某些 experts 不学习**：检查 load balancing，考虑增加 shared experts

**Q2: All-to-All 通信成为瓶颈怎么办？**

A: 优化策略：
1. **减少 EP degree**：牺牲一些 model parallel，减少通信
2. **使用 ETP 而非独立 EP**：借用 TP 维度
3. **增加 local experts**：增加 expert 数量，但减少 EP degree
4. **优化网络**：确保使用高速互联（NVLink/InfiniBand）

**Q3: 如何调试 MoE 模型？**

A: 调试技巧：
1. **强制均衡路由**：
```python
moe_args._debug_force_load_balance = True  # 测试用
```
2. **使用 for-loop experts**：
```python
moe_args.use_grouped_mm = False  # 更容易调试
```
3. **从小模型开始**：先用 4 experts 验证，再扩展到 16/32
4. **监控通信**：
```bash
export NCCL_DEBUG=INFO  # 查看 All-to-All 通信
```

---

## 12. 总结

### Expert Parallel 的核心要点

1. **MoE 的本质**：稀疏激活，参数多但计算少
   - 每个 token 只激活部分 experts（top-k）
   - 总参数可以很大，但激活参数较少

2. **Expert Parallel (EP)**：第 5 个并行维度
   - 将 experts 分片到多个 GPUs
   - 通过 All-to-All 通信分发和收集 tokens
   - 2D 模式：EP × TP（ExpertTensorParallel）

3. **核心组件**：
   - **Router**: 决定 token → experts 的映射
   - **GroupedExperts**: 使用 Grouped GEMM 高效执行
   - **TokenReorderer**: 按 expert 重新排序 tokens
   - **Load Balancing**: 保持 experts 负载均衡

4. **通信模式**：
   - **Token Dispatch**: All-to-All 分发 tokens
   - **Token Combine**: All-to-All 收集结果
   - **Permute/Unpermute**: 重新排列以适应 local experts

5. **性能优化**：
   - **Grouped GEMM**: 批量处理不同大小的矩阵乘法
   - **Auxiliary-Loss-Free Load Balancing**: 无需 auxiliary loss
   - **Shared Experts**: 所有 tokens 都经过，提升基础能力

### 搬桌子比喻总结

Expert Parallel 就像**专业工人分散到不同工地**：

```
Dense 模型：
  所有工人处理所有桌子 → 工人多时很慢

MoE 模型：
  每张桌子只找对应的专业工人 → 稀疏激活，高效

Expert Parallel:
  专业工人分散到多个工地 → 通过 All-to-All 分发桌子

  Token Dispatch: 把桌子送到对应的工地
  Local Processing: 每个工地的专家处理桌子
  Token Combine: 收集处理好的桌子
```

### 技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                Expert Parallel Stack                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TorchTitan (Integration Layer)                            │
│  ├─ MoE: Router + Experts + Shared Experts                 │
│  ├─ ExpertParallel: Token Dispatch/Combine                 │
│  ├─ ExpertTensorParallel: 2D (EP × TP)                     │
│  └─ Load Balancing: Auxiliary-Loss-Free                    │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  PyTorch DTensor (Distribution Layer)                      │
│  ├─ all_to_all_single: All-to-All 通信                     │
│  ├─ distribute_tensor: 权重分片 Shard(0)                   │
│  └─ DeviceMesh: EP mesh / EP×TP mesh                       │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  CUDA Kernels (Optimization Layer)                        │
│  ├─ torch._grouped_mm: Grouped GEMM                        │
│  ├─ generate_permute_indices: Triton kernel               │
│  └─ NCCL: All-to-All 集合通信                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. 参考资料

### 学术论文
- **Switch Transformers**: [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) - Google 的 MoE 论文
- **Auxiliary-Loss-Free Load Balancing**: [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) - 无 auxiliary loss 的负载均衡
- **GShard**: [arXiv:2006.16668](https://arxiv.org/abs/2006.16668) - 大规模 MoE 训练

### TorchTitan 源码
- `torchtitan/distributed/expert_parallel.py` - EP 实现
- `torchtitan/models/moe/moe.py` - MoE 层实现
- `torchtitan/models/moe/utils.py` - Permute/Unpermute
- `torchtitan/models/moe/kernels.py` - Triton kernels
- `torchtitan/components/optimizer.py:330-450` - Load Balancing

### PyTorch 文档
- [torch._grouped_mm](https://pytorch.org/docs/stable/generated/torch._grouped_mm.html) - Grouped GEMM
- [DTensor](https://pytorch.org/docs/stable/distributed.tensor.html) - 分布式 Tensor

---

**最后更新**：2025年11月25日
