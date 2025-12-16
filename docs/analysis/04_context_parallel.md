# Context Parallel (CP) 实现详解

## 目录
- [1. 什么是 Context Parallel？](#1-什么是-context-parallel)
- [2. 搬桌子的新比喻](#2-搬桌子的新比喻)
- [3. Ring Attention 原理](#3-ring-attention-原理)
- [4. 源码实现详解](#4-源码实现详解)
- [5. 性能分析](#5-性能分析)
- [6. 使用场景和最佳实践](#6-使用场景和最佳实践)

---

## 1. 什么是 Context Parallel？

### 1.1 长序列的挑战

在训练大语言模型时，我们经常面临**序列太长**的问题：

```python
# Llama3 8B 模型的 Attention 计算
batch_size = 8
seq_len = 8192  # 8K tokens
n_heads = 32
head_dim = 128

# Q, K, V 的形状
Q = [8, 8192, 32, 128]  # 需要 256 MB (bfloat16)
K = [8, 8192, 32, 128]  # 需要 256 MB
V = [8, 8192, 32, 128]  # 需要 256 MB

# Attention 矩阵
Attention_weights = Q @ K^T  # [8, 32, 8192, 8192]
                             # 需要 16 GB！ 😱
```

**问题**：
- **内存爆炸**：Attention 矩阵是 `O(seq_len²)`，序列越长，内存占用呈**平方增长**
- **单 GPU 放不下**：即使用 Flash Attention 优化，超长序列（> 32K）仍然会 OOM
- **Tensor Parallel 不够**：TP 只切分 heads，不切分 sequence

### 1.2 Context Parallel 的核心思想

**把序列切分到多个 GPU，每个 GPU 处理一段**

```
原始序列 (seq_len = 8192):
┌──────────────────────────────────────┐
│ Token 0, 1, 2, ..., 8191             │
└──────────────────────────────────────┘
    所有在 GPU 0

Context Parallel (CP = 4):
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ 0 - 2047│  │2048-4095│  │4096-6143│  │6144-8191│
└─────────┘  └─────────┘  └─────────┘  └─────────┘
   GPU 0        GPU 1        GPU 2        GPU 3

每个 GPU 处理 2048 个 tokens
```

**关键技术**：
- **Ring Attention**：用"接力"的方式让每个 GPU 看到完整的上下文
- **序列维度切分**：不是切模型，而是切输入
- **通信优化**：使用 All-Gather 或 All-to-All 交换 KV cache

---

## 2. 搬桌子的新比喻

### 2.1 场景回顾：什么是 TP？

回顾 Tensor Parallel：
- **桌子** = 神经网络层的权重
- **TP** = 把桌子本身切成几块，分散到多个 GPU

但 TP 不解决序列太长的问题！

### 2.2 Context Parallel：切分工作量

**Context Parallel 不切桌子，而是切工作量**

想象你要在一张**超大黑板**上写作业（这是序列）：

```
传统方式 (没有 CP):
你一个人在黑板上从左到右写 10000 个字
    ┌─────────────────────────────────────────────┐
    │ 字1, 字2, 字3, ..., 字10000                 │
    └─────────────────────────────────────────────┘
        一个人写，累死了 😓
        需要记住前面所有写过的字（内存爆炸）
```

**Context Parallel (CP = 4):**

```
把黑板分成 4 段，4 个人同时写

人1: │ 字1-2500    │
人2: │ 字2501-5000 │
人3: │ 字5001-7500 │
人4: │ 字7501-10000│

但问题来了：写字时要参考前面的内容！
比如人3写到"他"，要知道"他"指的是谁（在人1的部分）
```

### 2.3 Ring Attention：接力传递信息

**解决方案：像接力赛一样传递信息**

```
第1轮: 每个人拿着自己的纸条
人1: [段1]         人2: [段2]         人3: [段3]         人4: [段4]

第2轮: 传递纸条 (顺时针)
人1: [段1, 段2]    人2: [段2, 段3]    人3: [段3, 段4]    人4: [段4, 段1]
     ↑ 接到人2的纸条     ↑ 接到人3的       ↑ 接到人4的       ↑ 接到人1的

第3轮: 继续传递
人1: [段1,段2,段3] 人2: [段2,段3,段4] 人3: [段3,段4,段1] 人4: [段4,段1,段2]

第4轮: 最后一次传递
人1: [段1,2,3,4]   人2: [段2,3,4,1]   人3: [段3,4,1,2]   人4: [段4,1,2,3]
     ↑ 所有人都看到了完整的内容！
```

**关键点**：
- **分段处理**：每人只负责一段，减少单人工作量
- **接力传递**：通过多轮传递，让每人最终看到全部内容
- **并行计算**：4 个人同时工作，效率提升 4 倍

### 2.4 具体到 Attention 计算

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

传统方式 (seq_len = 8192):
Q: [batch, 8192, hidden]  在 GPU 0
K: [batch, 8192, hidden]  在 GPU 0
V: [batch, 8192, hidden]  在 GPU 0

计算 Q @ K^T: [batch, 8192, 8192]  需要 16 GB！

Context Parallel (CP = 4):
每个 GPU 只处理 2048 个 query tokens

GPU 0: Q[0:2048]    看到完整的 K, V
GPU 1: Q[2048:4096] 看到完整的 K, V
GPU 2: Q[4096:6144] 看到完整的 K, V
GPU 3: Q[6144:8192] 看到完整的 K, V

每个 GPU 的 Attention 矩阵: [batch, 2048, 8192]  只需要 4 GB
总内存: 4 GB × 4 = 16 GB (没变，但分散了！)
```

**为什么有效？**
- **Query 切分**：每个 GPU 只计算一部分 query 的 attention
- **KV 轮换**：通过 Ring 机制，让每个 GPU 看到完整的 K, V
- **内存降低**：单 GPU 内存从 16 GB 降到 4 GB

---

## 3. Ring Attention 原理

### 3.1 传统 Attention 的计算

```python
# 伪代码
def attention(Q, K, V):
    # Q: [batch, seq_len, hidden]
    # K, V: [batch, seq_len, hidden]

    scores = Q @ K.T / sqrt(d)       # [batch, seq_len, seq_len]
    weights = softmax(scores)        # [batch, seq_len, seq_len]
    output = weights @ V             # [batch, seq_len, hidden]
    return output
```

**问题**：`scores` 矩阵是 `O(seq_len²)`

### 3.2 Ring Attention 的计算流程

**核心思想**：把 K, V 切成多块，依次处理，最后合并

```python
# Context Parallel with Ring Attention (CP = 4)

# 初始状态：每个 GPU 有自己的一段
GPU 0: Q0 [0:2048],    K0 [0:2048],    V0 [0:2048]
GPU 1: Q1 [2048:4096], K1 [2048:4096], V1 [2048:4096]
GPU 2: Q2 [4096:6144], K2 [4096:6144], V2 [4096:6144]
GPU 3: Q3 [6144:8192], K3 [6144:8192], V3 [6144:8192]

# 每个 GPU 要计算自己的 Q 对完整 K, V 的 attention
# 但完整的 K, V 分散在 4 个 GPU 上

# === 第 1 轮：计算本地的 KV ===
GPU 0: output0 = attention(Q0, K0, V0)  # 部分结果
GPU 1: output1 = attention(Q1, K1, V1)
GPU 2: output2 = attention(Q2, K2, V2)
GPU 3: output3 = attention(Q3, K3, V3)

# === 第 2 轮：Ring 传递 KV ===
# 每个 GPU 把 KV 发给下一个 GPU（环形）
GPU 0 接收 K3, V3 (来自 GPU 3)
GPU 1 接收 K0, V0 (来自 GPU 0)
GPU 2 接收 K1, V1 (来自 GPU 1)
GPU 3 接收 K2, V2 (来自 GPU 2)

# 计算并累加
GPU 0: output0 += attention(Q0, K3, V3)
GPU 1: output1 += attention(Q1, K0, V0)
GPU 2: output2 += attention(Q2, K1, V1)
GPU 3: output3 += attention(Q3, K2, V2)

# === 第 3 轮：继续传递 ===
GPU 0 接收 K2, V2
GPU 1 接收 K3, V3
GPU 2 接收 K0, V0
GPU 3 接收 K1, V1

GPU 0: output0 += attention(Q0, K2, V2)
GPU 1: output1 += attention(Q1, K3, V3)
GPU 2: output2 += attention(Q2, K0, V0)
GPU 3: output3 += attention(Q3, K1, V1)

# === 第 4 轮：最后一次传递 ===
GPU 0 接收 K1, V1
GPU 1 接收 K2, V2
GPU 2 接收 K3, V3
GPU 3 接收 K0, V0

GPU 0: output0 += attention(Q0, K1, V1)
GPU 1: output1 += attention(Q1, K2, V2)
GPU 2: output2 += attention(Q2, K3, V3)
GPU 3: output3 += attention(Q3, K0, V0)

# 完成！每个 GPU 现在有完整的 attention 输出
```

**关键优化**：
- **重叠计算和通信**：在计算 round N 时，同时传递 round N+1 的 KV
- **因果掩码优化**：对于因果 attention，不需要传递所有 KV（只需要左边的）

### 3.3 Softmax 的数值稳定性

**挑战**：Softmax 需要看到所有 scores 才能归一化

```python
# 传统 Softmax
scores = Q @ K.T  # 需要完整的 scores 矩阵
max_score = max(scores)  # 找最大值
exp_scores = exp(scores - max_score)  # 数值稳定的 exp
weights = exp_scores / sum(exp_scores)  # 归一化
```

**Ring Attention 的解决方案**：**在线更新 Softmax**

```python
# 初始化
output = 0
sum_exp = 0
max_score = -inf

# 逐块处理 KV
for kv_chunk in [KV0, KV1, KV2, KV3]:
    # 计算当前 chunk 的 scores
    scores_chunk = Q @ kv_chunk.K.T

    # 更新全局最大值
    new_max = max(max_score, max(scores_chunk))

    # 重新缩放之前的结果 (因为最大值变了)
    scale_factor = exp(max_score - new_max)
    output *= scale_factor
    sum_exp *= scale_factor

    # 计算当前 chunk 的贡献
    exp_scores_chunk = exp(scores_chunk - new_max)
    sum_exp += sum(exp_scores_chunk)
    output += exp_scores_chunk @ kv_chunk.V

    # 更新最大值
    max_score = new_max

# 最终归一化
output /= sum_exp
```

**这个算法很巧妙**：
- **增量更新**：不需要一次性看到所有 scores
- **数值稳定**：始终用最新的 max_score 保证稳定性
- **支持流式计算**：可以边接收 KV 边计算

### 3.4 因果掩码优化

对于 **Causal Attention**（如 GPT），token 只能看到它**左边**的 tokens：

```
Token 0: 只能看 Token 0
Token 1: 只能看 Token 0, 1
Token 2: 只能看 Token 0, 1, 2
...
Token 2047: 可以看 Token 0 - 2047

假设 CP = 4, 每个 GPU 处理 2048 tokens:

GPU 0 (Token 0-2047):
  不需要接收其他 GPU 的 KV (因为右边的 token 都在未来)

GPU 1 (Token 2048-4095):
  只需要接收 GPU 0 的 KV (Token 0-2047)
  不需要 GPU 2, 3 的 KV

GPU 2 (Token 4096-6143):
  需要接收 GPU 0, 1 的 KV (Token 0-4095)
  不需要 GPU 3 的 KV

GPU 3 (Token 6144-8191):
  需要接收 GPU 0, 1, 2 的 KV (Token 0-6143)
```

**通信量优化**：
```
没有因果掩码: 每个 GPU 接收 3 次 KV (传递 3 轮)
有因果掩码:
  GPU 0: 0 次接收
  GPU 1: 1 次接收
  GPU 2: 2 次接收
  GPU 3: 3 次接收
  平均: 1.5 次接收 (减少 50% 通信量！)
```

---

## 4. 源码实现详解

### 4.1 核心 API：context_parallel

```python
# 来自: torchtitan/distributed/utils.py:198-220

def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,              # CP 的 device mesh
    cp_buffers: list[torch.Tensor],   # 需要在序列维度切分的 tensors
    cp_seq_dims: list[int],           # 每个 buffer 的序列维度索引
    cp_no_restore_buffers: set[torch.Tensor],  # 不需要恢复的 buffers
    cp_rotate_method: str,            # "allgather" 或 "alltoall"
):
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        print(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )

    # 设置轮换方法
    set_rotate_method(cp_rotate_method)

    # 返回 context manager
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )
```

**参数说明**：

1. **cp_mesh**：Context Parallel 的 device mesh
   - 例如：`world_mesh["cp"]` - CP 维度的 mesh
   - 如果 CP = 4，则包含 4 个 GPU

2. **cp_buffers**：需要切分的 tensors
   ```python
   # 来自: torchtitan/train.py:478-482
   cp_buffers = [inputs, labels]
   # inputs: [batch, seq_len, hidden]  - 输入序列
   # labels: [batch, seq_len]          - 标签序列

   if hasattr(model_parts[0], "freqs_cis"):
       # freqs_cis: [max_seq_len, head_dim] - RoPE 的频率
       cp_buffers += [m.freqs_cis for m in model_parts]
   ```

3. **cp_seq_dims**：每个 buffer 的序列维度
   ```python
   # 来自: torchtitan/train.py:479-482
   cp_seq_dims = [1, 1]  # inputs 和 labels 的 seq 维度都是 dim=1

   if hasattr(model_parts[0], "freqs_cis"):
       # freqs_cis 的 seq 维度是 dim=0
       cp_seq_dims += [0 for _ in model_parts]
   ```

4. **cp_no_restore_buffers**：不需要恢复的 buffers
   ```python
   # 来自: torchtitan/train.py:489
   cp_no_restore_buffers = {inputs, labels}
   ```
   - 输入和标签不需要恢复（因为它们已经被切分，后续不需要完整的）
   - freqs_cis 需要恢复（因为后续层需要完整的频率信息）

5. **cp_rotate_method**：KV 轮换的通信方式
   - `"allgather"`：每个 GPU All-Gather 其他 GPU 的 KV
   - `"alltoall"`：All-to-All 交换 KV chunks

### 4.2 使用方式

```python
# 来自: torchtitan/train.py:484-494

# 创建 CP context
optional_context_parallel_ctx = (
    dist_utils.create_context_parallel_ctx(
        cp_mesh=parallel_dims.world_mesh["cp"],
        cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
        cp_seq_dims=[1, 1] + [0 for _ in model_parts],
        cp_no_restore_buffers={inputs, labels},
        cp_rotate_method=job_config.parallelism.context_parallel_rotate_method,
    )
    if parallel_dims.cp_enabled
    else None
)

# 在训练中使用
with self.train_context(optional_context_parallel_ctx):
    pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
    loss = self.loss_fn(pred, labels)
    loss.backward()
```

**工作流程**：

1. **进入 context**：
   ```python
   with context_parallel(...):
       # 自动切分 cp_buffers 在序列维度
       # inputs: [batch, seq_len, hidden] → [batch, seq_len/CP, hidden]
       # labels: [batch, seq_len] → [batch, seq_len/CP]
   ```

2. **Forward pass**：
   - 模型的 Attention 层会自动使用 Ring Attention
   - 每个 GPU 只计算 `seq_len / CP` 个 query 的 attention
   - 通过 Ring 机制看到完整的 K, V

3. **退出 context**：
   ```python
   # 自动恢复 cp_buffers (除了 no_restore_buffers)
   # freqs_cis: [max_seq_len/CP, head_dim] → [max_seq_len, head_dim]
   ```

### 4.3 Attention Wrapper 与 CP 的配合

```python
# 来自: torchtitan/models/attention.py:86-127

class FlexAttentionWrapper(torch.nn.Module):
    """Wrapper around `flex_attention` to make it torch.compile and CP compatible.

    This wrapper serves two purposes:
    1) Invoke `torch.compile` with a valid mode "max-autotune-no-cudagraphs" to
       achieve good performance.
    2) Being a wrapper allows us to apply _ContextParallel to it.

    Note:
        The forward function must have q, k, v as the first three arguments, and
        block_mask as a keyword argument to be compatible with _ContextParallel.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        block_mask: BlockMask,
        scale: float | None = None,
        return_lse: bool = False,
    ):
        return FlexAttentionWrapper._compiled_flex_attn(
            q, k, v,
            block_mask=block_mask,
            scale=scale,
            return_lse=return_lse,
        )
```

**为什么需要 Wrapper？**

1. **CP 需要 nn.Module**：
   - `F.scaled_dot_product_attention` 不是 nn.Module
   - CP 的 Ring Attention 需要 hook 到 nn.Module 的 forward

2. **参数约定**：
   - Forward 的前 3 个参数必须是 `q, k, v`
   - CP 的 Ring 机制会自动 hook 这些参数

3. **自动应用 Ring Attention**：
   ```python
   # 在 CP context 内
   output = attention_wrapper(q, k, v, block_mask=mask)
   # CP 自动将其转换为 Ring Attention 计算
   ```

### 4.4 Device Mesh 的构建

```python
# 来自: torchtitan/distributed/parallel_dims.py:147-190

def _build_mesh_without_ep(self) -> DeviceMesh:
    dims = []
    names = []
    for d, name in zip(
        [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
        ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
    ):
        if d > 1:
            dims.append(d)
            names.append(name)

    # 例如: dims=[2, 2, 2], names=["dp_shard", "cp", "tp"]
    # 表示 8 个 GPU，2路 FSDP，2路 CP，2路 TP
    mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

    # 创建组合 mesh
    dp_shard_cp_mesh_dim_names = []
    if self.dp_shard_enabled:
        dp_shard_cp_mesh_dim_names.append("dp_shard")
    if self.cp_enabled:
        dp_shard_cp_mesh_dim_names.append("cp")

    if dp_shard_cp_mesh_dim_names:
        mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
            mesh_dim_name="dp_shard_cp"
        )

    return mesh
```

**为什么 CP 和 FSDP 组合？**

```
假设 8 个 GPU，dp_shard = 2, cp = 2, tp = 2

mesh = [
    [                                      # dp_shard group 0
        [GPU0, GPU1],  # cp group 0, tp group
        [GPU2, GPU3],  # cp group 1, tp group
    ],
    [                                      # dp_shard group 1
        [GPU4, GPU5],  # cp group 0, tp group
        [GPU6, GPU7],  # cp group 1, tp group
    ]
]

FSDP 在 dp_shard 维度做参数切分
CP 在 cp 维度做序列切分
TP 在 tp 维度做模型切分

三者正交，可以组合使用！
```

### 4.5 配置选项

```toml
# 来自: torchtitan/models/llama3/train_configs/llama3_8b.toml:43

[parallelism]
context_parallel_degree = 1  # CP 并行度，1 表示禁用

context_parallel_rotate_method = "allgather"  # "allgather" 或 "alltoall"
```

**两种轮换方法**：

1. **All-Gather**：
   ```
   每个 GPU 依次 All-Gather 其他 GPU 的 KV

   Round 1: GPU 0 all-gather GPU 0's KV (no-op)
   Round 2: GPU 0 all-gather GPU 1's KV
   Round 3: GPU 0 all-gather GPU 2's KV
   Round 4: GPU 0 all-gather GPU 3's KV

   优点：实现简单，每轮只有一个 GPU 发送
   缺点：通信量大 (每个 GPU 都要接收完整的 KV)
   ```

2. **All-to-All**：
   ```
   每个 GPU 同时发送和接收不同的 KV chunks

   优点：通信更均衡，可以重叠
   缺点：实现复杂，需要精确的通信调度
   ```

---

## 5. 性能分析

### 5.1 内存节省

**传统方式** (无 CP)：

```python
# Llama3 8B, seq_len = 32768
batch_size = 8
seq_len = 32768
n_heads = 32
head_dim = 128

# 单个 Attention 层的内存
Q = [8, 32768, 32, 128]  = 1 GB
K = [8, 32768, 32, 128]  = 1 GB
V = [8, 32768, 32, 128]  = 1 GB

# Flash Attention 的工作内存 (简化)
# 虽然不保存完整的 attention matrix，但仍需要大量临时内存
Work_memory ≈ 4 GB

Total ≈ 7 GB per layer
```

**Context Parallel (CP = 4)**：

```python
# 每个 GPU 处理 8192 tokens
Q_local = [8, 8192, 32, 128]  = 256 MB
K_chunk = [8, 8192, 32, 128]  = 256 MB (轮换的)
V_chunk = [8, 8192, 32, 128]  = 256 MB (轮换的)

Work_memory ≈ 1 GB

Total ≈ 1.8 GB per layer per GPU
```

**节省比例**：
```
7 GB / 1.8 GB = 3.9x 内存节省
```

### 5.2 通信开销

**CP 的通信量**：

假设：
- CP = 4
- seq_len = 32768
- hidden_dim = 4096
- dtype = bfloat16 (2 bytes)

**每个 Transformer Layer 的通信**：

```python
# KV cache 大小 (每个 chunk)
kv_chunk_size = seq_len / CP * hidden_dim * 2 * 2
              = 32768 / 4 * 4096 * 2 * 2
              = 256 MB

# Ring Attention 需要传递 (CP - 1) 轮
# 因果掩码优化后，平均传递 (CP - 1) / 2 轮

# 没有因果掩码优化
total_comm = kv_chunk_size * (CP - 1)
           = 256 MB * 3
           = 768 MB per layer

# 有因果掩码优化
total_comm_causal = kv_chunk_size * (CP - 1) / 2
                  = 256 MB * 1.5
                  = 384 MB per layer

# Llama3 8B 有 32 layers
total_comm_per_fwd = 384 MB * 32
                   = 12 GB
```

**对比 Tensor Parallel (TP)**：

```python
# TP = 4 的通信量 (每层 2 次 All-Reduce)
tp_comm_per_layer = 2 * batch_size * seq_len * hidden_dim * 2
                  = 2 * 8 * 32768 * 4096 * 2
                  = 4 GB per layer

tp_comm_total = 4 GB * 32 = 128 GB

CP 通信量 (12 GB) << TP 通信量 (128 GB)
```

**为什么 CP 通信量更少？**
- CP 只传递 KV cache，不传递完整的激活
- TP 需要在每个线性层后做 All-Reduce
- CP 的通信只在 Attention 层

### 5.3 计算效率

**CP 不改变计算量**：

```
传统 Attention: O(seq_len² * hidden_dim)
Ring Attention:  O(seq_len² * hidden_dim)

计算量相同！只是分散到多个 GPU
```

**但有额外开销**：

1. **通信延迟**：
   ```
   需要传递 (CP - 1) 轮 KV
   每轮延迟 ≈ 256 MB / bandwidth
   ```

2. **同步开销**：
   ```
   Ring 的每一轮需要同步
   slow GPU 会拖慢整个 ring
   ```

### 5.4 扩展性分析

**理想加速比**：

```
CP = 2: 2x 序列长度，通信开销 ~10%  → 实际 1.8x
CP = 4: 4x 序列长度，通信开销 ~20%  → 实际 3.2x
CP = 8: 8x 序列长度，通信开销 ~35%  → 实际 5.2x
```

**影响因素**：

1. **网络带宽**：
   - NVLink (900 GB/s): 通信开销小，扩展性好
   - InfiniBand (200 GB/s): 通信开销中等
   - PCIe (64 GB/s): 通信开销大，扩展性差

2. **序列长度**：
   - seq_len = 8K: 计算时间短，通信占比高
   - seq_len = 32K: 计算时间长，通信占比低
   - seq_len = 128K: 计算主导，通信开销可忽略

3. **CP 并行度**：
   - CP = 2: 通信 1 轮，开销最小
   - CP = 4: 通信 3 轮，开销适中
   - CP = 8: 通信 7 轮，开销较大

**最佳实践**：

```
短序列 (< 8K):    不建议用 CP
中等序列 (8K-32K):  CP = 2 或 4
长序列 (32K-128K):  CP = 4 或 8
超长序列 (> 128K):  CP = 8 或 16
```

---

## 6. 使用场景和最佳实践

### 6.1 何时应该使用 Context Parallel？

**推荐使用的场景**：

✅ **超长序列训练 (> 8K tokens)**
   - Llama3 with seq_len = 32K
   - 长文档理解
   - 代码生成（长上下文）

✅ **内存受限**
   - 单 GPU 放不下长序列的 Attention
   - 即使用了 Flash Attention 仍然 OOM

✅ **与 FSDP 组合**
   - CP 处理序列，FSDP 处理参数
   - 两者正交，可以完美组合

✅ **有高速互联**
   - NVLink: 900 GB/s (H100)
   - InfiniBand: 200 GB/s
   - Ring 需要频繁通信

**不推荐使用的场景**：

❌ **短序列 (< 8K)**
   - 内存足够，不需要 CP
   - 通信开销得不偿失

❌ **只有 PCIe 连接**
   - 带宽低 (64 GB/s)
   - 通信成为瓶颈

❌ **非因果 Attention**
   - 无法使用因果掩码优化
   - 通信量翻倍

❌ **推理场景**
   - 推理通常 batch_size = 1，不需要并行
   - KV cache 已经是增量的

### 6.2 配置方法

**TOML 配置**：

```toml
[training]
seq_len = 32768  # 长序列

[parallelism]
data_parallel_shard_degree = 8   # FSDP
context_parallel_degree = 4      # CP = 4
tensor_parallel_degree = 1       # 可选，通常 CP 与 TP 不同时用

context_parallel_rotate_method = "allgather"  # 或 "alltoall"
```

**序列长度要求**：

```python
# 来自: torchtitan/distributed/parallel_dims.py:252-259

@property
def seq_len_divisor(self):
    # Sequence Parallel requires that seq_len be divisible by TP degree.
    # Context Parallel requires that seq_len be divisible by 2 * CP degree
    return self.tp * (self.cp * 2)
```

**配置示例**：

```toml
# 场景1: 中等序列 + FSDP + CP
[training]
seq_len = 16384  # 16K tokens

[parallelism]
data_parallel_shard_degree = 8
context_parallel_degree = 2
tensor_parallel_degree = 1

# seq_len 必须能被 2 * CP = 4 整除 ✓ (16384 % 4 = 0)
```

```toml
# 场景2: 长序列 + FSDP + CP + TP
[training]
seq_len = 32768  # 32K tokens

[parallelism]
data_parallel_shard_degree = 4
context_parallel_degree = 4
tensor_parallel_degree = 2

# seq_len 必须能被 TP * 2 * CP = 16 整除 ✓ (32768 % 16 = 0)
```

### 6.3 与其他并行的组合

**推荐组合**：

| 场景 | FSDP | CP | TP | PP | 说明 |
|------|------|----|----|----|----|
| **短序列小模型** | ✓ | ✗ | ✗ | ✗ | 只用 FSDP |
| **长序列小模型** | ✓ | ✓ | ✗ | ✗ | FSDP + CP |
| **短序列大模型** | ✓ | ✗ | ✓ | ✗ | FSDP + TP |
| **长序列大模型** | ✓ | ✓ | ✓ | ✗ | FSDP + CP + TP |
| **超大模型** | ✓ | ✓ | ✓ | ✓ | 4D 并行 |

**配置示例（Llama3 70B + 32K context）**：

```toml
[model]
name = "llama3"
flavor = "70B"

[training]
seq_len = 32768
local_batch_size = 1

[parallelism]
# 256 GPUs = 32 FSDP × 4 CP × 2 TP
data_parallel_shard_degree = 32
context_parallel_degree = 4
tensor_parallel_degree = 2
pipeline_parallel_degree = 1

context_parallel_rotate_method = "allgather"
```

**为什么这样组合？**
- **FSDP (32路)**：处理 70B 参数，每个 GPU 约 2.2B 参数
- **CP (4路)**：处理 32K 序列，每个 GPU 约 8K tokens
- **TP (2路)**：减少单层内存，加速通信

### 6.4 调试和验证

**如何验证 CP 是否生效？**

1. **检查内存使用**：
   ```bash
   # 不启用 CP
   context_parallel_degree = 1
   # 观察 nvidia-smi，记录峰值内存

   # 启用 CP
   context_parallel_degree = 4
   # 峰值内存应该降低约 3-4x
   ```

2. **检查序列切分**：
   ```python
   # 在模型 forward 中打印输入形状
   print(f"Input shape: {inputs.shape}")

   # 不启用 CP
   # Input shape: [batch, 32768, hidden]

   # 启用 CP = 4
   # Input shape: [batch, 8192, hidden]  ← 序列被切分了
   ```

3. **Profiling 通信**：
   ```python
   from torch.profiler import profile, ProfilerActivity

   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       model(input)

   prof.export_chrome_trace("trace.json")
   # 在 chrome://tracing 中查看 all_gather 的频率和耗时
   ```

**常见问题**：

❓ **启用 CP 后训练变慢？**
- 检查序列长度是否足够长 (需要 > 8K)
- 检查网络连接是否为 NVLink/IB
- 检查 CP 并行度是否太高 (CP > 8 通常不推荐)

❓ **Loss 不收敛？**
- CP 应该是数值等价的，loss 应该和不用 CP 一致
- 检查是否正确设置了 `cp_no_restore_buffers`
- 检查 batch_size 和 learning rate 是否需要调整

❓ **OOM 错误？**
- CP 降低了 Attention 的内存，但没有降低模型参数内存
- 需要配合 FSDP 使用
- 检查 `cp_buffers` 是否包含了所有需要切分的 tensor

### 6.5 性能优化技巧

**1. 选择合适的 rotate_method**：

```toml
# All-Gather (默认)
context_parallel_rotate_method = "allgather"
# 优点：实现简单稳定
# 适用：CP <= 4, 网络带宽充足

# All-to-All
context_parallel_rotate_method = "alltoall"
# 优点：通信更均衡
# 适用：CP > 4, 需要更好的扩展性
```

**2. 配合 Flash Attention**：

```python
# CP 与 Flash Attention 是正交的
# Flash Attention 减少内存，CP 切分序列
# 两者结合效果最好

# 使用 flex_attention (自动启用 Flash Attention)
[model.llama3]
attn_type = "flex"  # 或 "sdpa" (也会用 Flash Attention)
```

**3. 优化 CP 并行度**：

```python
# 经验公式
optimal_CP = ceil(seq_len / 8192)

# 示例
seq_len = 16384  → CP = 2
seq_len = 32768  → CP = 4
seq_len = 65536  → CP = 8
seq_len = 131072 → CP = 16
```

**4. 平衡 CP 和 TP**：

```
总 GPU 数量固定时，需要在 CP 和 TP 之间权衡

32 GPUs, 选择：
- CP = 1, TP = 8, FSDP = 4  → 适合短序列大模型
- CP = 4, TP = 2, FSDP = 4  → 适合长序列大模型
- CP = 8, TP = 1, FSDP = 4  → 适合超长序列

通信量对比：
CP = 8: 12 GB per forward (主要在 Attention)
TP = 8: 128 GB per forward (遍布所有层)

一般来说：CP 的通信效率更高
```

---

## 7. 总结

### 7.1 核心要点

用**接力赛**总结 Context Parallel：

```
传统 Attention = 一个人看完整本书
    内存爆炸（要记住整本书）

Context Parallel = 4 个人接力读书
    人1读 第1章，传给人2
    人2读 第2章，传给人3
    ...
    每个人只需要记住 1/4 的内容
    但通过接力，每个人最终理解了整本书
```

**三大核心技术**：

1. **序列切分**：把输入序列切成多块，每个 GPU 处理一块
2. **Ring Attention**：通过环形传递 KV，让每个 GPU 看到完整上下文
3. **在线 Softmax**：增量更新 Softmax，支持流式计算

### 7.2 性能特点

**内存节省**：
- CP = 4: **3-4x** 内存节省
- CP = 8: **6-8x** 内存节省
- 可以训练**更长的序列**（32K → 128K → 1M）

**通信开销**：
- CP 的通信量 **远小于** TP
- 因果掩码优化可以减少 **50%** 通信
- 需要高速互联（NVLink / InfiniBand）

**计算效率**：
- **不增加计算量**
- 有通信延迟（10-35%）
- 长序列时通信占比小，效率高

### 7.3 使用建议

**推荐使用**：
- ✅ 长序列训练 (> 8K tokens)
- ✅ 内存受限场景
- ✅ 配合 FSDP 使用
- ✅ 有 NVLink 互联

**不推荐使用**：
- ❌ 短序列 (< 8K tokens)
- ❌ PCIe 连接
- ❌ 推理场景
- ❌ 非因果 Attention

**配置要点**：
```toml
[training]
seq_len = 32768  # 必须能被 2 * CP 整除

[parallelism]
context_parallel_degree = 4
context_parallel_rotate_method = "allgather"

# 推荐组合
data_parallel_shard_degree = 8  # FSDP
tensor_parallel_degree = 1      # TP（可选）
```

### 7.4 与其他并行的对比

| 特性 | Data Parallel | Tensor Parallel | Context Parallel |
|------|--------------|-----------------|------------------|
| **切分对象** | 数据 | 模型 | 序列 |
| **内存节省** | 参数 | 参数 + 激活 | 激活 (Attention) |
| **通信量** | 中 | 大 | 小 |
| **适用场景** | 通用 | 大模型 | 长序列 |
| **实现复杂度** | 简单 | 中等 | 复杂 |
| **数值等价性** | ✓ | ✓ | ✓ |

### 7.5 未来发展方向

**可能的改进**：

1. **更高效的 Ring 算法**：
   - 当前：顺序传递 KV
   - 未来：并行传递多个 KV chunks

2. **自适应 CP**：
   - 当前：固定的 CP 并行度
   - 未来：根据序列长度自动调整

3. **与 KV cache 优化结合**：
   - 当前：完整传递 KV
   - 未来：只传递增量 KV (适合推理)

4. **支持更多 Attention 变体**：
   - 当前：主要支持 Causal Attention
   - 未来：Sliding Window, Sparse Attention 等

---

## 8. 参考资料

**源码文件**：
- `torchtitan/distributed/utils.py:198-220` - CP context 创建
- `torchtitan/train.py:478-494` - CP 的使用
- `torchtitan/models/attention.py` - Attention Wrapper
- `torchtitan/distributed/parallel_dims.py` - 并行维度管理

**PyTorch 官方资源**：
- [Experimental Context Parallel API](https://pytorch.org/docs/stable/distributed.tensor.experimental.html)
- [Ring Attention Implementation](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/experimental/_attention.py)

**相关论文**：
- Ring Attention with Blockwise Transformers for Near-Infinite Context (Liu et al., 2023)
- Blockwise Parallel Transformer for Large Context Models (Liu et al., 2023)
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)

**相关文档**：
- `docs/analysis/02_tensor_parallel_implementation.md` - Tensor Parallel 详解
- `docs/analysis/03_async_tensor_parallel.md` - Async TP 详解
- `docs/converging.md` - 收敛性验证（包含 CP 测试）

---

## 附录：高级话题

### A.1 Ring Attention 的数学推导

**问题**：如何在不保存完整 attention matrix 的情况下计算 Softmax？

**关键洞察**：Softmax 可以增量更新

```python
# 传统 Softmax
scores = [s1, s2, s3, s4]  # 所有 scores
max_s = max(scores)
exp_scores = exp(scores - max_s)
softmax = exp_scores / sum(exp_scores)

# 增量 Softmax
# Step 1: 只有 s1
max_s = s1
exp_s1 = exp(s1 - max_s) = 1
sum_exp = 1
result = exp_s1 / sum_exp = 1

# Step 2: 加入 s2
max_s_new = max(max_s, s2)
# 重新缩放之前的结果
exp_s1 *= exp(max_s - max_s_new)
sum_exp *= exp(max_s - max_s_new)
# 加入新的 score
exp_s2 = exp(s2 - max_s_new)
sum_exp += exp_s2
# 更新
max_s = max_s_new

# Step 3, 4: 类似...
```

**应用到 Attention**：

```python
def ring_attention(Q, K_chunks, V_chunks):
    output = 0
    sum_exp = 0
    max_score = -inf

    for K_chunk, V_chunk in zip(K_chunks, V_chunks):
        # 计算当前 chunk 的 scores
        scores = Q @ K_chunk.T  # [batch, q_len, k_len]

        # 更新全局最大值
        chunk_max = scores.max(dim=-1, keepdim=True)
        new_max = torch.maximum(max_score, chunk_max)

        # 重新缩放
        exp_old_max = torch.exp(max_score - new_max)
        exp_new_scores = torch.exp(scores - new_max)

        # 更新累加器
        output = output * exp_old_max + exp_new_scores @ V_chunk
        sum_exp = sum_exp * exp_old_max + exp_new_scores.sum(dim=-1, keepdim=True)

        max_score = new_max

    # 最终归一化
    output = output / sum_exp
    return output
```

### A.2 Load Balancing 选项

```python
# 来自: torchtitan/models/flux/infra/parallelize.py:54-56

from torch.distributed.tensor.experimental._attention import _cp_options

_cp_options.enable_load_balance = False
```

**什么是 Load Balancing？**

- 在非因果 Attention 中，每个 GPU 计算的 attention 矩阵大小不同
- Load Balancing 尝试平衡各个 GPU 的计算量
- 但会增加通信复杂度

**何时禁用？**
- 因果 Attention：已经自然平衡（后面的 token 看更多）
- Flux 模型：不使用因果掩码，但仍禁用以简化实现

### A.3 CP 与 Variable Length Attention

Context Parallel 也可以与 Variable Length Attention 结合：

```python
# 不同文档有不同长度
batch = [
    "文档1: 1000 tokens",
    "文档2: 5000 tokens",
    "文档3: 200 tokens",
    "文档4: 3000 tokens",
]

# Padding 到最大长度 5000
padded_batch = pad(batch, max_len=5000)

# 使用 CP = 4 切分
# 每个 GPU 处理 1250 tokens
# 但大部分是 padding！

# 优化：Variable Length Attention
# 动态处理每个文档，不浪费计算在 padding 上
```

这是未来的研究方向，可以进一步提升效率。
