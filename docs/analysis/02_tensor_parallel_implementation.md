# Tensor Parallel (TP) 实现详解

## 目录
- [1. 什么是 Tensor Parallel？](#1-什么是-tensor-parallel)
- [2. 搬桌子的比喻](#2-搬桌子的比喻)
- [3. Tensor Parallel 的三种模式](#3-tensor-parallel-的三种模式)
- [4. 在 Transformer 中的应用](#4-在-transformer-中的应用)
- [5. 源码实现详解](#5-源码实现详解)
- [6. 通信优化](#6-通信优化)

---

## 1. 什么是 Tensor Parallel？

### 1.1 基本概念

**Tensor Parallel (TP)** 是一种模型并行技术，它将**单个神经网络层的权重矩阵**切分到多个 GPU 上。

**核心思想**：一个矩阵乘法 `Y = X @ W`，可以切分成多个小矩阵乘法并行计算。

### 1.2 为什么需要 TP？

当模型太大时：
- **单层权重就放不进一个 GPU** - 比如一个 Linear 层权重是 [8192, 8192]，fp16 需要 128MB
- **Data Parallel 不够用** - DP 只是把数据切分，每个 GPU 还是要存完整的模型
- **Pipeline Parallel 太粗粒度** - PP 是按层切分，但单层太大还是放不下

这时候就需要 **TP：把单层权重切碎，分散到多个 GPU**。

---

## 2. 搬桌子的比喻

### 2.1 场景设定

想象你需要搬一张**超级大的桌子**（就是神经网络的一层）：

- **桌面** = 权重矩阵 W
- **桌腿** = 不同的操作（查询/键/值/输出）
- **你和朋友们** = 多个 GPU

### 2.2 Column-wise Parallel (列切分)

**就像竖着切桌子**：

```
原始桌子：          切成两半后：
┌─────────┐        ┌────┐  ┌────┐
│         │   →    │ 左 │  │ 右 │
│  完整   │        │ 半 │  │ 半 │
│  桌面   │        │ 桌 │  │ 桌 │
└─────────┘        └────┘  └────┘
  GPU 0            GPU 0   GPU 1
```

**特点**：
- 你负责左半边，朋友负责右半边
- **最后要把两半拼起来** (All-Reduce)
- 输入数据大家都有一份完整的 (需要先复制)

### 2.3 Row-wise Parallel (行切分)

**就像横着切桌子**：

```
原始桌子：        切成两半后：
┌─────────┐      ┌─────────┐ GPU 0
│         │      │  上半   │
│  完整   │  →   │  桌面   │
│  桌面   │      ├─────────┤
│         │      │  下半   │ GPU 1
└─────────┘      │  桌面   │
                 └─────────┘
```

**特点**：
- 你负责上半部分，朋友负责下半部分
- **输入数据要先切分** (Scatter)
- **最后把结果拼起来** (All-Gather 或保持切分状态)

### 2.4 实际搬桌子的流程

假设有一个 FFN 层：`output = W2 @ (silu(W1 @ x) * W3 @ x)`

```
第1步：W1 和 W3 列切分（竖着切）
    输入 x 是完整的 [batch, seq, dim]
    ┌────┐              ┌────┐
    │ W1 │  GPU 0       │ W3 │  GPU 0
    │左半│              │左半│
    ├────┤          +   ├────┤
    │ W1 │  GPU 1       │ W3 │  GPU 1
    │右半│              │右半│
    └────┘              └────┘
    结果是切分的 [batch, seq, dim/2]

第2步：每个 GPU 算自己的激活
    GPU 0: silu(W1_left @ x) * W3_left @ x
    GPU 1: silu(W1_right @ x) * W3_right @ x

第3步：W2 行切分（横着切）
    ┌─────────┐ GPU 0
    │ W2 上半 │
    ├─────────┤
    │ W2 下半 │ GPU 1
    └─────────┘
    输入已经是切分的 [batch, seq, dim/2]

第4步：All-Reduce 汇总结果
    GPU 0 和 GPU 1 各算一半，然后加起来
```

---

## 3. Tensor Parallel 的三种模式

### 3.1 ColwiseParallel (列并行)

**权重矩阵按列切分**

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:169-173

"output": ColwiseParallel(
    input_layouts=Shard(1),      # 输入在 seq 维度已经是切分的
    output_layouts=Shard(-1),    # 输出在最后一个维度切分
    use_local_output=False,      # 是否保持切分状态
)
```

**数学原理**：
```
原始: Y = X @ W,  W: [Din, Dout]

切分后:
W = [W1 | W2]  (竖着切)
Y = X @ [W1 | W2] = [X @ W1 | X @ W2]

GPU 0: Y1 = X @ W1  → 得到 Dout/2 列
GPU 1: Y2 = X @ W2  → 得到 Dout/2 列
```

**通信**：
- **输入需要复制** (Replicate) 或者保持 Shard 状态
- **输出可以保持切分** 或做 All-Reduce

### 3.2 RowwiseParallel (行并行)

**权重矩阵按行切分**

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:164-167

"tok_embeddings": RowwiseParallel(
    input_layouts=Replicate(),   # 输入是完整的 token ids
    output_layouts=Shard(1),     # 输出在 seq 维度切分
)
```

**数学原理**：
```
原始: Y = X @ W,  W: [Din, Dout]

切分后:
    ┌ W1 ┐  (横着切)
W = │    │
    └ W2 ┘

Y = X @ W = [X1 | X2] @ W
  = X1 @ W1 + X2 @ W2  (需要 All-Reduce)

GPU 0: Y1 = X1 @ W1
GPU 1: Y2 = X2 @ W2
最终: Y = Y1 + Y2  (All-Reduce)
```

**通信**：
- **输入需要切分** (Shard)
- **输出需要 All-Reduce** 求和

### 3.3 SequenceParallel (序列并行)

**LayerNorm 在序列维度并行**

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:205,214

"attention_norm": SequenceParallel(),
"ffn_norm": SequenceParallel(),
```

**原理**：
- LayerNorm 在序列维度是**独立计算**的
- 每个 GPU 处理 `seq_len / TP` 个 token
- **不需要通信**，因为每个 token 的 norm 只依赖自己

**好处**：
- 减少内存占用
- 与 TP 的 Shard(1) 布局无缝衔接

---

## 4. 在 Transformer 中的应用

### 4.1 整体架构

```
Token IDs (完整)
    ↓
[Embedding - RowwiseParallel]
    ↓ 输出 Shard(1) - 在 seq 维度切分
┌─────────────────────────────┐
│   Transformer Block × N     │
│                             │
│  [LayerNorm - SeqParallel]  │
│         ↓ 保持 Shard(1)      │
│  [Attention - 混合模式]      │
│     wq, wk, wv: Colwise     │
│     wo: Rowwise             │
│         ↓ 输出 Shard(1)      │
│  [LayerNorm - SeqParallel]  │
│         ↓ 保持 Shard(1)      │
│  [FFN - 混合模式]           │
│     w1, w3: Colwise         │
│     w2: Rowwise             │
│         ↓ 输出 Shard(1)      │
└─────────────────────────────┘
    ↓ 保持 Shard(1)
[Final Norm - SeqParallel]
    ↓ 保持 Shard(1)
[Output Linear - ColwiseParallel]
    ↓
Logits (Shard(-1) 或 Replicate)
```

### 4.2 Attention 层的 TP 策略

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:206-213

layer_plan = {
    "attention": prepare_module_input(
        input_layouts=(Shard(1), None, None),        # x 在 seq 维度切分
        desired_input_layouts=(Replicate(), None, None),  # 需要完整的 x
    ),
    "attention.wq": colwise_parallel(),   # Q 投影按列切分
    "attention.wk": colwise_parallel(),   # K 投影按列切分
    "attention.wv": colwise_parallel(),   # V 投影按列切分
    "attention.wo": rowwise_parallel(output_layouts=Shard(1)),  # O 投影按行切分
}
```

**为什么这样设计？**

1. **wq, wk, wv 列切分**：
   ```
   原始: Q = X @ Wq, K = X @ Wk, V = X @ Wv
         Wq: [dim, n_heads * head_dim]

   切分后 (TP=2):
   GPU 0: Q0 = X @ Wq[:, :n_heads/2 * head_dim]  → 前一半 heads
   GPU 1: Q1 = X @ Wq[:, n_heads/2 * head_dim:]  → 后一半 heads

   每个 GPU 负责一部分 attention heads
   ```

2. **Multi-Head Attention 并行**：
   ```python
   # 来自: torchtitan/models/llama3/model/model.py:229-237

   bs, seqlen, _ = x.shape
   xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

   # 使用 -1 推断实际的 local heads 数量
   # TP 会自动把 heads 切分
   xq = xq.view(bs, seqlen, -1, self.head_dim)  # -1 会变成 n_heads/TP
   xk = xk.view(bs, seqlen, -1, self.head_dim)  # -1 会变成 n_kv_heads/TP
   xv = xv.view(bs, seqlen, -1, self.head_dim)
   ```

   **关键设计**：
   - 每个 GPU 处理 `n_heads / TP` 个完整的 attention heads
   - 不需要跨 GPU 通信来计算 attention
   - heads 之间是独立的，完美适合并行

3. **wo 行切分**：
   ```
   输入: Attention 输出 [batch, seq, n_heads/TP, head_dim]
        reshape 成 [batch, seq, (n_heads/TP) * head_dim]

   Wo 按行切分:
   GPU 0: Y0 = X0 @ Wo_upper  (X0 是前一半 heads 的输出)
   GPU 1: Y1 = X1 @ Wo_lower  (X1 是后一半 heads 的输出)

   最终: Y = Y0 + Y1  (All-Reduce)
   ```

   **All-Reduce 位置**：在 wo 之后，把各个 GPU 的结果加起来

### 4.3 FFN 层的 TP 策略

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:215-221

"feed_forward": prepare_module_input(
    input_layouts=(Shard(1),),           # 输入在 seq 维度切分
    desired_input_layouts=(Replicate(),), # 需要完整的 x
),
"feed_forward.w1": colwise_parallel(),   # Gate 投影按列切分
"feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),  # Down 投影按行切分
"feed_forward.w3": colwise_parallel(),   # Up 投影按列切分
```

**FFN 公式**：
```python
# 来自: torchtitan/models/llama3/model/model.py:310-311

def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**并行化分解**：

```
第1步：w1 和 w3 列切分
  原始维度: x [batch, seq, dim]
           w1, w3: [dim, hidden_dim]

  切分后:
  GPU 0: w1_left [dim, hidden_dim/2], w3_left [dim, hidden_dim/2]
  GPU 1: w1_right [dim, hidden_dim/2], w3_right [dim, hidden_dim/2]

  计算:
  GPU 0: gate0 = silu(x @ w1_left)  → [batch, seq, hidden_dim/2]
         up0 = x @ w3_left          → [batch, seq, hidden_dim/2]
         h0 = gate0 * up0

  GPU 1: gate1 = silu(x @ w1_right)
         up1 = x @ w3_right
         h1 = gate1 * up1

第2步：w2 行切分
  w2: [hidden_dim, dim]
  切分后:
  GPU 0: w2_upper [hidden_dim/2, dim]
  GPU 1: w2_lower [hidden_dim/2, dim]

  计算:
  GPU 0: out0 = h0 @ w2_upper  → [batch, seq, dim]
  GPU 1: out1 = h1 @ w2_lower  → [batch, seq, dim]

第3步：All-Reduce
  output = out0 + out1
```

**为什么这样设计？**
1. **w1, w3 列切分**：hidden_dim 很大（通常是 dim 的 4 倍），切分可以显著减少内存
2. **激活函数在本地计算**：`silu` 和乘法都在切分后的维度上，不需要通信
3. **w2 行切分 + All-Reduce**：自然地汇总各个 GPU 的结果

### 4.4 Embedding 和 Output 的 TP 策略

**Embedding 层**：
```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:164-167

"tok_embeddings": RowwiseParallel(
    input_layouts=Replicate(),    # token ids 在所有 GPU 上都一样
    output_layouts=Shard(1),      # 输出在 seq 维度切分
)
```

**为什么用 RowwiseParallel？**
- Embedding 表是 `[vocab_size, dim]`
- 按行切分相当于把词表分成几份
- 每个 GPU 负责一部分词的 embedding
- **但是输出不做 All-Reduce**！而是保持 Shard(1) 状态

**这是一个巧妙的优化**：
```
假设 vocab_size = 50000, dim = 4096, TP = 2

按行切分 embedding:
GPU 0: embedding[0:25000, :]     → [25000, 4096]
GPU 1: embedding[25000:50000, :] → [25000, 4096]

查表过程:
输入 token_ids = [1, 2, 100, 30000, ...]

GPU 0: 查表 embedding[token_ids, :] → 只有 1, 2, 100 有值，30000 = 0
GPU 1: 查表 embedding[token_ids, :] → 只有 30000 有值，1, 2, 100 = 0

合并: 每个位置只有一个 GPU 有非零值，所以可以直接 "拼接" 而不是求和
```

**但实际上**：RowwiseParallel 内部会做 **All-Reduce**，因为 embedding 查表在数学上等价于矩阵乘法。

**Output 层**：
```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:169-173

"output": ColwiseParallel(
    input_layouts=Shard(1),                    # 输入在 seq 维度切分
    output_layouts=Shard(-1) if loss_parallel else Replicate(),
    use_local_output=not loss_parallel,
)
```

**两种模式**：

1. **Loss Parallel (loss_parallel=True)**：
   - 输出保持 Shard(-1)，在 vocab 维度切分
   - 每个 GPU 计算一部分 vocab 的 logits
   - Loss 计算也并行化（需要特殊的 loss 实现）

2. **普通模式 (loss_parallel=False)**：
   - 输出 Replicate，做 All-Reduce 得到完整的 logits
   - 所有 GPU 都有完整的 vocab logits
   - Loss 计算在本地

---

## 5. 源码实现详解

### 5.1 核心入口：apply_tp 函数

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:149-234

def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
):
    """Apply tensor parallelism."""

    # 1. 并行化顶层模块
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(...),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(...),
        },
    )

    # 2. 选择并行策略（普通或 float8）
    if enable_float8_tensorwise_tp:
        rowwise_parallel = Float8RowwiseParallel
        colwise_parallel = Float8ColwiseParallel
        prepare_module_input = PrepareFloat8ModuleInput
    else:
        rowwise_parallel = RowwiseParallel
        colwise_parallel = ColwiseParallel
        prepare_module_input = PrepareModuleInput

    # 3. 对每个 Transformer Block 应用 TP
    for transformer_block in model.layers.values():
        layer_plan = {
            # Attention 部分
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(...),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),

            # FFN 部分
            "ffn_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(...),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )
```

**关键函数**：
- `parallelize_module`：PyTorch 提供的 TP API
- `prepare_module_input`：处理输入布局转换（Shard → Replicate）

### 5.2 Device Mesh 的构建

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

    # 例如: dims=[2, 4], names=["dp_shard", "tp"]
    # 表示 8 个 GPU，2路 FSDP，4路 TP
    mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

    return mesh
```

**Device Mesh 示例**：
```
假设 8 个 GPU，配置为 dp_shard=2, tp=4

mesh = [[GPU0, GPU1, GPU2, GPU3],   # dp_shard group 0
        [GPU4, GPU5, GPU6, GPU7]]   # dp_shard group 1

tp_mesh = mesh["tp"]
  → GPU0-1-2-3 是一个 TP group
  → GPU4-5-6-7 是另一个 TP group

每个 TP group 内做 Tensor Parallel
不同 TP group 之间是 Data Parallel
```

### 5.3 PrepareModuleInput 的作用

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:206-209

"attention": prepare_module_input(
    input_layouts=(Shard(1), None, None),        # 实际的输入布局
    desired_input_layouts=(Replicate(), None, None),  # 期望的输入布局
)
```

**作用**：插入通信操作，转换 tensor 布局

**工作流程**：
1. 前一层输出是 `Shard(1)` - 在 seq 维度切分
2. Attention 需要 `Replicate()` - 完整的输入
3. `prepare_module_input` 自动插入 **All-Gather** 操作

**相当于**：
```python
# 伪代码
x = previous_layer(x)  # 输出 Shard(1)
x = all_gather(x, dim=1)  # 转换为 Replicate
output = attention(x)  # 在完整的 x 上计算
```

### 5.4 TP 中的通信原语

**1. All-Gather**：
```
每个 GPU 有一部分数据，收集所有部分得到完整数据

输入 (TP=2):
GPU 0: [1, 2]
GPU 1: [3, 4]

All-Gather 后:
GPU 0: [1, 2, 3, 4]
GPU 1: [1, 2, 3, 4]
```

**2. Reduce-Scatter**：
```
每个 GPU 有完整数据，求和后每个 GPU 保留一部分

输入 (TP=2):
GPU 0: [1, 2, 3, 4]
GPU 1: [5, 6, 7, 8]

Reduce-Scatter 后:
GPU 0: [1+5, 2+6] = [6, 8]
GPU 1: [3+7, 4+8] = [10, 12]
```

**3. All-Reduce**：
```
All-Reduce = Reduce-Scatter + All-Gather
每个 GPU 有完整数据，求和后每个 GPU 都有完整的和

输入 (TP=2):
GPU 0: [1, 2, 3, 4]
GPU 1: [5, 6, 7, 8]

All-Reduce 后:
GPU 0: [6, 8, 10, 12]
GPU 1: [6, 8, 10, 12]
```

**在 TP 中的使用**：
- **ColwiseParallel → RowwiseParallel**：输出保持 Shard，下一层输入是 Shard，无通信
- **RowwiseParallel 输出 Replicate**：做 All-Reduce
- **Shard → Replicate**：做 All-Gather

---

## 6. 通信优化

### 6.1 通信与计算重叠

TorchTitan 支持 **Async TP**（异步张量并行）：

```python
# 来自: torchtitan/distributed/tensor_parallel.py:15-29

def maybe_enable_async_tp(job_config: JobConfig, tp_mesh: DeviceMesh):
    if not job_config.parallelism.enable_async_tensor_parallel:
        return

    # 需要启用 compile
    if not (job_config.compile.enable and "model" in job_config.compile.components):
        raise RuntimeError(
            "Async TP requires 'model' in --compile.components"
        )

    # 启用 micro-pipeline 和对称内存
    torch._inductor.config._micro_pipeline_tp = True
    enable_symm_mem_for_group(tp_mesh.get_group().group_name)
```

**原理**：
- **Micro-pipeline**：把一个 layer 的计算分成多个小块
- **通信与计算重叠**：在计算 chunk N 时，启动 chunk N+1 的通信
- **对称内存**：GPU 之间共享内存空间，减少拷贝

**效果**：
- 减少 All-Reduce 的延迟
- 提升 TP 的扩展效率

### 6.2 Sequence Parallel 优化

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:205,214

"attention_norm": SequenceParallel(),
"ffn_norm": SequenceParallel(),
```

**好处**：
1. **减少内存**：LayerNorm 的激活在序列维度切分
2. **无额外通信**：因为 LayerNorm 在 token 级别独立计算
3. **与 TP 布局一致**：前后都是 Shard(1)，无需转换

**配合使用**：
```
RowwiseParallel (wo) → Shard(1) 输出
    ↓ (无通信)
SequenceParallel (ffn_norm) → 在 Shard(1) 上直接计算
    ↓ (需要 All-Gather)
PrepareModuleInput → 转换为 Replicate
    ↓
ColwiseParallel (w1, w3) → 在 Replicate 上计算
```

### 6.3 TP 通信量分析

假设：
- 模型维度：`dim = 4096`
- TP 并行度：`TP = 4`
- Batch size：`B = 8`
- Sequence length：`S = 2048`
- Transformer layers：`L = 32`

**单个 Transformer Block 的通信量**：

1. **Attention 部分**：
   - `prepare_module_input` (Attention)：All-Gather on `[B, S, dim]`
     - 通信量 = `B * S * dim * sizeof(dtype)` = 8 * 2048 * 4096 * 2 = 128 MB
   - `wo` (RowwiseParallel)：All-Reduce on `[B, S, dim]`
     - 通信量 = `B * S * dim * sizeof(dtype)` = 128 MB

2. **FFN 部分**：
   - `prepare_module_input` (FFN)：All-Gather on `[B, S, dim]`
     - 通信量 = 128 MB
   - `w2` (RowwiseParallel)：All-Reduce on `[B, S, dim]`
     - 通信量 = 128 MB

**单个 Block 总通信量**：`4 * 128 MB = 512 MB`

**整个模型**：`512 MB * 32 layers = 16 GB`

**优化方向**：
- 使用 Async TP 重叠通信和计算
- 增大 batch size 和 sequence length，提高计算/通信比

### 6.4 Loss Parallel

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:169-173

"output": ColwiseParallel(
    input_layouts=Shard(1),
    output_layouts=Shard(-1) if loss_parallel else Replicate(),
    use_local_output=not loss_parallel,
)
```

**普通模式 (loss_parallel=False)**：
```
output 层做 All-Reduce → 所有 GPU 有完整的 [B, S, vocab_size] logits
每个 GPU 独立计算 loss
```
- **通信量大**：vocab_size 通常很大（32000 - 128000）
- **冗余计算**：所有 GPU 都算相同的 loss

**Loss Parallel 模式 (loss_parallel=True)**：
```
output 层保持 Shard(-1) → 每个 GPU 有 [B, S, vocab_size/TP] logits
每个 GPU 计算部分 vocab 的 loss contribution
最后 All-Reduce loss 的梯度
```
- **通信量小**：只通信 loss 的梯度（标量或小向量）
- **内存节省**：不需要存完整的 logits

**适用场景**：
- 大词表模型（vocab_size > 100k）
- 高 TP 并行度（TP >= 4）

---

## 7. 总结

### 7.1 TP 的核心思想

用**搬桌子**的比喻总结：

1. **列切分 (Colwise)**：竖着切桌子，每人搬一部分，最后**拼起来**
2. **行切分 (Rowwise)**：横着切桌子，先把输入**分开**，每人搬自己的部分，最后**加起来**
3. **序列并行 (Sequence)**：每人负责一段序列，**独立处理**，不需要交流

### 7.2 TP 在 Transformer 中的黄金组合

```
Attention:
  wq, wk, wv (Colwise) → 切分 heads
  wo (Rowwise) → 汇总 heads

FFN:
  w1, w3 (Colwise) → 切分 hidden_dim
  w2 (Rowwise) → 汇总结果
```

**设计原则**：
- **独立计算尽量并行**：Attention heads 独立，FFN 隐藏层独立
- **减少通信次数**：Colwise → Rowwise 可以保持 Shard，减少通信
- **保持 Shard(1) 布局**：在序列维度切分，与 Sequence Parallel 协同

### 7.3 TP 的优缺点

**优点**：
- ✅ 可以处理**单层放不进一个 GPU** 的超大模型
- ✅ 与 FSDP、PP 可以**组合使用**，实现 3D 并行
- ✅ **通信量可控**：只在 TP group 内通信，不跨节点

**缺点**：
- ❌ **需要高速互联**：TP group 内需要 NVLink/IB，否则通信成为瓶颈
- ❌ **代码侵入性**：需要手动标注哪些层用 Colwise/Rowwise
- ❌ **调试困难**：多 GPU 并行，需要理解分布式语义

### 7.4 何时使用 TP？

**推荐使用 TP**：
- 单卡放不下模型（模型 > 80GB for A100）
- 有高速 GPU 互联（NVLink, NVSwitch）
- 需要与 FSDP 组合使用

**不推荐使用 TP**：
- 模型可以用 FSDP 放进单卡
- 只有 PCIe 连接的 GPU
- 通信带宽受限的环境

**最佳实践**：
```
小模型 (< 10B):  只用 FSDP
中模型 (10B - 70B):  FSDP + TP (TP=2 or 4)
大模型 (> 70B):  FSDP + TP + PP (3D 并行)
```

### 7.5 相关配置

```toml
# 来自: 任意 train_configs/*.toml

[parallelism]
# TP 并行度
tp = 4

# 是否启用 Async TP (需要 compile)
enable_async_tensor_parallel = true

# 是否禁用 Loss Parallel
disable_loss_parallel = false
```

**经验值**：
- **TP = 1**：不使用 TP
- **TP = 2, 4**：适合单机多卡（8x A100/H100）
- **TP = 8**：适合 NVSwitch 连接的大型节点
- **TP > 8**：很少使用，通信开销大

---

## 8. 参考资料

**源码文件**：
- `torchtitan/distributed/tensor_parallel.py` - TP 入口和 Async TP
- `torchtitan/distributed/parallel_dims.py` - 并行维度管理
- `torchtitan/models/llama3/infra/parallelize.py` - Llama3 的 TP 应用
- `torchtitan/models/llama3/model/model.py` - Transformer 模型定义

**PyTorch 官方文档**：
- [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [DTensor](https://pytorch.org/docs/stable/distributed.tensor.html)

**相关论文**：
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism
