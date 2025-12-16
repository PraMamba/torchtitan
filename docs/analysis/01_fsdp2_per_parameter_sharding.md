# FSDP2 Per-Parameter Sharding 实现详解

## 目录
- [1. 什么是 FSDP2？](#1-什么是-fsdp2)
- [2. 搬桌子的比喻：从 TP 到 FSDP](#2-搬桌子的比喻从-tp-到-fsdp)
- [3. FSDP2 的核心创新：Per-Parameter Sharding](#3-fsdp2-的核心创新per-parameter-sharding)
- [4. Reshard After Forward 策略](#4-reshard-after-forward-策略)
- [5. 源码实现详解](#5-源码实现详解)
- [6. 内存管理优化](#6-内存管理优化)
- [7. 与其他并行策略的组合](#7-与其他并行策略的组合)

---

## 1. 什么是 FSDP2？

### 1.1 基本概念

**FSDP2 (Fully Sharded Data Parallel 2)** 是 PyTorch 对 FSDP 的重写版本，它将**模型参数、梯度、优化器状态**全部切分到多个 GPU 上。

**核心思想**：在 Data Parallel 的基础上，不仅切分数据，还切分模型的所有状态。

### 1.2 为什么需要 FSDP？

传统的 Data Parallel (DDP) 有个问题：

```
假设你有一个 70B 参数的模型，fp16 需要 140GB 内存

DDP 的做法：
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     GPU 0       │  │     GPU 1       │  │     GPU 2       │
│  完整模型 140GB │  │  完整模型 140GB │  │  完整模型 140GB │
│  不同的数据     │  │  不同的数据     │  │  不同的数据     │
└─────────────────┘  └─────────────────┘  └─────────────────┘

问题：每个 GPU 都存完整的模型，内存浪费！
```

FSDP 的改进：

```
FSDP 的做法（8 GPUs）：
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     GPU 0       │  │     GPU 1       │  │     GPU 7       │
│  1/8 模型 17.5GB│  │  1/8 模型 17.5GB│  │  1/8 模型 17.5GB│
│  不同的数据     │  │  不同的数据     │  │  不同的数据     │
└─────────────────┘  └─────────────────┘  └─────────────────┘

好处：每个 GPU 只存 1/8 的参数，内存大幅降低！
```

### 1.3 FSDP1 vs FSDP2：关键区别

| 特性 | FSDP1 | FSDP2 |
|-----|-------|-------|
| **参数表示** | `FlatParameter`（扁平化） | `DTensor`（保持原始结构） |
| **参数管理** | 所有参数压缩成一个大 tensor | 每个参数独立管理 |
| **State Dict** | 需要通信，复杂的重构逻辑 | 无需通信，直接保存 |
| **Composability** | 差（FlatParameter 难以处理） | 好（每个参数可独立操作） |
| **Meta Init** | 需要 `param_init_fn` | `to_empty()` 后初始化 |
| **内存管理** | 使用 `recordStream`（不确定性） | 自定义系统（确定性，更低内存） |

**最重要的改进**：FSDP2 的 **Per-Parameter Sharding** - 每个参数单独分片，而不是打包成大块。

---

## 2. 搬桌子的比喻：从 TP 到 FSDP

### 2.1 回顾 TP：单层的桌子切分

还记得 [TP 文档](./02_tensor_parallel_implementation.md) 中搬桌子的比喻吗？

**Tensor Parallel**：把**单张桌子**（一层的权重矩阵）切分

```
单个 Linear 层的权重矩阵（桌子）：
┌─────────────────┐
│                 │
│   W [4096x4096] │  → 竖着切成 4 份（TP=4）
│                 │
└─────────────────┘

每个 GPU 负责一部分：
GPU0: W[:, 0:1024]     (左 1/4)
GPU1: W[:, 1024:2048]  (左中 1/4)
GPU2: W[:, 2048:3072]  (右中 1/4)
GPU3: W[:, 3072:4096]  (右 1/4)
```

### 2.2 FSDP：整个房子的桌子切分

如果 TP 是切分"单张桌子"，那么 **FSDP 就是切分"整个房子里的所有桌子"**。

想象你有一个巨大的房子（Transformer 模型），里面有很多房间（层），每个房间有多张桌子（参数矩阵）。

```
Transformer 模型（房子）：
┌────────────────────────────────────┐
│  Embedding 层（玄关）              │
│    ├─ tok_embeddings 桌子          │
├────────────────────────────────────┤
│  TransformerBlock 0（第1间房）     │
│    ├─ attention.wq 桌子            │
│    ├─ attention.wk 桌子            │
│    ├─ attention.wv 桌子            │
│    ├─ attention.wo 桌子            │
│    ├─ ffn.w1 桌子                  │
│    ├─ ffn.w2 桌子                  │
│    └─ ffn.w3 桌子                  │
├────────────────────────────────────┤
│  TransformerBlock 1（第2间房）     │
│    └─ ... （同样的桌子）            │
├────────────────────────────────────┤
│  ... 重复 32 次                    │
├────────────────────────────────────┤
│  Output 层（后院）                 │
│    ├─ norm 桌子                    │
│    └─ output 桌子                  │
└────────────────────────────────────┘
```

### 2.3 FSDP1 的搬法：打包搬运（FlatParameter）

**FSDP1 的策略**：把每个房间的所有桌子**绑在一起**，变成一个大包裹。

```
TransformerBlock 0 的所有参数 → 打包成 FlatParameter

原来：
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ wq   │ │ wk   │ │ wv   │ │ wo   │  （7 张独立的桌子）
└──────┘ └──────┘ └──────┘ └──────┘
┌──────┐ ┌──────┐ ┌──────┐
│ w1   │ │ w2   │ │ w3   │
└──────┘ └──────┘ └──────┘

FSDP1 打包后：
┌────────────────────────────────────┐
│  [wq | wk | wv | wo | w1 | w2 | w3] │  （一个大 tensor）
└────────────────────────────────────┘
      FlatParameter (全部压平)

然后切分成 4 份（假设 FSDP = 4）：
GPU0: FlatParam[0:N/4]
GPU1: FlatParam[N/4:N/2]
GPU2: FlatParam[N/2:3N/4]
GPU3: FlatParam[3N/4:N]
```

**问题**：
- ❌ wq、wk 等参数的**边界信息丢失**
- ❌ 无法单独操作某个参数（比如冻结 wq）
- ❌ 保存 checkpoint 时需要复杂的重构逻辑

### 2.4 FSDP2 的搬法：逐个切分（Per-Parameter）

**FSDP2 的策略**：**每张桌子单独切分**，保持桌子的原始形状。

```
TransformerBlock 0 的参数 → 每个参数独立切分

原来（每张桌子完整）：
GPU0: ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
      │ wq   │ │ wk   │ │ wv   │ │ wo   │
      └──────┘ └──────┘ └──────┘ └──────┘
      ┌──────┐ ┌──────┐ ┌──────┐
      │ w1   │ │ w2   │ │ w3   │
      └──────┘ └──────┘ └──────┘

FSDP2 切分后（每张桌子切成 4 份，FSDP=4）：
GPU0: ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐  (每张桌子的上 1/4)
      │wq│ │wk│ │wv│ │wo│ │w1│ │w2│ │w3│
      │¼ │ │¼ │ │¼ │ │¼ │ │¼ │ │¼ │ │¼ │
      └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘

GPU1: ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐  (每张桌子的 2/4)
      │wq│ │wk│ │wv│ │wo│ │w1│ │w2│ │w3│
      │¼ │ │¼ │ │¼ │ │¼ │ │¼ │ │¼ │ │¼ │
      └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘

GPU2: 同理 (3/4)
GPU3: 同理 (4/4)
```

**关键特点**：
- ✅ 每个参数**保持原始形状**（wq 还是 wq，不会混到其他参数里）
- ✅ 可以**单独操作**每个参数
- ✅ Checkpoint 保存时，直接保存分片，**无需通信**

### 2.5 DTensor：魔法标签

FSDP2 用 **DTensor** 来表示分片的参数。DTensor 就像给每张桌子贴了个**魔法标签**：

```
普通 Tensor:
wq = Tensor([4096, 4096])  # 完整的权重矩阵

DTensor (FSDP2):
wq = DTensor(
    local_tensor=Tensor([1024, 4096]),  # 本地只存 1/4
    device_mesh=DeviceMesh([GPU0, GPU1, GPU2, GPU3]),
    placements=[Shard(0)],  # 在第 0 维切分
)

魔法能力：
- wq.shape  → 返回 [4096, 4096]  (全局形状)
- wq.to_local()  → 返回 [1024, 4096]  (本地分片)
- wq.full_tensor()  → All-Gather，得到完整的 [4096, 4096]
```

这个"魔法标签"让你可以**像使用完整 tensor 一样使用分片 tensor**！

---

## 3. FSDP2 的核心创新：Per-Parameter Sharding

### 3.1 什么是 Per-Parameter Sharding？

**Per-Parameter Sharding** = 每个参数单独分片，独立管理。

对比：

```
FSDP1 (FlatParameter):
┌────────────────────────────────┐
│ TransformerBlock 参数全部打包   │  → 一个通信 bucket
└────────────────────────────────┘

FSDP2 (Per-Parameter):
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ wq  │ │ wk  │ │ wv  │ │ wo  │  → 每个参数单独
└─────┘ └─────┘ └─────┘ └─────┘
┌─────┐ ┌─────┐ ┌─────┐
│ w1  │ │ w2  │ │ w3  │  → 但通信时可以合并
└─────┘ └─────┘ └─────┘
```

### 3.2 参数在内存中的布局

假设一个 TransformerBlock 有 7 个参数，FSDP degree = 4：

```
GPU 0 的内存布局：
┌────────────────────────────────────────┐
│  Sharded Parameters (分片参数)         │
│  ├─ wq: DTensor[1024, 4096] Shard(0)   │
│  ├─ wk: DTensor[256, 4096]  Shard(0)   │  (GQA: n_kv_heads < n_heads)
│  ├─ wv: DTensor[256, 4096]  Shard(0)   │
│  ├─ wo: DTensor[4096, 1024] Shard(1)   │
│  ├─ w1: DTensor[3584, 4096] Shard(0)   │  (hidden_dim = 14336/4)
│  ├─ w2: DTensor[4096, 3584] Shard(1)   │
│  └─ w3: DTensor[3584, 4096] Shard(0)   │
└────────────────────────────────────────┘

GPU 1/2/3: 同样的结构，但存储不同的分片
```

**关键点**：
1. 每个参数都是 **DTensor**，有自己的 placement 信息
2. 所有参数默认在 **dim-0** 切分（`Shard(0)`）
3. 每个 GPU 只存 **1/N** 的参数

### 3.3 Forward 时的通信模式

FSDP2 的 forward 分 3 步：

```
步骤 1: All-Gather 收集参数
   每个 GPU 只有 1/4 的参数 → All-Gather → 所有 GPU 都有完整参数

GPU0  GPU1  GPU2  GPU3         GPU0  GPU1  GPU2  GPU3
 wq¼   wq¼   wq¼   wq¼          wq    wq    wq    wq
  │     │     │     │            │     │     │     │
  └─────┴─────┴─────┘            │     │     │     │
         ↓                       ↓     ↓     ↓     ↓
    All-Gather              完整的参数，可以计算了！

步骤 2: 本地计算
   output = layer(input)  # 每个 GPU 用完整的参数计算

步骤 3: 释放参数（Reshard）
   丢弃刚才 All-Gather 来的部分，只保留自己的 1/4

GPU0  GPU1  GPU2  GPU3
 wq    wq    wq    wq
  ↓     ↓     ↓     ↓
 wq¼   wq¼   wq¼   wq¼   (释放 3/4，只保留 1/4)
```

**为什么要 Reshard？**
- **节省内存**：计算完就释放，不占用显存
- **Trade-off**：Backward 时需要再次 All-Gather

### 3.4 Backward 时的通信模式

Backward 也是 3 步：

```
步骤 1: All-Gather 参数（重新收集）
   Backward 需要用到完整的参数，再做一次 All-Gather

步骤 2: 本地计算梯度
   grad = backward(output_grad)  # 计算得到完整的梯度

步骤 3: Reduce-Scatter 梯度
   完整的梯度 → Reduce-Scatter → 每个 GPU 保留 1/4

GPU0     GPU1     GPU2     GPU3
grad_wq  grad_wq  grad_wq  grad_wq  (完整梯度)
  │        │        │        │
  └────────┴────────┴────────┘
              ↓
      Reduce-Scatter (求和 + 切分)
              ↓
  ┌────────┬────────┬────────┬────────┐
  │        │        │        │        │
grad¼    grad¼    grad¼    grad¼     (每个 GPU 的梯度)
GPU0     GPU1     GPU2     GPU3
```

**Reduce-Scatter 的作用**：
- 把所有 GPU 的梯度**求和**（因为是 Data Parallel）
- 然后**切分**，每个 GPU 只保留 1/4

### 3.5 完整的训练循环

```
初始状态：
  每个 GPU 有 1/4 的参数 (sharded)

─────────── Forward ───────────
1. All-Gather 参数
   wq: [1024, 4096] → [4096, 4096]

2. 计算 forward
   output = layer(input)

3. Reshard（如果 reshard_after_forward=True）
   wq: [4096, 4096] → [1024, 4096]  (释放 3/4)

─────────── Backward ──────────
4. All-Gather 参数（如果之前释放了）
   wq: [1024, 4096] → [4096, 4096]

5. 计算 backward
   grad_wq = backward(...)

6. Reduce-Scatter 梯度
   grad_wq: [4096, 4096] → [1024, 4096]
   并且求和所有 GPU 的梯度

7. Reshard 参数
   wq: [4096, 4096] → [1024, 4096]

─────────── Optimizer ─────────
8. 本地更新参数
   wq[本地分片] -= lr * grad_wq[本地分片]

结束状态：
  每个 GPU 有更新后的 1/4 参数 (sharded)
```

---

## 4. Reshard After Forward 策略

### 4.1 什么是 Reshard After Forward？

**Reshard After Forward** = Forward 结束后，是否立即释放参数？

```
reshard_after_forward = True (默认):
Forward:  Shard → All-Gather → 计算 → Reshard → Shard
Backward: Shard → All-Gather → 计算 → Reduce-Scatter → Shard

reshard_after_forward = False:
Forward:  Shard → All-Gather → 计算 → [保持完整]
Backward: [完整] → 计算 → Reduce-Scatter → Shard

区别：是否在 Forward 和 Backward 之间保持参数完整
```

### 4.2 Trade-off 分析

**reshard_after_forward = True**（ZeRO-3）：

```
优点：
✅ 内存占用低：Forward 后立即释放，只存激活值
✅ 适合大模型：可以训练更大的模型

缺点：
❌ 通信量大：Backward 需要再次 All-Gather
❌ 延迟高：多一次通信

时间线：
Forward:  [AG] → [Compute] → [Free]
          ----    --------    -----
Backward:              [AG] → [Compute] → [RS]
                       ----    --------    ----
                        ↑
                 额外的 All-Gather！
```

**reshard_after_forward = False**（ZeRO-2）：

```
优点：
✅ 通信量小：Backward 不需要 All-Gather
✅ 延迟低：少一次通信

缺点：
❌ 内存占用高：Forward 和 Backward 之间保持完整参数
❌ 不适合大模型：内存可能不够

时间线：
Forward:  [AG] → [Compute] → [Keep in memory]
          ----    --------
Backward:                     [Compute] → [RS]
                              --------    ----
                              无需 All-Gather！
```

### 4.3 TorchTitan 的智能策略

TorchTitan 使用**动态策略**，根据是否启用 Pipeline Parallel 自动选择：

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:281-293

match reshard_after_forward_policy:
    case "always":
        reshard_after_forward = True
    case "never":
        reshard_after_forward = False
    case "default":
        # 智能决策：
        # PP 启用时 → reshard_after_forward = False
        # PP 禁用时 → reshard_after_forward = True
        reshard_after_forward = not pp_enabled
```

**为什么 PP 时选择 False？**

Pipeline Parallel 有 microbatch，每个 microbatch 都要做 forward + backward：

```
PP 启用 + reshard_after_forward=True:
Microbatch 1: [AG] → [Fwd] → [Free] → [AG] → [Bwd] → [RS]
Microbatch 2: [AG] → [Fwd] → [Free] → [AG] → [Bwd] → [RS]
Microbatch 3: [AG] → [Fwd] → [Free] → [AG] → [Bwd] → [RS]
               ↑↑↑↑↑          非常多的 All-Gather！↑↑↑↑

PP 启用 + reshard_after_forward=False:
Microbatch 1: [AG] → [Fwd] → [Keep] → [Bwd] → [RS]
Microbatch 2:        [Fwd] → [Keep] → [Bwd]        (复用参数)
Microbatch 3:        [Fwd] → [Keep] → [Bwd]        (复用参数)
               ↑     只需要一次 All-Gather！
```

**优化**：最后一层（norm + output）总是 `reshard_after_forward=False`，因为 FSDP 会立即 prefetch。

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:307-314

# 优化：最后的 norm + output 不需要 reshard
fully_shard(
    [model.norm, model.output],
    **fsdp_config,
    reshard_after_forward=reshard_after_forward_policy == "always",
    # 注意：只有明确设置 "always" 才会 reshard
)
```

### 4.4 内存占用对比

假设 Llama3 70B 模型，bf16，FSDP=8：

```
参数量：70B
参数内存：70B × 2 bytes = 140 GB

每个 GPU 的参数分片：140 GB / 8 = 17.5 GB

reshard_after_forward = True:
  Forward 峰值：17.5 GB (sharded) + 激活值
  Backward 峰值：17.5 GB (sharded) + 激活值

reshard_after_forward = False:
  Forward 峰值：140 GB (full) + 激活值  😱
  Backward 峰值：140 GB (full) + 激活值

差距：140 GB - 17.5 GB = 122.5 GB！
```

**结论**：对于大模型，`reshard_after_forward=True` 是必须的。

---

## 5. 源码实现详解

### 5.1 核心入口：apply_fsdp 函数

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:250-316

def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """应用 FSDP2 到模型"""

    # 1. 配置 Mixed Precision
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,    # 参数存储精度（如 bf16）
        reduce_dtype=reduce_dtype,  # 梯度规约精度（如 fp32）
    )

    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # 2. 配置 CPU Offload（可选）
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # 3. 决定 reshard_after_forward 策略
    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # PP 启用时不 reshard，避免每个 microbatch 都 All-Gather
            reshard_after_forward = not pp_enabled

    # 4. 分层应用 fully_shard
    # 4.1 Embedding 层
    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # 4.2 每个 TransformerBlock
    for layer_id, transformer_block in model.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # 4.3 最后的 norm + output（优化：不 reshard）
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            # 只有明确 "always" 才 reshard
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    # 4.4 整个模型（root module）
    fully_shard(model, **fsdp_config)
```

### 5.2 fully_shard 的工作原理

`fully_shard` 是 FSDP2 的核心 API，来自 `torch.distributed.fsdp`：

```python
# PyTorch 官方 API

@contract(state_cls=FSDPState)
def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
) -> nn.Module:
    """
    对 module 应用 FSDP2

    关键特性：
    1. 将 module.parameters() 转换为 DTensor
    2. 在 dim-0 切分每个参数
    3. 注册 forward/backward hooks 来管理通信
    4. 动态 class swap（添加 FSDPModule 方法）
    """
```

**工作流程**：

```
输入: module (普通 nn.Module)

步骤 1: 参数转换
  for param in module.parameters():
      if param not in nested_fsdp_modules:
          param → DTensor(Shard(0))  # 在 dim-0 切分

步骤 2: 注册 Hooks
  module.register_forward_pre_hook(all_gather_hook)
  module.register_forward_hook(reshard_hook if reshard_after_forward)
  module.register_backward_hook(reduce_scatter_hook)

步骤 3: 动态 Class Swap
  原来: type(module) = TransformerBlock
  现在: type(module) = FSDPTransformerBlock (继承自 FSDPModule + TransformerBlock)

  FSDPModule 提供的方法：
  - set_requires_gradient_sync()
  - set_modules_to_forward_prefetch()
  - ...

输出: module (带 FSDP 功能的 nn.Module)
```

### 5.3 分层 Sharding 的策略

TorchTitan 采用**嵌套 FSDP**，每个 TransformerBlock 是一个 FSDP unit：

```
Model 层次结构：
Transformer
├─ tok_embeddings          ← fully_shard (Unit 1)
├─ layers[0]               ← fully_shard (Unit 2)
│   ├─ attention
│   │   ├─ wq
│   │   ├─ wk
│   │   ├─ wv
│   │   └─ wo
│   └─ feed_forward
│       ├─ w1
│       ├─ w2
│       └─ w3
├─ layers[1]               ← fully_shard (Unit 3)
│   └─ ...
├─ ...
├─ layers[31]              ← fully_shard (Unit 34)
├─ norm + output           ← fully_shard (Unit 35)
└─ (root)                  ← fully_shard (Unit 36)

每个 Unit 的参数独立分片，独立通信
```

**为什么要嵌套？**

1. **通信粒度**：TransformerBlock 是合适的通信单位（~100-500MB）
2. **内存管理**：每个 Block 完成后可以释放，内存占用更可控
3. **Prefetch**：可以在计算 Block N 时 prefetch Block N+1

### 5.4 DTensor 参数的生命周期

```
初始化阶段：
1. 模型在 meta device 创建
   with torch.device("meta"):
       model = Transformer(...)

   此时参数没有实际内存

2. 应用 TP（如果启用）
   apply_tp(model, tp_mesh, ...)

   参数变为 DTensor，但仍在 meta device

3. 应用 FSDP
   apply_fsdp(model, dp_mesh, ...)

   参数变为 DTensor，但仍在 meta device
   Placement: [Shard(0)] on dp_mesh

4. 分配实际内存
   model.to_empty(device="cuda")

   每个 GPU 分配 1/N 的参数内存

5. 初始化权重
   model.init_weights()

   在分片上初始化（每个 GPU 初始化自己的 1/N）

训练阶段：
6. Forward
   - All-Gather 参数
   - 计算
   - Reshard（可选）

7. Backward
   - All-Gather 参数（如果之前释放了）
   - 计算梯度
   - Reduce-Scatter 梯度

8. Optimizer
   - 本地更新分片参数
```

### 5.5 Mixed Precision Policy

FSDP2 支持灵活的混合精度：

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:276

mp_policy = MixedPrecisionPolicy(
    param_dtype=param_dtype,    # 参数存储的精度
    reduce_dtype=reduce_dtype,  # All-Reduce 的精度
)
```

**三种精度配置**：

```
配置 1: Full BF16
  param_dtype = bf16
  reduce_dtype = bf16

  优点: 内存占用最低
  缺点: 数值精度低，可能影响收敛

配置 2: Mixed Precision (推荐)
  param_dtype = bf16
  reduce_dtype = fp32

  优点: 内存低 + 梯度规约精度高
  缺点: All-Reduce 需要 upcast

配置 3: Full FP32
  param_dtype = fp32
  reduce_dtype = fp32

  优点: 精度最高
  缺点: 内存占用 2x
```

**实际执行**：

```
Forward (param_dtype = bf16):
  wq: DTensor[bf16]  → 计算用 bf16

Backward (reduce_dtype = fp32):
  grad_wq: Tensor[bf16]  → upcast to fp32
  All-Reduce (fp32)      → 高精度求和
  grad_wq: Tensor[fp32]  → downcast to bf16 (存储)
```

---

## 6. 内存管理优化

### 6.1 FSDP1 的问题：recordStream

FSDP1 使用 `recordStream` 来管理 GPU 内存：

```python
# FSDP1 伪代码

def all_gather_hook(module):
    # All-Gather 参数
    full_param = all_gather(sharded_param)

    # 记录到当前 stream
    full_param.record_stream(torch.cuda.current_stream())

    # 使用参数计算
    output = module(input, full_param)

    # full_param 什么时候释放？
    # → 由 CUDA caching allocator 决定（不确定！）
```

**问题**：
- ❌ **不确定性**：内存释放时间不确定
- ❌ **峰值内存高**：可能同时保留多个 layer 的参数
- ❌ **CPU 同步**：`limit_all_gathers` 需要 CPU 同步来限制内存

### 6.2 FSDP2 的改进：自定义内存管理

FSDP2 使用**确定性的内存管理**：

```python
# FSDP2 伪代码

class FSDPState:
    def __init__(self):
        self.param_buffer_pool = []  # 参数缓冲池

    def all_gather_hook(self, module):
        # 从缓冲池分配内存
        full_param = self.param_buffer_pool.allocate()

        # All-Gather
        all_gather(sharded_param, out=full_param)

        # 计算
        output = module(input, full_param)

        return output

    def reshard_hook(self, module, output):
        # 立即释放，放回缓冲池
        self.param_buffer_pool.free(full_param)

        # 确定性释放！无需等待 CUDA
```

**优点**：
- ✅ **确定性释放**：Forward 结束立即释放
- ✅ **内存复用**：缓冲池复用内存，减少分配
- ✅ **无 CPU 同步**：不需要 `limit_all_gathers`

### 6.3 内存占用对比

假设 Llama3 70B，bf16，8 GPUs，32 layers：

```
FSDP1 (recordStream):
  峰值内存 = 参数 + 梯度 + 优化器状态 + 激活 + ??? (不确定)

  可能的情况：
  - 好的情况: ~25 GB/GPU
  - 坏的情况: ~35 GB/GPU (多个 layer 的参数未释放)

FSDP2 (自定义管理):
  峰值内存 = 参数 + 梯度 + 优化器状态 + 激活 (确定)

  确定的值: ~22 GB/GPU

内存节省: ~3-13 GB/GPU (13%-37%)
```

**实际测试**（来自官方 PR）：

> Llama3 7B on 8x H100s:
> - FSDP1: 峰值 24.3 GB/GPU
> - FSDP2: 峰值 22.6 GB/GPU
> - **节省 7% 内存**

---

## 7. 与其他并行策略的组合

### 7.1 FSDP + TP (2D 并行)

最常见的组合：**FSDP 切分层，TP 切分单层权重**

```
假设 64 GPUs, DP=8, TP=8

Device Mesh:
         TP0  TP1  TP2  TP3  TP4  TP5  TP6  TP7
     ┌────────────────────────────────────────┐
DP0  │  0    1    2    3    4    5    6    7  │  ← FSDP Group 0
DP1  │  8    9   10   11   12   13   14   15  │  ← FSDP Group 1
DP2  │ 16   17   18   19   20   21   22   23  │  ← FSDP Group 2
...  │ ...                                 ... │
DP7  │ 56   57   58   59   60   61   62   63  │  ← FSDP Group 7
     └────────────────────────────────────────┘
     ↑                                         ↑
     TP Group 0                         TP Group 7
```

**参数分布**：

```
以 wq [4096, 4096] 为例：

第 1 步: TP 切分
  GPU 0-7:  wq[:, 0:512]    (TP 在列维度切，每个 GPU 512 列)
  GPU 8-15: wq[:, 0:512]    (同样的切分，不同的 FSDP group)
  ...

第 2 步: FSDP 切分（在 TP 的基础上）
  GPU 0:    wq[0:512, 0:512]      (TP 切分后的 1/8)
  GPU 8:    wq[512:1024, 0:512]   (FSDP 在行维度切)
  GPU 16:   wq[1024:1536, 0:512]
  ...
  GPU 56:   wq[3584:4096, 0:512]

每个 GPU 存储: 512 × 512 = 1/64 的参数！
```

**通信模式**：

```
Forward:
1. FSDP All-Gather (在 DP group 内)
   GPU 0 收集 GPU 0,8,16,...,56 的分片
   → 得到 wq[:, 0:512] (TP 切分后的完整部分)

2. TP 计算
   每个 TP group 计算自己的部分

3. TP All-Reduce (在 TP group 内)
   GPU 0-7 做 All-Reduce

4. FSDP Reshard (可选)
   释放 All-Gather 来的部分
```

### 7.2 FSDP + PP (2D 并行)

FSDP 与 Pipeline Parallel 的组合：

```
假设 64 GPUs, DP=8, PP=8

每个 PP stage 有 8 个 GPUs，在这 8 个 GPUs 上做 FSDP

Stage 0 (Layers 0-3):   GPU 0-7    (FSDP group)
Stage 1 (Layers 4-7):   GPU 8-15   (FSDP group)
Stage 2 (Layers 8-11):  GPU 16-23  (FSDP group)
...
Stage 7 (Layers 28-31): GPU 56-63  (FSDP group)
```

**关键优化**：`reshard_after_forward = False`

```
为什么？
  Pipeline 有 microbatch，每个 microbatch 都要 forward + backward
  如果每次都 reshard，会有大量重复的 All-Gather

1F1B Schedule (4 microbatches):
Stage 0: F0 → F1 → F2 → F3 → B0 → B1 → B2 → B3
         ↑
         如果 reshard_after_forward=True:
         F0: AG → Compute → Free
         F1: AG → Compute → Free  (重复 AG！)
         F2: AG → Compute → Free
         F3: AG → Compute → Free
         B0: AG → Compute → RS    (又一次 AG！)

         如果 reshard_after_forward=False:
         F0: AG → Compute → Keep
         F1:      Compute → Keep  (复用！)
         F2:      Compute → Keep
         F3:      Compute → Keep
         B0:      Compute → RS    (无需 AG！)

通信量: 8次 AG → 1次 AG (节省 87.5%)
```

### 7.3 FSDP + TP + PP (3D 并行)

最复杂的组合，用于超大模型：

```
Llama3 405B on 512 H100s
Config: DP=8, TP=8, PP=8

Device Mesh (3D):
         TP0  TP1  TP2  TP3  TP4  TP5  TP6  TP7
     ┌────────────────────────────────────────┐
DP0  │  PP0 的 8 个 GPUs (Stage 0)            │
DP1  │  PP0 的 8 个 GPUs (Stage 0)            │
...  │  ...                                   │
DP7  │  PP0 的 8 个 GPUs (Stage 0)            │
     ├────────────────────────────────────────┤
DP0  │  PP1 的 8 个 GPUs (Stage 1)            │
...  │  ...                                   │
     └────────────────────────────────────────┘

总共: 8 (DP) × 8 (TP) × 8 (PP) = 512 GPUs
```

**参数分布** (wq [4096, 4096])：

```
每个 GPU 存储: 4096/8 × 4096/8/8 = 32K 参数
总参数被切分 8×8×8 = 512 份
```

### 7.4 HSDP (Hybrid Sharded Data Parallel)

HSDP = FSDP 的变种，支持 2D DP mesh：

```
假设 64 GPUs, 配置为 HSDP (replicate=8, shard=8)

Device Mesh (2D):
            Shard0 Shard1 Shard2 ... Shard7
     ┌──────────────────────────────────────┐
Rep0 │   0     1     2    ...    7          │  ← Replica 0
Rep1 │   8     9    10    ...   15          │  ← Replica 1
Rep2 │  16    17    18    ...   23          │
...  │  ...                      ...         │
Rep7 │  56    57    58    ...   63          │  ← Replica 7
     └──────────────────────────────────────┘
```

**通信模式**：

```
Forward/Backward:
1. All-Gather 在 Shard 维度 (8 GPUs)
   GPU 0 收集 GPU 0,1,2,...,7 的分片

2. Reduce-Scatter 在 Shard 维度 (8 GPUs)
   GPU 0-7 做 Reduce-Scatter

3. All-Reduce 在 Replicate 维度 (8 GPUs)
   GPU 0,8,16,...,56 做 All-Reduce

优点：
- 减少通信域大小 (8 vs 64)
- 更好的网络拓扑利用 (intra-node vs inter-node)
```

**配置方式**：

```python
# TorchTitan 配置

[parallelism]
data_parallel_replicate_degree = 8  # Replicate 维度
data_parallel_shard_degree = 8      # Shard 维度

# 自动构建 2D mesh:
# mesh = DeviceMesh([[0-7], [8-15], ..., [56-63]])
```

---

## 8. 实战案例分析

### 8.1 Llama3 8B (8 GPUs)

**配置**：

```toml
[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 1
pipeline_parallel_degree = 1

[training]
mixed_precision_param = "bfloat16"
mixed_precision_reduce = "float32"
```

**参数分布**：

```
模型参数: 8B × 2 bytes = 16 GB
每个 GPU: 16 GB / 8 = 2 GB

内存占用 (per GPU):
- 参数 (sharded): 2 GB
- 梯度 (sharded): 2 GB
- 优化器状态 (sharded): 4 GB (Adam: 2x 参数)
- 激活值: ~6 GB (seq_len=8192, batch=2)
─────────────────
总计: ~14 GB

H100 80GB → 利用率 17.5%
```

**性能**：

```
吞吐量: 5,762 tokens/sec/GPU (FSDP only)
MFU: ~45%

瓶颈: 计算不饱和（模型太小）
```

### 8.2 Llama3 70B (256 GPUs)

**配置**：

```toml
[parallelism]
data_parallel_shard_degree = 32  # FSDP
tensor_parallel_degree = 8       # TP

[training]
mixed_precision_param = "bfloat16"
mixed_precision_reduce = "float32"
```

**参数分布**：

```
模型参数: 70B × 2 bytes = 140 GB

TP 切分: 每个 TP group 有完整的参数
  → 每个 TP group: 140 GB

FSDP 切分: 在 TP group 内切分
  → 每个 GPU: 140 GB / 32 = 4.375 GB

内存占用 (per GPU):
- 参数 (sharded): 4.375 GB
- 梯度 (sharded): 4.375 GB
- 优化器状态: 8.75 GB
- 激活值: ~25 GB
─────────────────
总计: ~42.5 GB

H100 80GB → 利用率 53%
```

**性能**：

```
吞吐量: 829 tokens/sec/GPU
MFU: ~52%

瓶颈: TP 通信
优化: 启用 Async TP → 876 tokens/sec/GPU (↑5.7%)
```

### 8.3 Llama3 405B (512 GPUs)

**配置**：

```toml
[parallelism]
data_parallel_shard_degree = 8   # FSDP
tensor_parallel_degree = 8       # TP
pipeline_parallel_degree = 8     # PP
fsdp_reshard_after_forward = "default"  # → False (因为 PP)

[training]
mixed_precision_param = "bfloat16"
mixed_precision_reduce = "float32"
```

**参数分布**：

```
模型参数: 405B × 2 bytes = 810 GB
层数: 126 layers

PP 切分: 126 / 8 = 15.75 layers/stage
  → 每个 stage: 810 GB / 8 = 101.25 GB

TP 切分: 在每个 stage 内
  → 每个 TP group: 101.25 GB (完整 stage)

FSDP 切分: 在 TP group 内
  → 每个 GPU: 101.25 GB / 8 = 12.66 GB

内存占用 (per GPU):
- 参数 (sharded): 12.66 GB
- 梯度 (sharded): 12.66 GB
- 优化器状态: 25.32 GB
- 激活值: ~20 GB (PP 减少激活)
─────────────────
总计: ~70.6 GB

H100 80GB → 利用率 88% ✅
```

**性能**：

```
Schedule: 1F1B
  吞吐量: 100 tokens/sec/GPU
  MFU: ~38%

Schedule: Interleaved 1F1B (2 stages/rank)
  吞吐量: 128 tokens/sec/GPU
  MFU: ~48%

提升: +28%
```

---

## 9. 调试与优化

### 9.1 常见问题

**Q1: OOM (Out of Memory)**

```
原因：
1. reshard_after_forward = False 但模型太大
2. 激活值太多（seq_len 或 batch_size 太大）
3. 没有启用 Activation Checkpointing

解决：
1. 设置 reshard_after_forward = "always"
2. 减少 batch_size 或 seq_len
3. 启用 Activation Checkpointing:
   [activation_checkpoint]
   mode = "selective"  # 或 "full"
```

**Q2: 通信瓶颈**

```
现象: GPU 利用率低，大量时间在通信

原因：
1. FSDP degree 太大（通信域太大）
2. 网络带宽不足
3. All-Gather 没有与计算重叠

解决：
1. 使用 HSDP 减小通信域
2. 检查网络拓扑（InfiniBand 是否正常）
3. 启用 gradient_as_bucket_view（默认启用）
```

**Q3: 数值问题**

```
现象: Loss 是 NaN 或发散

原因：
1. mixed_precision_reduce = bf16（精度不够）
2. 学习率太大
3. 梯度爆炸

解决：
1. 设置 mixed_precision_reduce = "float32"
2. 调整学习率
3. 启用梯度裁剪:
   [training]
   gradient_clip_norm = 1.0
```

### 9.2 性能调优技巧

**技巧 1: 调整 FSDP degree**

```python
# 不同 FSDP degree 的 trade-off

FSDP = 2:
  内存: 低（每个 GPU 1/2 参数）
  通信: 少（2 GPUs 通信）
  吞吐: 高

FSDP = 8:
  内存: 中（每个 GPU 1/8 参数）
  通信: 中（8 GPUs 通信）
  吞吐: 中

FSDP = 64:
  内存: 高（每个 GPU 1/64 参数）
  通信: 多（64 GPUs 通信）
  吞吐: 低

经验值:
- 单机 (8 GPUs): FSDP = 8
- 多机 (64 GPUs): FSDP = 8, TP = 8 (HSDP 更好)
- 大规模 (512 GPUs): FSDP = 8, TP = 8, PP = 8
```

**技巧 2: 优化 reshard 策略**

```toml
# 配置文件

[parallelism]
fsdp_reshard_after_forward = "default"  # 推荐

# 或者根据场景选择:
# "always"  - 内存最低，适合超大模型
# "never"   - 速度最快，适合内存充足
# "default" - 智能选择（PP 时 = never）
```

**技巧 3: 启用 Float8 + FSDP**

```toml
[model]
converters = ["float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true  # Float8 All-Gather
precompute_float8_dynamic_scale_for_fsdp = true

效果:
- 通信量减半（fp8 vs bf16）
- 计算加速（H100 Tensor Core）
- Llama3 70B: +16% 吞吐
```

### 9.3 监控指标

**关键指标**：

```python
# 1. 内存占用
device_mem_stats = device_memory_monitor.get_peak_stats()
logger.info(f"Peak memory: {device_mem_stats.max_reserved_gib:.2f} GiB")

# 2. 通信时间
# 使用 PyTorch Profiler 查看 All-Gather / Reduce-Scatter 时间

# 3. MFU (Model FLOPs Utilization)
mfu = achieved_flops / peak_flops
# 目标: MFU > 40%

# 4. 吞吐量
tokens_per_sec = tokens_processed / time_elapsed
# 目标: 越高越好
```

---

## 10. 总结

### 10.1 FSDP2 的核心优势

用**搬桌子**的比喻总结：

1. **Per-Parameter Sharding**：每张桌子单独切分，而不是打包成大捆
   - ✅ 保持参数的原始形状
   - ✅ 可以单独操作每个参数

2. **DTensor**：给每张桌子贴魔法标签
   - ✅ 像使用完整 tensor 一样使用分片 tensor
   - ✅ 无需通信的 state dict

3. **Reshard After Forward**：用完就扔，需要再拿
   - ✅ 极低的内存占用
   - ✅ 灵活的策略（PP 时智能优化）

4. **确定性内存管理**：不再依赖 CUDA caching allocator
   - ✅ 内存占用可预测
   - ✅ 峰值内存更低

### 10.2 何时使用 FSDP？

**推荐使用 FSDP**：
- ✅ 单 GPU 放不下模型
- ✅ 需要训练大模型（> 10B 参数）
- ✅ 有多个 GPU（≥ 8）
- ✅ 需要与 TP/PP 组合使用

**不推荐使用 FSDP**：
- ❌ 模型很小（< 1B 参数）
- ❌ 只有 1-2 个 GPU
- ❌ 需要极致的单机性能（用 DDP）

### 10.3 最佳实践

```
小模型 (< 10B):
  → FSDP 8

中模型 (10B - 70B):
  → FSDP 8-32 + TP 2-8

大模型 (70B - 405B):
  → FSDP 8 + TP 8 + PP 8

超大模型 (> 405B):
  → FSDP 8 + TP 8 + PP 16 + Float8
```

### 10.4 相关配置

```toml
# 基础配置
[parallelism]
data_parallel_shard_degree = 8     # FSDP degree
fsdp_reshard_after_forward = "default"  # Reshard 策略

# HSDP (可选)
data_parallel_replicate_degree = 8  # Replicate degree

# Mixed Precision
[training]
mixed_precision_param = "bfloat16"  # 参数精度
mixed_precision_reduce = "float32"  # 梯度规约精度

# CPU Offload (可选)
enable_cpu_offload = false  # 一般不需要

# Activation Checkpointing (推荐)
[activation_checkpoint]
mode = "selective"  # 或 "full"
```

### 10.5 性能数据

来自 TorchTitan benchmarks：

| 模型 | GPUs | 配置 | 吞吐 (tok/s/GPU) | MFU |
|-----|------|------|-----------------|-----|
| Llama3 8B | 8 | FSDP 8 | 5,762 | 45% |
| Llama3 70B | 256 | FSDP 32, TP 8 | 829 | 52% |
| Llama3 405B | 512 | FSDP 8, TP 8, PP 8 | 128 | 48% |

---

## 11. 参考资料

**源码文件**：
- `torchtitan/models/llama3/infra/parallelize.py:250-316` - FSDP 应用
- `torch.distributed.fsdp.fully_shard` - FSDP2 核心 API
- `torch.distributed.tensor.DTensor` - 分布式 Tensor

**PyTorch 官方文档**：
- [FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [DTensor](https://pytorch.org/docs/stable/distributed.tensor.html)
- [FSDP2 设计文档](https://github.com/pytorch/pytorch/issues/114299)

**相关文档**：
- [docs/fsdp.md](../fsdp.md) - FSDP1 → FSDP2 迁移指南
- [02_tensor_parallel_implementation.md](./02_tensor_parallel_implementation.md) - TP 实现

**学术论文**：
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

---

**最后更新**：2025年1月

**文档版本**：1.0
