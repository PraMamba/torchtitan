# Activation Checkpointing 激活检查点详解

## 目录
- [1. 什么是 Activation Checkpointing？](#1-什么是-activation-checkpointing)
- [2. 搬桌子的比喻：草稿纸策略](#2-搬桌子的比喻草稿纸策略)
- [3. 三种 AC 模式对比](#3-三种-ac-模式对比)
- [4. Full AC 实现](#4-full-ac-实现)
- [5. Selective AC - Layer 层级](#5-selective-ac---layer-层级)
- [6. Selective AC - Operator 算子级](#6-selective-ac---operator-算子级)
- [7. 源码实现详解](#7-源码实现详解)
- [8. 与 torch.compile 的交互](#8-与-torchcompile-的交互)
- [9. Memory Budget 模式](#9-memory-budget-模式)

---

## 1. 什么是 Activation Checkpointing？

### 1.1 基本概念

**Activation Checkpointing (AC)** = 在反向传播时，重新计算激活值，而不是在前向传播时全部保存。

**核心思想**：用**计算换内存** - 舍弃部分激活值，需要时重新计算。

### 1.2 为什么需要 AC？

训练深度学习模型时，内存占用主要来自三部分：

```
GPU 内存占用：
┌────────────────────────────────┐
│ 1. 模型参数 (Parameters)       │  20%
│    - weights, biases           │
├────────────────────────────────┤
│ 2. 优化器状态 (Optimizer)      │  40%
│    - momentum, variance        │
├────────────────────────────────┤
│ 3. 激活值 (Activations) 💥     │  40%
│    - 中间计算结果              │
│    - 需要用于反向传播          │
└────────────────────────────────┘

问题：激活值占用大量内存！
```

**具体例子** - Llama3 8B 训练：

```
配置：
- Batch size = 2
- Sequence length = 8192
- Hidden dim = 4096
- 32 layers

激活值内存占用：
每层的激活值 ≈ batch × seq_len × hidden_dim × sizeof(dtype)
             ≈ 2 × 8192 × 4096 × 2 bytes
             ≈ 128 MB

总激活值 ≈ 128 MB × 32 layers = 4 GB

如果没有 AC：
  需要保存所有层的激活值 → 4 GB
  可用于增大 batch size 或 seq_len 受限

使用 AC：
  只保存少量激活值 → 可能降到 1 GB
  可用内存增加 → batch size 可以增大 2-4x
```

### 1.3 AC 的权衡

```
不使用 AC:
  优点: ✅ 快（不需要重新计算）
  缺点: ❌ 内存占用高

使用 AC:
  优点: ✅ 内存占用低（可以训练更大模型/batch）
  缺点: ❌ 慢（需要重新计算激活值）

Time-Memory Tradeoff:
  Full AC:      最省内存，最慢（~20% 慢）
  Selective AC: 平衡（~10% 慢）
  No AC:        最快，最耗内存
```

---

## 2. 搬桌子的比喻：草稿纸策略

### 2.1 回顾搬桌子的场景

还记得我们用搬桌子比喻训练过程吗？（[FSDP 文档](./01_fsdp2_per_parameter_sharding.md)）

```
Forward Pass (正向搬桌子):
房间 A → 房间 B → 房间 C → 房间 D

每经过一个房间，都会产生一些"中间状态"：
- 搬到哪了？
- 用了什么工具？
- 桌子现在的位置？

这些"中间状态" = Activation（激活值）
```

### 2.2 不使用 AC：全部记录（内存爆炸）

**场景**：搬桌子时，在每个房间都详细记录。

```
搬桌子过程（Forward）：
房间 A → 房间 B → 房间 C → 房间 D
  ↓        ↓        ↓        ↓
[笔记1]  [笔记2]  [笔记3]  [笔记4]
写满了   写满了   写满了   写满了
10 页    10 页    10 页    10 页

总共：40 页笔记（内存）

检查工作（Backward）：
需要用这些笔记回顾每一步
笔记全在手边，查阅很快 ✅

问题：
❌ 笔记本用完了（内存不足）
❌ 背着 40 页笔记很重（内存限制）
```

### 2.3 Full AC：只记关键点（极致省内存）

**场景**：只在最开始记录，需要时重新搬一遍。

```
搬桌子过程（Forward）：
房间 A → 房间 B → 房间 C → 房间 D
  ↓
[笔记1]  [丢弃]  [丢弃]  [丢弃]
只保留
起点

总共：只有 1 页笔记（节省内存！）

检查工作（Backward）：
需要房间 C 的信息？
  → 从起点重新搬一遍：A → B → C（重新计算）
  → 查看信息
  → 继续检查

需要房间 B 的信息？
  → 从起点重新搬一遍：A → B（重新计算）
  → 查看信息

优点：
✅ 只需要 1 页笔记（内存占用极低）

缺点：
❌ 每次都要重新搬（计算时间增加 ~20%）
```

### 2.4 Selective AC - Layer：隔几个房间记录

**场景**：每隔 N 个房间做一次记录。

```
搬桌子过程（Forward）- 每 2 个房间记录：
房间 A → 房间 B → 房间 C → 房间 D → ... → 房间 H
  ↓                  ↓                         ↓
[笔记1]            [笔记2]                   [笔记3]
记录              记录                       记录

总共：3 页笔记（省了一半内存）

检查工作（Backward）：
需要房间 C 的信息？
  → 直接查笔记2 ✅（已保存）

需要房间 B 的信息？
  → 从笔记1重新搬：A → B（只需重算 1 步）
  → 查看信息

优点：
✅ 笔记减少 50%（内存占用中等）
✅ 重新计算的次数少（速度损失 ~10%）

缺点：
⚠️ 需要权衡记录频率
```

### 2.5 Selective AC - Operator：聪明的草稿纸

**场景**：根据重要性决定记录什么。

```
搬桌子过程中的不同操作：
1. 测量桌子尺寸         → 简单，重算很快
2. 拆解桌子腿           → 简单，重算很快
3. 用起重机搬主体 🏗️    → 复杂！重算很慢！
4. 调整桌子方向         → 简单，重算很快
5. 精密拼装 🔧          → 复杂！重算很慢！

策略：
✅ 保存：复杂操作的结果（起重机、精密拼装）
❌ 丢弃：简单操作的结果（测量、拆解）

总共：只记录关键操作（最优的内存-速度平衡）

检查工作（Backward）：
需要起重机的信息？
  → 直接查笔记 ✅（已保存，因为重算太慢）

需要测量的信息？
  → 重新测量一次 ✅（很快，不值得保存）

优点：
✅ 内存占用低（只保存重要的）
✅ 速度损失小（重算的都是快速操作）

缺点：
⚠️ 需要知道哪些操作"重要"
```

### 2.6 实际对应关系

```
草稿纸比喻 → 技术实现：

1. 笔记 = Activation（激活值）
   - Forward 时产生
   - Backward 时使用

2. 重新搬桌子 = Recompute（重新计算）
   - 没有笔记时
   - 从 checkpoint 开始重新 forward

3. 笔记本容量 = GPU 内存
   - 有限的资源
   - 需要权衡使用

4. 搬桌子的时间 = 训练时间
   - Recompute 增加时间
   - 但换来内存节省
```

---

## 3. 三种 AC 模式对比

### 3.1 模式总览

TorchTitan 支持 4 种 AC 模式：

```python
# 来自: torchtitan/config/job_config.py:586

mode: Literal["selective", "full", "memory_budget", "none"] = "selective"
```

| 模式 | 内存节省 | 速度损失 | 适用场景 |
|------|---------|---------|---------|
| **None** | 0% | 0% | 小模型、内存充足 |
| **Selective (Layer)** | 30-50% | ~10% | 中等模型（推荐） |
| **Selective (Op)** | 40-60% | ~12% | 大模型、最优平衡 |
| **Full** | 50-70% | ~20% | 超大模型、内存紧张 |
| **Memory Budget** | 自定义 | 自定义 | 高级优化、自动搜索 |

### 3.2 Full AC

**策略**：每个 TransformerBlock 都丢弃所有激活值。

```
Forward (32 layers):
Layer 0:  [Compute] → [Save input only] → [Discard activations]
Layer 1:  [Compute] → [Save input only] → [Discard activations]
...
Layer 31: [Compute] → [Save input only] → [Discard activations]

保存：
- 每层的输入（32 个 checkpoint）
- 最终的输出

Backward (32 layers):
Layer 31: [Recompute forward] → [Compute backward]
Layer 30: [Recompute forward] → [Compute backward]
...
Layer 0:  [Recompute forward] → [Compute backward]

每层都需要重新计算一次 forward！
```

**配置**：

```toml
[activation_checkpoint]
mode = "full"
```

### 3.3 Selective AC - Layer (每 N 层)

**策略**：每隔 N 层保存激活值，中间层丢弃。

```
Forward (32 layers, N=2):
Layer 0:  [Compute] → ✅ Save activations
Layer 1:  [Compute] → ❌ Discard activations
Layer 2:  [Compute] → ✅ Save activations
Layer 3:  [Compute] → ❌ Discard activations
...
Layer 30: [Compute] → ✅ Save activations
Layer 31: [Compute] → ❌ Discard activations

保存：16 层的激活值（节省 50%）

Backward:
Layer 31: ❌ 需要从 Layer 30 重新计算
Layer 30: ✅ 直接使用保存的激活值
Layer 29: ❌ 需要从 Layer 28 重新计算
Layer 28: ✅ 直接使用保存的激活值
...

每层最多重算 1 次 forward
```

**配置**：

```toml
[activation_checkpoint]
mode = "selective"
selective_ac_option = "2"  # 每 2 层保存一次
# 或 "3" 每 3 层, "4" 每 4 层...
```

### 3.4 Selective AC - Operator (算子级)

**策略**：保存"昂贵"的算子结果，丢弃"便宜"的。

```
TransformerBlock 内部操作：
┌─────────────────────────────────────┐
│ 1. LayerNorm            → Recompute │  (简单)
│ 2. QKV Projection (mm)  → SAVE ✅   │  (矩阵乘法，重算慢)
│ 3. Attention (SDPA) 🔥  → SAVE ✅   │  (复杂，重算很慢)
│ 4. Output Proj (mm)     → SAVE ✅   │  (矩阵乘法)
│ 5. Add & Norm           → Recompute │  (简单)
│ 6. FFN W1 (mm)          → SAVE ✅   │  (矩阵乘法)
│ 7. Activation (SiLU)    → Recompute │  (简单)
│ 8. FFN W2 (mm)          → SAVE ✅   │  (矩阵乘法)
│ 9. Add                  → Recompute │  (简单)
└─────────────────────────────────────┘

保存：5 个关键算子的输出
丢弃：4 个简单操作的输出

内存节省：~50%
速度损失：~12%（重算的都是简单操作）
```

**配置**：

```toml
[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"
```

### 3.5 内存占用对比

假设 Llama3 8B，batch=2，seq_len=8192：

```
No AC:
  Activations: ~4 GB
  可用内存: 80 GB - 4 GB - 12 GB (参数+优化器) = 64 GB

Selective (Layer, N=2):
  Activations: ~2 GB (节省 50%)
  可用内存: 80 GB - 2 GB - 12 GB = 66 GB
  → 可增大 batch size 30%

Selective (Op):
  Activations: ~1.8 GB (节省 55%)
  可用内存: 80 GB - 1.8 GB - 12 GB = 66.2 GB
  → 可增大 batch size 35%

Full AC:
  Activations: ~1 GB (节省 75%)
  可用内存: 80 GB - 1 GB - 12 GB = 67 GB
  → 可增大 batch size 70%
```

---

## 4. Full AC 实现

### 4.1 核心原理

Full AC 使用 PyTorch 的 `checkpoint_wrapper`：

```python
# 来自: torchtitan/distributed/activation_checkpoint.py:139-155

def _apply_full_ac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    """Apply full activation checkpointing to the module."""

    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=ac_config.preserve_rng_state,  # 保持随机性
        determinism_check=ac_config.determinism_check,    # 确定性检查
        early_stop=ac_config.early_stop,                  # 早停优化
        debug=ac_config.debug,                            # Debug 模式
    )
```

**checkpoint_wrapper 做了什么？**

```python
# 伪代码，展示原理

class CheckpointWrapper(nn.Module):
    def __init__(self, module):
        self.module = module

    def forward(self, *args):
        # 1. 保存输入
        saved_inputs = args

        # 2. 正常执行 forward（但不保存中间激活）
        with torch.no_grad():
            output = self.module(*args)

        # 3. 注册 backward hook
        output.register_hook(lambda grad_output:
            self._backward_with_recompute(saved_inputs, grad_output)
        )

        return output

    def _backward_with_recompute(self, saved_inputs, grad_output):
        # Backward 时：
        # 1. 重新计算 forward（这次保存激活）
        with torch.enable_grad():
            output = self.module(*saved_inputs)

        # 2. 计算梯度
        output.backward(grad_output)
```

### 4.2 应用到 Transformer

```python
# 来自: torchtitan/distributed/activation_checkpoint.py:323-332

# 对每个 TransformerBlock 应用 Full AC
for layer_id, transformer_block in model.layers.named_children():
    transformer_block = _apply_full_ac(transformer_block, ac_config)
    model.layers.register_module(layer_id, transformer_block)
```

**效果**：

```
未包装的 TransformerBlock:
forward():
  x = layer_norm(x)          → 保存 x
  q, k, v = qkv_proj(x)      → 保存 q, k, v
  attn_out = attention(q,k,v) → 保存 attn_out
  x = out_proj(attn_out)     → 保存 x
  ...

包装后的 TransformerBlock:
forward():
  [保存输入 x0]
  with no_grad():
    x = layer_norm(x)        → 不保存
    q, k, v = qkv_proj(x)    → 不保存
    attn_out = attention()   → 不保存
    ...
  [只保存最终输出]

backward():
  [重新计算整个 forward]
  with enable_grad():
    x = layer_norm(x)        → 保存（用于梯度）
    q, k, v = qkv_proj(x)    → 保存
    ...
  [计算梯度]
```

### 4.3 内存节省分析

```
TransformerBlock 激活值大小：
  layer_norm:    batch × seq × hidden ≈ 128 MB
  qkv_proj:      3 × 128 MB = 384 MB
  attention:     batch × heads × seq × seq ≈ 512 MB
  out_proj:      128 MB
  ffn_norm:      128 MB
  ffn_w1/w3:     2 × 256 MB = 512 MB
  ffn_w2:        128 MB
  ──────────────────────────────────────
  总计: ~2 GB / layer

Full AC:
  只保存输入: 128 MB / layer
  节省: 2 GB - 128 MB = 1.87 GB / layer

32 层总节省: 1.87 GB × 32 = 60 GB！
```

---

## 5. Selective AC - Layer 层级

### 5.1 Layer SAC 原理

**策略**：每 N 层保存一次完整激活。

```python
# 来自: torchtitan/distributed/activation_checkpoint.py:26-48

_layer_sac_count = 0  # 全局计数器

def _apply_layer_sac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    global _layer_sac_count
    _layer_sac_count += 1  # 每调用一次 +1

    ac_freq = int(ac_config.selective_ac_option)  # 例如 "2"

    if _layer_sac_count % ac_freq == 0:
        # 第 0, 2, 4, 6, ... 层：不使用 AC，保存激活
        return module
    else:
        # 第 1, 3, 5, 7, ... 层：使用 AC，丢弃激活
        return ptd_checkpoint_wrapper(module, ...)
```

**应用示例** (N=2)：

```
Layer 0:  No AC  → 保存所有激活 ✅
Layer 1:  AC     → 丢弃激活 ❌
Layer 2:  No AC  → 保存所有激活 ✅
Layer 3:  AC     → 丢弃激活 ❌
...
Layer 30: No AC  → 保存所有激活 ✅
Layer 31: AC     → 丢弃激活 ❌

保存: 16 层的激活
丢弃: 16 层的激活
节省: 50%
```

### 5.2 Backward 重计算

```
Backward 时（从后向前）：

Layer 31 (AC):
  需要激活 → 从 Layer 30 的输出重新计算
  重算次数: 1 次

Layer 30 (No AC):
  直接使用保存的激活 ✅
  重算次数: 0 次

Layer 29 (AC):
  需要激活 → 从 Layer 28 的输出重新计算
  重算次数: 1 次

Layer 28 (No AC):
  直接使用保存的激活 ✅
  重算次数: 0 次

...

平均重算次数: 16 / 32 = 0.5 次/层
```

### 5.3 调优策略

**选择 N 的准则**：

```
N = 1: 每层都保存
  - 内存节省: 0%
  - 速度损失: 0%
  - 相当于 No AC

N = 2: 每 2 层保存一次（推荐）
  - 内存节省: 50%
  - 速度损失: ~10%
  - 平衡点

N = 3: 每 3 层保存一次
  - 内存节省: 67%
  - 速度损失: ~15%

N = 4: 每 4 层保存一次
  - 内存节省: 75%
  - 速度损失: ~18%
  - 接近 Full AC

N = ∞: 相当于 Full AC
  - 内存节省: ~80%
  - 速度损失: ~20%
```

**实际选择**：

```toml
# 内存充足，追求速度
[activation_checkpoint]
mode = "selective"
selective_ac_option = "2"  # 或 "1"（几乎不用 AC）

# 内存紧张，愿意牺牲速度
[activation_checkpoint]
mode = "selective"
selective_ac_option = "4"  # 或使用 "full"
```

---

## 6. Selective AC - Operator 算子级

### 6.1 Op SAC 原理

**策略**：根据算子的重算代价决定保存还是丢弃。

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:34-44

# 定义"昂贵"的算子（必须保存）
_op_sac_save_list = {
    torch.ops.aten.mm.default,  # 矩阵乘法（Matmul）
    torch.ops.aten._scaled_dot_product_efficient_attention.default,  # Attention
    torch.ops.aten._scaled_dot_product_flash_attention.default,      # Flash Attention
    torch.ops._c10d_functional.reduce_scatter_tensor.default,        # 通信算子
    torch.ops.aten.max.default,  # Max（用于 float8 量化）
    torch._higher_order_ops.flex_attention,  # Flex Attention
}
```

**为什么这些算子需要保存？**

```
1. torch.ops.aten.mm.default (矩阵乘法):
   计算量: O(N³)  (N = 4096)
   重算代价: 非常高 😱
   → 必须保存 ✅

2. Attention (SDPA/Flash):
   计算量: O(N² × d)  (N = 8192, d = 128)
   重算代价: 极高 😱😱
   → 必须保存 ✅

3. Reduce-Scatter (通信):
   代价: 网络通信
   重算代价: 高（需要重新通信）
   → 必须保存 ✅

4. LayerNorm, Add, SiLU:
   计算量: O(N)
   重算代价: 很低 ✅
   → 可以丢弃，需要时重算
```

### 6.2 自定义 Policy

```python
# 来自: torchtitan/distributed/activation_checkpoint.py:97-123

def _get_custom_policy(meta):
    def _custom_policy(ctx, func, *args, **kwargs):
        # 规则 1: 永远不要丢弃 GPU → CPU 的拷贝
        if (
            func == torch.ops.aten._to_copy.default
            and "cuda" in str(args[0].device)
            and "device" in kwargs
            and str(kwargs["device"]) == "cpu"
        ):
            return CheckpointPolicy.MUST_SAVE

        # 规则 2: Matmul 的智能策略
        if func == torch.ops.aten.mm.default:
            # 检查 shape（某些特定 shape 强制重算）
            if args[1].shape in mm_recompute_shapes:
                return CheckpointPolicy.PREFER_RECOMPUTE

            # 每隔一个 mm 保存一次（节省内存）
            meta["mm_count"] += 1
            if meta["mm_count"] % 2 == 0:
                return CheckpointPolicy.PREFER_RECOMPUTE

        # 规则 3: 在 save_list 中的算子 → 保存
        to_save = func in op_sac_save_list
        return (
            CheckpointPolicy.MUST_SAVE if to_save
            else CheckpointPolicy.PREFER_RECOMPUTE
        )

    return _custom_policy
```

**三种 Policy**：

```
MUST_SAVE:
  - 必须保存，不可丢弃
  - 用于：GPU→CPU 拷贝

MUST_RECOMPUTE:
  - 必须丢弃，强制重算
  - 用于：调试、特定优化

PREFER_RECOMPUTE:
  - 优先丢弃，可以重算
  - 用于：大部分算子（默认）
```

### 6.3 实际执行流程

```
Forward Pass (TransformerBlock):

1. LayerNorm(x)
   Policy: PREFER_RECOMPUTE
   → 丢弃输出 ❌

2. QKV Projection (3个 mm)
   Policy: MUST_SAVE (在 save_list)
   → 保存 q, k, v ✅

3. Attention (SDPA)
   Policy: MUST_SAVE (在 save_list)
   → 保存 attn_output ✅

4. Output Projection (mm)
   Policy: MUST_SAVE (在 save_list)
   → 保存输出 ✅

5. Add
   Policy: PREFER_RECOMPUTE
   → 丢弃 ❌

6. LayerNorm
   Policy: PREFER_RECOMPUTE
   → 丢弃 ❌

7. FFN W1/W3 (2个 mm)
   Policy: MUST_SAVE
   → 保存 ✅

8. SiLU Activation
   Policy: PREFER_RECOMPUTE
   → 丢弃 ❌

9. FFN W2 (mm)
   Policy: MUST_SAVE
   → 保存 ✅

总结:
  保存: 7 个关键算子
  丢弃: 4 个简单算子
  内存节省: ~40-50%
```

### 6.4 与 Matmul 的特殊处理

```python
# 每隔一个 mm 保存一次
if func == torch.ops.aten.mm.default:
    meta["mm_count"] += 1
    if meta["mm_count"] % 2 == 0:
        return CheckpointPolicy.PREFER_RECOMPUTE  # 丢弃

为什么？
  TransformerBlock 有很多 mm:
  - QKV: 3 个 mm
  - Output: 1 个 mm
  - FFN: 3 个 mm (w1, w2, w3)

  全部保存: 内存占用高
  全部丢弃: 重算代价高

  折中: 保存一半，丢弃一半
    → 内存占用中等
    → 重算代价中等
```

---

## 7. 源码实现详解

### 7.1 入口函数

```python
# 来自: torchtitan/distributed/activation_checkpoint.py:286-334

def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    use_flex_attn: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
) -> None:
    """Apply activation checkpointing to the model."""

    # 特殊模式：Memory Budget (自动搜索最优策略)
    if ac_config.mode == "memory_budget":
        assert model_compile_enabled, "Memory budget 需要 compile"
        torch._functorch.config.activation_memory_budget = ac_config.memory_budget
        return

    # 标准模式：对每个 TransformerBlock 应用 AC
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block,
            ac_config,
            base_fqn=f"layers.{layer_id}",
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
            op_sac_save_list=op_sac_save_list,
        )
        model.layers.register_module(layer_id, transformer_block)
```

### 7.2 TransformerBlock 的包装

```python
# 来自: torchtitan/distributed/activation_checkpoint.py:233-283

def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_config: ACConfig,
    ...
) -> nn.Module:
    # 1. 检查模式
    if ac_config.mode == "full":
        return _apply_full_ac(module, ac_config)

    # 2. Selective AC
    assert ac_config.mode == "selective"

    # 2.1 判断是 Layer SAC 还是 Op SAC
    use_op_sac = (ac_config.selective_ac_option == "op")
    use_layer_sac = ac_config.selective_ac_option.isdigit()  # 例如 "2"

    if use_op_sac:
        # 2.2 Op SAC
        if use_flex_attn:
            # Flex Attention 特殊处理（避免与 compile 冲突）
            return _apply_op_sac_to_transformer_block_with_flex(...)
        else:
            return _apply_op_sac(module, ac_config, ...)

    # 2.3 Layer SAC
    return _apply_layer_sac(module, ac_config)
```

### 7.3 Checkpoint Wrapper 原理

```python
# PyTorch 内部实现（简化版）

def checkpoint(function, *args, preserve_rng_state=True):
    """Checkpoint 函数的核心逻辑"""

    class CheckpointFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            # Forward: 不保存中间结果
            ctx.save_for_backward(*args)  # 只保存输入

            with torch.no_grad():
                outputs = function(*args)

            return outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            # Backward: 重新计算
            inputs = ctx.saved_tensors

            # 重新执行 forward（这次保留梯度）
            with torch.enable_grad():
                detached_inputs = [x.detach().requires_grad_() for x in inputs]
                outputs = function(*detached_inputs)

            # 计算梯度
            torch.autograd.backward(outputs, grad_outputs)

            # 返回输入的梯度
            grads = [x.grad for x in detached_inputs]
            return tuple(grads)

    return CheckpointFunction.apply(*args)
```

### 7.4 Selective Checkpoint Context

```python
# 来自: torchtitan/distributed/activation_checkpoint.py:125-136

def _apply_op_sac(module, ac_config, ...):
    # 创建 selective checkpoint context
    def selective_checkpointing_context_fn():
        meta = defaultdict(int)  # 状态追踪（mm 计数等）
        return create_selective_checkpoint_contexts(
            _get_custom_policy(meta)
        )

    # 包装 module
    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,  # 自定义策略
        ...
    )
```

**create_selective_checkpoint_contexts 做了什么？**

```python
# PyTorch 内部（简化）

def create_selective_checkpoint_contexts(policy_fn):
    """创建上下文，拦截每个算子调用"""

    class SelectiveCheckpoint:
        def __enter__(self):
            # 注册 dispatch hook
            self.handle = torch._C._register_dispatch_key_hook(
                self._hook
            )

        def _hook(self, func, *args, **kwargs):
            # 每个算子调用时：
            # 1. 调用 policy_fn 判断是否保存
            policy = policy_fn(ctx, func, *args, **kwargs)

            if policy == CheckpointPolicy.MUST_SAVE:
                # 保存：正常执行，保留梯度
                return func(*args, **kwargs)
            else:
                # 丢弃：执行但不保留梯度
                with torch.no_grad():
                    return func(*args, **kwargs)

        def __exit__(self, ...):
            self.handle.remove()
```

---

## 8. 与 torch.compile 的交互

### 8.1 问题：AC 与 Compile 的冲突

```
问题：
  torch.compile 需要完整的计算图
  AC 会在运行时重新计算 → 破坏计算图

示例：
  # 没有 AC
  model = torch.compile(model)  # 编译整个模型 ✅

  # 有 AC
  model = checkpoint_wrapper(model)
  model = torch.compile(model)  # 编译失败或效果差 ❌
```

### 8.2 TorchTitan 的解决方案

**策略 1：先 AC，后 Compile**

```python
# 来自: torchtitan/models/llama3/infra/parallelize.py:96-108

# 1. 先应用 AC
apply_ac(model, ac_config, ...)

# 2. 再 compile（每个 TransformerBlock）
if model_compile_enabled:
    apply_compile(model, compile_config)

# 为什么按这个顺序？
# - AC 包装后，每个 TransformerBlock 是独立的 checkpoint 单元
# - Compile 单独编译每个 TransformerBlock
# - 不会破坏 AC 的重计算逻辑
```

**策略 2：Flex Attention 的特殊处理**

```python
# 来自: torchtitan/distributed/activation_checkpoint.py:158-230

if use_flex_attn:
    # Flex Attention 必须 compile 才能高效
    # 但 AC 会破坏 compile

    # 解决方案：分模块处理
    if hasattr(module, "moe"):
        # MoE 层：都用 Op SAC
        wrap_submodule("moe", full_ac=False)

        if model_compile_enabled:
            wrap_submodule("attention", full_ac=False)  # Op SAC
        else:
            wrap_submodule("attention", full_ac=True)   # Full AC
    else:
        # Dense 层
        if model_compile_enabled:
            # 整个 block 用 Op SAC（保留 compile）
            module = _apply_op_sac(module, ...)
        else:
            # 分开包装
            wrap_submodule("feed_forward", full_ac=False)  # Op SAC
            wrap_submodule("attention", full_ac=True)      # Full AC
```

**为什么这样做？**

```
Flex Attention 需求：
  ✅ 必须 compile
  ❌ 不能有 Full AC（破坏图）
  ✅ 可以有 Op SAC（保留图结构）

策略：
  compile = True:  用 Op SAC（全部算子级）
  compile = False: Attention 用 Full AC，FFN 用 Op SAC
```

### 8.3 Compile 与 AC 的性能对比

```
Llama3 8B (8 GPUs):

No AC + No Compile:
  速度: 5,762 tok/s/GPU
  内存: 24 GB

No AC + Compile:
  速度: 6,667 tok/s/GPU (+15.7%)
  内存: 24 GB

Selective AC + No Compile:
  速度: 5,186 tok/s/GPU (-10%)
  内存: 18 GB (-25%)

Selective AC + Compile:
  速度: 6,000 tok/s/GPU (+4.1%)
  内存: 18 GB (-25%)

结论：
  AC + Compile 可以兼得：
  - 内存节省 25%
  - 速度仍有提升
```

---

## 9. Memory Budget 模式

### 9.1 什么是 Memory Budget？

**Memory Budget** = 自动搜索最优的 AC 策略，给定内存预算。

```
传统 AC:
  手动选择：Full / Selective(2) / Selective(op)
  问题：不知道哪个最优

Memory Budget:
  自动搜索：在内存预算内，找最快的策略

示例：
  内存预算 = 20 GB
  → 自动尝试不同的 AC 组合
  → 找到最优：某些层 Full AC，某些层 Selective
```

### 9.2 配置

```toml
[activation_checkpoint]
mode = "memory_budget"
memory_budget = 0.8  # 80% 的可用内存

# 可视化搜索过程（可选）
visualize_memory_budget_pareto = true

# 必须启用 compile
[compile]
enable = true
components = ["model"]
```

### 9.3 工作原理

```python
# 来自: torchtitan/distributed/activation_checkpoint.py:311-321

if ac_config.mode == "memory_budget":
    # 1. 必须启用 compile（依赖编译器分析）
    assert model_compile_enabled

    # 2. 设置内存预算
    torch._functorch.config.activation_memory_budget = ac_config.memory_budget

    # 3. （可选）可视化 Pareto 曲线
    if ac_config.visualize_memory_budget_pareto:
        torch._functorch.config.visualize_memory_budget_pareto = True

    # Compile 时，PyTorch 会自动：
    # - 分析每个算子的内存和计算代价
    # - 搜索最优的 checkpoint 策略
    # - 生成满足预算的最快代码
```

### 9.4 Pareto 曲线

```
Memory Budget 会生成 Pareto 曲线：

Speed (tokens/sec)
  ▲
  │                   ○ Full AC
  │                  /
  │                 /
  │                ○ Selective (op)
  │               /
  │              /
  │             ○ Selective (2)
  │            /
  │           /
  │          ○ No AC
  │
  └──────────────────────────────► Memory Usage (GB)
     10      15      20      25      30

给定内存预算 20 GB:
  → 选择 Selective (op)（在预算内，速度最快）
```

### 9.5 适用场景

```
推荐使用 Memory Budget：
✅ 大规模实验（有时间搜索）
✅ 固定硬件（一次搜索，多次使用）
✅ 追求极致性能

不推荐：
❌ 快速原型（搜索耗时）
❌ 硬件多变（每次都要重新搜索）
❌ 简单任务（Selective 已足够）
```

---

## 10. 实战案例

### 10.1 Llama3 8B (8 GPUs)

**场景**：单机训练，内存充足。

```toml
# 配置 1: No AC（Baseline）
[activation_checkpoint]
mode = "none"

# 内存占用: 24 GB
# 速度: 5,762 tok/s/GPU
# Batch size: 2

# 配置 2: Selective (Layer, N=2)
[activation_checkpoint]
mode = "selective"
selective_ac_option = "2"

# 内存占用: 18 GB (-25%)
# 速度: 5,186 tok/s/GPU (-10%)
# Batch size: 可增至 3 (+50%)

# 配置 3: Selective (Op) - 推荐
[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"

# 内存占用: 17 GB (-29%)
# 速度: 5,300 tok/s/GPU (-8%)
# Batch size: 可增至 3 (+50%)

# 配置 4: Full AC
[activation_checkpoint]
mode = "full"

# 内存占用: 15 GB (-37.5%)
# 速度: 4,610 tok/s/GPU (-20%)
# Batch size: 可增至 4 (+100%)
```

**选择建议**：

```
内存充足 (> 30 GB 可用):
  → mode = "selective", selective_ac_option = "2"
  → 或不使用 AC

内存紧张 (20-30 GB):
  → mode = "selective", selective_ac_option = "op"

内存非常紧张 (< 20 GB):
  → mode = "full"
```

### 10.2 Llama3 70B (256 GPUs)

**场景**：多机训练，追求吞吐。

```toml
[parallelism]
data_parallel_shard_degree = 32
tensor_parallel_degree = 8

[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"  # Op SAC（最优平衡）

[compile]
enable = true
components = ["model"]

# 效果:
# - 内存占用: 42 GB/GPU（可接受）
# - 速度: 接近无 AC 的 90%
# - 可训练 seq_len = 8192
```

### 10.3 Llama3 405B (512 GPUs)

**场景**：超大模型，内存极度紧张。

```toml
[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 8
pipeline_parallel_degree = 8

[activation_checkpoint]
mode = "full"  # Full AC（最省内存）

[compile]
enable = true

# 效果:
# - 内存占用: 70 GB/GPU（勉强放下）
# - 速度: 慢 20%（可接受）
# - 没有 AC 根本无法训练
```

### 10.4 调试场景

```toml
# 调试时：关闭 AC（更快迭代）
[activation_checkpoint]
mode = "none"

# 验证数值正确性
[activation_checkpoint]
mode = "selective"
determinism_check = "deterministic"  # 检查重算是否一致

# Debug 模式（打印详细信息）
[activation_checkpoint]
mode = "selective"
debug = true
```

---

## 11. 调试与优化

### 11.1 常见问题

**Q1: AC 后训练变慢了很多**

```
症状:
  启用 AC 后，速度慢 > 30%

原因:
1. 使用了 Full AC（预期 20% 慢）
2. 重算代价很高的算子（如多次 recompute Attention）
3. 与 torch.compile 冲突

解决:
1. 使用 Selective (op) 而不是 Full
2. 检查 op_sac_save_list，确保关键算子被保存
3. 启用 compile:
   [compile]
   enable = true
```

**Q2: 内存没有减少**

```
症状:
  启用 AC 后，内存占用没变化

原因:
1. 其他部分占内存（参数、优化器）
2. AC 没有正确应用
3. 使用 Selective(1) 相当于不用 AC

检查:
1. 对比激活值内存（不是总内存）:
   device_memory_monitor.get_peak_stats()
2. 确认 AC 模式:
   logger.info(f"Applied {mode} AC")
3. 调整 selective_ac_option:
   "2" → "4" 或 "full"
```

**Q3: 数值不一致**

```
症状:
  启用 AC 后，loss 不同或训练不稳定

原因:
1. RNG 状态不一致（Dropout 等）
2. 重算时数值误差累积

解决:
1. 启用 preserve_rng_state:
   [activation_checkpoint]
   preserve_rng_state = true
2. 使用确定性检查:
   determinism_check = "deterministic"
3. 检查是否有 in-place 操作
```

**Q4: 与 FSDP/TP 冲突**

```
症状:
  AC + FSDP 后出错或速度很慢

原因:
1. AC 在 FSDP 之前应用（顺序错误）
2. AC 包装了整个模型（应该包装 layer）

解决:
1. 确保顺序：
   apply_tp() → apply_ac() → apply_compile() → apply_fsdp()
2. 只包装 TransformerBlock:
   for layer in model.layers:
       layer = checkpoint_wrapper(layer)
```

### 11.2 性能优化技巧

**技巧 1: 选择合适的 AC 模式**

```python
# 决策树
if memory_is_sufficient:
    mode = "selective"
    selective_ac_option = "2"  # 最小速度损失
elif memory_is_tight:
    mode = "selective"
    selective_ac_option = "op"  # 最优平衡
else:  # memory_is_very_tight
    mode = "full"  # 最省内存
```

**技巧 2: 微调 op_sac_save_list**

```python
# 添加自定义算子到 save_list
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    # 添加：如果你的模型有特殊的昂贵算子
    torch.ops.my_custom.expensive_op.default,
}

# 移除：如果某个算子重算很快
_op_sac_save_list = {
    # torch.ops.aten.mm.default,  # 移除 mm（如果重算快）
    ...
}
```

**技巧 3: 与 Batch Size 联动**

```python
# 没有 AC
batch_size = 2
activation_memory = 4 GB

# 有 AC (节省 50%)
batch_size = 3  # 增大 50%
activation_memory = 3 GB (节省 1 GB)
# 总吞吐: batch_size ↑ 50%, 速度 ↓ 10%
# 净收益: +35% 吞吐！
```

**技巧 4: Profile 激活值内存**

```python
# 开启内存 profiling
torch.cuda.memory._record_memory_history()

# 训练几步
for i, batch in enumerate(dataloader):
    if i == 10:
        break
    loss = model(batch)
    loss.backward()

# 查看内存快照
torch.cuda.memory._dump_snapshot("memory.pickle")

# 分析激活值占比
# 使用 PyTorch Memory Profiler Visualizer
```

### 11.3 监控指标

**关键指标**：

```python
# 1. 激活值内存占用
import torch.cuda

before_forward = torch.cuda.memory_allocated()
output = model(input)
after_forward = torch.cuda.memory_allocated()
activation_memory = after_forward - before_forward

logger.info(f"Activation memory: {activation_memory / 1e9:.2f} GB")

# 2. Recompute 开销
import time

start = time.time()
for _ in range(100):
    loss = model(input)
    loss.backward()
forward_backward_time = time.time() - start

logger.info(f"Forward+Backward time: {forward_backward_time:.2f}s")

# 3. 速度损失百分比
no_ac_speed = 5762  # tok/s/GPU (baseline)
with_ac_speed = 5186  # tok/s/GPU (with AC)
slowdown = (1 - with_ac_speed / no_ac_speed) * 100

logger.info(f"AC slowdown: {slowdown:.1f}%")
```

---

## 12. 总结

### 12.1 AC 的核心思想

用**草稿纸**的比喻总结：

1. **No AC**：记录所有步骤
   - ✅ 查阅快（速度快）
   - ❌ 笔记多（内存高）

2. **Full AC**：只记起点
   - ✅ 笔记少（内存低）
   - ❌ 重新推导（速度慢 20%）

3. **Selective (Layer)**：隔几步记录
   - ✅ 笔记中等（内存中等）
   - ✅ 重新推导少（速度慢 10%）

4. **Selective (Op)**：只记重要的
   - ✅ 笔记少（内存低）
   - ✅ 重新推导少（速度慢 12%）
   - **最优平衡**

### 12.2 选择建议

```
场景 1: 小模型 (< 10B)，内存充足
  → mode = "none" 或 "selective", option = "2"

场景 2: 中模型 (10B-70B)，标准训练
  → mode = "selective", option = "op"  （推荐）

场景 3: 大模型 (> 70B)，内存紧张
  → mode = "full"

场景 4: 研究优化，追求极致
  → mode = "memory_budget"
```

### 12.3 配置速查

```toml
# 推荐配置（适合大多数场景）
[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"
preserve_rng_state = true

# 调试配置
[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"
determinism_check = "deterministic"
debug = true

# 极致内存优化
[activation_checkpoint]
mode = "full"
preserve_rng_state = true

# 关闭 AC（调试时）
[activation_checkpoint]
mode = "none"
```

### 12.4 与其他技术的关系

```
FSDP:
  切分参数，节省参数内存

AC:
  丢弃激活，节省激活内存

FSDP + AC:
  → 可训练 2-4x 更大的模型
  → 或 2-4x 更大的 batch

Compile:
  加速计算

AC + Compile:
  → 可兼得（需要正确配置）
  → TorchTitan 已优化集成
```

### 12.5 关键源码

```
核心文件:
- torchtitan/distributed/activation_checkpoint.py
  - apply_ac: 入口函数
  - _apply_full_ac: Full AC
  - _apply_layer_sac: Layer SAC
  - _apply_op_sac: Operator SAC

配置:
- torchtitan/config/job_config.py:585-613
  - ActivationCheckpoint 配置类

应用:
- torchtitan/models/llama3/infra/parallelize.py:96-104
  - 在模型初始化时应用 AC
```

---

## 13. 参考资料

**源码文件**：
- `torchtitan/distributed/activation_checkpoint.py` - AC 实现
- `torchtitan/config/job_config.py:585-613` - AC 配置
- `torchtitan/models/llama3/infra/parallelize.py:34-44` - Op SAC save list

**PyTorch 官方文档**：
- [Checkpoint](https://pytorch.org/docs/stable/checkpoint.html)
- [Activation Checkpointing](https://pytorch.org/docs/stable/distributed.algorithms.html#activation-checkpointing)
- [Memory Efficient Training](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

**相关文档**：
- [01_fsdp2_per_parameter_sharding.md](./01_fsdp2_per_parameter_sharding.md) - FSDP2 实现
- [02_tensor_parallel_implementation.md](./02_tensor_parallel_implementation.md) - TP 实现

**学术论文**：
- Training Deep Nets with Sublinear Memory Cost (Gradient Checkpointing)
- Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization

---

**最后更新**：2025年1月

**文档版本**：1.0
