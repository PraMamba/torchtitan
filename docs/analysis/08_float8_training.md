# TorchTitan Float8 Training 实现详解

## 目录
1. [什么是 Float8 Training](#1-什么是-float8-training)
2. [搬桌子比喻：压缩搬运](#2-搬桌子比喻压缩搬运)
3. [Float8 vs BFloat16/Float32](#3-float8-vs-bfloat16float32)
4. [两种 Scaling 策略](#4-两种-scaling-策略)
5. [Float8 与 FSDP 的结合](#5-float8-与-fsdp-的结合)
6. [Float8 与 TP 的结合](#6-float8-与-tp-的结合)
7. [源码实现详解](#7-源码实现详解)
8. [配置和使用](#8-配置和使用)
9. [性能数据](#9-性能数据)
10. [最佳实践](#10-最佳实践)
11. [总结](#11-总结)
12. [参考资料](#12-参考资料)

---

## 1. 什么是 Float8 Training

### 核心思想

**Float8 Training** 是一种**低精度训练技术**，通过使用 8 位浮点数（Float8）代替传统的 16 位浮点数（BFloat16）或 32 位浮点数（Float32），在保持模型精度的同时：

1. **加速计算**：利用 GPU 的 FP8 Tensor Core，比 BF16 Tensor Core 更快
2. **节省带宽**：通信和内存访问的数据量减半（8 bit vs 16 bit）
3. **降低内存**：模型参数和激活值占用更少显存

### Float8 格式

Float8 有两种常见格式，PyTorch 主要使用 **E4M3**（4 bits 指数，3 bits 尾数）：

```
Float8 E4M3: [S][EEEE][MMM]
  - 1 bit 符号位 (Sign)
  - 4 bits 指数 (Exponent)
  - 3 bits 尾数 (Mantissa)

对比：
  - BFloat16: [S][EEEEEEEE][MMMMMMM]  (8 bits 指数, 7 bits 尾数)
  - Float32:  [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]  (8 bits 指数, 23 bits 尾数)
```

**关键特点**：
- ✅ **指数范围适中**：4 bits 指数可以表示较大范围的数值
- ⚠️ **精度有限**：只有 3 bits 尾数，需要通过 **scales（缩放因子）** 来保持精度
- 🚀 **硬件加速**：H100 GPU 的 FP8 Tensor Core 峰值性能是 BF16 的 2 倍

---

## 2. 搬桌子比喻：压缩搬运

延续我们的"搬桌子"比喻系列，Float8 Training 就像**压缩搬运**。

### 场景：从仓库搬桌子到工地

**传统方式（BFloat16）**：
- 每张桌子用**标准货车**运输
- 每辆货车能装 **2 张桌子**
- 精确记录每张桌子的重量（kg）

```
货车 1: [桌子A: 45.3kg] [桌子B: 52.7kg]
货车 2: [桌子C: 38.9kg] [桌子D: 61.2kg]
```

**Float8 方式（压缩搬运）**：
- 每张桌子用**小型货车**运输（省油、更快）
- 每辆货车仍能装 **2 张桌子**（体积减半）
- 但是！重量记录精度降低，只能记录到 **整数 kg**
- 为了保持精度，我们记录一个 **缩放比例 (scale)**

```
小货车 1: [桌子A: 45kg] [桌子B: 53kg]  缩放比例: 1.0x
小货车 2: [桌子C: 39kg] [桌子D: 61kg]  缩放比例: 1.0x

实际重量 = 记录重量 × 缩放比例
```

### 为什么需要 Scale（缩放比例）？

假设桌子重量范围是 0-100kg，但我们只能用 **8 位整数（0-255）** 表示：

**不用 Scale 的问题**：
```
桌子A: 45.3kg → 存储为 45 (损失 0.3kg)
桌子B: 0.5kg  → 存储为 0  (❌ 完全丢失！)
```

**使用 Scale 的解决方案**：
```
找到这批桌子的最大绝对值：max = 100kg
计算 scale = 255 / 100 = 2.55

桌子A: 45.3kg × 2.55 = 115.5 → 存储为 116 → 恢复为 116 / 2.55 = 45.5kg ✓
桌子B:  0.5kg × 2.55 = 1.3   → 存储为 1   → 恢复为 1 / 2.55 = 0.39kg ✓
```

通过 **动态调整 scale**，我们可以充分利用 Float8 的表示范围！

### Tensorwise vs Rowwise Scaling

继续我们的比喻：

**Tensorwise Scaling（整车一个比例）**：
```
货车 1: [桌子A: 45kg] [桌子B: 53kg]  统一缩放比例: 1.0x
货车 2: [桌子C: 39kg] [桌子D: 61kg]  统一缩放比例: 1.0x

优点：简单快速，只需记录一个比例
缺点：如果某张桌子特别重（比如 200kg），其他轻桌子的精度会受影响
```

**Rowwise Scaling（每张桌子一个比例）**：
```
桌子A: 45kg  缩放比例: 1.1x
桌子B: 53kg  缩放比例: 0.9x
桌子C: 39kg  缩放比例: 1.2x
桌子D: 61kg  缩放比例: 0.8x

优点：每张桌子都有最优精度
缺点：需要记录更多比例，计算开销稍大
```

---

## 3. Float8 vs BFloat16/Float32

### 数值表示能力对比

| 数据类型 | 位数 | 指数位 | 尾数位 | 范围 | 精度 |
|---------|-----|-------|-------|------|------|
| **Float32** | 32 | 8 | 23 | ±3.4e38 | ~7 位十进制 |
| **BFloat16** | 16 | 8 | 7 | ±3.4e38 | ~3 位十进制 |
| **Float8 E4M3** | 8 | 4 | 3 | ±240 | ~1 位十进制 |

### 为什么 Float8 能训练深度模型？

虽然 Float8 精度很低，但在训练中：

1. **梯度更新是累积的**：单次计算精度低，但多次累积后精度足够
2. **Scale 动态调整**：通过 `max(abs(tensor))` 动态计算 scale，充分利用表示范围
3. **关键操作保持高精度**：优化器状态、梯度累积仍用 Float32

### 计算示例：Float8 矩阵乘法

**传统 BFloat16 矩阵乘法**：
```python
# 不需要 scale
output = torch.mm(input_bf16, weight_bf16)
```

**Float8 矩阵乘法**：
```python
# 需要 scale 来恢复正确的数值范围
output = torch._scaled_mm(
    input_fp8,              # Float8 输入
    weight_fp8,             # Float8 权重
    scale_a=scale_input,    # 输入的 scale
    scale_b=scale_weight,   # 权重的 scale
)
```

**Scale 的计算**：
```python
# 计算输入的 scale
amax_input = torch.max(torch.abs(input_bf16))
scale_input = 255.0 / amax_input  # Float8 E4M3 的最大值是 240，这里简化为 255

# 量化到 Float8
input_fp8 = (input_bf16 * scale_input).to(torch.float8_e4m3fn)

# 同样计算权重的 scale
amax_weight = torch.max(torch.abs(weight_bf16))
scale_weight = 255.0 / amax_weight
weight_fp8 = (weight_bf16 * scale_weight).to(torch.float8_e4m3fn)

# Float8 矩阵乘法
output_scaled = torch._scaled_mm(input_fp8, weight_fp8, scale_a=scale_input, scale_b=scale_weight)

# 输出已经自动 descale 回正确的数值范围
```

---

## 4. 两种 Scaling 策略

TorchTitan 支持两种 Float8 scaling 策略，对应 TorchAO 的两种 recipe。

### 4.1 Tensorwise Scaling（张量级缩放）

**定义**：整个 tensor 使用一个 scale。

```python
# Tensorwise scaling
amax = torch.max(torch.abs(tensor))  # 整个 tensor 的最大绝对值
scale = 255.0 / amax
tensor_fp8 = (tensor * scale).to(torch.float8_e4m3fn)
```

**优点**：
- ✅ **计算简单**：只需计算一个 amax
- ✅ **通信高效**：FSDP all-gather 时，每个参数只需通信一个 scale
- ✅ **速度快**：开销小，适合大规模训练

**缺点**：
- ⚠️ **精度受限**：如果 tensor 中有极端值，其他值的精度会受影响
- ⚠️ **不适合 outliers**：当 tensor 中有少数异常大/小的值时，精度损失明显

**适用场景**：
- 大规模分布式训练（FSDP + TP）
- 追求最大吞吐量
- 模型权重分布相对均匀

### 4.2 Rowwise Scaling（行级缩放）

**定义**：对于矩阵的每一行，使用独立的 scale。

```python
# Rowwise scaling（假设 tensor 是 2D）
amax_per_row = torch.max(torch.abs(tensor), dim=1, keepdim=True)[0]  # 每行的最大绝对值
scale_per_row = 255.0 / amax_per_row
tensor_fp8 = (tensor * scale_per_row).to(torch.float8_e4m3fn)
```

**优点**：
- ✅ **精度更高**：每行独立缩放，不受其他行影响
- ✅ **鲁棒性强**：对 outliers 不敏感
- ✅ **收敛更好**：在一些任务上收敛曲线更接近 BF16

**缺点**：
- ⚠️ **计算开销大**：需要计算每行的 amax
- ⚠️ **通信开销大**：FSDP/TP 通信时，需要传输更多 scales
- ⚠️ **编译友好性**：需要 `torch.compile` 来优化性能

**适用场景**：
- 对精度要求高的任务
- 权重分布不均匀（存在 outliers）
- 追求收敛质量而非极致速度

### 两种策略的性能对比

**Llama3 70B (256 H100s, FSDP=32, TP=8)**：

| 配置 | TPS/GPU | 相对 BF16 加速 |
|-----|---------|--------------|
| BFloat16 baseline | 597 | 1.00x |
| Float8 tensorwise | 810 | **1.36x** |
| Float8 rowwise | 600 | 1.00x |

**观察**：
- Tensorwise 在这个规模下有显著加速（1.36x）
- Rowwise 速度接近 BF16（因为计算和通信开销抵消了 FP8 Tensor Core 的优势）
- 如果配合 AsyncTP，tensorwise 可以达到 1.16x 加速（相对 BF16 + AsyncTP）

---

## 5. Float8 与 FSDP 的结合

### 5.1 传统 FSDP 的问题

回顾 FSDP 的工作流程：

```
Forward:
  1. All-Gather 权重分片（BFloat16）  ← 通信瓶颈
  2. 计算（BFloat16）
  3. Reshard 释放内存

Backward:
  1. All-Gather 权重分片（BFloat16）  ← 通信瓶颈
  2. 计算梯度（BFloat16）
  3. Reduce-Scatter 梯度（BFloat16） ← 通信瓶颈
  4. Reshard
```

**通信开销巨大**：All-Gather 和 Reduce-Scatter 是通信密集型操作。

### 5.2 Float8 All-Gather 优化

**核心思想**：在 All-Gather 之前，将权重从 BFloat16 转换为 Float8，通信量减半！

```
┌─────────────────────────────────────────────────────────────┐
│                     FSDP Float8 Forward                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Rank 0: [param_shard_0] (BFloat16, 存储)                  │
│  Rank 1: [param_shard_1] (BFloat16, 存储)                  │
│  Rank 2: [param_shard_2] (BFloat16, 存储)                  │
│  Rank 3: [param_shard_3] (BFloat16, 存储)                  │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 1: 本地量化（Cast to Float8）                         │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Rank 0: [param_shard_0_fp8] + scale_0                     │
│  Rank 1: [param_shard_1_fp8] + scale_1                     │
│  Rank 2: [param_shard_2_fp8] + scale_2                     │
│  Rank 3: [param_shard_3_fp8] + scale_3                     │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 2: All-Gather Float8 权重（通信量减半！）              │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  每个 Rank 都有: [param_fp8_full] = concat([shard_0_fp8,   │
│                                              shard_1_fp8,   │
│                                              shard_2_fp8,   │
│                                              shard_3_fp8])  │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 3: 计算全局 scale（All-Reduce scales）                │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  global_scale = max(scale_0, scale_1, scale_2, scale_3)    │
│  → 通过 All-Reduce 通信                                     │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 4: Float8 矩阵乘法                                    │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  output = torch._scaled_mm(input_fp8, param_fp8_full,      │
│                            scale_a=scale_input,             │
│                            scale_b=global_scale)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Precompute Scale for FSDP

**问题**：每个参数单独 All-Reduce scale，通信次数太多！

**解决方案**：`precompute_float8_dynamic_scale_for_fsdp`

```python
# 原始方式：每个参数单独 All-Reduce scale
for param in model.parameters():
    local_amax = torch.max(torch.abs(param))
    global_amax = torch.distributed.all_reduce(local_amax, op=ReduceOp.MAX)  # ← N 次通信！
    scale = 255.0 / global_amax
```

**优化方式**：将所有 scales 合并成一个 All-Reduce

```python
# TorchAO 的优化：一次 All-Reduce 通信所有 scales
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

# 在 optimizer step 之后调用
precompute_float8_dynamic_scale_for_fsdp(model)

# 原理：
# 1. 收集所有参数的 local amax
# 2. 打包成一个 tensor: [amax_0, amax_1, ..., amax_N]
# 3. 一次 All-Reduce 通信
# 4. 为每个参数计算 global scale
```

**性能提升**：
- ❌ 不优化：N 个参数 = N 次小的 All-Reduce（latency 高）
- ✅ 优化后：1 次大的 All-Reduce（latency 低，bandwidth 利用率高）

### 5.4 配置示例

```toml
[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true   # 启用 Float8 all-gather
precompute_float8_dynamic_scale_for_fsdp = true  # 优化 scale 通信
```

---

## 6. Float8 与 TP 的结合

### 6.1 TP 中的通信模式

回顾 TP 的通信模式（以 Colwise Parallel 为例）：

```
Input: [batch, seq_len, hidden]  (Replicate)
Weight: [hidden, ffn_dim]  (Shard on dim=1, 列切分)

Forward:
  1. 输入在所有 TP ranks 上是相同的（Replicate）
  2. 每个 rank 计算 matmul(input, weight_shard)
  3. 输出: [batch, seq_len, ffn_dim] (Shard on dim=-1)
```

**关键问题**：输入是 Replicate 的，但在 TP 场景下，我们需要计算**全局的 scale**。

### 6.2 Float8 TP 的实现

**Tensorwise Float8 TP**：

```
┌─────────────────────────────────────────────────────────────┐
│               Float8 TP Colwise Forward                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input (Replicate):  [batch, seq, hidden]                  │
│                                                             │
│  Rank 0: weight_shard_0 [hidden, ffn_dim/4]                │
│  Rank 1: weight_shard_1 [hidden, ffn_dim/4]                │
│  Rank 2: weight_shard_2 [hidden, ffn_dim/4]                │
│  Rank 3: weight_shard_3 [hidden, ffn_dim/4]                │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 1: 计算 Input 的全局 scale（需要在 TP group 通信）    │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  local_amax_input = max(abs(input))  # 每个 rank 相同      │
│  global_amax_input = local_amax_input  # TP 中 input 是 replicate 的 │
│  scale_input = 255.0 / global_amax_input                   │
│  input_fp8 = cast_to_fp8(input, scale_input)               │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 2: 计算 Weight 的全局 scale（需要在 TP group 通信）   │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Rank 0: local_amax_0 = max(abs(weight_shard_0))           │
│  Rank 1: local_amax_1 = max(abs(weight_shard_1))           │
│  Rank 2: local_amax_2 = max(abs(weight_shard_2))           │
│  Rank 3: local_amax_3 = max(abs(weight_shard_3))           │
│                                                             │
│  global_amax_weight = All-Reduce(local_amax, op=MAX)       │
│  → 在 TP group 内通信                                       │
│                                                             │
│  scale_weight = 255.0 / global_amax_weight                 │
│  weight_fp8 = cast_to_fp8(weight_shard, scale_weight)      │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 3: Float8 矩阵乘法                                    │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  output_fp8 = torch._scaled_mm(input_fp8, weight_fp8,      │
│                                scale_a=scale_input,         │
│                                scale_b=scale_weight)        │
│                                                             │
│  Output: [batch, seq, ffn_dim/4] (Shard on dim=-1)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Rowwise Float8 TP**：

对于 Rowwise scaling，通信开销更大，因为每一行都需要通信 scale：

```python
# Rowwise scaling in TP
amax_per_row_local = max(abs(weight_shard), dim=1)  # 每行的 local amax
# 需要 All-Reduce 每一行的 amax（通信量大！）
amax_per_row_global = torch.distributed.all_reduce(amax_per_row_local, op=ReduceOp.MAX)
```

**这就是为什么 Rowwise Float8 在 TP 中性能提升不明显**：通信开销抵消了 FP8 Tensor Core 的优势。

### 6.3 Float8 All-Gather for TP

在某些 TP 模式下（例如 Sequence Parallel），输入也是 Shard 的，需要 All-Gather：

```
┌─────────────────────────────────────────────────────────────┐
│       Float8 TP with Sequence Parallel (All-Gather)         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input (Shard on seq_len):                                 │
│    Rank 0: [batch, seq_len/4, hidden]                      │
│    Rank 1: [batch, seq_len/4, hidden]                      │
│    Rank 2: [batch, seq_len/4, hidden]                      │
│    Rank 3: [batch, seq_len/4, hidden]                      │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 1: Cast input to Float8 + compute scale              │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  local_amax = max(abs(input_shard))                        │
│  global_amax = All-Reduce(local_amax, op=MAX)  ← 通信 scale│
│  scale_input = 255.0 / global_amax                         │
│  input_fp8_shard = cast_to_fp8(input_shard, scale_input)   │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 2: Float8 All-Gather（通信量减半！）                  │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  input_fp8_full = All-Gather(input_fp8_shard)              │
│  → 每个 rank: [batch, seq_len, hidden] (Float8)           │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Step 3: Float8 matmul                                     │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  output = torch._scaled_mm(input_fp8_full, weight_fp8, ...)│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 源码实现详解

### 7.1 Float8LinearConverter 类

文件：`torchtitan/components/quantization/float8.py`

这是 TorchTitan 中负责将模型转换为 Float8 的核心类。

```python
class Float8LinearConverter(QuantizationConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config, parallel_dims)
        float8_config: Float8Linear = job_config.quantize.linear.float8

        # 1. 检查硬件支持（需要 SM89 或更高，即 H100+）
        if has_cuda_capability(8, 9) or (
            float8_config.emulate and not model_compile_enabled
        ):
            pass
        else:
            raise ValueError(
                "Float8 is only supported on SM89 or later (H100+)."
            )

        # 2. 导入 TorchAO 的 Float8LinearConfig
        from torchao.float8 import Float8LinearConfig as TorchAOFloat8LinearConfig

        # 3. 根据 recipe_name 或手动配置创建 config
        if float8_config.recipe_name is not None:
            # 使用预定义的 recipe（tensorwise, rowwise, rowwise_with_gw_hp）
            self.config = TorchAOFloat8LinearConfig.from_recipe_name(
                float8_config.recipe_name
            )
            self.precompute_scale = False
        else:
            # 手动配置 tensorwise scaling
            enable_fsdp_float8_all_gather = (
                parallel_dims.dp_shard_enabled
                and float8_config.enable_fsdp_float8_all_gather
            )
            self.config = TorchAOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
                emulate=float8_config.emulate,
            )
            # 是否启用 precompute_scale 优化
            self.precompute_scale = (
                enable_fsdp_float8_all_gather
                and float8_config.precompute_float8_dynamic_scale_for_fsdp
            )

        # 4. 初始化过滤函数（哪些层不转换为 Float8）
        self.filter_fn = self._init_filter_fn(float8_config)

        self.enabled = True
```

**关键点**：
1. **硬件检查**：Float8 需要 H100+ GPU（SM89），否则只能用 `emulate=True` 模拟（性能差）
2. **Recipe 选择**：可以用预定义 recipe（tensorwise, rowwise）或手动配置
3. **FSDP 优化**：通过 `enable_fsdp_float8_all_gather` 启用 Float8 all-gather
4. **Precompute Scale**：通过 `precompute_float8_dynamic_scale_for_fsdp` 减少通信次数

### 7.2 模型转换：convert 方法

```python
def convert(self, model: nn.Module):
    """
    将模型的 nn.Linear 层转换为 Float8Linear。
    """
    if not self.enabled:
        return

    from torchao.float8 import convert_to_float8_training

    # 调用 TorchAO 的转换函数
    convert_to_float8_training(
        model,
        config=self.config,
        module_filter_fn=self.filter_fn,  # 过滤不需要转换的层
    )
    logger.info(
        f"Swapped to Float8Linear layers with enable_fsdp_float8_all_gather="
        f"{self.config.enable_fsdp_float8_all_gather}"
    )
```

**convert_to_float8_training 做了什么？**

1. 遍历模型的所有 `nn.Linear` 层
2. 根据 `module_filter_fn` 决定是否转换（例如跳过 `output` 层）
3. 将 `nn.Linear` 替换为 `Float8Linear`
4. `Float8Linear` 的 forward 会自动处理 Float8 量化和 scaled_mm

### 7.3 Precompute Scale 优化

```python
def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
    """
    在 optimizer step 之后调用，预计算所有参数的 Float8 scales。
    """
    if not self.enabled:
        return

    if not self.precompute_scale:
        return

    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

    models = [model] if isinstance(model, nn.Module) else model
    for m in models:
        precompute_float8_dynamic_scale_for_fsdp(m)
```

**precompute_float8_dynamic_scale_for_fsdp 的实现原理**：

```python
# 伪代码：TorchAO 中的实现
def precompute_float8_dynamic_scale_for_fsdp(model):
    # 1. 收集所有 Float8Linear 层的参数
    params = []
    for module in model.modules():
        if isinstance(module, Float8Linear):
            params.append(module.weight)

    # 2. 计算每个参数的 local amax
    local_amaxs = []
    for param in params:
        local_amax = torch.max(torch.abs(param))
        local_amaxs.append(local_amax)

    # 3. 打包成一个 tensor，一次性 All-Reduce
    local_amaxs_tensor = torch.stack(local_amaxs)
    global_amaxs_tensor = torch.distributed.all_reduce(
        local_amaxs_tensor,
        op=ReduceOp.MAX
    )

    # 4. 为每个参数缓存 scale
    for i, param in enumerate(params):
        global_amax = global_amaxs_tensor[i]
        scale = 255.0 / global_amax
        param._float8_scale = scale  # 缓存 scale
```

**性能对比**：

| 方法 | 通信次数 | Latency |
|-----|---------|---------|
| 不优化 | N 次 All-Reduce（N = 参数数量） | 高 |
| Precompute | 1 次 All-Reduce | 低 |

对于 Llama3 70B（~80 个 Linear 层），从 80 次通信降到 1 次！

### 7.4 Filter FQNs：选择性转换

**为什么需要过滤？**

并非所有 Linear 层都适合 Float8：
1. **小矩阵**：矩阵太小时，量化开销 > FP8 Tensor Core 收益
2. **精度敏感层**：某些层（如 output projection）对精度要求高

**配置示例**：

```toml
[quantize.linear.float8]
filter_fqns = ["output", "attention.wk"]  # 不转换这些层
```

**Auto Filter**：

TorchTitan 支持自动过滤小矩阵：

```toml
[quantize.linear.float8]
filter_fqns = ["auto_filter_small_kn"]  # 自动过滤 K,N 维度过小的层
```

**实现原理**：

```python
def _init_filter_fn(self, float8_config: Float8Linear):
    use_auto_filter = "auto_filter_small_kn" in float8_config.filter_fqns
    if use_auto_filter:
        from torchao.float8 import _auto_filter_for_recipe

        # 根据 recipe 自动决定阈值
        return _auto_filter_for_recipe(
            recipe_name,
            filter_fqns=float8_config.filter_fqns,
        )

    # 手动过滤
    return partial(module_filter_fn, filter_fqns=float8_config.filter_fqns)
```

**Auto filter 的阈值**（基于 H100 microbenchmark）：

| Recipe | K 阈值 | N 阈值 |
|--------|-------|-------|
| tensorwise | K ≥ 2048 | N ≥ 2048 |
| rowwise | K ≥ 4096 | N ≥ 4096 |

只有当矩阵的 K 和 N 都超过阈值时，才转换为 Float8。

### 7.5 与并行策略的集成

在 `torchtitan/models/llama3/infra/parallelize.py` 中：

```python
def parallelize_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    # 1. 应用 TP
    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # Tensorwise Float8 支持 Float8 all-gather in TP
        # Rowwise Float8 使用高精度通信
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,  # ← 传递给 TP
        )

    # 2. 应用 AC
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint, ...)

    # 3. 应用 torch.compile
    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    # 4. 应用 FSDP
    if parallel_dims.fsdp_enabled:
        apply_fsdp(model, ...)
```

**关键顺序**：
1. **先 TP，后 FSDP**：这样 Float8 量化发生在 TP 通信时
2. **先 AC，后 Compile**：确保 checkpoint wrapper 能被编译
3. **Float8 转换在所有并行策略之前**：通过 `model_converter` 机制

---

## 8. 配置和使用

### 8.1 Tensorwise Float8 配置

**最常用的配置**（推荐用于大规模训练）：

```toml
[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
filter_fqns = ["auto_filter_small_kn"]  # 自动过滤小矩阵

[compile]
enable = true
components = ["model", "loss"]  # Float8 需要 compile 才能达到最佳性能
```

**命令行启动**：

```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh \
  --model.converters="quantize.linear.float8" \
  --quantize.linear.float8.enable_fsdp_float8_all_gather \
  --quantize.linear.float8.precompute_float8_dynamic_scale_for_fsdp \
  --compile.enable
```

### 8.2 Rowwise Float8 配置

**追求精度的配置**（适合小规模或精度敏感任务）：

```toml
[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
recipe_name = "rowwise"  # 使用 rowwise scaling
# 不启用 enable_fsdp_float8_all_gather（rowwise 通信开销大）

[compile]
enable = true
components = ["model", "loss"]  # Rowwise 更依赖 compile 优化
```

**命令行启动**：

```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh \
  --model.converters="quantize.linear.float8" \
  --quantize.linear.float8.recipe_name=rowwise \
  --compile.enable
```

### 8.3 手动过滤特定层

**跳过精度敏感层**：

```toml
[quantize.linear.float8]
filter_fqns = ["output", "attention.wk", "attention.wv"]
```

**如何确定哪些层需要过滤？**

1. **查看 TorchAO 的 microbenchmark**：[torchao/float8 performance](https://github.com/pytorch/ao/tree/main/torchao/float8#performance)
2. **实验验证**：训练时监控 loss 曲线，如果 Float8 收敛明显变差，尝试过滤更多层
3. **经验规则**：
   - `output` projection 通常需要过滤（影响最终 logits）
   - 小于 2048x2048 的矩阵建议过滤
   - MoE 的 gate 层通常需要高精度

### 8.4 Llama3 各模型配置

**Llama3 8B (8 GPUs)**：

```toml
[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 1

[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
filter_fqns = ["auto_filter_small_kn"]

[compile]
enable = true

[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"
```

**Llama3 70B (256 GPUs)**：

```toml
[parallelism]
data_parallel_shard_degree = 32
tensor_parallel_degree = 8

[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
filter_fqns = ["output"]

[compile]
enable = true

[activation_checkpoint]
mode = "full"
```

**Llama3 405B (512 GPUs)**：

```toml
[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 8
pipeline_parallel_degree = 8
enable_async_tensor_parallel = true

[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
filter_fqns = ["output"]

[compile]
enable = true

[activation_checkpoint]
mode = "full"
```

---

## 9. 性能数据

### 9.1 Llama3 8B (8 H100s)

| 配置 | TPS/GPU | 显存 (GiB) | 相对 Baseline 加速 |
|-----|---------|-----------|-----------------|
| FSDP (baseline) | 5,762 | 68.2 | 1.00x |
| FSDP + compile | 6,667 | 77.0 | 1.16x |
| FSDP + compile + Float8 | **8,532** | 76.8 | **1.48x** |

**观察**：
- Float8 在小规模（单机 8 卡）也有显著加速（1.48x）
- 显存占用几乎不变（因为激活值仍是 BF16，只有权重通信用 Float8）

### 9.2 Llama3 70B (256 H100s)

**配置**：FSDP=32, TP=8, local batch size=16, Full AC

| 配置 | TPS/GPU | 显存 (GiB) | 相对 Baseline 加速 |
|-----|---------|-----------|-----------------|
| FSDP + TP + compile (baseline) | 597 | 65.5 | 1.00x |
| + Float8 tensorwise | **810** | 64.8 | **1.36x** |
| + Float8 tensorwise + AsyncTP | **942** | 64.8 | **1.58x** |

**观察**：
- Float8 在大规模训练中收益更大（通信瓶颈明显）
- 配合 AsyncTP，可以达到 1.58x 加速！
- 显存占用略微降低（因为 FSDP all-gather 的临时缓冲区减小）

### 9.3 Llama3 405B (512 H100s)

**配置**：FSDP=8, TP=8, PP=8, AsyncTP, local batch size=32, Full AC, Interleaved 1F1B

| 配置 | TPS/GPU | 显存 (GiB) |
|-----|---------|-----------|
| FSDP + TP + PP + compile + Float8 + AsyncTP | **128** | 77.2 |

**说明**：
- 405B 必须使用 Float8 才能在 512 卡上高效训练
- Float8 节省的通信带宽使得 3D 并行更高效

### 9.4 Float8 Tensorwise vs Rowwise

**Llama3 70B (256 H100s, FSDP=32, TP=8)**

| Scaling 策略 | TPS/GPU | 相对 BF16 加速 | 收敛性 |
|------------|---------|--------------|-------|
| BFloat16 baseline | 597 | 1.00x | ✓ |
| Float8 tensorwise | 810 | 1.36x | ✓ (与 BF16 基本一致) |
| Float8 rowwise | 600 | 1.00x | ✓✓ (略好于 BF16) |

**观察**：
- Tensorwise 速度快，收敛性好（推荐）
- Rowwise 速度与 BF16 接近（通信开销抵消了收益），但收敛性稍好

### 9.5 Float8 + AsyncTP 的叠加效果

**Llama3 70B (256 H100s)**

| 配置 | TPS/GPU | 相对 Vanilla TP 加速 |
|-----|---------|-------------------|
| Vanilla TP (BF16) | 597 | 1.00x |
| Vanilla TP + Float8 tensorwise | 810 | 1.36x |
| AsyncTP (BF16) | 652 | 1.09x |
| AsyncTP + Float8 tensorwise | **942** | **1.58x** |

**观察**：
- Float8 和 AsyncTP 的加速效果可以**叠加**！
- Float8 降低通信量，AsyncTP 隐藏通信延迟，两者互补

---

## 10. 最佳实践

### 10.1 什么时候使用 Float8？

✅ **推荐使用**：
1. **大规模分布式训练**：世界大小 ≥ 64 GPUs，通信瓶颈明显
2. **TP 并行度高**：TP ≥ 8，通信量大
3. **大矩阵为主**：模型中大部分 Linear 层的 K, N ≥ 2048
4. **H100+ GPU**：有硬件 FP8 Tensor Core 支持

❌ **不推荐使用**：
1. **小规模训练**：单机 ≤ 8 GPUs，通信不是瓶颈
2. **小模型**：模型 < 1B 参数，矩阵太小
3. **精度要求极高**：科学计算、金融模型等
4. **老硬件**：A100 或更早的 GPU（可以用 `emulate=True` 测试，但没有加速）

### 10.2 Tensorwise vs Rowwise 如何选择？

| 场景 | 推荐策略 | 理由 |
|-----|---------|------|
| 大规模训练（≥256 GPUs） | **Tensorwise** | 通信高效，速度快 |
| 中小规模训练（<256 GPUs） | Rowwise 或 不用 Float8 | Rowwise 通信开销大，收益不明显 |
| 追求极致吞吐量 | **Tensorwise** | 最快 |
| 追求收敛质量 | **Rowwise** | 精度高，鲁棒性强 |
| 模型有 outliers | **Rowwise** | 每行独立缩放，不受极端值影响 |
| 配合 AsyncTP | **Tensorwise** | 两者叠加效果最好 |

### 10.3 调优 Checklist

1. **启用 torch.compile**：Float8 需要 compile 来融合量化/反量化 kernel
   ```toml
   [compile]
   enable = true
   components = ["model", "loss"]
   ```

2. **使用 Auto Filter**：自动跳过小矩阵
   ```toml
   [quantize.linear.float8]
   filter_fqns = ["auto_filter_small_kn"]
   ```

3. **启用 Precompute Scale**（Tensorwise）：减少通信次数
   ```toml
   [quantize.linear.float8]
   precompute_float8_dynamic_scale_for_fsdp = true
   ```

4. **过滤 Output Layer**：保持最终输出的高精度
   ```toml
   [quantize.linear.float8]
   filter_fqns = ["output", "auto_filter_small_kn"]
   ```

5. **配合 AsyncTP**：叠加加速效果
   ```toml
   [parallelism]
   enable_async_tensor_parallel = true
   ```

6. **监控收敛性**：对比 BF16 baseline 的 loss 曲线
   - 如果 Float8 loss 明显偏高，尝试：
     - 过滤更多层（`filter_fqns`）
     - 切换到 Rowwise scaling
     - 降低学习率

### 10.4 常见问题

**Q1: 为什么我的 Float8 没有加速？**

A: 可能的原因：
1. 矩阵太小：大部分 Linear 层被 auto_filter 过滤了
2. 没有启用 compile：Float8 kernel 没有融合
3. 通信不是瓶颈：小规模训练（<64 GPUs）
4. 使用 Rowwise：在中小规模下，Rowwise 通信开销大

**Q2: Float8 会影响收敛吗？**

A: 一般不会。在大多数任务上：
- Tensorwise Float8：收敛曲线与 BF16 基本一致
- Rowwise Float8：收敛曲线略好于 BF16（精度更高）

但在某些精度敏感任务（例如长序列、小 batch size），可能需要：
- 过滤精度敏感层（如 output）
- 使用 Rowwise scaling
- 微调学习率

**Q3: Float8 支持哪些并行策略？**

A: Float8 与 TorchTitan 的所有并行策略兼容：
- ✅ FSDP：支持 Float8 all-gather
- ✅ TP：Tensorwise 支持 Float8 通信，Rowwise 使用高精度通信
- ✅ PP：支持（但 Float8 主要优化通信，对 PP 收益有限）
- ✅ CP：支持
- ✅ AsyncTP：完美配合，叠加加速

**Q4: 如何调试 Float8？**

1. **对比 BF16 baseline**：
   ```bash
   # 先跑 BF16 baseline
   ./run_train.sh  # 不加 --model.converters

   # 再跑 Float8
   ./run_train.sh --model.converters="quantize.linear.float8" ...
   ```

2. **检查哪些层被转换**：
   ```python
   # 在训练脚本中打印模型结构
   print(model)
   # Float8Linear 会显示 Float8Linear 而不是 nn.Linear
   ```

3. **监控通信量**：
   ```bash
   # 使用 NCCL 调试
   export NCCL_DEBUG=INFO
   # 查看 all-gather/reduce-scatter 的 size（Float8 应该是 BF16 的一半）
   ```

4. **Profiling**：
   ```toml
   [profiling]
   enable_profiling = true
   # 检查 Float8 kernel 的时间占比
   ```

---

## 11. 总结

### Float8 Training 的核心要点

1. **本质**：用 8 位浮点数代替 16 位，通过**动态 scale** 保持精度
   - Float8 E4M3: 4 bits 指数，3 bits 尾数
   - Scale = 255 / max(abs(tensor))

2. **两种 Scaling 策略**：
   - **Tensorwise**：整个 tensor 一个 scale（快，适合大规模）
   - **Rowwise**：每行一个 scale（精度高，通信开销大）

3. **与分布式训练结合**：
   - **FSDP Float8 all-gather**：通信量减半
   - **TP Float8**：权重和激活都用 Float8（Tensorwise）
   - **Precompute Scale**：将 N 次 All-Reduce 合并为 1 次

4. **性能提升**：
   - Llama3 8B (8 GPUs): **1.48x** 加速
   - Llama3 70B (256 GPUs): **1.36x** 加速（Float8）→ **1.58x**（Float8 + AsyncTP）
   - Llama3 405B (512 GPUs): 必须使用 Float8 才能高效训练

5. **最佳实践**：
   - ✅ 大规模训练（≥64 GPUs）
   - ✅ 启用 torch.compile
   - ✅ 使用 auto_filter 跳过小矩阵
   - ✅ Tensorwise + Precompute Scale + AsyncTP 组合

### 搬桌子比喻总结

Float8 Training 就像**压缩搬运**：

```
传统 BF16: 标准货车运输，精确记录重量（kg，小数点后 1 位）
Float8:    小型货车运输，记录重量 + 缩放比例
           → 货车更快、更省油
           → 通过缩放比例恢复精度

Tensorwise: 整车一个比例（简单快速）
Rowwise:    每张桌子一个比例（精度更高，但需要记录更多比例）
```

### 技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                   Float8 Training Stack                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TorchTitan (Integration Layer)                            │
│  ├─ Float8LinearConverter: 模型转换                         │
│  ├─ Precompute Scale: 通信优化                              │
│  └─ Filter FQNs: 选择性应用                                 │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  TorchAO (Implementation Layer)                            │
│  ├─ Float8Linear: Float8 矩阵乘法                           │
│  ├─ Float8LinearConfig: 配置管理                            │
│  ├─ convert_to_float8_training: 模型转换                    │
│  └─ Recipes: tensorwise, rowwise, ...                      │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  PyTorch (Kernel Layer)                                    │
│  ├─ torch._scaled_mm: Float8 矩阵乘法 kernel                │
│  ├─ torch.compile: Kernel 融合                              │
│  └─ torch.float8_e4m3fn: Float8 数据类型                    │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  CUDA (Hardware Layer)                                     │
│  ├─ FP8 Tensor Core: 硬件加速（H100+）                      │
│  └─ CUTLASS: 高性能 GEMM library                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. 参考资料

### TorchTitan 文档
- [docs/float8.md](../../docs/float8.md) - Float8 使用指南
- [benchmarks/llama3_h100_202412_torchtitan.md](../../benchmarks/llama3_h100_202412_torchtitan.md) - 性能 Benchmark

### TorchAO 文档
- [torchao/float8](https://github.com/pytorch/ao/tree/main/torchao/float8) - Float8 实现和 API
- [torchao/float8 Performance](https://github.com/pytorch/ao/tree/main/torchao/float8#performance) - Microbenchmark

### PyTorch 文档
- [torch.float8_e4m3fn](https://pytorch.org/docs/stable/tensors.html#torch.float8_e4m3fn) - Float8 数据类型
- [torch._scaled_mm](https://pytorch.org/docs/stable/generated/torch._scaled_mm.html) - Float8 矩阵乘法

### 学术论文
- **FP8 Formats for Deep Learning**: [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
- **FP8 Training**: NVIDIA 的 FP8 训练白皮书

### 源码位置
- `torchtitan/components/quantization/float8.py` - Float8 转换器
- `torchtitan/config/job_config.py:667-689` - Float8 配置
- `torchtitan/models/llama3/infra/parallelize.py:69-86` - Float8 与 TP 集成

---

**最后更新**：2025年11月25日
