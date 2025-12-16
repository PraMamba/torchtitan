# TorchTitan torch.compile 实现详解

## 目录
1. [什么是 torch.compile](#1-什么是-torchcompile)
2. [搬桌子比喻：流水线自动化](#2-搬桌子比喻流水线自动化)
3. [torch.compile 的工作原理](#3-torchcompile-的工作原理)
4. [TorchTitan 中的编译策略](#4-torchtitan-中的编译策略)
5. [模型编译实现](#5-模型编译实现)
6. [Loss 编译实现](#6-loss-编译实现)
7. [与并行策略的结合](#7-与并行策略的结合)
8. [源码实现详解](#8-源码实现详解)
9. [配置和使用](#9-配置和使用)
10. [性能数据](#10-性能数据)
11. [最佳实践](#11-最佳实践)
12. [总结](#12-总结)
13. [参考资料](#13-参考资料)

---

## 1. 什么是 torch.compile

### 核心思想

**torch.compile** 是 PyTorch 2.0 引入的**编译器技术**，通过将 Python 代码编译为优化的机器码，提升训练和推理性能。

**核心优化**：
1. **Kernel Fusion（算子融合）**：将多个小算子合并成一个大算子，减少内存访问
2. **Graph Optimization（图优化）**：消除冗余计算，优化计算顺序
3. **Code Generation（代码生成）**：生成高度优化的 CUDA kernel

**传统 Eager Mode vs Compile Mode**：

```python
# Eager Mode（传统方式）
def model_forward(x):
    x = layer_norm(x)      # Kernel 1: LayerNorm
    x = linear(x)          # Kernel 2: Linear
    x = gelu(x)            # Kernel 3: GELU
    return x

# 每个操作独立执行：
# 1. 从 HBM 读取 x → LayerNorm → 写回 HBM
# 2. 从 HBM 读取 x → Linear → 写回 HBM
# 3. 从 HBM 读取 x → GELU → 写回 HBM
# → 6 次内存访问（3 读 + 3 写）

# Compile Mode（编译后）
@torch.compile
def model_forward(x):
    x = layer_norm(x)
    x = linear(x)
    x = gelu(x)
    return x

# 编译器融合算子：
# 1. 从 HBM 读取 x → LayerNorm + Linear + GELU（融合） → 写回 HBM
# → 2 次内存访问（1 读 + 1 写）
# 内存访问减少 67%！
```

### 为什么 torch.compile 能加速？

**核心原因**：**内存带宽是瓶颈**！

在 GPU 上，**内存带宽（Memory Bandwidth）** 往往是瓶颈，而不是计算能力：

```
H100 GPU 规格：
  - 计算能力（BF16）: 1000 TFLOPS
  - 内存带宽: 3.35 TB/s

假设一个简单的 elementwise 操作：y = x + 1
  - 计算量：N 次加法
  - 内存访问：读 N 个元素，写 N 个元素

瓶颈分析：
  - 如果 N = 1M，数据大小 = 2MB（BF16）
  - 计算时间 ≈ 1M / 1000e12 ≈ 1 纳秒
  - 内存访问时间 ≈ 2MB / 3.35TB/s ≈ 0.6 微秒
  - 内存访问时间 >> 计算时间（600x）
```

**torch.compile 的优化策略**：
1. **减少内存访问次数**：通过算子融合
2. **提高内存访问效率**：通过生成优化的 CUDA kernel
3. **消除冗余计算**：通过图优化

---

## 2. 搬桌子比喻：流水线自动化

延续我们的"搬桌子"比喻系列，torch.compile 就像**流水线自动化**。

### 场景：工厂搬运桌子到仓库

**传统方式（Eager Mode）**：
每道工序独立进行，工人手动搬运

```
┌─────────────────────────────────────────────────────────────┐
│              传统手工搬运（Eager Mode）                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  工人 A: 从车间拿桌子 → 放到临时区 A                         │
│           ↓                                                 │
│  工人 B: 从临时区 A 拿桌子 → 打磨 → 放到临时区 B             │
│           ↓                                                 │
│  工人 C: 从临时区 B 拿桌子 → 上漆 → 放到临时区 C             │
│           ↓                                                 │
│  工人 D: 从临时区 C 拿桌子 → 搬到仓库                       │
│                                                             │
│  问题：                                                     │
│   - 需要 4 个临时存储区（内存访问多）                        │
│   - 每次都要从临时区拿桌子（带宽浪费）                       │
│   - 工人之间等待时间长（延迟高）                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**torch.compile 方式（Compile Mode）**：
流水线自动化，一气呵成

```
┌─────────────────────────────────────────────────────────────┐
│           自动化流水线（Compile Mode）                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  自动化机械臂:                                              │
│    从车间拿桌子 → 直接打磨 → 直接上漆 → 直接送仓库          │
│                   (一条流水线完成所有工序)                  │
│                                                             │
│  优化:                                                      │
│   ✅ 只需 1 个临时区（桌子在流水线上）                       │
│   ✅ 无需多次拿放桌子（内存访问少）                          │
│   ✅ 流水线连续运行（延迟低）                                │
│   ✅ 机械臂比人工更精准（代码优化）                          │
│                                                             │
│  编译过程 = 设计流水线（一次性成本）                         │
│  执行过程 = 流水线运行（持续收益）                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 编译的代价

**初次编译**（设计流水线）：
```
第一次运行: 需要时间设计流水线（编译开销）
  - 分析每道工序的顺序
  - 设计最优的流水线布局
  - 制造专用的机械臂
  → 第一批桌子会慢一些

后续运行: 流水线已就绪（无编译开销）
  → 后续所有桌子都很快
```

**这就是为什么 torch.compile 有 warmup 开销**：第一次迭代慢（编译），后续迭代快（复用编译结果）。

---

## 3. torch.compile 的工作原理

### 3.1 编译流程

torch.compile 使用 **TorchDynamo** 和 **TorchInductor** 两个组件：

```
┌─────────────────────────────────────────────────────────────┐
│                torch.compile 编译流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Python 代码                                             │
│     ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ TorchDynamo (动态追踪)                             │    │
│  │  - 捕获 Python bytecode                            │    │
│  │  - 构建计算图（FX Graph）                          │    │
│  │  - 处理 control flow、dynamic shapes              │    │
│  └────────────────────────────────────────────────────┘    │
│     ↓                                                       │
│  2. FX Graph (中间表示)                                     │
│     ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Graph Optimization (图优化)                        │    │
│  │  - 常量折叠（Constant Folding）                    │    │
│  │  - 死代码消除（Dead Code Elimination）             │    │
│  │  - 算子融合（Operator Fusion）                     │    │
│  └────────────────────────────────────────────────────┘    │
│     ↓                                                       │
│  3. Optimized Graph                                        │
│     ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ TorchInductor (代码生成)                           │    │
│  │  - 生成 Triton kernel（GPU）                       │    │
│  │  - 生成 C++ kernel（CPU）                          │    │
│  │  - 调用 cuBLAS/cuDNN 等库                          │    │
│  └────────────────────────────────────────────────────┘    │
│     ↓                                                       │
│  4. Compiled CUDA/C++ Code                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 关键概念

**TorchDynamo**：
- 在 Python 解释器层面工作
- 捕获 Python bytecode，构建计算图
- 支持动态控制流（if/for/while）
- 处理 dynamic shapes（变长序列）

**TorchInductor**：
- PyTorch 的默认 backend
- 使用 Triton 生成 GPU kernel
- 支持 kernel fusion、memory planning
- 生成高度优化的代码

**fullgraph=True**：
- 要求编译整个函数为一个图
- 如果遇到无法编译的代码，会报错（而不是回退到 eager mode）
- 提供更好的性能和确定性

### 3.3 算子融合示例

**LayerNorm + Linear + GELU 融合**：

```python
# Eager Mode: 3 个独立的 kernel
x = layer_norm(x)       # Kernel 1
x = linear(x)           # Kernel 2
x = gelu(x)             # Kernel 3

# Compile Mode: 融合为 1 个 kernel
@torch.compile
def fused_block(x):
    x = layer_norm(x)
    x = linear(x)
    x = gelu(x)
    return x

# 生成的伪代码（Triton）：
@triton.jit
def fused_kernel(x_ptr, out_ptr, ...):
    # 1. 加载 x 从 HBM
    x = tl.load(x_ptr + offsets)

    # 2. LayerNorm（寄存器中）
    mean = tl.sum(x) / N
    var = tl.sum((x - mean) ** 2) / N
    x = (x - mean) / tl.sqrt(var + eps)

    # 3. Linear（寄存器中）
    x = tl.dot(x, weight)

    # 4. GELU（寄存器中）
    x = x * 0.5 * (1 + tl.erf(x / 1.414))

    # 5. 写回 out 到 HBM
    tl.store(out_ptr + offsets, x)
```

**性能对比**：

| 方式 | Kernel 数 | HBM 访问次数 | 延迟 |
|-----|----------|-------------|------|
| Eager Mode | 3 | 6 次（3 读 + 3 写） | 高 |
| Compile Mode | 1 | 2 次（1 读 + 1 写） | 低 |

---

## 4. TorchTitan 中的编译策略

### 4.1 两个可编译组件

TorchTitan 支持编译两个组件：
1. **Model（模型）**：编译 Transformer layers
2. **Loss（损失函数）**：编译 cross-entropy loss

```toml
[compile]
enable = true
components = ["model", "loss"]  # 可选: ["model"], ["loss"], ["model", "loss"]
backend = "inductor"  # 默认 backend
```

### 4.2 Per-Block 编译策略

**为什么不编译整个模型？**

TorchTitan 使用 **Per-TransformerBlock 编译**，而不是编译整个模型：

```python
# ❌ 不推荐：编译整个模型
model = torch.compile(entire_model)

# ✅ 推荐：编译每个 TransformerBlock
for layer_id, transformer_block in model.layers.named_children():
    transformer_block = torch.compile(transformer_block, fullgraph=True)
    model.layers.register_module(layer_id, transformer_block)
```

**Per-Block 编译的优势**：

1. **编译时间短**：
   - 整个模型：80 层 → 编译 1 个巨大的图（可能需要几分钟）
   - Per-Block：80 层 → 编译 1 层，复用 80 次（只需几秒）

2. **内存占用低**：
   - 编译整个模型需要大量内存存储中间图
   - Per-Block 只需编译一个小图

3. **与 FSDP 兼容**：
   - FSDP 会在每层前后插入 all-gather/reduce-scatter hooks
   - Per-Block 编译可以将 hooks 排除在编译图之外

4. **与 AC 兼容**：
   - Activation Checkpointing 会在某些层插入 checkpoint boundaries
   - Per-Block 编译不会影响 AC 的效果

### 4.3 fullgraph=True

TorchTitan 使用 `fullgraph=True`，要求每个 Block 完整编译：

```python
transformer_block = torch.compile(
    transformer_block,
    backend="inductor",
    fullgraph=True  # ← 关键参数
)
```

**fullgraph=True 的作用**：
- ✅ 确保整个 Block 被编译（没有回退到 eager mode）
- ✅ 提供确定性的性能（要么全部编译，要么报错）
- ✅ 避免 graph break（图断裂）

**什么会导致 graph break？**

```python
# 示例：graph break
def transformer_block(x):
    x = attention(x)        # ← 可编译

    if x.shape[0] > 100:    # ← Graph break!（动态控制流）
        x = extra_layer(x)

    x = ffn(x)              # ← 可编译
    return x

# fullgraph=True 会报错，提示有 graph break
# 需要修改代码，消除动态控制流
```

### 4.4 编译时机

**编译发生在哪里？**

在 TorchTitan 的并行策略应用顺序中，编译发生在特定的位置：

```python
def parallelize_llama(model, parallel_dims, job_config):
    # 1. 应用 TP
    if parallel_dims.tp_enabled:
        apply_tp(model, ...)

    # 2. 应用 Activation Checkpointing
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, ...)

    # 3. 应用 torch.compile ← 在这里！
    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    # 4. 应用 FSDP
    if parallel_dims.fsdp_enabled:
        apply_fsdp(model, ...)
```

**为什么这个顺序？**

1. **先 TP，后 compile**：
   - TP 修改了模型结构（Colwise/Rowwise）
   - 编译看到的是 TP 之后的结构

2. **先 AC，后 compile**：
   - AC 插入 checkpoint wrappers
   - 编译时会自动识别并处理 AC

3. **先 compile，后 FSDP**：
   - FSDP 插入 all-gather/reduce-scatter hooks
   - 这些 hooks 不需要被编译（是通信操作）
   - 如果先 FSDP，后 compile，会导致 graph break

---

## 5. 模型编译实现

### 5.1 apply_compile 函数

文件：`torchtitan/models/llama3/infra/parallelize.py:236-248`

```python
def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        # 编译每个 TransformerBlock
        transformer_block = torch.compile(
            transformer_block,
            backend=compile_config.backend,  # 默认 "inductor"
            fullgraph=True                   # 要求完整编译
        )
        # 替换原来的 module
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")
```

### 5.2 编译的是什么？

**TransformerBlock 的结构**：

```python
class TransformerBlock(nn.Module):
    def __init__(self, ...):
        self.attention = Attention(...)
        self.feed_forward = FeedForward(...)
        self.attention_norm = RMSNorm(...)
        self.ffn_norm = RMSNorm(...)

    def forward(self, x):
        # Attention block
        h = x + self.attention(self.attention_norm(x))

        # FFN block
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
```

**编译后，整个 forward 方法变成一个优化的 kernel**：

```
Before compile:
  RMSNorm → Attention → Add (残差)
    ↓        ↓          ↓
  3 kernels

  RMSNorm → FFN → Add (残差)
    ↓       ↓      ↓
  3 kernels

Total: ~6 kernels

After compile:
  [RMSNorm + Attention + Add + RMSNorm + FFN + Add] (融合)
    ↓
  1-2 kernels (取决于融合程度)
```

### 5.3 与 Activation Checkpointing 的交互

如果启用了 AC，TransformerBlock 会被包装：

```python
# 应用 AC 后
transformer_block = CheckpointWrapper(transformer_block)

# 应用 compile 后
transformer_block = torch.compile(CheckpointWrapper(transformer_block))
```

**torch.compile 如何处理 AC？**

```python
class CheckpointWrapper(nn.Module):
    def forward(self, x):
        if self.training:
            return checkpoint(self._checkpoint_wrapped_module, x)  # ← 特殊处理
        else:
            return self._checkpoint_wrapped_module(x)

# torch.compile 识别 checkpoint 函数，将其视为边界：
# 1. 编译 checkpoint 内部的代码（_checkpoint_wrapped_module）
# 2. checkpoint 本身不编译（保留其重计算逻辑）
```

---

## 6. Loss 编译实现

### 6.1 build_cross_entropy_loss 函数

文件：`torchtitan/components/loss.py:26-32`

```python
def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),  # [batch*seq, vocab]
        labels.flatten(0, 1)         # [batch*seq]
    )


def build_cross_entropy_loss(job_config: JobConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = cross_entropy_loss

    # 编译 loss function
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)

    return loss_fn
```

### 6.2 为什么编译 Loss？

**Cross-Entropy Loss 的计算**：

```python
# 未编译的 cross-entropy
def cross_entropy_loss(pred, labels):
    pred = pred.flatten(0, 1)        # Kernel 1: Reshape
    pred = pred.float()              # Kernel 2: Cast to float32
    labels = labels.flatten(0, 1)    # Kernel 3: Reshape

    # Kernel 4-6: cross_entropy 内部
    # 4. exp(pred)
    # 5. sum(exp(pred))
    # 6. log(sum) - pred[labels]

    return loss

# Total: 6+ kernels
```

**编译后**：

```python
@torch.compile
def cross_entropy_loss(pred, labels):
    pred = pred.flatten(0, 1).float()
    labels = labels.flatten(0, 1)
    return torch.nn.functional.cross_entropy(pred, labels)

# 编译器融合：
# [Reshape + Cast + Reshape + exp + sum + log] → 1-2 kernels
```

**性能提升**：
- Llama3 8B: 编译 loss 可以额外提升 ~2-3% 速度
- 虽然提升不大，但几乎没有额外成本（loss 函数很小，编译很快）

---

## 7. 与并行策略的结合

### 7.1 torch.compile + FSDP

**挑战**：FSDP 插入了 all-gather/reduce-scatter hooks

```
Forward (FSDP):
  1. All-Gather 权重分片            ← 通信 hook
  2. Compute (TransformerBlock)     ← 可编译
  3. Reshard 权重                   ← 通信 hook
```

**解决方案**：先 compile，后 FSDP

```python
# 正确顺序
apply_compile(model, ...)   # 编译 TransformerBlock
apply_fsdp(model, ...)      # FSDP 包装，插入 hooks

# torch.compile 只编译 TransformerBlock.forward
# FSDP hooks 在编译图之外（不会被编译）
```

**为什么有效？**

```python
# FSDP 的实现
class FSDPTransformerBlock(nn.Module):
    def forward(self, x):
        # 1. All-Gather (not compiled)
        params = all_gather(self.sharded_params)

        # 2. Compute (compiled)
        with torch.no_grad():
            out = self.compiled_block(x, params)  # ← 已编译

        # 3. Reshard (not compiled)
        free(params)

        return out
```

### 7.2 torch.compile + TP

**TP 不会导致 graph break**：

```python
# TP 修改了 Linear 层的计算
class ColwiseParallel(nn.Linear):
    def forward(self, x):
        # x: [batch, seq, hidden]  (Replicate)
        # weight: [hidden, ffn_dim]  (Shard on dim=1)

        # Local matmul
        out = F.linear(x, self.weight)  # ← 可编译

        # 输出: [batch, seq, ffn_dim/tp_size]  (Shard)
        return out

# torch.compile 可以正常编译这个 forward
```

**性能提升**：
- TP 通信（all-reduce）不会被编译
- 但 TP 计算部分会被编译和优化

### 7.3 torch.compile + AsyncTP

**AsyncTP 必须配合 torch.compile**：

```python
def maybe_enable_async_tp(job_config, tp_mesh):
    if not (job_config.compile.enable and "model" in job_config.compile.components):
        raise AssertionError(
            "Async TP requires 'model' in --compile.components and --compile.enable"
        )

    # 启用 AsyncTP
    torch._inductor.config.make_comms_ops_symbolic = True
```

**为什么？**

AsyncTP 的通信重叠需要编译器支持：

```python
# 编译后的伪代码
def async_tp_forward(x):
    # 1. 启动通信（异步）
    comm_handle = async_all_reduce_start(...)

    # 2. 并行计算（与通信重叠）
    y = compute(x)

    # 3. 等待通信完成
    result = async_all_reduce_wait(comm_handle)

    return result

# torch.compile 需要识别异步通信 pattern，才能生成正确的代码
```

### 7.4 torch.compile + Float8

**Float8 量化/反量化可以被融合**：

```python
# 未编译
def float8_linear(x, weight_fp8, scale):
    # Kernel 1: Cast input to FP8
    x_fp8 = cast_to_fp8(x, scale_input)

    # Kernel 2: FP8 matmul
    out = torch._scaled_mm(x_fp8, weight_fp8, ...)

    # Kernel 3: Cast output to BF16
    out = out.to(torch.bfloat16)

    return out

# 编译后
@torch.compile
def float8_linear(x, weight_fp8, scale):
    x_fp8 = cast_to_fp8(x, scale_input)
    out = torch._scaled_mm(x_fp8, weight_fp8, ...)
    out = out.to(torch.bfloat16)
    return out

# 编译器融合 cast 和 matmul，减少中间 tensor
```

**性能提升**：
- Float8 + compile: **1.48x** 加速（相对 FSDP baseline）
- Float8 不compile: ~1.2x 加速
- compile 对 Float8 的效果更明显！

---

## 8. 源码实现详解

### 8.1 Compile 配置类

文件：`torchtitan/config/job_config.py:655-664`

```python
@dataclass
class Compile:
    enable: bool = False
    """Whether to apply torch.compile"""

    components: list[Literal["model", "loss"]] = field(
        default_factory=lambda: ["model", "loss"]
    )
    """Which components to compile"""

    backend: str = "inductor"
    """Compiler backend (default: inductor)"""
```

**配置选项**：
- `enable`: 是否启用编译
- `components`: 编译哪些组件（["model"], ["loss"], 或 ["model", "loss"]）
- `backend`: 编译 backend（默认 "inductor"，也可以用 "eager", "aot_eager" 等）

### 8.2 模型编译调用位置

文件：`torchtitan/models/llama3/infra/parallelize.py:89-108`

```python
def parallelize_llama(model, parallel_dims, job_config):
    # ...前面应用 TP...

    # 检查是否启用 model 编译
    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    # 应用 Activation Checkpointing
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,  # ← 传递给 AC
            ...
        )

    # 应用 torch.compile（在 AC 之后，FSDP 之前）
    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    # 应用 FSDP
    if parallel_dims.fsdp_enabled:
        apply_fsdp(model, ...)
```

### 8.3 Per-Block 编译实现

文件：`torchtitan/models/llama3/infra/parallelize.py:236-248`

```python
def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # 遍历所有 TransformerBlock
    for layer_id, transformer_block in model.layers.named_children():
        # 编译单个 Block
        transformer_block = torch.compile(
            transformer_block,
            backend=compile_config.backend,
            fullgraph=True  # 要求完整编译，遇到 graph break 报错
        )

        # 替换原 module
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")
```

**关键点**：
1. 只编译 `model.layers` 中的 TransformerBlock
2. 不编译 embedding、output projection、RoPE 等
3. 使用 `fullgraph=True` 确保没有 graph break

### 8.4 Loss 编译实现

文件：`torchtitan/components/loss.py:26-32`

```python
def build_cross_entropy_loss(job_config: JobConfig, **kwargs):
    del kwargs
    loss_fn = cross_entropy_loss

    # 检查是否编译 loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        # 编译 loss function
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)

    return loss_fn
```

**对比 model 编译**：
- Model: 编译每个 TransformerBlock（per-block）
- Loss: 编译整个 loss function（per-function）

### 8.5 MoE 模型的特殊处理

对于 MoE 模型（Llama4），需要特殊处理：

文件：`torchtitan/models/llama4/infra/parallelize.py:507-540`

```python
def apply_compile(model: nn.Module, compile_config: CompileConfig):
    # MoE 需要的特殊配置
    torch._dynamo.config.capture_scalar_outputs = True
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    for layer_id, transformer_block in model.layers.named_children():
        if transformer_block.moe_enabled:
            # MoE 层的 FSDP(GroupedExperts) 会导致 graph break
            # 需要特殊处理：在 FSDP hooks 周围编织 compile wrappers

            # 解包 CheckpointWrapper
            if isinstance(transformer_block, CheckpointWrapper):
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            # 找到 MoE submodule
            for attr_name, submod in block.named_children():
                if isinstance(submod, moe_module.MoE):
                    # 对 MoE 内部的专家进行编译
                    # （代码省略，较复杂）
                    ...
        else:
            # 非 MoE 层，正常编译
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True
            )
            model.layers.register_module(layer_id, transformer_block)
```

**MoE 的挑战**：
- MoE 使用 GroupedExperts（分组专家）
- FSDP 对 GroupedExperts 的包装会导致 graph break
- 需要细粒度地控制哪些部分编译

---

## 9. 配置和使用

### 9.1 基础配置

**启用 torch.compile（model + loss）**：

```toml
[compile]
enable = true
components = ["model", "loss"]
backend = "inductor"
```

**命令行启动**：

```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh \
  --compile.enable \
  --compile.components="model,loss"
```

### 9.2 只编译 model

```toml
[compile]
enable = true
components = ["model"]  # 只编译 model，不编译 loss
```

**使用场景**：
- 调试 loss 函数时
- Loss 函数比较复杂，可能有 graph break

### 9.3 只编译 loss

```toml
[compile]
enable = true
components = ["loss"]  # 只编译 loss，不编译 model
```

**使用场景**：
- 调试模型结构时
- 模型有 graph break，但 loss 可以编译

### 9.4 Llama3 各模型推荐配置

**Llama3 8B (8 GPUs)**：

```toml
[parallelism]
data_parallel_shard_degree = 8

[compile]
enable = true
components = ["model", "loss"]

[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"
```

**Llama3 70B (256 GPUs)**：

```toml
[parallelism]
data_parallel_shard_degree = 32
tensor_parallel_degree = 8

[compile]
enable = true
components = ["model", "loss"]

[activation_checkpoint]
mode = "full"

[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
```

**Llama3 405B (512 GPUs)**：

```toml
[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 8
pipeline_parallel_degree = 8
enable_async_tensor_parallel = true  # AsyncTP 必须配合 compile

[compile]
enable = true
components = ["model", "loss"]

[activation_checkpoint]
mode = "full"

[model]
converters = ["quantize.linear.float8"]
```

---

## 10. 性能数据

### 10.1 Llama3 8B (8 H100s)

**配置**：FSDP=8, local batch size=2, Selective AC

| 配置 | TPS/GPU | 显存 (GiB) | 相对 Baseline 加速 |
|-----|---------|-----------|-----------------|
| FSDP (baseline) | 5,762 | 82.4 | 1.00x |
| **FSDP + compile** | **6,667** | 77.0 | **1.16x** |
| FSDP + compile + Float8 | 8,532 | 76.8 | 1.48x |

**观察**：
- 单独使用 compile: **1.16x** 加速
- compile 降低了显存占用（82.4 → 77.0 GiB）
  - 原因：编译后的 kernel 更高效，中间 tensor 更少

### 10.2 Llama3 70B (128 H100s)

**配置**：FSDP=8, TP=1, local batch size=2, Selective AC

| 配置 | TPS/GPU | 显存 (GiB) | 相对 Baseline 加速 |
|-----|---------|-----------|-----------------|
| FSDP (baseline) | 5,605 | 67.0 | 1.00x |
| **FSDP + compile** | **6,514** | 62.0 | **1.16x** |
| FSDP + compile + Float8 | 8,380 | 61.8 | 1.49x |

**观察**：
- compile 的加速比与 8B 模型一致（~1.16x）
- 显存节省更明显（67.0 → 62.0 GiB）

### 10.3 Llama3 70B (256 H100s)

**配置**：FSDP=32, TP=8, local batch size=16, Full AC, Float8

| 配置 | TPS/GPU |
|-----|---------|
| **FSDP + TP + compile + Float8** | **829** |

### 10.4 Llama3 405B (512 H100s)

**配置**：FSDP=8, TP=8, PP=8, AsyncTP, Full AC, Float8, Interleaved 1F1B

| 配置 | TPS/GPU |
|-----|---------|
| **FSDP + TP + PP + compile + Float8 + AsyncTP** | **128** |

### 10.5 编译开销分析

**首次迭代（包含编译时间）**：

| 模型 | 未编译 | 编译（首次） | 编译（后续） |
|-----|-------|-----------|-----------|
| Llama3 8B | 1.0 sec | ~10 sec | 0.86 sec |
| Llama3 70B | 4.0 sec | ~20 sec | 3.44 sec |

**编译 warmup**：
- 第一次迭代：慢（需要编译）
- 后续迭代：快（复用编译结果）
- Warmup 通常 1-2 个迭代即可

**长期训练收益**：
```
假设训练 10000 steps：
  - 未编译：10000 × 1.0 sec = 10000 sec
  - 编译：10 sec (首次) + 9999 × 0.86 sec = 8609 sec
  - 节省时间：10000 - 8609 = 1391 sec (23 分钟)
  - 加速比：1.16x
```

---

## 11. 最佳实践

### 11.1 什么时候使用 torch.compile？

✅ **推荐使用**：
1. **生产训练**：长时间训练（> 1000 steps），编译开销可忽略
2. **大模型**：模型越大，编译收益越明显
3. **重复结构**：Transformer 这种重复结构，per-block 编译很高效
4. **配合优化技术**：Float8、AsyncTP 等都需要 compile 才能达到最佳性能

❌ **不推荐使用**：
1. **快速实验**：只跑几个 steps 验证想法，编译开销不值得
2. **模型调试**：频繁修改模型结构，每次都需要重新编译
3. **复杂控制流**：模型有大量 if/for/while，可能有 graph break
4. **动态 shapes**：输入 shape 频繁变化，会触发重新编译

### 11.2 如何处理 Graph Break？

**检测 graph break**：

```python
import torch._dynamo as dynamo

# 启用 verbose 模式
torch._dynamo.config.verbose = True

# 运行模型
model = torch.compile(model, fullgraph=True)
output = model(input)

# 查看编译日志，如果有 graph break 会报告
```

**常见 graph break 原因**：

1. **动态控制流**：
```python
# ❌ Bad: 动态 if
if x.shape[0] > 100:
    x = extra_layer(x)

# ✅ Good: 静态配置
if self.config.use_extra_layer:  # 静态配置
    x = extra_layer(x)
```

2. **Python 内置函数**：
```python
# ❌ Bad: len(), print()
for i in range(len(layers)):  # len() 可能导致 graph break
    x = layers[i](x)

# ✅ Good: 使用 tensor 操作
for layer in layers:
    x = layer(x)
```

3. **In-place 修改**：
```python
# ❌ Bad: in-place 修改
x[:, 0] = 0

# ✅ Good: 非 in-place
x = x.clone()
x[:, 0] = 0
```

### 11.3 调试技巧

**1. 检查是否真的被编译了**：

```python
# 添加日志
logger.info(f"Compiling {module_name}")

# 查看编译的 graph
torch._dynamo.explain(model)(input)
```

**2. 对比编译前后的性能**：

```bash
# 不编译
./run_train.sh --compile.enable=false --steps=100

# 编译
./run_train.sh --compile.enable --steps=100

# 对比 TPS/GPU
```

**3. 使用 profiling 查看 kernel 数量**：

```python
# 启用 profiling
with torch.profiler.profile() as prof:
    output = model(input)

# 查看 kernel 调用
print(prof.key_averages().table())

# 编译后应该看到更少的 kernel 调用
```

### 11.4 Compile + Float8 + AsyncTP 组合

**最佳配置**（Llama3 70B, 256 GPUs）：

```toml
[parallelism]
data_parallel_shard_degree = 32
tensor_parallel_degree = 8
enable_async_tensor_parallel = true  # 必须配合 compile

[compile]
enable = true
components = ["model", "loss"]

[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true

[activation_checkpoint]
mode = "full"
```

**性能提升**：
- 只有 FSDP + TP: 597 TPS/GPU
- + compile: 829 TPS/GPU (1.39x)
- + Float8: 810 TPS/GPU (1.36x)
- + compile + Float8 + AsyncTP: **942 TPS/GPU (1.58x)**

### 11.5 监控编译效果

**训练日志中查看**：

```
[2025-01-25 10:00:00] Compiling each TransformerBlock with torch.compile
[2025-01-25 10:00:00] Compiling the loss function with torch.compile
[2025-01-25 10:00:10] step:    1 | loss: 10.5 | memory: 77.0 GiB | dt: 10.5s (compiling)
[2025-01-25 10:00:11] step:    2 | loss: 10.3 | memory: 77.0 GiB | dt: 0.9s
[2025-01-25 10:00:12] step:    3 | loss: 10.1 | memory: 77.0 GiB | dt: 0.9s
```

**关键指标**：
- 第 1 步 dt 很大（编译开销）
- 后续步骤 dt 稳定且更小（编译收益）

---

## 12. 总结

### torch.compile 的核心要点

1. **本质**：通过编译器优化（算子融合、图优化、代码生成）提升性能
   - 减少 kernel 数量（融合）
   - 减少内存访问（内存带宽瓶颈）
   - 生成优化的 CUDA code

2. **TorchTitan 的编译策略**：
   - **Per-Block 编译**：只编译 TransformerBlock（高效、兼容并行）
   - **fullgraph=True**：确保完整编译（无 graph break）
   - **编译两个组件**：model 和 loss

3. **性能提升**：
   - Llama3 8B/70B: **1.16x** 加速
   - 配合 Float8: **1.48x** 加速
   - 配合 Float8 + AsyncTP: **1.58x** 加速

4. **最佳实践**：
   - ✅ 生产训练必开（长期收益）
   - ✅ 与 Float8、AsyncTP 组合使用
   - ✅ Per-block 编译（而非整个模型）
   - ⚠️ 注意编译顺序：TP → AC → compile → FSDP

### 搬桌子比喻总结

torch.compile 就像**流水线自动化**：

```
传统方式（Eager Mode）:
  工人手工搬运 → 每道工序独立 → 需要多个临时区 → 效率低

torch.compile (Compile Mode):
  自动化流水线 → 所有工序连续 → 一个流水线完成 → 效率高

编译过程 = 设计流水线（一次性成本）
执行过程 = 流水线运行（持续收益）
```

### 技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                   torch.compile Stack                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TorchTitan (Integration Layer)                            │
│  ├─ apply_compile: Per-Block 编译                          │
│  ├─ build_cross_entropy_loss: Loss 编译                    │
│  └─ 编译顺序: TP → AC → compile → FSDP                     │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  PyTorch (Compiler Stack)                                  │
│  ├─ TorchDynamo: 捕获 Python bytecode → FX Graph           │
│  ├─ Graph Optimizer: 算子融合、常量折叠、死代码消除        │
│  └─ TorchInductor: Triton codegen → CUDA kernel            │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Backend (Execution Layer)                                 │
│  ├─ Triton: GPU kernel 编程语言                            │
│  ├─ CUDA: NVIDIA GPU 运行时                                │
│  └─ cuBLAS/cuDNN: 优化的数学库                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. 参考资料

### PyTorch 文档
- [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html) - 官方 API 文档
- [TorchDynamo](https://pytorch.org/docs/stable/dynamo/index.html) - 动态追踪文档
- [TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) - Inductor 介绍

### TorchTitan 文档
- [benchmarks/llama3_h100_202412_torchtitan.md](../../benchmarks/llama3_h100_202412_torchtitan.md) - 性能 Benchmark

### 博客文章
- [PyTorch 2.0: Our next generation release](https://pytorch.org/blog/pytorch-2.0-release/) - PyTorch 2.0 发布博客
- [How PyTorch 2.0 works](https://pytorch.org/get-started/pytorch-2.0/) - torch.compile 工作原理

### 源码位置
- `torchtitan/models/llama3/infra/parallelize.py:236-248` - apply_compile 实现
- `torchtitan/components/loss.py:26-32` - Loss 编译
- `torchtitan/config/job_config.py:655-664` - Compile 配置

---

**最后更新**：2025年11月25日
