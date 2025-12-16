# TorchTitan 训练框架并行策略分析

## 📚 文档索引

本目录包含 TorchTitan 训练框架各种并行策略的详细分析文档，面向分布式训练 infra 初学者，用通俗易懂的比喻和详细的源码分析帮助理解。

### 已完成的文档

| 文档 | 主题 | 大小 | 核心比喻 | 关键技术 |
|-----|------|------|---------|---------|
| [01_fsdp2_per_parameter_sharding.md](./01_fsdp2_per_parameter_sharding.md) | FSDP2 参数分片 | 60 KB | 搬桌子：整个房子 | DTensor, Per-Parameter, Reshard策略 |
| [02_tensor_parallel_implementation.md](./02_tensor_parallel_implementation.md) | Tensor Parallel 基础实现 | 24 KB | 搬桌子：竖切/横切 | ColwiseParallel, RowwiseParallel, SequenceParallel |
| [03_async_tensor_parallel.md](./03_async_tensor_parallel.md) | Async TP 优化 | 27 KB | 搬桌子流水线 | Micro-Pipeline, Symmetric Memory, 通信与计算重叠 |
| [04_context_parallel.md](./04_context_parallel.md) | Context Parallel 长序列 | 34 KB | 接力赛读书 | Ring Attention, 序列切分, 在线 Softmax |
| [05_pipeline_parallel.md](./05_pipeline_parallel.md) | Pipeline Parallel 层切分 | 33 KB | 工厂流水线 | 1F1B, Interleaved Schedule, Bubble 优化 |
| [06_distributed_checkpointing.md](./06_distributed_checkpointing.md) | 分布式检查点 | 72 KB | 搬桌子：拍照存档 | DCP, Async Save, State Dict 管理 |
| [07_activation_checkpointing.md](./07_activation_checkpointing.md) | 激活检查点 | 38 KB | 搬桌子：草稿纸 | Full AC, Selective AC, Op SAC, Memory Budget |
| [08_float8_training.md](./08_float8_training.md) | Float8 训练 | 45 KB | 搬桌子：压缩搬运 | Tensorwise/Rowwise Scaling, Float8 All-Gather |
| [09_torch_compile.md](./09_torch_compile.md) | torch.compile 编译优化 | 39 KB | 搬桌子：流水线自动化 | Kernel Fusion, Per-Block Compile, fullgraph |
| [10_expert_parallel_moe.md](./10_expert_parallel_moe.md) | Expert Parallel (MoE) | 56 KB | 搬桌子：专业工人分工 | EP, ETP, Router, Grouped GEMM, Load Balancing |
| [11_optimizer_in_backward.md](./11_optimizer_in_backward.md) | Optimizer in Backward | 37 KB | 搬桌子：立即搬走 | Post Accumulate Hook, 梯度立即释放, 内存优化 |
| [12_gradient_accumulation.md](./12_gradient_accumulation.md) | Gradient Accumulation | 51 KB | 搬桌子：分批搬运 | Microbatch, Loss Scaling, 梯度累加, 有效 Batch Size |
| [13_mixed_precision_training.md](./13_mixed_precision_training.md) | Mixed Precision Training | 50 KB | 搬桌子：精细测量vs粗略测量 | BF16, FP32, FSDP MixedPrecision, torch.autocast |

**总计**：13 份文档，566 KB

---

## 🎯 快速导航

### 按使用场景选择

**问题：整个模型太大，单 GPU 放不下**
→ 阅读 [01_fsdp2_per_parameter_sharding.md](./01_fsdp2_per_parameter_sharding.md)

**问题：单层权重太大，一个 GPU 放不下**
→ 阅读 [02_tensor_parallel_implementation.md](./02_tensor_parallel_implementation.md)

**问题：TP 通信开销大，想要优化**
→ 阅读 [03_async_tensor_parallel.md](./03_async_tensor_parallel.md)

**问题：序列太长（> 8K tokens），Attention 内存爆炸**
→ 阅读 [04_context_parallel.md](./04_context_parallel.md)

**问题：模型层数太多，总参数量太大**
→ 阅读 [05_pipeline_parallel.md](./05_pipeline_parallel.md)

**问题：需要保存和恢复训练状态**
→ 阅读 [06_distributed_checkpointing.md](./06_distributed_checkpointing.md)

**问题：内存不足，激活值太大**
→ 阅读 [07_activation_checkpointing.md](./07_activation_checkpointing.md)

**问题：通信瓶颈，想要加速训练**
→ 阅读 [08_float8_training.md](./08_float8_training.md)

**问题：计算效率低，想要优化 kernel 性能**
→ 阅读 [09_torch_compile.md](./09_torch_compile.md)

**问题：MoE 模型，专家数量太多，单 GPU 放不下**
→ 阅读 [10_expert_parallel_moe.md](./10_expert_parallel_moe.md)

**问题：梯度占用内存太大，想要节省内存**
→ 阅读 [11_optimizer_in_backward.md](./11_optimizer_in_backward.md)

**问题：激活值内存太大，或需要更大的 batch size**
→ 阅读 [12_gradient_accumulation.md](./12_gradient_accumulation.md)

**问题：想要加速训练并节省内存（参数、梯度、激活值）**
→ 阅读 [13_mixed_precision_training.md](./13_mixed_precision_training.md)

### 按并行维度

```
┌─────────────────────────────────────────────────────┐
│                     World GPUs                      │
│                                                     │
│  ┌──────────────────────────────────────────┐      │
│  │          Pipeline Parallel (PP)          │      │
│  │        (按层切分，Stage 0 → Stage N)      │      │
│  │                                          │      │
│  │  ┌────────────────────────────────┐     │      │
│  │  │    Data Parallel (FSDP)        │     │      │
│  │  │  (参数分片，梯度同步)           │     │      │
│  │  │                                │     │      │
│  │  │  ┌──────────────────────┐     │     │      │
│  │  │  │ Tensor Parallel (TP) │     │     │      │
│  │  │  │  (单层权重切分)      │     │     │      │
│  │  │  │                      │     │     │      │
│  │  │  │  ┌────────────┐     │     │     │      │
│  │  │  │  │  Context   │     │     │     │      │
│  │  │  │  │  Parallel  │     │     │     │      │
│  │  │  │  │ (序列切分) │     │     │     │      │
│  │  │  │  └────────────┘     │     │     │      │
│  │  │  └──────────────────────┘     │     │      │
│  │  │                                │     │      │
│  │  │  ┌──────────────────────┐     │     │      │
│  │  │  │ Expert Parallel (EP) │     │     │      │
│  │  │  │  (专家切分, MoE)     │     │     │      │
│  │  │  └──────────────────────┘     │     │      │
│  │  └────────────────────────────────┘     │      │
│  └──────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

---

## 📖 文档内容概览

### 01. FSDP2 Per-Parameter Sharding

**核心问题**：整个模型太大，单 GPU 放不下

**解决方案**：每个参数单独分片，分散到多个 GPU

**三种模式**：
- **ZeRO-3** (reshard_after_forward=True)：内存占用低
- **ZeRO-2** (reshard_after_forward=False)：通信量少
- **HSDP**：2D data parallel mesh，减小通信域

**应用场景**：
- 所有需要多 GPU 训练的场景
- 与 TP、PP、CP 组合使用

**关键源码**：
- `torchtitan/models/llama3/infra/parallelize.py:250-316`
- `torch.distributed.fsdp.fully_shard`

---

### 02. Tensor Parallel 基础实现

**核心问题**：单层权重矩阵太大

**解决方案**：切分权重矩阵

**三种模式**：
- **ColwiseParallel**：按列切（竖着切桌子）
- **RowwiseParallel**：按行切（横着切桌子）
- **SequenceParallel**：LayerNorm 在序列维度并行

**应用场景**：
- Attention: wq/wk/wv 用 Colwise，wo 用 Rowwise
- FFN: w1/w3 用 Colwise，w2 用 Rowwise

**关键源码**：
- `torchtitan/distributed/tensor_parallel.py`
- `torchtitan/models/llama3/infra/parallelize.py:149-234`

---

### 03. Async Tensor Parallel 优化

**核心问题**：TP 的通信和计算串行，效率低

**解决方案**：通信与计算重叠

**三大技术**：
1. **Micro-Pipeline**：把计算切成小块，流水线执行
2. **Symmetric Memory**：GPU 间共享内存，零拷贝通信
3. **Torch.compile**：自动生成优化代码

**性能提升**：
- Llama3 70B + Float8: **1.16x** 加速
- Llama3 8B: **1.06-1.10x** 加速

**关键源码**：
- `torchtitan/distributed/tensor_parallel.py:15-29`

---

### 04. Context Parallel 长序列

**核心问题**：序列太长，Attention 内存 O(seq_len²)

**解决方案**：序列切分 + Ring Attention

**核心算法**：
- **Ring 传递 KV**：环形传递，让每个 GPU 看到完整上下文
- **在线 Softmax**：增量更新，不需要存储完整 attention matrix
- **因果掩码优化**：减少 50% 通信量

**内存节省**：
- CP = 4: **3-4x** 内存节省
- 可训练 32K → 128K → 1M 序列

**关键源码**：
- `torchtitan/distributed/utils.py:198-220`
- `torchtitan/models/attention.py`

---

### 05. Pipeline Parallel 层切分

**核心问题**：层数太多，总参数量太大

**解决方案**：按层切分成多个 stage

**三种 Schedule**：
- **1F1B**：交替 Forward/Backward，内存低
- **Interleaved 1F1B**：虚拟 stage，Bubble 更少
- **ZeroBubble**：理论 0 Bubble，实现复杂

**性能对比** (Llama3 405B)：
- 1F1B: 100 TPS/GPU
- Interleaved 1F1B: 128 TPS/GPU (**+28%**)

**关键源码**：
- `torchtitan/distributed/pipeline_parallel.py:41-153`

---

### 06. Distributed Checkpointing 分布式检查点

**核心问题**：需要保存和恢复训练状态

**解决方案**：每个 GPU 只保存自己的参数分片

**三种模式**：
- **Disabled**：同步保存（简单，慢）
- **Async**：异步保存（快，需要 CPU 内存）
- **Async + Pinned Mem**：最快（几乎零阻塞）

**应用场景**：
- 训练中保存 checkpoint（恢复训练）
- 训练结束导出模型（HF 格式推理）
- Fault Tolerance 支持

**关键源码**：
- `torchtitan/components/checkpoint.py:118-846`
- `torch.distributed.checkpoint.save/async_save`

---

### 07. Activation Checkpointing 激活检查点

**核心问题**：激活值占用内存太大

**解决方案**：用计算换内存，重算激活值

**三种模式**：
- **Full AC**：每层都 checkpoint（最省内存）
- **Selective AC - Layer**：每 N 层 checkpoint（平衡内存和计算）
- **Selective AC - Op**：只 checkpoint 贵的算子（最灵活）

**应用场景**：
- 大模型训练（Llama3 70B/405B）
- 长序列训练（8K → 32K）
- 显存不足时的首选优化

**性能数据**：
- Op SAC：内存节省 **40%**，速度损失 **12%**
- Full AC：内存节省 **60%**，速度损失 **25%**

**关键源码**：
- `torchtitan/distributed/activation_checkpoint.py`
- `torch.utils.checkpoint.checkpoint`

---

### 08. Float8 Training 低精度训练

**核心问题**：通信瓶颈，训练速度慢

**解决方案**：用 8 位浮点数代替 16 位，加速计算和通信

**两种 Scaling 策略**：
- **Tensorwise**：整个 tensor 一个 scale（快，适合大规模）
- **Rowwise**：每行一个 scale（精度高，通信开销大）

**应用场景**：
- 大规模分布式训练（≥64 GPUs）
- 通信密集场景（TP ≥ 8）
- H100+ GPU 硬件

**性能数据**：
- Llama3 8B (8 GPUs): **1.48x** 加速
- Llama3 70B (256 GPUs): **1.36x** 加速 → **1.58x**（配合 AsyncTP）

**关键源码**：
- `torchtitan/components/quantization/float8.py`
- `torch._scaled_mm`

---

### 09. torch.compile 编译优化

**核心问题**：内存带宽瓶颈，kernel 调用开销大

**解决方案**：编译器优化（算子融合、图优化、代码生成）

**核心优化**：
- **Kernel Fusion**：将多个算子融合为一个（减少内存访问）
- **Graph Optimization**：消除冗余计算，优化计算顺序
- **Per-Block Compile**：只编译 TransformerBlock（高效、兼容并行）

**应用场景**：
- 生产训练（长时间训练）
- 配合 Float8、AsyncTP 使用
- Transformer 等重复结构

**性能数据**：
- Llama3 8B/70B: **1.16x** 加速
- + Float8: **1.48x** 加速
- + Float8 + AsyncTP: **1.58x** 加速

**关键源码**：
- `torchtitan/models/llama3/infra/parallelize.py:236-248`
- `torchtitan/components/loss.py:26-32`

---

### 10. Expert Parallel (MoE) 专家并行

**核心问题**：MoE 模型专家数量太多，单 GPU 放不下

**解决方案**：将专家分散到多个 GPU，用 All-to-All 通信分发 tokens

**核心技术**：
- **Token Dispatch**：All-to-All 将 tokens 发送到对应专家所在的 GPU
- **Token Combine**：All-to-All 收集处理后的 tokens
- **Expert Tensor Parallel**：2D 并行（EP × TP）
- **Grouped GEMM**：变长批次的高效矩阵乘法

**应用场景**：
- Llama4 17Bx16E, 1Bx8E 等 MoE 模型
- 专家数量 > GPU 数量的场景
- 需要稀疏激活节省计算的场景

**性能数据**：
- Llama4 17Bx16E: 支持 16 专家并行训练
- Load Balancing: Auxiliary-loss-free 方法，无额外损失函数
- Grouped GEMM: 比循环调用 Linear 快 2-3x

**关键源码**：
- `torchtitan/distributed/expert_parallel.py`
- `torchtitan/models/moe/moe.py`
- `torchtitan/models/moe/utils.py`

---

### 11. Optimizer in Backward 内存优化

**核心问题**：梯度占用大量内存，反向传播峰值内存高

**解决方案**：在梯度计算完成后立即更新参数并释放梯度

**核心技术**：
- **Post Accumulate Grad Hook**：梯度计算完成后的回调函数
- **逐参数优化**：每个参数一个独立的优化器实例
- **立即释放**：optimizer.step() 和 zero_grad() 在 backward 中执行
- **内存节省**：梯度内存从 16GB 降至接近 0

**应用场景**：
- 内存受限的训练（单 GPU 内存不足）
- FSDP 训练（完美配合）
- 不需要梯度裁剪的场景

**限制**：
- ❌ 不兼容梯度裁剪（梯度被立即清零）
- ❌ 不兼容 Pipeline Parallel（stage 间不同步）
- ❌ 不兼容 Expert Parallel（破坏协调时序）

**性能数据**：
- Llama3 8B (8 GPUs): 节省梯度内存 **2 GB per GPU** (25%)
- 与 Activation Checkpointing 组合: 节省 **5-6 GB per GPU** (35-40%)
- 速度影响: 轻微减速 **1-2%**

**关键源码**：
- `torchtitan/components/optimizer.py:131-177`
- `torch.Tensor.register_post_accumulate_grad_hook`

---

### 12. Gradient Accumulation 梯度累积

**核心问题**：激活值内存占用太大，或需要更大的有效 batch size

**解决方案**：将大 batch 分成多个小 microbatch，分批计算梯度并累加

**核心技术**：
- **Microbatch**：单次 forward/backward 的小批次数据
- **Loss Scaling**：自动缩放 loss，确保梯度正确
- **梯度累加**：PyTorch 自动累加梯度到 param.grad
- **统一更新**：所有 microbatch 完成后统一执行 optimizer.step()

**应用场景**：
- 内存受限（激活值太大）
- 需要大 batch size（训练稳定性）
- 与 FSDP、AC 组合使用

**性能数据**：
- Llama3 8B: 节省激活值内存 **60-90%**（取决于配置）
- 与 AC 组合: 节省 **76%** 激活值内存
- 速度影响: +0-10%（可能加速）

**核心公式**：
```
Effective Batch Size = Local Batch Size × Gradient Accumulation Steps × DP Degree
Gradient Accumulation Steps = Global Batch Size / (Local Batch Size × DP Degree)
```

**关键源码**：
- `torchtitan/train.py:542-609` (训练循环)
- `torchtitan/components/loss.py:35-66` (Loss 缩放)

---

### 13. Mixed Precision Training 混合精度训练

**核心问题**：训练速度慢，内存占用大（参数、梯度、激活值都用 FP32）

**解决方案**：计算和存储使用 BF16，关键操作使用 FP32

**核心技术**：
- **BF16**：16 位浮点数，范围与 FP32 相同
- **参数和计算**：BF16（节省 50% 内存，加速 2-8x）
- **梯度规约**：FP32（避免精度损失）
- **优化器状态**：FP32（保证更新精度）

**两种模式**：
- **FSDP 模式**：通过 `param_dtype` 和 `reduce_dtype`
- **AMP 模式**：通过 `torch.autocast`（仅 DDP/单设备）

**应用场景**：
- 几乎所有大模型训练（≥1B 参数）
- A100/H100 等支持 BF16 的硬件
- 与 FSDP、TP、AC、Float8 等技术组合

**性能数据**：
- Llama3 8B: 内存节省 **43%**，加速 **52%**
- 通信加速: **42%**（BF16 通信）
- 精度影响: 极小（< 0.1%）

**核心配置**：
```toml
[training]
dtype = "float32"                    # 初始化: FP32
mixed_precision_param = "bfloat16"   # 参数: BF16
mixed_precision_reduce = "float32"   # 规约: FP32 (必须！)
```

**关键源码**：
- `torchtitan/config/job_config.py:234-253` (配置)
- `torchtitan/distributed/utils.py:238-258` (模式选择)
- `torchtitan/models/llama3/infra/parallelize.py:117-125` (FSDP 应用)

---

## 🔧 并行策略组合指南

### 单一并行 (8 GPUs)

| 模型 | 推荐配置 |
|-----|---------|
| Llama3 8B | FSDP 8 |
| Llama3 8B (长序列 128K) | FSDP 4, CP 2 |

### 2D 并行 (64-256 GPUs)

| 模型 | 推荐配置 |
|-----|---------|
| Llama3 70B | FSDP 8, TP 8 |
| Llama3 70B + Float8 + AsyncTP | FSDP 32, TP 8 |

### 3D 并行 (256-512 GPUs)

| 模型 | 推荐配置 |
|-----|---------|
| Llama3 405B | FSDP 8, TP 8, PP 8 |
| Llama3 405B + Interleaved | FSDP 8, TP 8, PP 8 (2 stages/rank) |

### 4D 并行 (512+ GPUs)

| 模型 | 推荐配置 |
|-----|---------|
| Llama3 405B (长序列 128K) | FSDP 4, TP 8, PP 8, CP 2 |
| Llama3 405B (超长序列 256K) | FSDP 2, TP 8, PP 8, CP 4 |

---

## 📊 性能 Benchmark 总结

来自 `benchmarks/llama3_h100_202412_torchtitan.md`

### Llama3 8B (8 H100s)

| 配置 | TPS/GPU | 加速比 |
|-----|---------|--------|
| FSDP | 5,762 | 1.0x |
| FSDP + compile | 6,667 | 1.16x |
| FSDP + compile + Float8 | 8,532 | 1.48x |

### Llama3 70B (256 H100s)

| 配置 | TPS/GPU | 加速比 |
|-----|---------|--------|
| FSDP 32, TP 8 | 829 | 1.0x |
| + AsyncTP | 876 | 1.06x |

### Llama3 405B (512 H100s)

| 配置 | TPS/GPU | 加速比 |
|-----|---------|--------|
| FSDP 8, TP 8, PP 8 (1F1B) | 100 | 1.0x |
| FSDP 8, TP 8, PP 8 (Interleaved) | 128 | 1.28x |

---

## 🎓 学习路线建议

### 初学者路线

1. **从 FSDP 开始**：[01_fsdp2_per_parameter_sharding.md](./01_fsdp2_per_parameter_sharding.md)
   - 最基础的数据并行技术
   - 理解参数分片的概念
   - 掌握搬桌子的比喻（整个房子）

2. **学习 TP**：[02_tensor_parallel_implementation.md](./02_tensor_parallel_implementation.md)
   - 单层权重切分
   - 理解 Colwise/Rowwise 的本质
   - 掌握搬桌子的比喻（单张桌子）

3. **理解 CP**：[04_context_parallel.md](./04_context_parallel.md)
   - 长序列的必备技术
   - Ring Attention 算法很巧妙
   - 接力赛的比喻很形象

4. **掌握 PP**：[05_pipeline_parallel.md](./05_pipeline_parallel.md)
   - 大模型的关键技术
   - 理解 Bubble 和 Schedule
   - 工厂流水线的比喻

5. **优化 TP**：[03_async_tensor_parallel.md](./03_async_tensor_parallel.md)
   - 高级优化技术
   - 理解通信与计算重叠
   - 掌握 Micro-Pipeline

6. **节省内存**：[07_activation_checkpointing.md](./07_activation_checkpointing.md)
   - 内存优化的核心技术
   - 理解计算与内存的权衡
   - 掌握草稿纸策略的比喻

7. **加速训练**：[08_float8_training.md](./08_float8_training.md)
   - 低精度训练技术
   - 理解 Float8 的 Scaling 策略
   - 掌握压缩搬运的比喻

8. **编译优化**：[09_torch_compile.md](./09_torch_compile.md)
   - torch.compile 编译器技术
   - 理解算子融合和图优化
   - 掌握流水线自动化的比喻

9. **保存训练**：[06_distributed_checkpointing.md](./06_distributed_checkpointing.md)
   - Checkpoint 的必备知识
   - 理解 DCP 的优势
   - 掌握异步保存技术

10. **MoE 模型**：[10_expert_parallel_moe.md](./10_expert_parallel_moe.md)
   - MoE 和 Expert Parallel 基础
   - 理解 All-to-All 通信模式
   - 掌握专业工人分工的比喻
   - Grouped GEMM 优化技术

11. **内存优化**：[11_optimizer_in_backward.md](./11_optimizer_in_backward.md)
   - Optimizer in Backward 机制
   - 理解 Post Accumulate Grad Hook
   - 掌握立即搬走的比喻
   - 内存优化原理和限制

12. **梯度累积**：[12_gradient_accumulation.md](./12_gradient_accumulation.md)
   - Gradient Accumulation 原理
   - 理解 Microbatch 和 Loss Scaling
   - 掌握分批搬运的比喻
   - 与 FSDP、AC 的完美配合

13. **混合精度训练**：[13_mixed_precision_training.md](./13_mixed_precision_training.md)
   - Mixed Precision Training 原理
   - 理解 BF16 vs FP32 的权衡
   - 掌握精细测量 vs 粗略测量的比喻
   - FSDP 模式 vs AMP 模式

### 进阶路线

**问题驱动学习**：
1. 遇到模型太大 → 学习 FSDP
2. 遇到单层太大 → 学习 TP
3. 遇到序列太长 → 学习 CP
4. 遇到层数太多 → 学习 PP
5. 遇到性能瓶颈 → 学习 Async TP
6. 遇到内存不足 → 学习 Activation Checkpointing
7. 遇到通信瓶颈 → 学习 Float8 Training
8. 遇到计算效率低 → 学习 torch.compile
9. 需要保存恢复 → 学习 Distributed Checkpointing
10. 训练 MoE 模型 → 学习 Expert Parallel
11. 梯度内存太大 → 学习 Optimizer in Backward
12. 激活值内存太大 → 学习 Gradient Accumulation
13. 训练速度慢/内存占用高 → 学习 Mixed Precision Training

**实践建议**：
1. 先在小模型上实验（Llama3 8B）
2. 逐步增加并行维度（1D → 2D → 3D）
3. 使用 profiling 分析瓶颈
4. 根据 Benchmark 调优配置

---

## 🔍 术语表

### 并行相关

| 术语 | 全称 | 含义 |
|-----|------|------|
| **DP** | Data Parallel | 数据并行 |
| **DDP** | Distributed Data Parallel | 分布式数据并行 |
| **FSDP** | Fully Sharded Data Parallel | 全分片数据并行 |
| **HSDP** | Hybrid Sharded Data Parallel | 混合分片数据并行 |
| **TP** | Tensor Parallel | 张量并行 |
| **CP** | Context Parallel | 上下文并行 |
| **PP** | Pipeline Parallel | 流水线并行 |
| **EP** | Expert Parallel | 专家并行（MoE） |
| **ETP** | Expert Tensor Parallel | 专家张量并行（2D: EP × TP） |

### FSDP 相关

| 术语 | 含义 |
|-----|------|
| **DTensor** | 分布式 Tensor（带分片信息） |
| **Shard(dim)** | 在某个维度切分 |
| **Replicate** | 复制（每个 GPU 都有完整副本） |
| **All-Gather** | 收集所有分片，得到完整 tensor |
| **Reduce-Scatter** | 求和后切分 |
| **Reshard** | 重新分片（释放内存） |
| **ZeRO-2** | reshard_after_forward=False |
| **ZeRO-3** | reshard_after_forward=True |

### TP 相关

| 术语 | 含义 |
|-----|------|
| **Colwise** | 列并行（权重按列切分） |
| **Rowwise** | 行并行（权重按行切分） |
| **Shard(dim)** | 在某个维度切分 |
| **Replicate** | 复制（每个 GPU 都有完整副本） |
| **All-Reduce** | 全规约（所有 GPU 求和） |

### CP 相关

| 术语 | 含义 |
|-----|------|
| **Ring Attention** | 环形 Attention（KV 轮换传递） |
| **Rotate Method** | 轮换方法（allgather 或 alltoall） |
| **Load Balance** | 负载均衡 |

### PP 相关

| 术语 | 含义 |
|-----|------|
| **Stage** | 流水线阶段（模型的一部分） |
| **Microbatch** | 微批次 |
| **Bubble** | 气泡（GPU 空闲时间） |
| **1F1B** | 1 Forward 1 Backward |
| **Interleaved** | 交错式（虚拟 stage） |

### AC 相关

| 术语 | 含义 |
|-----|------|
| **AC** | Activation Checkpointing（激活检查点） |
| **Full AC** | 全激活检查点（每层都 checkpoint） |
| **Selective AC** | 选择性激活检查点（部分层/算子） |
| **Op SAC** | 算子级选择性 AC（只 checkpoint 贵的算子） |
| **Recomputation** | 重计算（反向时重新计算激活值） |
| **CheckpointPolicy** | 检查点策略（MUST_SAVE/PREFER_RECOMPUTE） |

### Float8 相关

| 术语 | 含义 |
|-----|------|
| **Float8** | 8 位浮点数（E4M3: 4 bits 指数, 3 bits 尾数） |
| **Scale** | 缩放因子（255 / max(abs(tensor))） |
| **Tensorwise Scaling** | 整个 tensor 一个 scale |
| **Rowwise Scaling** | 每行一个 scale |
| **Float8 All-Gather** | 用 Float8 格式进行 all-gather 通信 |
| **Precompute Scale** | 预计算 scales，减少通信次数 |
| **FP8 Tensor Core** | H100+ GPU 的 Float8 硬件加速单元 |

### torch.compile 相关

| 术语 | 含义 |
|-----|------|
| **torch.compile** | PyTorch 2.0 的编译器（编译 Python 代码为优化 kernel） |
| **Kernel Fusion** | 算子融合（将多个算子合并为一个） |
| **TorchDynamo** | 捕获 Python bytecode，构建计算图 |
| **TorchInductor** | 代码生成 backend（生成 Triton kernel） |
| **fullgraph** | 要求完整编译，遇到 graph break 报错 |
| **Graph Break** | 无法编译的代码点（如动态控制流） |
| **Per-Block Compile** | 只编译 TransformerBlock（而非整个模型） |

### MoE / Expert Parallel 相关

| 术语 | 含义 |
|-----|------|
| **MoE** | Mixture of Experts（混合专家模型） |
| **Expert** | 专家网络（通常是 FFN） |
| **Router** | 路由网络（决定 token 去哪个专家） |
| **Top-K Routing** | 每个 token 选择 top-k 个专家 |
| **Token Dispatch** | 将 tokens 分发到对应的专家（All-to-All） |
| **Token Combine** | 收集专家处理后的 tokens（All-to-All） |
| **Grouped GEMM** | 批量处理变长矩阵乘法 |
| **Load Balancing** | 负载均衡（确保每个专家处理相似数量的 tokens） |
| **Auxiliary Loss** | 辅助损失（用于负载均衡，但增加训练目标） |
| **Expert Bias** | 专家偏置（Auxiliary-loss-free 负载均衡方法） |
| **Shared Experts** | 共享专家（所有 tokens 都经过） |
| **Token Reorderer** | Token 重排序器（按专家分组） |
| **Permute/Unpermute** | Token 排列/还原（优化 Grouped GEMM） |
| **Token Group Alignment** | Token 对齐（padding 到 8/16/32 倍数） |

### Optimizer in Backward 相关

| 术语 | 含义 |
|-----|------|
| **Optimizer in Backward** | 在反向传播中执行优化器更新（内存优化技术） |
| **Post Accumulate Grad Hook** | 梯度累积完成后的回调函数 |
| **register_post_accumulate_grad_hook** | PyTorch API，注册梯度完成后的 hook |
| **Gradient Memory** | 梯度占用的内存（约等于参数大小） |
| **Per-Parameter Optimizer** | 每个参数一个独立的优化器实例 |
| **Early Step** | 提前执行 optimizer.step()（在 backward 中） |
| **Immediate Release** | 立即释放梯度内存 |

### Gradient Accumulation 相关

| 术语 | 含义 |
|-----|------|
| **Gradient Accumulation** | 梯度累积（分批计算梯度并累加） |
| **Microbatch** | 单次 forward/backward 的小批次数据 |
| **Local Batch Size** | 每个 GPU 每次 forward 的样本数 |
| **Global Batch Size** | 所有 GPU 的总有效 batch size |
| **Effective Batch Size** | 有效 batch size（等于 global batch size） |
| **Gradient Accumulation Steps** | 累积多少个 microbatch |
| **Loss Scaling** | 缩放 loss（除以 accumulation steps） |
| **RescaleAccumulatedLoss** | 自动缩放 loss 的包装类 |

### Mixed Precision Training 相关

| 术语 | 含义 |
|-----|------|
| **Mixed Precision Training** | 混合精度训练（计算用低精度，关键操作用高精度） |
| **BF16 / bfloat16** | Brain Float 16（16 位浮点数，8 位指数，7 位尾数） |
| **FP32 / float32** | 32 位浮点数（8 位指数，23 位尾数） |
| **FP16 / float16** | 16 位浮点数（5 位指数，10 位尾数） |
| **param_dtype** | 参数精度（存储和计算权重的数据类型） |
| **reduce_dtype** | 规约精度（梯度 all-reduce/reduce-scatter 的数据类型） |
| **Mixed Precision Policy** | FSDP 的混合精度策略（param_dtype + reduce_dtype） |
| **torch.autocast** | PyTorch 自动混合精度上下文管理器（AMP 模式） |
| **AMP** | Automatic Mixed Precision（自动混合精度） |
| **Loss Scaling (MP)** | 损失缩放（FP16 训练防止下溢，BF16 不需要） |
| **Master Weights** | 主权重（优化器中保存的 FP32 副本） |
| **Dynamic Range** | 动态范围（BF16: ≈1e38，FP16: ≈6e4） |
| **Precision Loss** | 精度损失（低精度累加时的误差） |
| **Underflow** | 下溢（数值太小无法表示） |

---

## 🛠️ 配置速查表

### Llama3 8B (8 GPUs)

```toml
[training]
local_batch_size = 2
seq_len = 8192

[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 1
pipeline_parallel_degree = 1
context_parallel_degree = 1

[compile]
enable = true                # 启用编译
components = ["model", "loss"]  # 编译 model 和 loss

[activation_checkpoint]
mode = "selective"

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
filter_fqns = ["auto_filter_small_kn"]
```

### Llama3 70B (64 GPUs)

```toml
[training]
local_batch_size = 16
seq_len = 8192

[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 8
enable_async_tensor_parallel = false
pipeline_parallel_degree = 1
context_parallel_degree = 1

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
filter_fqns = ["output"]
```

### Llama3 405B (512 GPUs)

```toml
[model]
converters = ["float8"]

[training]
local_batch_size = 8
seq_len = 8192

[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 8
enable_async_tensor_parallel = true
pipeline_parallel_degree = 8
pipeline_parallel_schedule = "Interleaved1F1B"
pipeline_parallel_microbatch_size = 1
context_parallel_degree = 1

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
filter_fqns = ["output"]
```

---

## 📚 进一步阅读

### TorchTitan 官方文档

- [README.md](../../README.md) - 项目介绍
- [docs/composability.md](../composability.md) - 并行策略组合
- [docs/converging.md](../converging.md) - 收敛性验证
- [benchmarks/](../../benchmarks/) - 性能 Benchmark

### PyTorch 官方文档

- [Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [Pipeline Parallel](https://pytorch.org/docs/stable/distributed.pipelining.html)
- [FSDP](https://pytorch.org/docs/stable/fsdp.html)

### 学术论文

**Tensor Parallel**:
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

**Context Parallel**:
- Ring Attention with Blockwise Transformers for Near-Infinite Context
- FlashAttention: Fast and Memory-Efficient Exact Attention

**Pipeline Parallel**:
- GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism
- PipeDream: Generalized Pipeline Parallelism for DNN Training
- Zero Bubble Pipeline Parallelism

---

## ✨ 文档特点

所有文档都具有以下特点：

1. **通俗易懂**
   - 用生活化比喻（搬桌子、接力赛、工厂流水线）
   - 避免过度抽象的概念
   - 逐步递进，由浅入深

2. **图文并茂**
   - ASCII 图展示数据流
   - 时间线图展示执行过程
   - 表格对比不同方案

3. **源码结合**
   - 每个概念都有对应的源码位置
   - 关键函数的详细解析
   - 配置示例和参数说明

4. **实战导向**
   - 真实的性能数据
   - 最佳实践建议
   - 调试技巧和常见问题

5. **面向初学者**
   - 假设读者是 infra 初学者
   - 详细解释基础概念
   - 提供学习路线建议

---

## 🙏 致谢

这些文档基于 TorchTitan 开源项目和 PyTorch 分布式训练框架。感谢 Meta AI 团队的贡献！

如有问题或建议，欢迎提 Issue 或 PR。

---

**最后更新**：2025年11月25日

**注**：
- Float8 Training 是大规模分布式训练加速的关键技术，建议在 ≥64 GPUs 的场景下使用。
- torch.compile 是生产训练的标配，建议始终启用（除非调试模型结构）。
- Expert Parallel 是 MoE 模型训练的第 5 个并行维度，支持 Llama4 等稀疏激活模型的高效训练。
- Optimizer in Backward 是内存优化的利器，可节省 25% 梯度内存，但不兼容梯度裁剪、PP 和 EP。
- Gradient Accumulation 是训练大模型的核心技术，可节省 60-90% 激活值内存，几乎所有大模型训练都会使用。
- Mixed Precision Training 是几乎所有大模型训练的标配技术，可节省 43% 内存并加速 52%，强烈推荐使用 BF16（而非 FP16）配合 FP32 梯度规约。
