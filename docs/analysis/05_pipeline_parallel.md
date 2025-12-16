# Pipeline Parallel (PP) 实现详解

## 目录
- [1. 什么是 Pipeline Parallel？](#1-什么是-pipeline-parallel)
- [2. 搬桌子的流水线比喻](#2-搬桌子的流水线比喻)
- [3. Pipeline Schedule 详解](#3-pipeline-schedule-详解)
- [4. 源码实现详解](#4-源码实现详解)
- [5. 性能分析](#5-性能分析)
- [6. 使用场景和最佳实践](#6-使用场景和最佳实践)

---

## 1. 什么是 Pipeline Parallel？

### 1.1 为什么需要 Pipeline Parallel？

回顾我们学过的并行方式：

| 并行方式 | 切分对象 | 适用场景 |
|---------|---------|---------|
| Data Parallel (FSDP) | 数据 + 参数 | 通用 |
| Tensor Parallel (TP) | 单层权重 | 单层太大 |
| Context Parallel (CP) | 序列 | 序列太长 |

**但还有一个问题**：即使单层能放进 GPU，**所有层加起来还是太大**！

```python
# Llama3 405B 模型
num_layers = 126
hidden_dim = 16384
intermediate_size = 53248

# 单层参数量
params_per_layer = 4 * hidden_dim^2 + 3 * hidden_dim * intermediate_size
                 ≈ 3.6B parameters

# 总参数量
total_params = 126 * 3.6B ≈ 405B parameters

# 内存需求 (fp16)
memory = 405B * 2 bytes = 810 GB

# H100 80GB 需要 810 / 80 = 10+ GPUs 才能放下参数！
```

**Pipeline Parallel 的思路**：把模型**按层切分**，每个 GPU 负责一部分层。

### 1.2 Pipeline Parallel 的核心思想

**把 Transformer 的层分成多个 stage，每个 stage 放在不同的 GPU 上**

```
原始模型 (32 layers):
┌─────────────────────────────────────┐
│ Embedding → Layer0 → ... → Layer31 → Output │
└─────────────────────────────────────┘
    全部在 GPU 0 (放不下！)

Pipeline Parallel (PP = 4):
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
│ Embedding │ → │ Layer8-15 │ → │ Layer16-23│ → │ Layer24-31│
│ Layer0-7  │   │           │   │           │   │ Norm,Output│
└───────────┘   └───────────┘   └───────────┘   └───────────┘
   GPU 0          GPU 1          GPU 2          GPU 3
   (Stage 0)      (Stage 1)      (Stage 2)      (Stage 3)
```

**每个 GPU 只需要存储 1/4 的模型参数！**

### 1.3 Pipeline Parallel vs 其他并行

| 特性 | FSDP | TP | CP | PP |
|------|------|----|----|-----|
| **切分对象** | 参数 (scatter) | 参数 (partition) | 序列 | 层 |
| **通信类型** | All-Gather / Reduce-Scatter | All-Reduce | Ring (KV) | P2P (activations) |
| **通信量** | 大 | 中 | 小 | 小 |
| **内存节省** | 参数 | 参数 + 激活 | 激活 | 参数 |
| **适用场景** | 通用 | 单层太大 | 序列太长 | 层数太多 |

**PP 的特点**：
- ✅ **通信量小**：只传递层之间的激活值
- ✅ **实现简单**：按层切分，不需要修改单层逻辑
- ❌ **有 Bubble**：GPU 会有空闲时间

---

## 2. 搬桌子的流水线比喻

### 2.1 场景设定

继续用搬桌子的比喻。这次想象你要组装一张**超级大的桌子**，需要 4 个步骤：

```
步骤 1: 切割木材 (Embedding + Layer 0-7)
步骤 2: 打磨木材 (Layer 8-15)
步骤 3: 上漆 (Layer 16-23)
步骤 4: 组装 (Layer 24-31 + Output)
```

### 2.2 传统方式：一个人完成所有步骤

```
时间线:
人1:  [切割桌子1] → [打磨桌子1] → [上漆桌子1] → [组装桌子1]
                                                     ↓
人1:  [切割桌子2] → [打磨桌子2] → [上漆桌子2] → [组装桌子2]

总耗时: 2 × 4步骤 = 8 个时间单位
```

**问题**：一个人要掌握所有工序，效率低下。

### 2.3 Pipeline Parallel：流水线作业

```
4 个人分工，每人负责一道工序

时间线:
时刻1: 人1[切割桌子1]
时刻2: 人1[切割桌子2]  人2[打磨桌子1]
时刻3: 人1[切割桌子3]  人2[打磨桌子2]  人3[上漆桌子1]
时刻4: 人1[切割桌子4]  人2[打磨桌子3]  人3[上漆桌子2]  人4[组装桌子1]
时刻5:                 人2[打磨桌子4]  人3[上漆桌子3]  人4[组装桌子2]
时刻6:                                人3[上漆桌子4]  人4[组装桌子3]
时刻7:                                               人4[组装桌子4]

4 张桌子耗时: 7 个时间单位
传统方式: 4 × 4 = 16 个时间单位
加速比: 16 / 7 = 2.3x 🚀
```

### 2.4 Bubble：流水线的空闲时间

仔细看上面的时间线，你会发现：

```
时刻1: 人1[工作]  人2[空闲]  人3[空闲]  人4[空闲]  ← 开始阶段
时刻2: 人1[工作]  人2[工作]  人3[空闲]  人4[空闲]
时刻3: 人1[工作]  人2[工作]  人3[工作]  人4[空闲]
时刻4: 人1[工作]  人2[工作]  人3[工作]  人4[工作]  ← 满载
时刻5: 人1[空闲]  人2[工作]  人3[工作]  人4[工作]  ← 结束阶段
时刻6: 人1[空闲]  人2[空闲]  人3[工作]  人4[工作]
时刻7: 人1[空闲]  人2[空闲]  人3[空闲]  人4[工作]
```

**Bubble = 空闲时间**
- 开始阶段：后面的 stage 在等前面的输出
- 结束阶段：前面的 stage 没有新任务了

**Bubble 比例**：
```
总时间槽: 4 人 × 7 时刻 = 28
实际工作: 4 张桌子 × 4 步骤 = 16
Bubble: 28 - 16 = 12
Bubble 比例: 12 / 28 = 43%  😰
```

### 2.5 减少 Bubble 的方法

**方法 1：增加 Microbatch 数量**

```
增加到 8 张桌子:

时刻1:  人1[桌子1]
时刻2:  人1[桌子2]  人2[桌子1]
时刻3:  人1[桌子3]  人2[桌子2]  人3[桌子1]
时刻4:  人1[桌子4]  人2[桌子3]  人3[桌子2]  人4[桌子1]
时刻5:  人1[桌子5]  人2[桌子4]  人3[桌子3]  人4[桌子2]
时刻6:  人1[桌子6]  人2[桌子5]  人3[桌子4]  人4[桌子3]
时刻7:  人1[桌子7]  人2[桌子6]  人3[桌子5]  人4[桌子4]
时刻8:  人1[桌子8]  人2[桌子7]  人3[桌子6]  人4[桌子5]
时刻9:              人2[桌子8]  人3[桌子7]  人4[桌子6]
时刻10:                        人3[桌子8]  人4[桌子7]
时刻11:                                   人4[桌子8]

总时间槽: 4 × 11 = 44
实际工作: 8 × 4 = 32
Bubble 比例: (44 - 32) / 44 = 27%

比 4 张桌子的 43% 少多了！
```

**公式**：
```
Bubble 比例 = (PP - 1) / (PP - 1 + n_microbatches)

n_microbatches = 4:  Bubble = 3/7 = 43%
n_microbatches = 8:  Bubble = 3/11 = 27%
n_microbatches = 16: Bubble = 3/19 = 16%
n_microbatches = 32: Bubble = 3/35 = 8.5%
```

**方法 2：Interleaved Schedule（虚拟 Stage）**

```
每个人学会两道工序！

人1: 切割 + 组装 (Stage 0 + Stage 3)
人2: 打磨 + 上漆 (Stage 1 + Stage 2)

流程:
桌子1: 人1切割 → 人2打磨 → 人2上漆 → 人1组装
桌子2: 人1切割 → 人2打磨 → 人2上漆 → 人1组装

时间线更密集，Bubble 更少！
```

这就是 **Interleaved 1F1B** 的思想。

---

## 3. Pipeline Schedule 详解

### 3.1 GPipe Schedule

**最简单的 schedule：所有 Forward 完成后，再做所有 Backward**

```
Forward pass (所有 microbatch):
F0 → F1 → F2 → F3 → ...

然后:
Backward pass (所有 microbatch):
B3 → B2 → B1 → B0 → ...

时间线 (4 stages, 4 microbatches):
       Stage0  Stage1  Stage2  Stage3
时刻1: [F0]
时刻2: [F1]    [F0]
时刻3: [F2]    [F1]    [F0]
时刻4: [F3]    [F2]    [F1]    [F0]
时刻5:         [F3]    [F2]    [F1]
时刻6:                 [F3]    [F2]
时刻7:                         [F3]
时刻8:                         [B3]    ← Backward 开始
时刻9:                 [B3]    [B2]
时刻10:        [B3]    [B2]    [B1]
时刻11:[B3]    [B2]    [B1]    [B0]
时刻12:[B2]    [B1]    [B0]
时刻13:[B1]    [B0]
时刻14:[B0]
```

**问题**：
- 需要保存所有 microbatch 的激活值
- 内存消耗大 = O(n_microbatches)
- Bubble 比例高

### 3.2 1F1B Schedule

**交替执行 Forward 和 Backward，减少内存**

```
时间线 (4 stages, 4 microbatches):
       Stage0  Stage1  Stage2  Stage3
时刻1: [F0]
时刻2: [F1]    [F0]
时刻3: [F2]    [F1]    [F0]
时刻4: [F3]    [F2]    [F1]    [F0]
时刻5: [B0]    [F3]    [F2]    [F1]    ← Stage0 开始 Backward
时刻6: [B1]    [B0]    [F3]    [F2]
时刻7: [B2]    [B1]    [B0]    [F3]
时刻8: [B3]    [B2]    [B1]    [B0]    ← 稳态: 1F1B
时刻9:         [B3]    [B2]    [B1]
时刻10:                [B3]    [B2]
时刻11:                        [B3]
```

**1F1B 的含义**：**1 Forward 1 Backward**
- 稳态阶段：每个 stage 交替执行 1 次 Forward 和 1 次 Backward
- 内存消耗 = O(PP)，而不是 O(n_microbatches)

**对比 GPipe**：
| 特性 | GPipe | 1F1B |
|------|-------|------|
| **内存** | O(n_microbatches) | O(PP) |
| **Bubble** | 相同 | 相同 |
| **实现复杂度** | 简单 | 中等 |

### 3.3 Interleaved 1F1B Schedule

**每个 GPU 持有多个 stage（虚拟 stage），进一步减少 Bubble**

```
配置: PP = 2, 每个 rank 持有 2 个 stage
      Rank 0: Stage 0, Stage 2
      Rank 1: Stage 1, Stage 3

模型流程: Stage0 → Stage1 → Stage2 → Stage3

时间线 (2 ranks, 4 virtual stages, 4 microbatches):
       Rank0(S0,S2)  Rank1(S1,S3)
时刻1: [F0_S0]
时刻2: [F1_S0]       [F0_S1]
时刻3: [F0_S2]       [F1_S1]       ← Rank0 执行 Stage2
时刻4: [F2_S0]       [F0_S3]
时刻5: [F1_S2]       [F2_S1]
时刻6: [F3_S0]       [F1_S3]
时刻7: [F2_S2]       [F3_S1]
时刻8: [B0_S2]       [F2_S3]       ← Backward 开始
...
```

**为什么 Bubble 更少？**

```
普通 1F1B (PP = 4, 4 ranks):
Bubble = 3 个 stage 的 warm-up + 3 个 stage 的 cool-down

Interleaved 1F1B (PP = 2, 2 ranks, 2 stages/rank):
Bubble = 1 个 rank 的 warm-up + 1 个 rank 的 cool-down
       = 只有 1 个单位的 bubble (而不是 3)
```

**内存 trade-off**：
- 每个 rank 持有 2 个 stage → 需要存储 2 组激活
- 内存 = O(PP × stages_per_rank)
- 但 Bubble 大幅减少

### 3.4 ZeroBubble Schedule

**更激进的调度，理论上 0 Bubble**

```
核心思想:
1. 拆分 Backward 为 B 和 W
   - B: Backward 计算梯度
   - W: Weight update (梯度乘以学习率)

2. 重排 B 和 W 的顺序，填满 Bubble

传统 1F1B:
[F F F F B B B B] → 有 Bubble

ZeroBubble:
[F F B F B F B F B W W W W] → 无 Bubble
```

**实现复杂度高**，需要精细的调度。TorchTitan 支持 `ZBVZeroBubble` 和 `InterleavedZeroBubble`。

### 3.5 Schedule 对比总结

| Schedule | Bubble 比例 | 内存 | 复杂度 | 适用场景 |
|----------|-------------|------|--------|---------|
| **GPipe** | 高 | O(n_mb) | 低 | 教学/简单场景 |
| **1F1B** | 中 | O(PP) | 中 | 默认选择 |
| **Interleaved 1F1B** | 低 | O(PP × stages) | 中高 | 大模型 |
| **ZeroBubble** | ~0 | 中 | 高 | 极致性能 |

---

## 4. 源码实现详解

### 4.1 核心入口：pipeline_llm

```python
# 来自: torchtitan/distributed/pipeline_parallel.py:41-153

def pipeline_llm(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_args: BaseModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    """
    将模型切分成 pipeline stages，并构建 schedule。

    返回:
        - pp_schedule: Pipeline schedule
        - model_parts: 每个 stage 的模型部分
        - has_first_stage: 当前 rank 是否有第一个 stage
        - has_last_stage: 当前 rank 是否有最后一个 stage
    """
    pp_mesh = parallel_dims.world_mesh["pp"]

    # 1. 确定 schedule 类型
    schedule_class = get_schedule_class(
        job_config.parallelism.pipeline_parallel_schedule
    )
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)

    # 2. 计算虚拟 stage 数量
    num_layers = model_args.n_layers
    layers_per_stage = job_config.parallelism.pipeline_parallel_layers_per_stage

    if layers_per_stage is not None:
        # 根据每个 stage 的层数计算总 stage 数
        num_virtual_stages = math.ceil(num_layers / layers_per_stage)
    else:
        # 默认：单 stage schedule 每个 rank 1 个 stage
        #       多 stage schedule 每个 rank 2 个 stage
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank

    # 3. 生成每个 stage 的模块名
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_virtual_stages, num_layers, input_weight, output_weight
    )

    # 4. 切分模型
    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        job_config.parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

    # 5. 对每个 stage 应用其他并行化 (FSDP, TP, etc.)
    for i, m in enumerate(model_parts):
        m = parallelize_fn(m, parallel_dims, job_config)
        model_parts[i] = m
        stages[i].submod = m

    # 6. 构建 schedule
    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    # 7. 返回
    has_first_stage = any(stage.is_first for stage in stages)
    has_last_stage = any(stage.is_last for stage in stages)

    return pp_schedule, model_parts, has_first_stage, has_last_stage
```

### 4.2 模型切分：generate_llm_fqn_per_model_part

```python
# 来自: torchtitan/distributed/pipeline_parallel.py:226-334

def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """
    为每个 stage 生成模块名列表。

    Args:
        num_stages: Pipeline stage 数量
        num_layers: Transformer 层数
        input_weight: Embedding 的权重（用于负载均衡）
        output_weight: Output 层的权重

    Returns:
        每个 stage 的模块名列表
    """
    # 例如: num_stages=4, num_layers=32

    # 第一个 stage: ["tok_embeddings", "layers.0", ..., "layers.7"]
    # 第二个 stage: ["layers.8", ..., "layers.15"]
    # 第三个 stage: ["layers.16", ..., "layers.23"]
    # 最后一个 stage: ["layers.24", ..., "layers.31", "norm", "output"]

    module_names_per_stage = []

    # 计算有效层数（包括 embedding 和 output 的权重）
    num_effective_layers = num_layers + input_weight + output_weight

    # 均匀分配
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # 计算这个 stage 的层数
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        if stage_idx == 0:
            # 第一个 stage: 包含 embedding
            stage_modules.append("tok_embeddings")
            remaining = effective_layers_for_stage - input_weight
            for _ in range(remaining):
                stage_modules.append(f"layers.{current_layer}")
                current_layer += 1

        elif stage_idx == num_stages - 1:
            # 最后一个 stage: 包含 output
            remaining = effective_layers_for_stage - output_weight
            for _ in range(remaining):
                stage_modules.append(f"layers.{current_layer}")
                current_layer += 1
            stage_modules.extend(["norm", "output"])

        else:
            # 中间 stage: 只有 transformer 层
            for _ in range(effective_layers_for_stage):
                stage_modules.append(f"layers.{current_layer}")
                current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage
```

**示例**：
```python
# Llama3 8B, 32 layers, 4 stages
generate_llm_fqn_per_model_part(4, 32)

# 返回:
[
    ["tok_embeddings", "layers.0", ..., "layers.7"],   # Stage 0: 8 layers
    ["layers.8", ..., "layers.15"],                    # Stage 1: 8 layers
    ["layers.16", ..., "layers.23"],                   # Stage 2: 8 layers
    ["layers.24", ..., "layers.31", "norm", "output"], # Stage 3: 8 layers
]
```

### 4.3 实际模型切分：pipeline_module_split

```python
# 来自: torchtitan/distributed/pipeline_parallel.py:337-475

def pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    根据模块名切分模型，创建 PipelineStage。
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()

    def _build_stage_from_modules(stage_idx, module_names, num_stages):
        # 深拷贝整个模型
        model = copy.deepcopy(whole_model)

        # 只保留这个 stage 需要的模块
        modules_to_keep = set(module_names)
        for module_name, module_value in model.named_children():
            if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
                # 处理 layers
                layers_to_keep = {...}
                # 删除不需要的层
            elif module_name not in modules_to_keep:
                # 设置为 None
                setattr(model, module_name, None)

        # 创建 PipelineStage
        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    # 计算当前 rank 负责哪些 stage
    def _get_stage_indices():
        stages_per_rank = num_stages // pp_degree

        if style == "loop":  # Interleaved schedule
            # Rank 0: Stage 0, 4, 8, ...
            # Rank 1: Stage 1, 5, 9, ...
            return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
        elif style == "v":   # ZeroBubble V-shaped
            # Rank 0: Stage 0, Stage (N-1)
            # Rank 1: Stage 1, Stage (N-2)
            return stage_v_pairs[pp_rank]

    # 构建 stages
    stages = []
    models = []
    for stage_idx in _get_stage_indices():
        stage, model_chunk = _build_stage_from_modules(...)
        stages.append(stage)
        models.append(model_chunk)

    return stages, models
```

**关键点**：
- **深拷贝模型**：每个 stage 从完整模型深拷贝，然后删除不需要的部分
- **Stage 分配**：根据 schedule 类型（loop 或 v）确定每个 rank 负责哪些 stage
- **PipelineStage**：PyTorch 的 `torch.distributed.pipelining.PipelineStage` 封装

### 4.4 构建 Schedule：build_pipeline_schedule

```python
# 来自: torchtitan/distributed/pipeline_parallel.py:156-223

def build_pipeline_schedule(
    job_config: JobConfig, stages: list[PipelineStage], loss_fn: Callable
) -> _PipelineSchedule:
    """
    根据配置构建 pipeline schedule。
    """
    # 获取 schedule 类
    schedule_class = get_schedule_class(
        job_config.parallelism.pipeline_parallel_schedule
    )

    # 计算 microbatch 数量
    microbatch_size = job_config.parallelism.pipeline_parallel_microbatch_size
    batch_size = job_config.training.local_batch_size
    n_microbatches = batch_size // microbatch_size

    # 验证
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than stages "
            f"({num_total_stages}) which may result in a bubble."
        )

    # 创建 schedule
    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=rescale_accumulated_loss(loss_fn, n_microbatches),
        scale_grads=False,
    )

    return schedule
```

**关键参数**：
- **n_microbatches**：`batch_size / microbatch_size`
  - 越大，Bubble 越小
  - 但内存越大

- **rescale_accumulated_loss**：
  - Loss 会累加 n_microbatches 次
  - 需要除以 n_microbatches 得到平均

### 4.5 训练循环中的使用

```python
# 来自: torchtitan/train.py:496-527

if parallel_dims.pp_enabled:
    # Pipeline Parallel forward / backward
    with self.train_context(optional_context_parallel_ctx):
        targets, losses = (
            (labels, []) if self.pp_has_last_stage else (None, None)
        )

        if self.pp_has_first_stage:
            # 第一个 stage: 需要传入 input
            self.pp_schedule.step(
                inputs,
                **extra_inputs,
                **extra_kwargs,
                target=targets,
                losses=losses,
                return_outputs=False,
            )
        else:
            # 中间 / 最后 stage: 不需要传入 input
            self.pp_schedule.step(
                **extra_kwargs,
                target=targets,
                losses=losses,
                return_outputs=False,
            )

    # 汇总 loss
    loss = (
        torch.sum(torch.stack(losses)).to(self.device)
        if self.pp_has_last_stage
        else torch.tensor([-1.0], device=self.device)
    )
```

**关键点**：
- **pp_schedule.step()**：执行完整的 forward + backward
- **只有 first_stage 传入 input**：其他 stage 从上一个 stage 接收
- **只有 last_stage 计算 loss**：loss 在最后一个 stage 产生
- **losses 列表**：收集所有 microbatch 的 loss

---

## 5. 性能分析

### 5.1 官方 Benchmark 结果

来自 `benchmarks/llama3_h100_202412_torchtitan.md`

**Table 5: Llama 3.1 405B, 512 H100s (FSDP 8, TP 8, PP 8)**

| Schedule | TPS/GPU | Memory (GiB) |
|----------|---------|--------------|
| **1F1B** | 100 | 82.5 |
| **Interleaved 1F1B** | 128 | 72.7 |

**分析**：
- **Interleaved 1F1B 快 28%**
- **Interleaved 1F1B 内存更少**：因为 Bubble 小，不需要保存那么多激活

### 5.2 Bubble 比例计算

**公式**：
```
Bubble 比例 ≈ (PP - 1) / n_microbatches

1F1B with PP=8, n_microbatches=32:
  Bubble = 7 / 32 = 21.9%

Interleaved 1F1B with PP=8, n_microbatches=32, stages_per_rank=2:
  有效 PP = 8 / 2 = 4
  Bubble = 3 / 32 = 9.4%
```

**Interleaved 的优势**：Bubble 减少了 12.5%

### 5.3 通信开销

**PP 的通信特点**：
- **点对点通信**：Stage 之间传递激活值
- **通信量小**：只传 activations，不传 weights

```python
# 每个 stage 之间的通信量
activation_size = batch_size * seq_len * hidden_dim * sizeof(dtype)

# Llama3 405B
batch_size = 2, seq_len = 8192, hidden_dim = 16384, dtype = fp16
activation_size = 2 * 8192 * 16384 * 2 = 512 MB

# 对比 FSDP (传递权重)
weight_size = 405B * 2 = 810 GB

PP 通信量 << FSDP 通信量
```

### 5.4 内存分析

**每个 stage 的内存**：
```python
# Llama3 405B, PP = 8

# 参数内存
params_per_stage = 405B / 8 = 50.6B params
params_memory = 50.6B * 2 bytes = 101 GB

# 激活内存 (1F1B)
# 需要保存 PP 个 microbatch 的激活
activations_memory = PP * activation_size = 8 * 512 MB = 4 GB

# 梯度内存
gradients_memory = params_memory = 101 GB

# 优化器状态 (AdamW)
optimizer_memory = params_memory * 2 = 202 GB

# 总内存
total = 101 + 4 + 101 + 202 = 408 GB
```

**但实际只有 82.5 GB？**

因为配合了：
- **FSDP**：参数分散到 8 个 GPU
- **Float8**：减少激活和梯度内存
- **Activation Checkpointing**：减少激活内存

### 5.5 与其他并行的组合效果

**3D Parallelism: FSDP + TP + PP**

```
Llama3 405B on 512 H100s

配置: FSDP 8, TP 8, PP 8
      512 = 8 × 8 × 8 GPUs

每个 GPU 的内存:
- 参数: 405B / 8 (FSDP) / 8 (TP) / 8 (PP) = 0.79B params = 1.6 GB
- 激活: 被 TP 切分，再被 CP 切分
- 总计: 72-82 GB

吞吐: 100-128 TPS/GPU
```

---

## 6. 使用场景和最佳实践

### 6.1 何时应该使用 Pipeline Parallel？

**推荐使用的场景**：

✅ **超大模型 (> 70B)**
   - 层数太多，单 GPU 放不下所有层
   - 需要跨节点分布模型

✅ **与 FSDP + TP 组合**
   - 3D 或 4D 并行
   - 处理超大模型 (405B+)

✅ **节点间通信受限**
   - PP 的通信量比 FSDP 小
   - 适合 InfiniBand 带宽有限的场景

**不推荐使用的场景**：

❌ **小模型 (< 13B)**
   - FSDP 足够，不需要 PP
   - PP 会引入 Bubble 开销

❌ **Batch size 太小**
   - 无法创建足够的 microbatch
   - Bubble 比例太高

❌ **调试阶段**
   - PP 增加调试难度
   - 先用 FSDP 调通，再加 PP

### 6.2 配置方法

**基本配置**：

```toml
[parallelism]
pipeline_parallel_degree = 4  # PP 并行度

# Schedule 选择
pipeline_parallel_schedule = "1F1B"  # 或 "Interleaved1F1B"

# Microbatch 配置
pipeline_parallel_microbatch_size = 1

[training]
local_batch_size = 8  # 必须能被 microbatch_size 整除
```

**n_microbatches 计算**：
```python
n_microbatches = local_batch_size / pipeline_parallel_microbatch_size
              = 8 / 1 = 8 microbatches
```

### 6.3 Schedule 选择指南

**1F1B**（默认）：
```toml
pipeline_parallel_schedule = "1F1B"
```
- 简单稳定
- 适合大多数场景
- 每个 rank 1 个 stage

**Interleaved 1F1B**（推荐大模型）：
```toml
pipeline_parallel_schedule = "Interleaved1F1B"
```
- 更少 Bubble
- 每个 rank 多个 stage
- 需要更多 microbatch

**ZeroBubble**（极致性能）：
```toml
pipeline_parallel_schedule = "ZBVZeroBubble"
# 或
pipeline_parallel_schedule = "InterleavedZeroBubble"
```
- 理论 0 Bubble
- 实现复杂
- 需要仔细调参

### 6.4 Microbatch 数量调优

**经验法则**：

```python
# 最小 microbatch 数
min_microbatches = PP * 2  # 至少 2 倍 PP 数量

# 推荐 microbatch 数
recommended = PP * 4 ~ PP * 8

# 示例
PP = 8:
  min = 16, recommended = 32 ~ 64
```

**trade-off**：
- **microbatch 太少**：Bubble 大，效率低
- **microbatch 太多**：内存大，小 batch 通信开销占比高

### 6.5 层数分配调优

**自动分配** (默认)：

```toml
[parallelism]
# 不指定 layers_per_stage，自动均匀分配
```

**手动分配**：

```toml
[parallelism]
pipeline_parallel_layers_per_stage = 4
# Llama3 8B (32 layers), PP = 4:
# 每个 stage 8 层，总共 4 stages

# 或指定具体模块
module_fqns_per_model_part = [
    ["tok_embeddings", "layers.0", ..., "layers.9"],   # Stage 0: 10 layers
    ["layers.10", ..., "layers.19"],                    # Stage 1: 10 layers
    ["layers.20", ..., "layers.29"],                    # Stage 2: 10 layers
    ["layers.30", "layers.31", "norm", "output"],       # Stage 3: 2 layers + output
]
```

**负载均衡**：

```toml
[parallelism]
# 第一个 stage 少放层（因为有 embedding）
pipeline_parallel_first_stage_less_layers = 1

# 最后一个 stage 少放层（因为有 output）
pipeline_parallel_last_stage_less_layers = 1
```

### 6.6 与其他并行的组合

**推荐组合**：

| 模型大小 | GPU 数 | 配置 | 说明 |
|---------|--------|------|------|
| 8B | 8 | FSDP 8 | 只用 FSDP |
| 70B | 64 | FSDP 8, TP 8 | 2D |
| 405B | 256 | FSDP 4, TP 8, PP 8 | 3D |
| 405B + 长序列 | 512 | FSDP 8, TP 8, PP 8, CP 1-8 | 4D |

**配置示例 (Llama3 405B on 512 H100s)**：

```toml
[model]
name = "llama3"
flavor = "405B"
converters = ["float8"]

[training]
local_batch_size = 8
seq_len = 8192

[parallelism]
# 512 = 8 × 8 × 8
data_parallel_shard_degree = 8   # FSDP
tensor_parallel_degree = 8       # TP
pipeline_parallel_degree = 8     # PP
enable_async_tensor_parallel = true

# PP 配置
pipeline_parallel_schedule = "Interleaved1F1B"
pipeline_parallel_microbatch_size = 1

[activation_checkpoint]
mode = "full"

[compile]
enable = true
components = ["model", "loss"]
```

### 6.7 调试技巧

**1. 验证切分是否正确**：

```python
# 查看日志
# PP rank 0 is building stage_idx 0 with modules [tok_embeddings, layers.0, ...]
# PP rank 1 is building stage_idx 1 with modules [layers.8, ...]
```

**2. 检查 Bubble**：

```python
# 查看 warning
# "Number of microbatches (4) is less than stages (8) which may result in a bubble."
```

**3. Profiling**：

```bash
[profiling]
enable_profiling = true
```

用 `chrome://tracing` 查看：
- 各个 stage 的 forward/backward 时间
- 通信时间
- Bubble 时间

**常见问题**：

❓ **Loss 不对？**
- 检查 last_stage 是否正确计算 loss
- 检查 loss 是否正确 rescale

❓ **OOM？**
- 减少 microbatch 数量
- 增加 PP 并行度
- 启用 activation checkpointing

❓ **速度很慢？**
- 检查 microbatch 数量是否太少
- 考虑用 Interleaved schedule
- 检查通信是否成为瓶颈

---

## 7. 总结

### 7.1 核心要点

用**工厂流水线**总结 Pipeline Parallel：

```
传统方式 = 一个工人完成所有工序
    工序太多，记不住，效率低

Pipeline Parallel = 流水线作业
    每人负责一道工序
    工件在流水线上依次传递
    并行处理多个工件
```

**三大核心概念**：

1. **Stage**：模型的一部分（若干层）
2. **Microbatch**：数据的一部分
3. **Schedule**：forward/backward 的执行顺序

### 7.2 性能特点

**优点**：
- ✅ **通信量小**：只传 activations
- ✅ **实现简单**：按层切分
- ✅ **与其他并行组合好**：3D/4D 并行

**缺点**：
- ❌ **有 Bubble**：GPU 空闲时间
- ❌ **需要足够的 microbatch**
- ❌ **调试复杂**

### 7.3 Schedule 选择

| Schedule | Bubble | 内存 | 推荐场景 |
|----------|--------|------|---------|
| **1F1B** | 中 | 低 | 默认选择 |
| **Interleaved 1F1B** | 低 | 中 | 大模型 |
| **ZeroBubble** | ~0 | 中 | 极致性能 |

**实测性能 (405B)**：
- 1F1B: 100 TPS/GPU
- Interleaved 1F1B: 128 TPS/GPU (**+28%**)

### 7.4 使用建议

**推荐使用**：
- ✅ 超大模型 (> 70B)
- ✅ 与 FSDP + TP 组合
- ✅ 节点间通信受限

**不推荐使用**：
- ❌ 小模型 (< 13B)
- ❌ Batch size 太小
- ❌ 调试阶段

**配置要点**：
```toml
[parallelism]
pipeline_parallel_degree = 8
pipeline_parallel_schedule = "Interleaved1F1B"
pipeline_parallel_microbatch_size = 1

[training]
local_batch_size = 32  # = 32 microbatches，Bubble ≈ 22%
```

### 7.5 与其他并行的对比

| 特性 | FSDP | TP | CP | **PP** |
|------|------|----|----|--------|
| **切分对象** | 参数 | 单层权重 | 序列 | **层** |
| **通信量** | 大 | 中 | 小 | **小** |
| **内存节省** | 参数 | 参数+激活 | 激活 | **参数** |
| **额外开销** | 无 | 少 | 少 | **Bubble** |
| **适用场景** | 通用 | 单层大 | 序列长 | **层数多** |

---

## 8. 参考资料

**源码文件**：
- `torchtitan/distributed/pipeline_parallel.py` - PP 核心实现
- `torchtitan/train.py` - 训练循环中的使用
- `torchtitan/config/job_config.py` - 配置选项

**PyTorch 官方资源**：
- [Pipeline Parallelism](https://pytorch.org/docs/stable/distributed.pipelining.html)
- [Schedule 实现](https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/schedules.py)

**相关论文**：
- GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism
- PipeDream: Generalized Pipeline Parallelism for DNN Training
- Zero Bubble Pipeline Parallelism

**相关文档**：
- `docs/analysis/02_tensor_parallel_implementation.md` - TP 详解
- `docs/analysis/03_async_tensor_parallel.md` - Async TP 详解
- `docs/analysis/04_context_parallel.md` - CP 详解
- `benchmarks/llama3_h100_202412_torchtitan.md` - 性能 Benchmark
- `docs/converging.md` - 收敛性验证

---

## 附录：高级话题

### A.1 Custom Schedule

TorchTitan 支持从 CSV 文件加载自定义 schedule：

```toml
[parallelism]
pipeline_parallel_schedule_csv = "/path/to/schedule.csv"
```

CSV 格式定义了每个时间步每个 rank 执行什么操作。

### A.2 Virtual Stage 的内存 trade-off

**Interleaved 1F1B 的内存计算**：

```python
# PP = 4, stages_per_rank = 2

# 每个 rank 持有 2 个 stage
# 需要保存 2 组 forward 激活

# 1F1B:
# 内存 = PP 个激活 = 4 个激活

# Interleaved 1F1B:
# 内存 = PP / stages_per_rank * stages_per_rank = PP 个激活
# 但实际上因为 warm-up 阶段，可能需要更多

# 实测：Interleaved 内存反而更小
# 因为 Bubble 小，不需要那么长的 warm-up 阶段
```

### A.3 PP 与 Gradient Accumulation 的交互

```python
# 配置
local_batch_size = 8
microbatch_size = 1
gradient_accumulation_steps = 4

# 实际执行
# 每个 training step:
#   1. 执行 4 次 forward-backward (gradient_accumulation_steps)
#   2. 每次有 8 个 microbatch (n_microbatches)
#   3. 总共处理 4 × 8 = 32 个样本
#   4. 然后做一次 optimizer.step()
```

### A.4 PP 的通信模式

```
Stage 0          Stage 1          Stage 2          Stage 3

Forward:
[F0] ──send──→ [F0] ──send──→ [F0] ──send──→ [F0]
     activation     activation     activation

Backward:
[B0] ←──send── [B0] ←──send── [B0] ←──send── [B0]
     gradient      gradient      gradient
```

**通信原语**：
- **P2P Send/Recv**：相邻 stage 之间
- **异步通信**：可以与计算重叠

### A.5 Model-aware 切分

对于某些模型，可能需要特殊的切分策略：

```python
# 例如：Mixture of Experts 模型
# Expert 层很大，可能需要单独放在一个 stage

module_fqns_per_model_part = [
    ["tok_embeddings", "layers.0", ..., "layers.7"],   # 普通层
    ["layers.8"],                                      # Expert 层 (单独一个 stage)
    ["layers.9", ..., "layers.15"],                    # 普通层
    ["layers.16", ..., "layers.23", "norm", "output"], # 普通层
]
```

TorchTitan 的 `module_fqns_per_model_part` 配置支持这种灵活的切分。
