# Distributed Checkpointing 分布式检查点详解

## 目录
- [1. 什么是 Distributed Checkpointing？](#1-什么是-distributed-checkpointing)
- [2. 搬桌子的比喻：拍照存档](#2-搬桌子的比喻拍照存档)
- [3. DCP vs 传统 Checkpoint](#3-dcp-vs-传统-checkpoint)
- [4. Async Checkpoint 三种模式](#4-async-checkpoint-三种模式)
- [5. 源码实现详解](#5-源码实现详解)
- [6. State Dict 管理](#6-state-dict-管理)
- [7. HuggingFace 格式支持](#7-huggingface-格式支持)
- [8. 与并行策略的配合](#8-与并行策略的配合)

---

## 1. 什么是 Distributed Checkpointing？

### 1.1 基本概念

**Distributed Checkpointing (DCP)** = 在分布式训练中，每个 GPU 只保存自己的那部分参数，而不是每个 GPU 都保存完整的模型。

**核心思想**：就像 FSDP 在训练时切分参数一样，checkpoint 时也切分保存。

### 1.2 为什么需要 DCP？

传统的 checkpoint 有两个大问题：

```
问题 1: 内存爆炸
假设 Llama3 70B 模型，bf16，8 GPUs 训练

传统方式（每个 GPU 都保存完整模型）：
GPU 0: 收集所有参数 → 140 GB → 保存到磁盘
GPU 1: 收集所有参数 → 140 GB → 保存到磁盘
...
GPU 7: 收集所有参数 → 140 GB → 保存到磁盘

问题：
- ❌ 每个 GPU 需要临时分配 140 GB 内存（OOM！）
- ❌ 8 个 GPU 保存 8 份重复的文件（浪费！）
```

```
问题 2: 速度慢
单个 GPU 保存 140 GB 到磁盘需要很长时间
如果是同步保存，训练会被阻塞！

时间线（同步保存）：
Training → [Pause] → GPU 0 保存 140GB (5-10 minutes) → [Resume]
                                  ↑
                          训练暂停，GPU 闲置！
```

**DCP 的解决方案**：

```
DCP 方式：
GPU 0: 只保存自己的 1/8 参数 → 17.5 GB
GPU 1: 只保存自己的 1/8 参数 → 17.5 GB
...
GPU 7: 只保存自己的 1/8 参数 → 17.5 GB

所有 GPU 并行保存！

好处：
✅ 每个 GPU 只需临时分配 17.5 GB（不会 OOM）
✅ 8 个 GPU 并行保存，速度快 8 倍
✅ 磁盘总共只存一份模型（节省空间）
```

### 1.3 Checkpoint 包含什么？

一个完整的 checkpoint 包含 5 个部分：

```
checkpoint/step-1000/
├── __0_0.distcp          ← GPU 0 的 model 参数分片
├── __1_0.distcp          ← GPU 1 的 model 参数分片
├── ...
├── __7_0.distcp          ← GPU 7 的 model 参数分片
├── __0_optimizer_0.distcp ← GPU 0 的 optimizer 状态
├── ...
├── __7_optimizer_0.distcp ← GPU 7 的 optimizer 状态
├── __0_lr_scheduler.distcp ← LR scheduler 状态
├── __0_dataloader.distcp   ← DataLoader 状态（当前位置）
├── __0_train_state.distcp  ← 训练状态（step、ntokens_seen）
└── .metadata              ← 元数据（告诉 DCP 如何重建）
```

**5 个组件**：

1. **Model** (`model`): 模型参数（weights）
2. **Optimizer** (`optimizer`): 优化器状态（momentum、variance等）
3. **LR Scheduler** (`lr_scheduler`): 学习率调度器状态
4. **DataLoader** (`dataloader`): 数据加载器状态（当前读到哪了）
5. **Train State** (`train_state`): 训练状态（当前步数、见过多少 tokens）

---

## 2. 搬桌子的比喻：拍照存档

### 2.1 回顾搬桌子的场景

还记得我们用搬桌子比喻并行训练吗？（[FSDP 文档](./01_fsdp2_per_parameter_sharding.md)）

```
房子（模型）里有很多桌子（参数）：
TransformerBlock 0:
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │ wq   │ │ wk   │ │ wv   │ │ wo   │
  └──────┘ └──────┘ └──────┘ └──────┘
  ┌──────┐ ┌──────┐ ┌──────┐
  │ w1   │ │ w2   │ │ w3   │
  └──────┘ └──────┘ └──────┘

FSDP 切分：每张桌子切成 4 份（FSDP=4）
GPU 0: 每张桌子的第 1 块
GPU 1: 每张桌子的第 2 块
GPU 2: 每张桌子的第 3 块
GPU 3: 每张桌子的第 4 块
```

### 2.2 Checkpoint = 拍照存档

**场景**：你和朋友们正在搬家具（训练模型），突然需要休息一下，明天继续。

**怎么记住当前进度？** → 拍照存档！

### 2.3 传统方式：完整拍照（问题多多）

```
传统 Checkpoint（非分布式）：

步骤 1: 收集所有桌子碎片
GPU 0: 把我的碎片给 GPU 0
GPU 1: 把我的碎片给 GPU 0
GPU 2: 把我的碎片给 GPU 0
GPU 3: 把我的碎片给 GPU 0
       ↓
GPU 0: 拼成完整的房子

步骤 2: GPU 0 拍照保存
GPU 0: [咔嚓] → 保存到相册（磁盘）

问题：
❌ GPU 0 需要临时存储完整的房子（内存爆炸）
❌ 其他 GPU 闲置等待（浪费时间）
❌ GPU 0 拍照很慢（单线程 IO）
```

### 2.4 DCP 方式：分片拍照（高效）

```
Distributed Checkpoint（分布式）：

步骤 1: 每人拍自己的部分
GPU 0: [咔嚓] 拍我负责的桌子碎片 → photo_0.jpg
GPU 1: [咔嚓] 拍我负责的桌子碎片 → photo_1.jpg
GPU 2: [咔嚓] 拍我负责的桌子碎片 → photo_2.jpg
GPU 3: [咔嚓] 拍我负责的桌子碎片 → photo_3.jpg

所有人同时拍照！（并行保存）

步骤 2: 存档管理器记录拼图方法
管理器: 记录 {
    GPU 0 的照片 → 桌子的第 1/4 部分
    GPU 1 的照片 → 桌子的第 2/4 部分
    GPU 2 的照片 → 桌子的第 3/4 部分
    GPU 3 的照片 → 桌子的第 4/4 部分
}
→ 保存为 .metadata

恢复时：
读取 .metadata → 知道每张照片对应哪部分
每个 GPU 读取自己的照片 → 恢复自己负责的碎片
继续搬桌子！

好处：
✅ 每人只拍自己的部分（内存占用低）
✅ 并行拍照（速度快）
✅ 总共只存一套照片（节省空间）
```

### 2.5 进一步的比喻：Optimizer = 工具箱

训练不只有桌子（参数），还有：

```
1. 桌子（Model）：家具本身
   GPU 0: wq的1/4, wk的1/4, wv的1/4, ...

2. 工具箱（Optimizer）：搬桌子用的工具
   - 动量 (momentum): 手推车的速度
   - 方差 (variance): 手推车的方向调整
   GPU 0: wq的工具箱1/4, wk的工具箱1/4, ...

3. 说明书（LR Scheduler）：搬家计划
   - 当前学习率：搬桌子的力度
   - 调度状态：搬家进度

4. 搬家清单（DataLoader）：
   - 当前位置：搬到第 1000 个数据了

5. 进度记录（Train State）：
   - 当前步数：第 500 步
   - 总工作量：处理了 2M tokens
```

**Checkpoint 就是把这 5 样东西都拍照存档**！

---

## 3. DCP vs 传统 Checkpoint

### 3.1 对比表

| 特性 | 传统 Checkpoint | Distributed Checkpoint (DCP) |
|-----|----------------|------------------------------|
| **内存占用** | 每个 GPU 需要完整模型 | 每个 GPU 只需自己的分片 |
| **保存速度** | 串行，慢 | 并行，快 8x-100x |
| **磁盘占用** | N 个 GPU 可能保存 N 份 | 只保存 1 份（分片存储） |
| **可扩展性** | 受限于单 GPU 内存 | 可扩展到任意大模型 |
| **加载方式** | 所有 GPU 读取相同文件 | 每个 GPU 读取自己的分片 |
| **与并行策略** | 需要手动处理 | 自动处理 FSDP/TP/PP |

### 3.2 内存占用对比

假设 Llama3 70B，bf16，8 GPUs：

```
传统 Checkpoint:
  模型参数: 70B × 2 bytes = 140 GB
  优化器状态: 140 GB × 2 (Adam) = 280 GB
  总计: 420 GB

  保存时每个 GPU 需要：
  - 训练时分片: 52.5 GB (420GB / 8)
  - 收集完整状态: + 420 GB
  ─────────────────
  峰值: 472.5 GB  😱 (OOM!)

DCP:
  保存时每个 GPU 需要：
  - 训练时分片: 52.5 GB
  - 临时拷贝（async）: + 52.5 GB
  ─────────────────
  峰值: 105 GB  ✅ (可行!)

节省: 367.5 GB / GPU
```

### 3.3 保存速度对比

假设每个 GPU 写入带宽 = 1 GB/s：

```
传统 Checkpoint (单 GPU 保存):
  时间 = 420 GB / 1 GB/s = 420 seconds = 7 minutes

DCP (8 GPUs 并行):
  时间 = 52.5 GB / 1 GB/s = 52.5 seconds

加速: 8x
```

实际上，DCP 还有**异步保存**，训练可以继续进行！

---

## 4. Async Checkpoint 三种模式

### 4.1 问题：Checkpoint 阻塞训练

即使是 DCP，如果**同步保存**，训练还是会被阻塞：

```
同步保存（Disabled）:
Step 495: Training... ✅
Step 496: Training... ✅
Step 497: Training... ✅
Step 498: Training... ✅
Step 499: Training... ✅
Step 500: [Checkpoint!]
    └─ 暂停训练
    └─ 拷贝参数到 CPU/Staging
    └─ 写入磁盘 (52 seconds)
    └─ 恢复训练
Step 501: Training... (终于继续了)

浪费时间: 52 秒 × 训练频率
```

**解决方案**：**Async Checkpoint** - 保存的同时继续训练！

### 4.2 三种模式

TorchTitan 支持 3 种 checkpoint 模式：

```python
# 来自: torchtitan/config/job_config.py:525-541

async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
```

#### 模式 1: Disabled（同步保存）

```
Timeline:
Training → [Pause] → Copy → Save → [Resume] → Training

特点:
- 训练暂停，等待保存完成
- 最简单，最可靠
- 慢，阻塞训练

适用：
- 调试、小模型
- 保存频率低（如每 5000 步）
```

#### 模式 2: Async（异步保存）

```
Timeline:
Training → [Copy to CPU] → Training (continues)
                ↓
           [Save in background]

原理:
1. 拷贝 GPU tensor 到 CPU (快，几秒钟)
2. 训练继续
3. 后台线程把 CPU tensor 写入磁盘

特点:
- ✅ 训练几乎不阻塞（只有拷贝时间）
- ✅ 简单的异步实现
- ❌ 需要额外 CPU 内存（存储拷贝）
- ❌ 拷贝本身还是有开销

适用：
- 中等频率保存（每 500-1000 步）
- CPU 内存充足
```

#### 模式 3: Async with Pinned Memory（最快）

```
Timeline:
Training → [Stage to Pinned Mem] → Training (continues)
                ↓
           [Upload via multiprocess]

原理:
1. 拷贝 GPU tensor 到 Pinned Memory (超快，DMA)
2. 训练继续
3. 独立进程通过 Pinned Memory 上传到磁盘

特点:
- ✅ 训练几乎零阻塞
- ✅ 拷贝速度最快（DMA）
- ✅ 独立进程上传，完全不影响训练
- ❌ 需要 Pinned Memory（GPU可寻址的CPU内存）
- ❌ 实现复杂

适用：
- 高频保存（每 100-500 步）
- 大模型训练
- 追求极致性能
```

### 4.3 时间开销对比

假设 52.5 GB 参数（Llama3 70B / 8 GPUs）：

```
Disabled:
  Copy: 0 (直接写)
  Save: 52.5s (阻塞训练)
  ──────
  训练暂停: 52.5s

Async:
  Copy GPU→CPU: 5s (阻塞训练)
  Save: 52.5s (后台)
  ──────
  训练暂停: 5s

Async with Pinned Mem:
  Copy GPU→Pinned: 2s (阻塞训练，DMA)
  Save: 52.5s (独立进程)
  ──────
  训练暂停: 2s

效果对比:
  Disabled: 52.5s 暂停
  Async: 5s 暂停 (↓ 90%)
  Async+Pinned: 2s 暂停 (↓ 96%)
```

### 4.4 配置示例

```toml
# 模式 1: Disabled（同步）
[checkpoint]
enable = true
interval = 5000
async_mode = "disabled"

# 模式 2: Async
[checkpoint]
enable = true
interval = 1000
async_mode = "async"

# 模式 3: Async with Pinned Memory（推荐大模型）
[checkpoint]
enable = true
interval = 500
async_mode = "async_with_pinned_mem"
```

---

## 5. 源码实现详解

### 5.1 CheckpointManager 核心架构

```python
# 来自: torchtitan/components/checkpoint.py:118-175

class CheckpointManager:
    """管理 TorchTitan 的 checkpointing 逻辑"""

    def __init__(
        self,
        dataloader: BaseDataLoader,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],  # 额外的状态（如 train_state）
        checkpoint_config: CheckpointConfig,
        sd_adapter: BaseStateDictAdapter | None,  # HF 格式转换器
        ft_manager: FTManager | None = None,  # Fault Tolerance
    ):
        # 1. 包装模型为 ModelWrapper
        self.states = {
            MODEL: ModelWrapper(model_parts),  # 支持多个 model parts (PP)
            OPTIMIZER: optimizers,
            LR_SCHEDULER: lr_schedulers,
            DATALOADER: dataloader,
            **states,  # train_state 等
        }

        # 2. 配置 Async 模式
        self.async_mode = AsyncMode[checkpoint_config.async_mode.upper()]

        # 3. 配置 Pinned Memory Stager
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.stager = DefaultStager(
                StagingOptions(
                    use_pinned_memory=True,  # 使用 Pinned Memory
                    use_separate_process=True,  # 独立进程上传
                )
            )

        # 4. 配置清理策略
        self.keep_latest_k = checkpoint_config.keep_latest_k
        if self.keep_latest_k > 0:
            self.purge_thread = threading.Thread(
                target=purge_thread,  # 后台删除旧 checkpoint
                daemon=True,
            )
```

**关键组件**：

1. **ModelWrapper**：包装模型，支持 Pipeline Parallel 的多 model parts
2. **States dict**：统一管理 5 个组件的状态
3. **Stager**：Pinned Memory 管理器
4. **Purge thread**：后台清理线程

### 5.2 Save 流程

```python
# 来自: torchtitan/components/checkpoint.py:468-541

@torch.no_grad()
def save(self, curr_step: int, last_step: bool = False) -> None:
    """保存 checkpoint"""

    # 1. 检查是否需要保存
    if not self._should_save(curr_step, last_step):
        return

    # 2. 等待上一次异步保存完成
    self._async_wait()

    # 3. 创建 checkpoint ID
    checkpoint_id = self._create_checkpoint_id(curr_step)
    # checkpoint_id = "outputs/checkpoint/step-500"

    # 4. 获取要保存的状态
    states = self._flattened_model_states_sd()
    # states = {
    #     "model.layers.0.attention.wq": DTensor(...),
    #     "optimizer": {...},
    #     "lr_scheduler": {...},
    #     ...
    # }

    # 5. 根据模式保存
    if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
        # 模式 3: Async + Pinned Memory
        result = self.dcp_save(
            states,
            checkpoint_id=checkpoint_id,
            async_mode=self.async_mode,
        )
        self.save_future = result.upload_completion  # 上传完成 Future
        self.staging_future = result.staging_completion  # Staging 完成 Future

    elif self.async_mode == AsyncMode.ASYNC:
        # 模式 2: Async
        self.save_future = self.dcp_save(
            states,
            checkpoint_id=checkpoint_id,
            async_mode=self.async_mode,
        )

    else:
        # 模式 1: Disabled (同步)
        self.dcp_save(
            states,
            checkpoint_id=checkpoint_id,
            async_mode=AsyncMode.DISABLED,
        )

    # 6. 清理旧 checkpoint
    self._purge_stale_checkpoints()
```

### 5.3 DCP Save 的核心

```python
# 来自: torchtitan/components/checkpoint.py:340-426

def dcp_save(
    self,
    state_dict: dict[str, Any],
    checkpoint_id: str,
    async_mode: AsyncMode,
) -> Future | None:
    """使用 DCP API 保存"""

    # 根据模式调用不同的 DCP API
    if async_mode == AsyncMode.ASYNC:
        # 异步保存（模式 2）
        return dcp.async_save(
            state_dict,
            checkpoint_id=checkpoint_id,
            process_group=self.pg,  # Gloo backend (CPU)
        )

    elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
        # 异步 + Pinned Memory（模式 3）
        return dcp.async_save(
            state_dict,
            checkpoint_id=checkpoint_id,
            process_group=self.pg,
            async_checkpointer_type=AsyncCheckpointerType.PROCESS,  # 独立进程
            async_stager=self.stager,  # Pinned Memory Stager
        )

    else:
        # 同步保存（模式 1）
        return dcp.save(
            state_dict,
            checkpoint_id=checkpoint_id,
        )
```

**DCP API 的工作原理**：

```
dcp.save(state_dict, checkpoint_id) 做了什么？

1. 分析 state_dict 中的 DTensor
   state_dict = {
       "model.wq": DTensor(local=[1024, 4096], global=[4096, 4096], Shard(0)),
       "optimizer.wq.momentum": DTensor(local=[1024, 4096], Shard(0)),
       ...
   }

2. 每个 GPU 保存自己的 local tensor
   GPU 0: 保存 wq[0:1024, :] → __0_0.distcp
   GPU 1: 保存 wq[1024:2048, :] → __1_0.distcp
   ...

3. Rank 0 保存元数据
   .metadata = {
       "model.wq": {
           "shape": [4096, 4096],
           "chunks": [
               {"rank": 0, "offsets": [0, 0], "lengths": [1024, 4096]},
               {"rank": 1, "offsets": [1024, 0], "lengths": [1024, 4096]},
               ...
           ]
       },
       ...
   }

4. 所有 GPU barrier 同步
   确保所有人都保存完成
```

### 5.4 Load 流程

```python
# 来自: torchtitan/components/checkpoint.py:544-638

def load(self, step: int = -1) -> bool:
    """加载 checkpoint"""

    # 1. 查找要加载的 step
    if step == -1:
        step = self._find_load_step()  # 找最新的
    if step == -1:
        return False  # 没有 checkpoint

    # 2. 创建 checkpoint ID
    checkpoint_id = self._create_checkpoint_id(step)

    # 3. 决定加载什么
    if step == 0:
        # step 0 是初始化，只加载模型
        states = self.states[MODEL].state_dict()
    else:
        # 加载完整 checkpoint
        states = self._flattened_model_states_sd()

    # 4. 使用 DCP 加载
    self.dcp_load(states, checkpoint_id)

    return True
```

**DCP Load 的工作原理**：

```
dcp.load(state_dict, checkpoint_id) 做了什么？

1. Rank 0 读取元数据
   .metadata → 知道每个 tensor 的分片信息

2. 每个 GPU 读取自己的分片
   GPU 0: 读取 __0_0.distcp → wq[0:1024, :]
   GPU 1: 读取 __1_0.distcp → wq[1024:2048, :]
   ...

3. 填充到 state_dict
   state_dict["model.wq"] = DTensor(
       local=wq[0:1024, :],  # 只加载自己的部分
       global_shape=[4096, 4096],
       placement=Shard(0),
   )

4. 调用 load_state_dict
   model.load_state_dict(state_dict)
   optimizer.load_state_dict(state_dict)
   ...

完成！每个 GPU 只读取自己的分片
```

### 5.5 Keep Latest K 策略

```python
# 来自: torchtitan/components/checkpoint.py:824-846

def _purge_stale_checkpoints(self):
    """清理旧的 checkpoint"""

    if self.keep_latest_k > 0 and dist.get_rank() == 0:
        # 1. 扫描所有 checkpoint
        discovered_checkpoints = []
        for filename in os.listdir(self.folder):
            match = re.search(r"step-(\d+)", filename)
            if match:
                step = int(match.group(1))
                path = os.path.join(self.folder, filename)
                discovered_checkpoints.append((step, path))

        # 2. 按 step 排序
        discovered_checkpoints.sort()

        # 3. 删除旧的（保留最新的 k 个）
        to_delete = discovered_checkpoints[:-self.keep_latest_k]

        # 4. 发送到后台删除线程
        for _, path in to_delete:
            self.purge_queue.put(path)
```

**为什么用后台线程删除？**

```
删除大文件夹很慢（shutil.rmtree）:
checkpoint/step-500/ (52 GB) → 删除需要 30-60 秒

如果在主线程删除：
Training → [Pause] → Delete step-500 (60s) → [Resume]
                              ↑
                      浪费 1 分钟！

后台线程删除：
Training → Queue.put(step-500) → Training (continues)
                  ↓
        [Background thread deletes it]

训练不受影响！
```

---

## 6. State Dict 管理

### 6.1 ModelWrapper：处理 Pipeline Parallel

Pipeline Parallel 有个问题：多个 model parts 的参数会冲突。

```python
# 问题示例：PP=2

# Rank 0 (Stage 0: layers 0-15)
model_part_0.layers[0].wq → "layers.0.wq"

# Rank 1 (Stage 1: layers 16-31)
model_part_1.layers[0].wq → "layers.0.wq"  # 冲突！
                                            # 实际是 layers.16.wq
```

**ModelWrapper 的解决方案**：

```python
# 来自: torchtitan/components/checkpoint.py:58-82

class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]):
        # 支持单个或多个 model parts
        self.model = [model] if isinstance(model, nn.Module) else model

    def state_dict(self) -> dict[str, Any]:
        # 从所有 model parts 收集 state dict
        state_dict = {
            k: v
            for sd in map(get_model_state_dict, self.model)
            for k, v in sd.items()
        }
        # 自动合并，键不会冲突
        # 因为 Pipeline split 保证了参数名唯一
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        # 加载到所有 model parts
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),  # 允许部分加载
        )
        list(map(func, self.model))
```

**为什么 strict=False？**

```
PP Rank 0 (Stage 0):
  state_dict 包含: layers.0-15.*

PP Rank 1 (Stage 1):
  state_dict 包含: layers.16-31.*

加载时：
  Rank 0: 只加载 layers.0-15.*，忽略 layers.16-31.* (strict=False)
  Rank 1: 只加载 layers.16-31.*，忽略 layers.0-15.* (strict=False)

每个 rank 只加载自己需要的部分！
```

### 6.2 Optimizer State Dict Flattening

Optimizer 也有类似问题，需要 **flattening**：

```python
# 问题：Optimizer 的 state_dict 是基于 index 的

# PP Rank 0
optimizer.state_dict() = {
    "state": {
        0: {"momentum": ...},  # 对应 layers.0.wq
        1: {"momentum": ...},  # 对应 layers.0.wk
        ...
    },
    "param_groups": [{"params": [0, 1, ...]}]
}

# PP Rank 1
optimizer.state_dict() = {
    "state": {
        0: {"momentum": ...},  # 对应 layers.16.wq (不是 layers.0!)
        1: {"momentum": ...},
        ...
    },
    "param_groups": [{"params": [0, 1, ...]}]
}

# 冲突！两个 rank 都有 index 0，但指向不同参数
```

**解决方案：Flattening**

```python
# PyTorch DCP 提供的 flattening 功能

# 保存时：
optimizer_state_dict = {
    "state": {
        "model.layers.0.wq": {"momentum": ...},  # 用 FQN (全限定名) 而不是 index
        "model.layers.0.wk": {"momentum": ...},
        ...
    }
}

# 现在不同 rank 的 FQN 不会冲突了！
# Rank 0: model.layers.0.wq
# Rank 1: model.layers.16.wq

# DCP 会自动处理这个转换
# 在 OptimizersContainer 中启用：flatten_optimizer_state_dict=True
```

### 6.3 State Dict 的三种形式

TorchTitan 使用三种 state dict 形式：

```python
# 1. Native State Dict（原始）
model.state_dict() = {
    "tok_embeddings.weight": Tensor(...),
    "layers.0.attention.wq.weight": Tensor(...),
    ...
}

# 2. Sharded State Dict (FSDP/DCP)
get_model_state_dict(model) = {
    "tok_embeddings.weight": DTensor(Shard(0), ...),
    "layers.0.attention.wq.weight": DTensor(Shard(0), ...),
    ...
}

# 3. HuggingFace State Dict (转换后)
sd_adapter.to_hf(state_dict) = {
    "model.embed_tokens.weight": Tensor(...),  # 重命名
    "model.layers.0.self_attn.q_proj.weight": Tensor(...),
    ...
}
```

**使用场景**：

- **Sharded State Dict**: 训练中保存/加载（DCP 格式）
- **HuggingFace State Dict**: 导出给 HF Transformers 使用

---

## 7. HuggingFace 格式支持

### 7.1 为什么需要 HF 格式？

```
问题：训练完成后，想用 HuggingFace Transformers 推理

TorchTitan 格式:
checkpoint/step-10000/
├── __0_0.distcp
├── __1_0.distcp
├── ...
└── .metadata

HuggingFace 需要的格式:
checkpoint/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
└── model-00004-of-00004.safetensors

完全不同！
```

**TorchTitan 的解决方案**：`StateDictAdapter`

### 7.2 StateDictAdapter 工作原理

```python
# 来自: torchtitan/protocols/state_dict_adapter.py

class BaseStateDictAdapter:
    """State dict 转换器基类"""

    def to_hf(self, state_dict: dict) -> dict:
        """TorchTitan → HuggingFace"""
        raise NotImplementedError

    def from_hf(self, hf_state_dict: dict) -> dict:
        """HuggingFace → TorchTitan"""
        raise NotImplementedError

    def get_hf_storage_reader(self, path: str):
        """创建 HF 格式读取器"""
        raise NotImplementedError
```

**实际示例**（Llama3）：

```python
# 重命名规则
LLAMA3_KEY_MAPPING = {
    # TorchTitan → HuggingFace
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

def to_hf(state_dict):
    hf_state_dict = {}
    for tt_key, tensor in state_dict.items():
        # 应用重命名规则
        hf_key = apply_mapping(tt_key, LLAMA3_KEY_MAPPING)
        hf_state_dict[hf_key] = tensor
    return hf_state_dict
```

### 7.3 保存 HF 格式

```python
# 来自: torchtitan/components/checkpoint.py:364-421

# 配置
[checkpoint]
last_save_in_hf = true  # 最后一步保存为 HF 格式

# 保存流程
if to_hf:
    # 1. 转换 state dict
    state_dict = self.sd_adapter.to_hf(state_dict)

    # 2. 使用 HuggingFaceStorageWriter
    storage_writer = HuggingFaceStorageWriter(
        path=checkpoint_id,
        save_distributed=True,  # 分布式保存
        enable_consolidation=True,  # 合并分片
    )

    # 3. DCP 保存（仍然是分布式）
    dcp.save(state_dict, storage_writer=storage_writer)

    # 4. 合并成最终的 safetensors
    # checkpoint/
    # ├── model-00001-of-00004.safetensors
    # ├── model-00002-of-00004.safetensors
    # ├── model-00003-of-00004.safetensors
    # ├── model-00004-of-00004.safetensors
    # └── model.safetensors.index.json
```

### 7.4 从 HF 格式加载

```toml
# 配置
[checkpoint]
initial_load_in_hf = true
initial_load_path = "/path/to/hf/checkpoint"

# 或者使用默认 HF assets
[model]
hf_assets_path = "/path/to/hf/llama3-8b"
```

```python
# 加载流程
if from_hf:
    # 1. 转换 state dict（创建模板）
    hf_state_dict = self.sd_adapter.to_hf(state_dict)

    # 2. 使用 HF Storage Reader
    hf_storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_id)

    # 3. DCP 加载
    dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

    # 4. 转换回 TorchTitan 格式
    state_dict = self.sd_adapter.from_hf(hf_state_dict)

    # 5. 加载到模型
    model.load_state_dict(state_dict)
```

---

## 8. 与并行策略的配合

### 8.1 FSDP + Checkpoint

FSDP 已经把参数切分了，checkpoint 自然就是分布式的：

```
FSDP (8 GPUs):
GPU 0: wq[0:512, :]
GPU 1: wq[512:1024, :]
...
GPU 7: wq[3584:4096, :]

DCP Save:
GPU 0: 保存 wq[0:512, :] → __0_0.distcp
GPU 1: 保存 wq[512:1024, :] → __1_0.distcp
...
GPU 7: 保存 wq[3584:4096, :] → __7_0.distcp

DCP Load:
GPU 0: 读取 __0_0.distcp → wq[0:512, :]
GPU 1: 读取 __1_0.distcp → wq[512:1024, :]
...
GPU 7: 读取 __7_0.distcp → wq[3584:4096, :]

完美配合！无需额外通信
```

### 8.2 TP + Checkpoint

TP 切分单层权重，checkpoint 保存的是切分后的：

```
TP (4 GPUs):
GPU 0: wq[:, 0:1024]    (列切分，前 1/4)
GPU 1: wq[:, 1024:2048]
GPU 2: wq[:, 2048:3072]
GPU 3: wq[:, 3072:4096]

DCP Save:
GPU 0: 保存 wq[:, 0:1024] → __0_0.distcp
GPU 1: 保存 wq[:, 1024:2048] → __1_0.distcp
...

DCP Load:
GPU 0: 读取 __0_0.distcp → wq[:, 0:1024]
GPU 1: 读取 __1_0.distcp → wq[:, 1024:2048]
...

也是完美配合！
```

### 8.3 FSDP + TP (2D 并行)

2D 并行更复杂，但 DCP 自动处理：

```
配置: DP=8, TP=8 (64 GPUs)

参数布局 (wq [4096, 4096]):
GPU 0:  wq[0:512, 0:512]      (DP 的 1/8, TP 的 1/8)
GPU 1:  wq[0:512, 512:1024]   (DP 的 1/8, TP 的 2/8)
...
GPU 7:  wq[0:512, 3584:4096]  (DP 的 1/8, TP 的 8/8)
GPU 8:  wq[512:1024, 0:512]   (DP 的 2/8, TP 的 1/8)
...
GPU 63: wq[3584:4096, 3584:4096] (DP 的 8/8, TP 的 8/8)

DCP Save:
每个 GPU 保存自己的双切分块

Metadata:
{
    "model.wq": {
        "shape": [4096, 4096],
        "chunks": [
            {"rank": 0, "offsets": [0, 0], "lengths": [512, 512]},
            {"rank": 1, "offsets": [0, 512], "lengths": [512, 512]},
            ...
            {"rank": 63, "offsets": [3584, 3584], "lengths": [512, 512]},
        ]
    }
}

DCP 自动推断 placement: [Shard(0), Shard(1)]
```

### 8.4 PP + Checkpoint

Pipeline Parallel 最复杂，但有 ModelWrapper 处理：

```
PP (4 Stages, 16 GPUs, 每个 stage 4 GPUs FSDP):

Stage 0 (Rank 0-3):  layers 0-7
Stage 1 (Rank 4-7):  layers 8-15
Stage 2 (Rank 8-11): layers 16-23
Stage 3 (Rank 12-15): layers 24-31

State Dict:
Rank 0-3:
  "layers.0.wq", "layers.1.wq", ..., "layers.7.wq"

Rank 4-7:
  "layers.8.wq", "layers.9.wq", ..., "layers.15.wq"

参数名不冲突！

DCP Save:
所有 rank 保存自己的参数
Metadata 记录每个 layer 在哪个 rank

DCP Load:
每个 rank 读取自己的 layers
ModelWrapper 用 strict=False 忽略其他 layers
```

### 8.5 完整示例：3D 并行

```
Llama3 405B on 512 H100s
配置: DP=8, TP=8, PP=8

参数总量: 405B × 2 bytes = 810 GB

每个 GPU 的参数:
  810 GB / 512 = 1.58 GB

Checkpoint 结构:
checkpoint/step-1000/
├── __0_0.distcp           (1.58 GB, Rank 0)
├── __1_0.distcp           (1.58 GB, Rank 1)
├── ...
├── __511_0.distcp         (1.58 GB, Rank 511)
├── __0_optimizer_0.distcp (3.16 GB, Rank 0, Adam 2x)
├── ...
├── __511_optimizer_0.distcp (3.16 GB, Rank 511)
└── .metadata              (记录所有分片信息)

总大小:
  参数: 1.58 GB × 512 = 810 GB
  优化器: 3.16 GB × 512 = 1620 GB
  总计: 2430 GB (分布在 512 个文件中)

保存速度（Async + Pinned Mem）:
  每个 GPU 写入: 1.58 GB + 3.16 GB = 4.74 GB
  写入时间: 4.74 GB / 1 GB/s = 4.74 seconds
  训练暂停: ~2 seconds (Staging)

对比传统 checkpoint:
  需要收集完整模型: 2430 GB (OOM!)
  写入时间: 2430 GB / 1 GB/s = 40 minutes

效率提升:
  内存: 无限 (传统方式根本无法完成)
  速度: 500x (4.74s vs 40min)
```

---

## 9. 实战案例

### 9.1 Llama3 8B (8 GPUs)

**配置**：

```toml
[checkpoint]
enable = true
folder = "checkpoint"
interval = 500
async_mode = "async"
keep_latest_k = 10
```

**Checkpoint 大小**：

```
模型参数: 8B × 2 bytes = 16 GB
优化器状态: 16 GB × 2 (Adam) = 32 GB
总计: 48 GB

每个 GPU:
  参数: 16 GB / 8 = 2 GB
  优化器: 32 GB / 8 = 4 GB
  总计: 6 GB

Checkpoint 文件:
checkpoint/step-500/
├── __0_0.distcp (2 GB)
├── __0_optimizer_0.distcp (4 GB)
├── __1_0.distcp (2 GB)
├── __1_optimizer_0.distcp (4 GB)
├── ...
├── __7_0.distcp (2 GB)
├── __7_optimizer_0.distcp (4 GB)
└── .metadata (几 KB)

总大小: 48 GB (8 × 6 GB)
```

**性能**：

```
Async 模式:
  Copy to CPU: 6 GB → 2 seconds
  训练继续
  Background save: 6 GB @ 1 GB/s → 6 seconds

训练暂停: 2 seconds
吞吐损失: ~0.03% (假设 500 步需要 1 hour)
```

### 9.2 Llama3 70B (256 GPUs)

**配置**：

```toml
[checkpoint]
enable = true
interval = 1000
async_mode = "async_with_pinned_mem"
keep_latest_k = 5
```

**Checkpoint 大小**：

```
模型参数: 70B × 2 bytes = 140 GB
优化器状态: 140 GB × 2 = 280 GB
总计: 420 GB

每个 GPU:
  参数: 140 GB / 256 = 0.547 GB
  优化器: 280 GB / 256 = 1.094 GB
  总计: 1.641 GB

Checkpoint 文件数: 256 × 2 = 512 files
总大小: 420 GB
```

**性能**：

```
Async + Pinned Mem 模式:
  Stage to Pinned Memory: 1.641 GB → 1 second (DMA)
  训练继续
  Process upload: 1.641 GB @ 1 GB/s → 1.6 seconds

训练暂停: 1 second
几乎无感！
```

### 9.3 Llama3 405B (512 GPUs)

**配置**：

```toml
[checkpoint]
enable = true
interval = 500
async_mode = "async_with_pinned_mem"
keep_latest_k = 3
last_save_model_only = true
last_save_in_hf = true
export_dtype = "bfloat16"
```

**Checkpoint 策略**：

```
训练中 (每 500 步):
  保存完整 checkpoint (模型 + 优化器 + ...)
  格式: DCP 分布式
  用于恢复训练

最后一步 (step 10000):
  只保存模型
  格式: HuggingFace safetensors
  用于推理部署
  精度: bfloat16 (节省空间)
```

**Checkpoint 大小**：

```
训练中:
  模型: 405B × 2 bytes = 810 GB
  优化器: 810 GB × 2 = 1620 GB
  总计: 2430 GB
  每个 GPU: 4.75 GB

最后一步（模型 only + bf16）:
  模型: 405B × 2 bytes = 810 GB
  每个 GPU: 1.58 GB

HF 格式（合并后）:
  model-00001-of-00008.safetensors (100 GB)
  model-00002-of-00008.safetensors (100 GB)
  ...
  model-00008-of-00008.safetensors (100 GB + 10 GB)
  model.safetensors.index.json
```

---

## 10. 调试与优化

### 10.1 常见问题

**Q1: Checkpoint 保存很慢**

```
症状:
  Checkpoint 保存耗时 > 10 minutes

原因：
1. 使用 async_mode = "disabled"
2. 磁盘 IO 带宽不足
3. 文件系统不支持并行写入

解决：
1. 启用 async_mode = "async" 或 "async_with_pinned_mem"
2. 检查磁盘: iostat -x 1
3. 使用分布式文件系统（Lustre, GPFS）
4. 增加 interval，减少保存频率
```

**Q2: OOM during checkpoint**

```
症状:
  Checkpoint 时 CUDA out of memory

原因：
1. async_mode = "disabled" 需要临时内存
2. 同时有多个 async checkpoint 在进行
3. Pinned Memory 不足

解决：
1. 使用 async_mode = "async_with_pinned_mem"
2. 等待上一次 checkpoint 完成:
   checkpointer.maybe_wait_for_staging()
3. 调整 GC 策略
```

**Q3: Checkpoint 无法恢复训练**

```
症状:
  load checkpoint 失败或数值不对

原因：
1. 并行度变化（训练时 DP=8, 恢复时 DP=16）
2. Model 结构变化
3. Checkpoint 损坏

解决：
1. 保持并行度一致
2. 使用 initial_load_model_only=true（只加载模型）
3. 检查 .metadata 文件是否完整
4. 使用 keep_latest_k > 1 保留多个备份
```

**Q4: 磁盘空间不足**

```
症状:
  No space left on device

原因：
1. keep_latest_k = 0，保留所有 checkpoint
2. Checkpoint 太大
3. 清理线程没有及时删除

解决：
1. 设置 keep_latest_k = 3-5
2. 只在最后保存模型：last_save_model_only = true
3. 手动清理旧 checkpoint:
   rm -rf checkpoint/step-*
   (保留最新的几个)
```

### 10.2 性能优化技巧

**技巧 1: 选择合适的 async_mode**

```toml
# 小模型 (< 10B)
[checkpoint]
async_mode = "async"
interval = 1000

# 中等模型 (10B-70B)
[checkpoint]
async_mode = "async_with_pinned_mem"
interval = 500

# 大模型 (> 70B)
[checkpoint]
async_mode = "async_with_pinned_mem"
interval = 500
enable_first_step_checkpoint = true  # 第一步也保存（验证系统）
```

**技巧 2: 调整 keep_latest_k**

```toml
# 调试阶段
[checkpoint]
keep_latest_k = 3  # 只保留 3 个，快速迭代

# 长期训练
[checkpoint]
keep_latest_k = 10  # 保留 10 个，防止损坏

# 磁盘空间受限
[checkpoint]
keep_latest_k = 2  # 最少 2 个（不能为 1）
```

**技巧 3: 分离训练和导出 checkpoint**

```toml
# 训练中：每 500 步保存完整 checkpoint
[checkpoint]
interval = 500
last_save_model_only = false

# 最后：只保存模型 + HF 格式
last_save_model_only = true
last_save_in_hf = true
export_dtype = "bfloat16"  # 节省空间

效果:
  训练中: 可以随时恢复
  最后: 得到推理用的 HF checkpoint
```

**技巧 4: 利用 GC 优化内存**

```python
# 来自: torchtitan/components/checkpoint.py:423-424

# DCP 会自动在 checkpoint 后 GC
if enable_garbage_collection:
    GarbageCollection.collect("GC collection invoked by checkpointer.")

# 对于 async checkpoint，GC 在 _async_wait() 后调用
# 因为 async 时 CPU 内存仍被占用
```

### 10.3 监控指标

**关键指标**：

```python
# 1. Checkpoint 保存时间
begin = time.monotonic()
checkpointer.save(curr_step)
checkpoint_time = time.monotonic() - begin

logger.info(f"Checkpoint took {checkpoint_time:.2f} seconds")

# 2. Checkpoint 大小
checkpoint_id = f"checkpoint/step-{step}"
checkpoint_size = sum(
    os.path.getsize(os.path.join(checkpoint_id, f))
    for f in os.listdir(checkpoint_id)
)
logger.info(f"Checkpoint size: {checkpoint_size / 1e9:.2f} GB")

# 3. Staging 时间（Pinned Memory 模式）
if async_mode == "async_with_pinned_mem":
    checkpointer.staging_future.result()  # 等待 staging
    logger.info(f"Staging took {staging_time:.2f} seconds")

# 4. 磁盘使用
du -sh checkpoint/
```

---

## 11. 总结

### 11.1 DCP 的核心优势

用**搬桌子拍照**的比喻总结：

1. **分布式拍照**：每人拍自己的部分，并行保存
   - ✅ 内存占用低（不需要收集完整模型）
   - ✅ 速度快（并行 IO）
   - ✅ 可扩展（支持任意大模型）

2. **异步拍照**：拍照的同时继续搬桌子
   - ✅ 训练几乎不阻塞
   - ✅ 3 种模式适应不同场景

3. **智能管理**：
   - ✅ 自动清理旧照片（keep_latest_k）
   - ✅ 支持 HF 格式（导出推理）
   - ✅ 与所有并行策略无缝配合

### 11.2 使用建议

```
小模型训练 (< 10B, 单机):
  → async_mode = "async"
  → interval = 1000
  → keep_latest_k = 5

中等模型 (10B-70B, 多机):
  → async_mode = "async_with_pinned_mem"
  → interval = 500
  → keep_latest_k = 5

大模型 (> 70B, 大规模):
  → async_mode = "async_with_pinned_mem"
  → interval = 500
  → keep_latest_k = 3
  → last_save_in_hf = true (导出 HF)
```

### 11.3 配置速查

```toml
# 完整配置示例
[checkpoint]
# 基础
enable = true
folder = "checkpoint"
interval = 500

# Async 模式
async_mode = "async_with_pinned_mem"  # 或 "async", "disabled"

# 清理策略
keep_latest_k = 5  # 保留最新 5 个

# 初始加载
initial_load_path = "/path/to/pretrained"  # 可选
initial_load_model_only = true  # 只加载模型
initial_load_in_hf = false  # 是否从 HF 加载

# 最后保存
last_save_model_only = true  # 最后只保存模型
last_save_in_hf = true  # 保存为 HF 格式
export_dtype = "bfloat16"  # 导出精度

# 其他
enable_first_step_checkpoint = false  # 第一步是否保存
exclude_from_loading = []  # 加载时排除的组件
```

### 11.4 与并行策略的关系

```
FSDP:
  参数已经分片 → DCP 直接保存分片 → 完美配合

TP:
  单层权重切分 → DCP 保存切分后的 → 完美配合

PP:
  多个 model parts → ModelWrapper 统一管理 → 完美配合

FSDP + TP + PP (3D):
  参数在 3 个维度切分 → DCP 自动推断 placement → 完美配合

结论: DCP 与所有并行策略无缝集成！
```

### 11.5 关键源码

```
核心文件:
- torchtitan/components/checkpoint.py:118-846
  - CheckpointManager: 主类
  - ModelWrapper: PP 支持
  - save/load: 保存和加载

配置:
- torchtitan/config/job_config.py:421-550
  - Checkpoint 配置类

PyTorch DCP API:
- torch.distributed.checkpoint.save
- torch.distributed.checkpoint.async_save
- torch.distributed.checkpoint.load
```

---

## 12. 参考资料

**源码文件**：
- `torchtitan/components/checkpoint.py` - CheckpointManager 实现
- `torchtitan/config/job_config.py:421-550` - Checkpoint 配置
- `torchtitan/protocols/state_dict_adapter.py` - HF 格式转换

**PyTorch 官方文档**：
- [Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
- [Async Checkpoint](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)

**相关文档**：
- [01_fsdp2_per_parameter_sharding.md](./01_fsdp2_per_parameter_sharding.md) - FSDP2 实现
- [02_tensor_parallel_implementation.md](./02_tensor_parallel_implementation.md) - TP 实现
- [05_pipeline_parallel.md](./05_pipeline_parallel.md) - PP 实现

**学术论文**：
- PyTorch Distributed Checkpoint: Efficient State Persistence for Large-Scale Training

---

**最后更新**：2025年1月

**文档版本**：1.0
