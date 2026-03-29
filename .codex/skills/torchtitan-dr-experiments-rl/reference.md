# RL Reference

## File Map

- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/.gitignore`
  - Ignores `example_checkpoint/`, which is the expected local HF checkpoint cache for the README flow.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/README.md`
  - User-facing setup and launch guide for the RL experiment.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/__init__.py`
  - Re-exports the vLLM wrapper and registration helper.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/config_registry.py`
  - Config entrypoints for the RL controller.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/plugin.py`
  - vLLM model-name constant and registry glue.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/simple_grpo_sum_digits.py`
  - Controller, GPU allocation, training loop, and evaluation loop.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/sum_digits.py`
  - Demo task, answer extraction, and reward function.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/inference_example.py`
  - Standalone vLLM generation example.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/types.py`
  - Shared `Episode` dataclass.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/generator.py`
  - vLLM rollout actor and its compile/sampling configs.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/grader.py`
  - Reward-scoring actor.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/trainer.py`
  - TorchTitan-based policy trainer actor.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/utils.py`
  - Log-prob extraction, PPO/GRPO-style loss computation, and identity verification.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/attention.py`
  - Custom vLLM attention backend and adapter layer.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/parallelize.py`
  - RL-specific Qwen3 parallelization plan.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/vllm_wrapper.py`
  - vLLM-facing wrapper around a TorchTitan model.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/tests/test_attn_numerics.py`
  - End-to-end numerics parity test between vLLM generation and trainer-side log-prob recomputation.

## Key Types And Fields

### `Episode`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/types.py`.

- `policy_version: int`
- `prompt_token_ids: list[int]`
- `text: str`
- `token_ids: list[int]`
- `token_log_probs: list[float]`
- `expected_answer: str = ""`
- `reward: float = 0.0`
- `group_id: str = ""`
- `advantage: float = 0.0`

Operational meaning:
- generator sets prompt, completion, log-probs, `policy_version`, `expected_answer`, and `group_id`
- grader mutates `reward`
- controller mutates `advantage`
- trainer consumes all fields except `text` is only used for logging and sample inspection.

### `RLTrainer.Config`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/simple_grpo_sum_digits.py`.

- `model_spec: ModelSpec | None`
- `hf_assets_path: str`
- `num_steps: int`
- `dump_folder: str`
- `batch_invariant_mode: bool`
- `num_episodes_per_step: int`
- `log_samples: bool`
- `trainer: PolicyTrainer.Config`
- `generator: VLLMGenerator.Config`

Notable invariant:
- `model_spec` is expected to be injected by the config registry, then mutated so `parallelize_fn = parallelize_qwen3`.

### `PolicyTrainer.Config`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/trainer.py`.

- `optimizer: OptimizersContainer.Config`
- `lr_scheduler: LRSchedulersContainer.Config`
- `training: TrainingConfig`
- `parallelism: ParallelismConfig`
- `comm: CommConfig`
- `compile: CompileConfig`

### `VLLMGenerator.Config`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/generator.py`.

- `parallelism: ParallelismConfig`
- `sampling: SamplingConfig`
- `attention_backend: str = "FLASH_ATTN"`
- `model_dtype: str = "bfloat16"`
- `gpu_memory_limit: float = 0.9`
- `compile: GeneratorCompileConfig`
- `num_samples_per_prompt: int = 8`
- `seed: int | None = None`

Validation:
- `data_parallel_shard_degree` must be `1` or `-1`
- `data_parallel_replicate_degree` must be `1`

### `GeneratorCompileConfig`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/generator.py`.

- `backend: Literal["none", "eager", "inductor"]`
- `cudagraph_mode: Literal["none", "piecewise", "full", "full_and_piecewise"]`

Methods:
- `is_eager`
- `get_vllm_compilation_config()`

Important rule:
- piecewise or full-and-piecewise cudagraph capture requires compile backend not equal to `"none"`.

### `SamplingConfig`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/generator.py`.

- `temperature`
- `top_p`
- `max_tokens`

### `Provisioner`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/simple_grpo_sum_digits.py`.

- `total_gpus`
- `next_gpu`
- `available`
- `allocate(num_gpus) -> Callable[[], None]`

Its bootstrap closure sets `CUDA_VISIBLE_DEVICES` before CUDA initialization inside spawned processes.

## Main Functions And Responsibilities

### `register_model_to_vllm_model_registry(model_spec)`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/plugin.py`.

- Imports vLLM lazily.
- Builds a dynamic subclass of `TorchTitanVLLMModelWrapper` that closes over `model_spec`.
- Renames the class to `TorchTitanCausalLM`.
- Registers it into `ModelRegistry`.

### `RLTrainer._compute_world_size(p)`
- Multiplies replicate DP, shard DP, TP, PP, and CP.
- Ignores EP/ETP in the controller's mesh sizing.

### `RLTrainer.setup()`
- Computes separate trainer and generator world sizes.
- Allocates disjoint GPU ranges.
- Spawns Monarch proc meshes:
  - trainer mesh on GPUs
  - generator mesh on GPUs
  - grader mesh on CPU
- Initializes torch elastic env for trainer and generator meshes.
- Spawns actors and initializes TorchStore.
- Pushes trainer weights and pulls generator policy version `0`.

### `RLTrainer.evaluate(num_samples=20)`
- Builds held-out prompts from `SumDigitsTask(seed=99)`.
- Calls generator and scores only the first completion per prompt.
- Reports `accuracy`, `correct`, `total`, `format_rate`, `format_ok`.

### `RLTrainer.train()`
- Runs pre-training eval.
- For each step:
  - creates prompt/answer pairs
  - gets episodes from the generator
  - grades episodes
  - computes within-group advantages
  - calls trainer step
  - pushes and pulls weights
  - logs reward, correctness, avg token count, log-prob drift, and elapsed time
- Stops early if loss becomes non-finite.
- Runs post-training eval.

### `VLLMGenerator.__init__(...)`
- Registers the model with vLLM.
- Sets vLLM env vars.
- Optionally enables batch invariance.
- Constructs `EngineArgs` and `LLMEngine`.
- Records sampling fields and policy version.

### `VLLMGenerator.generate(prompt_texts, expected_answers)`
- Builds `SamplingParams` with `n = num_samples_per_prompt`, `logprobs=1`, `prompt_logprobs=1`, and `RequestOutputKind.FINAL_ONLY`.
- Adds requests to the engine and steps until completion.
- Sorts outputs by `request_id`.
- Creates one `group_id` per prompt using `pid`, `policy_version`, and index.
- Returns a flat list of `Episode` values.

### `VLLMGenerator.pull_model_state_dict(version)`
- Reads `"model_state_dict"` from TorchStore into `self._get_model().model.state_dict()`.
- Uses direct RDMA if available.
- Updates local `policy_version`.

### `Grader.score(episodes)`
- Calls `reward_fn([ep.text], ep.expected_answer)` per episode.
- Writes `ep.reward`.
- Logs reward mean and std.

### `PolicyTrainer.__init__(...)`
- Sets device from `LOCAL_RANK`.
- Calls `dist_utils.init_distributed(config.comm)`.
- Builds `ParallelDims`.
- Builds optional state-dict adapter.
- Builds and HF-loads both the trainable model and frozen reference model through `_build_model(...)`.
- Constructs optimizers and LR schedulers.
- Computes DP shard metadata: `dp_size`, `dp_rank`, `dp_enabled`.

### `PolicyTrainer._build_model(...)`
- Builds the model on meta device.
- Applies `model_spec.parallelize_fn(...)`.
- Materializes with `to_empty(...)` and `init_weights(...)`.
- Loads HF weights via `_load_initial_hf_weights(...)`.

### `PolicyTrainer._load_initial_hf_weights(model, checkpoint_path)`
- Creates HF `storage_reader` from `state_dict_adapter`.
- Uses DCP `load(...)` on adapter-shaped HF state dict.
- Converts back with `from_hf(...)`.
- Applies `set_model_state_dict(..., strict=True)`.

### `PolicyTrainer.push_model_state_dict()`
- Publishes `self.model.state_dict()` to TorchStore.
- Uses `transfer_dtype` only when generator dtype differs from training dtype.

### `PolicyTrainer.step(episodes)`
- Extracts rewards, advantages, prompt tokens, completion tokens, and rollout log-probs.
- Shards samples across DP ranks by striding over the flattened episode list.
- Computes reference log-probs with `compute_token_log_probs(...)`.
- Computes policy loss and metrics with `compute_policy_gradient_loss(...)`.
- Computes drift metrics with `verify_logprob_identity(...)`.
- All-reduces gradients across DP ranks if DP is enabled.
- Clips gradients, steps optimizer and scheduler, increments policy version, returns metrics.

### `compute_token_log_probs(...)`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/utils.py`.

- Concatenates prompt and generated tokens.
- Creates explicit `positions` to avoid symbolic-shape RoPE slice issues under compile.
- Runs model forward.
- Applies `log_softmax` in fp32.
- Slices out generated-token positions only.

### `compute_policy_gradient_loss(...)`
- Recomputes current-policy token log-probs for each sample.
- Computes per-token log-ratio vs reference model.
- Uses mean per-token ratio and mean KL per sample.
- Applies PPO clipping and entropy regularization.
- Returns total loss, metrics, and batch token log-probs.

### `verify_logprob_identity(...)`
- Checks bitwise equality between rollout-side and trainer-side token log-probs.
- Tracks max delta, average delta, and log-ratio drift.

### `replace_with_vllm_attention(model, tp_degree)`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/attention.py`.

- Requires `model.layers`.
- Computes local KV head count from total KV heads and TP degree.
- Replaces each block's `attention.inner_attention` with `VLLMAttention`.

### `parallelize_qwen3(...)`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/parallelize.py`.

- Applies TP via `apply_non_moe_tp(...)` when enabled.
- Optionally applies `apply_compile_dense_rl(...)`.
- Accepts `has_position_id` because generator and trainer call signatures differ.

### `create_torchtitan_config_from_vllm_config(vllm_config)`
Defined in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/vllm_wrapper.py`.

- Converts vLLM parallel config into TorchTitan `ParallelDims` and `ParallelismConfig`.
- Hard-codes `dp_shard=1`, `ep=1`, and `etp=1` for inference.

### `TorchTitanVLLMModelWrapper`
- Builds the TorchTitan model from `model_spec.model`.
- Replaces attention with vLLM-compatible attention.
- Applies TorchTitan parallelization inside constructor.
- Extends RoPE cache if `max_model_len` exceeds current cache.
- Loads HF weights during init, then exposes:
  - `embed_input_ids(...)`
  - `get_input_embeddings(...)`
  - `forward(...)`
  - `compute_logits(...)`
  - `load_weights_from_state_dict(...)`
  - `_initial_load_weights(...)`
  - `load_weights(...)` as a vLLM no-op acknowledgment path

Special behavior:
- monkeypatches `vllm.utils.torch_utils.weak_ref_tensor` to handle DTensor outputs during piecewise cudagraph capture.

## Test Contract

`/home/scbjtfy/torchtitan/torchtitan/experiments/rl/tests/test_attn_numerics.py` is the strongest behavioral contract in the module.

What it asserts:
- the RL registration path can boot a vLLM engine using `TorchTitanCausalLM`
- trainer-side model building through `parallelize_qwen3(...)` and HF weight loading matches generator-side behavior
- per-token log-probs on the same prompt plus generated completion are bitwise identical

Important test assumptions:
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- TP size of `2`
- custom attention backend enabled
- Qwen3 `0.6B` checkpoint available under the example path

## Operational Notes And Pitfalls

- `simple_grpo_sum_digits.py` and `inference_example.py` both mutate `config.model_spec.parallelize_fn` at runtime. If the model spec becomes immutable or reused elsewhere, this mutation point must be revisited.
- `PolicyTrainer._compute_world_size(...)` ignores EP/ETP while `PolicyTrainer` itself still stores them in `ParallelDims`; if the RL experiment expands expert parallel use, controller sizing must be revisited.
- The generator assumes `all_outputs.sort(key=lambda o: int(o.request_id))` is enough to recover prompt ordering.
- `verify_logprob_identity(...)` only compares the local DP shard inside `PolicyTrainer.step(...)`; global parity debugging still requires understanding sharding.
- The README says batch invariance only supports TP=1, but the example configs use TP values greater than `1`. Treat that as an experimental or unstable area rather than a guaranteed supported path.
