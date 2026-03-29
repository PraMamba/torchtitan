---
name: torchtitan-dr-experiments-rl
description: Use when changing TorchTitan's RL experiment, especially Monarch actor orchestration, vLLM generator integration, policy-gradient loss flow, TorchStore weight sync, or Qwen3-based trainer and inference behavior.
---

# TorchTitan Experiments RL

## Overview

`/home/scbjtfy/torchtitan/torchtitan/experiments/rl` is TorchTitan's reinforcement-learning experiment layer for running a TorchTitan model as both a training policy and a vLLM-backed rollout policy. The module combines a top-level controller in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/simple_grpo_sum_digits.py`, Monarch actors in `actors/`, a vLLM adapter stack in `models/`, a registration shim in `plugin.py`, and a toy sum-digits task used to demonstrate GRPO training. The design goal is to keep trainer and generator on the same TorchTitan model definition so rollout log-probs and trainer log-probs can be compared directly and weight synchronization can be done between two independently parallelized meshes.

## Public Surface

- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/__init__.py`
  - Re-exports `TorchTitanVLLMModelWrapper` and `register_model_to_vllm_model_registry(...)` for manual vLLM registration.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/config_registry.py`
  - `rl_grpo_qwen3_0_6b()`, `rl_grpo_qwen3_1_7b()`, `rl_grpo_qwen3_debug()` return complete `RLTrainer.Config` presets discoverable via `--module rl --config ...`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/simple_grpo_sum_digits.py`
  - `Provisioner`: allocates disjoint GPU ranges for Monarch meshes.
  - `RLTrainer.Config`: top-level config for model spec, trainer actor, generator actor, rollout count, logging, and HF asset path.
  - `RLTrainer`: orchestrates setup, evaluation, training, advantage calculation, and cross-mesh weight sync.
  - `main()`: async entrypoint using `ConfigManager`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/generator.py`
  - `GeneratorCompileConfig`, `SamplingConfig`, and `VLLMGenerator`.
  - `VLLMGenerator.generate(...)` returns a flat list of `Episode` values.
  - `VLLMGenerator.pull_model_state_dict(...)` refreshes generator weights from TorchStore.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/grader.py`
  - `Grader.score(...)` fills `Episode.reward`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/trainer.py`
  - `PolicyTrainer.Config`, `PolicyTrainer.push_model_state_dict(...)`, and `PolicyTrainer.step(...)`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/utils.py`
  - `compute_token_log_probs(...)`, `compute_policy_gradient_loss(...)`, and `verify_logprob_identity(...)`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/attention.py`
  - Registers the custom vLLM backend `PyTorchFlashAttentionBackend`.
  - Defines `VLLMAttention` and `replace_with_vllm_attention(...)`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/parallelize.py`
  - `parallelize_qwen3(...)` and `apply_non_moe_tp(...)` for the RL-specific TP plan.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/vllm_wrapper.py`
  - `create_torchtitan_config_from_vllm_config(...)` and `TorchTitanVLLMModelWrapper`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/plugin.py`
  - `VLLM_MODEL_NAME` and `register_model_to_vllm_model_registry(...)`.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/types.py`
  - `Episode` dataclass used across generator, grader, controller, and trainer.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/sum_digits.py`
  - `SumDigitsTask`, `extract_answer(...)`, and the reward function used by the example loop.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/inference_example.py`
  - Standalone vLLM inference smoke test using the RL registration path.
- `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/tests/test_attn_numerics.py`
  - Numerics contract that trainer and vLLM generator log-probs match bitwise for the same prompt plus generated tokens.

## Design Logic

- The module splits responsibilities by runtime role instead of by algorithm stage. `RLTrainer` is a controller, `VLLMGenerator` owns the inference engine, `Grader` owns reward assignment, and `PolicyTrainer` owns optimization. That keeps each mesh-specific concern isolated and lets Monarch place actors on different GPU groups.
- The experiment patches `config.model_spec.parallelize_fn` to `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/parallelize.py:parallelize_qwen3` in both `RLTrainer.__init__()` and `inference_example.py`. This is deliberate: the RL path needs a TP plan and attention replacement that are compatible with vLLM's paged-attention runtime, not the standard training parallelization path.
- vLLM registration is model-agnostic. `register_model_to_vllm_model_registry(...)` in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/plugin.py` builds a closure class around a `ModelSpec` and always registers it as `TorchTitanCausalLM`, which must match `hf_overrides["architectures"]` passed to `EngineArgs`.
- Weight sync is designed around TorchStore instead of direct actor-to-actor parameter copies. `PolicyTrainer.push_model_state_dict(...)` publishes the trainer-side state dict, and `VLLMGenerator.pull_model_state_dict(...)` pulls it into the wrapped vLLM model state dict. The transfer path optionally uses direct RDMA if available.
- The policy-gradient loss is written around per-token log-probs rather than full-sequence likelihoods. `compute_policy_gradient_loss(...)` averages log-ratios and KL terms across tokens per sample before applying PPO clipping so longer generations do not explode ratio magnitudes.
- The custom attention backend exists to keep vLLM and trainer numerics aligned. `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/attention.py` replaces vLLM's default attention kernel path with PyTorch's FlashAttention varlen API and wraps it in `VLLMAttention`, which also handles DTensor unwrap and re-wrap through `LocalMapAttention`.
- The experiment currently hard-codes several scope limits on purpose:
  - generator data parallelism is rejected in `VLLMGenerator.Config.__post_init__()`
  - MoE inference through the RL TP plan is rejected in `apply_non_moe_tp(...)`
  - pipeline parallelism is unsupported in `TorchTitanVLLMModelWrapper.supports_pp = False`
  - batch-invariant mode is documented as TP=1-only in the README even though the configs expose TP sizes.

## Core Data Structures

- `Episode` in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/types.py`
  - Carries `policy_version`, `prompt_token_ids`, `text`, `token_ids`, `token_log_probs`, `expected_answer`, `reward`, `group_id`, and `advantage`.
  - This is the contract between generator, grader, controller logic, and trainer.
- `RLTrainer.Config` in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/simple_grpo_sum_digits.py`
  - Top-level controller config containing `model_spec`, `hf_assets_path`, `num_steps`, `dump_folder`, `batch_invariant_mode`, `num_episodes_per_step`, `log_samples`, `trainer`, and `generator`.
- `PolicyTrainer.Config` in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/trainer.py`
  - Wraps optimizer, LR scheduler, `TrainingConfig`, `ParallelismConfig`, `CommConfig`, and `CompileConfig`.
- `VLLMGenerator.Config` in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/generator.py`
  - Encodes generator-side `parallelism`, `sampling`, `attention_backend`, `model_dtype`, `gpu_memory_limit`, `compile`, `num_samples_per_prompt`, and optional `seed`.
- `GeneratorCompileConfig` and `SamplingConfig` in `actors/generator.py`
  - Map directly onto vLLM `CompilationConfig` and `SamplingParams` creation.
- `ParallelDims` and `ParallelismConfig`
  - Used twice with different meanings: once in `PolicyTrainer` for trainer mesh setup, and once in `TorchTitanVLLMModelWrapper` by reconstructing TorchTitan parallel settings from `VllmConfig`.
- `TorchTitanVLLMModelWrapper` in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/vllm_wrapper.py`
  - Holds `state_dict_adapter`, `parallelize_fn`, `config`, `model`, `rope_config`, `parallel_dims`, and implements the vLLM model API.
- `VLLMAttention` in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/attention.py`
  - Adapts TorchTitan attention tensor layout to vLLM's flattened token-major interface.

See `/home/scbjtfy/torchtitan/.codex/skills/torchtitan-dr-experiments-rl/reference.md` for field-level detail and file-by-file API inventory.

## State Flow

1. Config selection:
   `ConfigManager.parse_args()` resolves `--module rl --config ...` to one of the presets in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/config_registry.py`, producing an `RLTrainer.Config`.
2. Controller setup:
   `RLTrainer.__init__()` swaps the model spec's `parallelize_fn` to `parallelize_qwen3` and builds `SumDigitsTask`. `RLTrainer.setup()` computes trainer and generator world sizes from each `ParallelismConfig`, allocates GPU ranges through `Provisioner`, spawns Monarch meshes, initializes torch-elastic envs, and spawns `PolicyTrainer`, `VLLMGenerator`, and `Grader`.
3. Initial weight sync:
   After `ts.initialize(...)`, the trainer publishes weights via `push_model_state_dict()`, and the generator pulls version `0` via `pull_model_state_dict(0)`.
4. Generator path:
   `VLLMGenerator.__init__()` registers the TorchTitan model with vLLM, sets environment variables, optionally enables batch-invariant mode, creates `EngineArgs`, and builds `LLMEngine`. `generate(...)` adds one vLLM request per prompt, steps until all outputs are finished, sorts outputs by request ID, and flattens them into `Episode` values grouped by `group_id`.
5. Grading and controller logic:
   `Grader.score(...)` calls the task reward function once per episode and fills `ep.reward`. Back in `RLTrainer.train()`, the controller groups episodes by `group_id`, computes GRPO advantages as reward minus within-group mean reward, and optionally logs the first sample per group.
6. Trainer path:
   `PolicyTrainer.__init__()` initializes distributed state, constructs `ParallelDims`, optionally builds a state-dict adapter, builds the policy model plus frozen reference model, loads HF weights through DCP, and creates optimizers and LR schedulers.
7. Loss computation:
   `PolicyTrainer.step(...)` shards episodes across DP ranks, computes reference log-probs with `compute_token_log_probs(...)`, computes policy loss and divergence stats with `compute_policy_gradient_loss(...)`, verifies rollout-vs-trainer log-prob identity with `verify_logprob_identity(...)`, all-reduces gradients across DP ranks if enabled, clips gradients, steps optimizer and scheduler, increments `policy_version`, and returns metrics.
8. Ongoing weight sync:
   `RLTrainer.train()` calls `trainer.push_model_state_dict()` then `generator.pull_model_state_dict(policy_version)` after each step so the generator always serves the latest policy.
9. Evaluation and smoke tests:
   `RLTrainer.evaluate()` reuses the generator and grader path on held-out prompts. `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/inference_example.py` exercises registration plus standalone generation, while `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/tests/test_attn_numerics.py` compares trainer-computed and vLLM log-probs bitwise.

## Error Handling And Side Effects

- `Provisioner.allocate(...)` raises `RuntimeError` if requested GPU count exceeds what remains.
- `GeneratorCompileConfig.__post_init__()` raises `ValueError` when piecewise/full-and-piecewise cudagraph modes are requested without torch.compile enabled.
- `VLLMGenerator.Config.__post_init__()` asserts that generator data-parallel shard and replicate degrees are effectively disabled.
- `PolicyTrainer._load_initial_hf_weights(...)` raises `FileNotFoundError` when the HF checkpoint directory does not exist.
- `apply_non_moe_tp(...)` raises `ValueError` if a transformer block has `moe_enabled=True`, because RL vLLM inference does not support Qwen3 MoE yet.
- `replace_with_vllm_attention(...)` raises `AttributeError` or `ValueError` if the model shape assumptions are broken (`.layers`, `.attention`, or incompatible KV-head partitioning).
- `TorchTitanVLLMModelWrapper.forward(...)` raises `NotImplementedError` for `inputs_embeds` and `ValueError` if no `input_ids` are provided.
- The module has several global side effects:
  - monkeypatches `vllm.utils.torch_utils.weak_ref_tensor` in `models/vllm_wrapper.py`
  - sets `VLLM_ATTENTION_BACKEND`, optionally `VLLM_BATCH_INVARIANT`, and in the inference/test scripts `VLLM_WORKER_MULTIPROC_METHOD=spawn`
  - mutates `config.model_spec.parallelize_fn` at runtime in both the controller and the inference example.

## Common Modification Scenarios

- Add a new RL config or model flavor:
  Start in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/config_registry.py`. Add a new config function that builds `RLTrainer.Config`, choose the model via `model_registry(...)`, and ensure any required `parallelize_fn` patch still matches the model family.
- Change rollout generation behavior or sampling defaults:
  Edit `VLLMGenerator.Config`, `SamplingConfig`, and `VLLMGenerator.generate(...)` in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/generator.py`. This is where request batching, sampling params, prompt ordering, group IDs, and per-token log-prob extraction are defined.
- Change policy-loss math or identity verification:
  Edit `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/actors/utils.py` and the call site in `PolicyTrainer.step(...)`. Keep `compute_token_log_probs(...)`, PPO clipping, KL computation, and verification metrics aligned if you change tensor layout or log-prob semantics.
- Replace the demo task or reward function:
  Edit `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/sum_digits.py` and the controller logic in `simple_grpo_sum_digits.py`. If the new task needs richer metadata than `expected_answer`, extend `Episode` in `types.py` and thread the new field through generator, grader, and trainer.
- Change how TorchTitan integrates with vLLM:
  Edit `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/plugin.py`, `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/vllm_wrapper.py`, and `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/attention.py`. The key junctions are dynamic registration, DTensor handling, RoPE cache extension, and layout conversion around vLLM attention.
- Extend RL support beyond dense Qwen3 TP:
  Start in `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/models/parallelize.py`. That file is the explicit temporary TP plan and is where MoE support, alternate model families, or trainer/generator parity changes belong.
- Debug trainer vs generator numerical drift:
  Use `/home/scbjtfy/torchtitan/torchtitan/experiments/rl/tests/test_attn_numerics.py`, `verify_logprob_identity(...)`, and the reference model path in `PolicyTrainer.step(...)`. The experiment treats log-prob parity as a first-order contract.
