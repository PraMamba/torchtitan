# Scripts Generate Reference

## File Index

### [`scripts/generate/README.md`](../../../scripts/generate/README.md)
- Declares the module as a "Model Generation Check", not a serving stack.
- Establishes two main usage modes:
  - single-GPU smoke test
  - multi-GPU TP smoke test with optional JSON stdout
- Points users to `python -m scripts.generate.test_generate --help` for the authoritative CLI.

### [`scripts/generate/run_llama_generate.sh`](../../../scripts/generate/run_llama_generate.sh)
- Defaults:
  - `NGPU=1`
  - `LOG_RANK=0`
  - `MODULE=llama3`
  - `CONFIG=llama3_debugmodel`
  - `CHECKPOINT_DIR=./outputs/checkpoint/`
  - `PROMPT=""`
- Argument handling:
  - strips `--prompt=...` from positional overrides
  - if the prompt value is an existing file, replaces it with file contents
  - forwards every other arg untouched to `scripts.generate.test_generate`
- Launch contract:
  - `torchrun --standalone`
  - `--nproc_per_node="${NGPU}"`
  - `--local-ranks-filter="${LOG_RANK}"`
  - module execution via `-m scripts.generate.test_generate`

### [`scripts/generate/_generation.py`](../../../scripts/generate/_generation.py)

#### `multinomial_sample_one(probs, rng=None) -> torch.Tensor`
- Uses `torch.empty_like(probs).exponential_(1, generator=rng)` to sample Gumbel-like noise.
- Returns `argmax(probs / q, dim=-1, keepdim=True)` cast to `torch.long`.
- Output shape is `(B, 1)` when input `probs` is `(B, vocab)`.

#### `logits_to_probs(logits, temperature=1.0, top_k=None) -> torch.Tensor`
- Divides logits by `max(temperature, 1e-5)` to avoid division by zero.
- If `top_k` is set:
  - computes `torch.topk(logits, k=min(top_k, logits.size(-1)))`
  - uses the smallest kept value as a pivot
  - masks all logits below the pivot to `-Inf`
- Returns `softmax(logits, dim=-1)`.

#### `generate_next_token(model, x, *, temperature=1.0, top_k=None, rng=None) -> torch.Tensor`
- Calls `model(x)` and assumes output shape `(B, T, vocab_size)`.
- Slices only `logits[:, -1, :]` for next-token sampling.
- Delegates filtering to `logits_to_probs(...)` and selection to `multinomial_sample_one(...)`.

#### `generate(model, input_ids, *, max_new_tokens, temperature=1.0, top_k=None, seed=None) -> torch.Tensor`
- Adds a batch dimension when `input_ids.ndim == 1`.
- Seeds a device-local `torch.Generator` when `seed` is provided.
- Clones `input_ids` into `generated_tokens`.
- Repeats:
  - call `generate_next_token(...)`
  - concatenate token along `dim=1`
- Returns the full sequence including prompt tokens.

### [`scripts/generate/test_generate.py`](../../../scripts/generate/test_generate.py)

#### Import-time behavior
- Appends `Path(__file__).parent.parent.resolve()` to `sys.path`.
- Imports local helper as `from generate._generation import generate`.
- Depends on:
  - `torchtitan.components.metrics.build_device_memory_monitor`
  - `torchtitan.config.ConfigManager`, `DebugConfig`
  - `torchtitan.distributed.ParallelDims`, `torchtitan.distributed.utils`
  - `torchtitan.tools.utils`, `torchtitan.tools.logging`
  - `torchtitan.components.tokenizer.HuggingFaceTokenizer`

#### `apply_tp_minus_sp(model, tp_mesh)`
- Applies TP to:
  - `"tok_embeddings"` with `RowwiseParallel(input_layouts=Replicate())`
  - `"output"` with `ColwiseParallel(output_layouts=Replicate())`
- Iterates `model.layers.items()` and applies:
  - `attention.wq`, `attention.wk`, `attention.wv` -> `ColwiseParallel()`
  - `attention.wo` -> `RowwiseParallel()`
  - `feed_forward.w1`, `feed_forward.w3` -> `ColwiseParallel()`
  - `feed_forward.w2` -> `RowwiseParallel()`
- Deliberately omits sequence parallelism; the comment ties that limitation to unevenly sharded sequences.

#### `test_generate(...)`

Parameters:
- `model_name: str`
- `config_name: str`
- `checkpoint_path: str`
- `prompt: str`
- keyword-only:
  - `temperature: float = 1.0`
  - `max_new_tokens: int = 32`
  - `batch_size: int = 1`
  - `top_k: int | None = None`
  - `seed: int | None = None`
  - `deterministic: bool = False`

Execution steps:
1. `init_logger()`
2. Build config with `ConfigManager.parse_args(["--module", model_name, "--config", config_name])`
3. Warn on empty prompt via `args.prompt`
4. Resolve `WORLD_SIZE` and `LOCAL_RANK`
5. Set current device through `device_module.set_device(device)`
6. Create `device_memory_monitor = build_device_memory_monitor()`
7. Build tokenizer with `HuggingFaceTokenizer.Config().build(tokenizer_path=config.hf_assets_path)`
8. Get `model_config = config.model_spec.model`
9. Run `model_config.update_from_config(trainer_config=config)`
10. Build the model on `"meta"` if `world_size > 1`, else directly on `device`
11. If multi-rank:
    - `dist_utils.init_distributed(config.comm)`
    - create TP-only `ParallelDims`
    - call `apply_tp_minus_sp(model, parallel_dims.get_mesh("tp"))`
12. Else:
    - create a fully size-1 `ParallelDims`
13. Build `DebugConfig(seed=seed, deterministic=deterministic)`
14. Call `dist_utils.set_determinism(...)` with `distinct_seed_mesh_dims=["pp"]`
15. Materialize params via `model.to_empty(device=device_type)`
16. Initialize weights via `model.init_weights()`
17. Switch to eval mode
18. Load checkpoint with `dcp.load(state_dict, checkpoint_id=checkpoint_path)`
19. Log peak reserved memory before generation
20. Encode prompt with BOS and no EOS, then repeat for batch size
21. Reset peak memory stats
22. Call `generate(...)`
23. On rank 0:
    - decode prompt and completion text per sample
    - log colored transcript
    - populate `output_data["metadata"]`
    - optionally `print(json.dumps(output_data, indent=4))` when `args.out`

Notable invariants and quirks:
- Uses `model.state_dict()` plus `dcp.load(...)`, so checkpoint compatibility is whatever the model's current state-dict structure expects.
- `input_ids` are moved with `.to(device_type)` rather than `.to(device)`.
- `tokens_per_sec` is computed as `(B * T) / elapsed_sec`, where `T` includes prompt plus generated tokens.
- The function is not purely library-safe because it reads global `args`.

#### CLI contract
- Required:
  - `--module`
  - `--config`
  - `--checkpoint`
- Optional:
  - `--temperature`
  - `--max_new_tokens`
  - `--batch_size`
  - `--top_k`
  - `--seed`
  - `--deterministic`
  - `--prompt`
  - `--out`
- Cleanup:
  - destroys the process group if `torch.distributed.is_initialized()`

## Dependency Relationships

- Config resolution comes from `torchtitan/config`.
- Tokenizer and memory monitor come from `torchtitan/components`.
- Device mesh and deterministic seeding come from `torchtitan/distributed`.
- Logger, colors, device type, and current-device setter come from `torchtitan/tools`.
- The hard-coded TP layout assumes a Llama-style dense decoder module structure like TorchTitan's `llama3` family.

## Modification Checklist

- If you change CLI flags, update both:
  - `scripts/generate/test_generate.py`
  - `scripts/generate/README.md`
- If you change prompt-file handling, update:
  - `scripts/generate/run_llama_generate.sh`
  - any docs/examples that pass `--prompt=...`
- If you change sampling semantics, update:
  - `scripts/generate/_generation.py`
  - throughput/report wording if output meaning changes
- If you change TP behavior, validate assumptions against:
  - the model family's actual submodule names
  - `ParallelDims` mesh expectations
  - checkpoint compatibility after wrapping
