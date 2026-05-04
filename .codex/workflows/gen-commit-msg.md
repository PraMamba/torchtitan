# Codex Workflow: gen-commit-msg

Migrated from `.claude/commands/gen-commit-msg.md`.

This is a Codex workflow/reference document, not a native Claude slash command.
Invoke it by asking Codex to follow this workflow, for example: "Follow
`.codex/workflows/gen-commit-msg.md`". Preserve safety checks and user-confirmation gates
for destructive GitHub or git operations.

---

---
name: gen-commit-msg
description: Generate intelligent commit messages based on staged changes. Invoke with /gen-commit-msg.
---

# Generate Commit Message

Generate a well-formatted commit message based on staged changes.

## Usage

```
/gen-commit-msg [--amend] [--scope <scope>]
```

## Workflow

### Step 1: Analyze Changes

```bash
git diff --cached --name-only
git diff --cached
git log --oneline -5
```

### Step 2: Categorize Changes

| Type | When to Use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change without feature/fix |
| `test` | Adding or fixing tests |
| `chore` | Build, deps, config changes |
| `perf` | Performance improvement |

### Step 3: Determine Scope

Infer scope from changed files:

- `torchtitan/distributed/` -> `distributed`
- `torchtitan/models/common/` -> `models`
- `torchtitan/models/<name>/` -> `<name>`
- `torchtitan/components/` -> `components`
- `torchtitan/config/` -> `config`
- `torchtitan/protocols/` -> `protocols`
- `torchtitan/experiments/` -> `experiments`
- `torchtitan/trainer.py` -> `trainer`
- `tests/` -> `tests`
- `docs/` -> `docs`
- Multiple areas -> omit scope or use broader term

### Step 4: Generate Message

**Format:**
```
<type>(<scope>): <subject>

<body>

[Optional sections:]
Key changes:
- change 1
- change 2
```

**Rules:**
- Subject: imperative mood, ~50-72 chars, no period
- Body: explain "why" not "what", wrap at 72 chars
- Key changes: bullet list of main modifications (for complex commits)

### Step 5: Confirm and Commit

Show preview, ask user to confirm, then execute:

```bash
git commit -m "$(cat <<'EOF'
<message>
EOF
)"
```

## Examples

**Single file fix:**
```
fix(distributed): handle missing mesh dimension in EP validation

Return early when EP is disabled instead of asserting on mesh size.
```

**Multi-file feature:**
```
feat(models): add shared attention mask utilities

Extract causal mask creation from model-specific code into
torchtitan/models/common/attention.py for reuse across all models.

Key changes:
- Add create_causal_mask() to common/attention.py
- Update llama3, qwen3 to use shared implementation
- Add unit tests for mask creation
```
