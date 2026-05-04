# Codex Compatibility Layer for TorchTitan Claude Workflows

This directory contains a Codex-compatible adaptation of the repository's
Claude Code workflow assets from `.claude/`.

## Migration policy

- **Information fidelity first:** migrated files preserve the useful source
  content from `.claude` even when an artifact is reference-oriented.
- **Codex validity required:** Claude-only semantics are wrapped or documented so
  they do not masquerade as native Codex behavior.
- **Additive migration:** this layer does not delete or modify `.claude`.
- **Hooks are reference-only:** `.claude/hooks` was intentionally not converted
  into executable Codex hook configuration.

## Directory layout

- `agents/` — Codex custom agents generated from `.claude/agents/*.md`.
- `rules/` — source-preserving scoped rule references from `.claude/rules/*.md`.
- `skills/` — Codex-adapted skills from `.claude/skills/*/SKILL.md`.
- `workflows/` — Codex-invokable workflow docs from `.claude/commands/*.md`.
- `PROJECT_GUIDE.md` — source-preserving project guidance assembled from
  `CLAUDE.md`, `.claude/CLAUDE.md`, and rule references.
- `AGENTS.md` — guidance scoped only to `.codex/` maintenance.
- `reference/` — settings and hook non-migration notes.

## Source-to-target mapping

| Source | Target | Adaptation |
|---|---|---|
| `.claude/agents/checkpoint-expert.md` | `.codex/agents/checkpoint-expert.toml` | Codex custom agent |
| `.claude/agents/code-verifier.md` | `.codex/agents/code-verifier.toml` | Codex custom agent |
| `.claude/agents/config-expert.md` | `.codex/agents/config-expert.toml` | Codex custom agent |
| `.claude/agents/distributed-expert.md` | `.codex/agents/distributed-expert.toml` | Codex custom agent |
| `.claude/agents/model-expert.md` | `.codex/agents/model-expert.toml` | Codex custom agent |
| `.claude/agents/planner.md` | `.codex/agents/planner.toml` | Codex custom agent |
| `.claude/agents/trainer-expert.md` | `.codex/agents/trainer-expert.toml` | Codex custom agent |
| `.claude/rules/code-style.md` | `.codex/rules/code-style.md` | Source-preserving scoped rule reference |
| `.claude/rules/config.md` | `.codex/rules/config.md` | Source-preserving scoped rule reference |
| `.claude/rules/distributed.md` | `.codex/rules/distributed.md` | Source-preserving scoped rule reference |
| `.claude/rules/experiments.md` | `.codex/rules/experiments.md` | Source-preserving scoped rule reference |
| `.claude/rules/models.md` | `.codex/rules/models.md` | Source-preserving scoped rule reference |
| `.claude/rules/testing.md` | `.codex/rules/testing.md` | Source-preserving scoped rule reference |
| `.claude/skills/add-experiment/SKILL.md` | `.codex/skills/add-experiment/SKILL.md` | Codex-adapted skill |
| `.claude/skills/add-model/SKILL.md` | `.codex/skills/add-model/SKILL.md` | Codex-adapted skill |
| `.claude/skills/debug-distributed/SKILL.md` | `.codex/skills/debug-distributed/SKILL.md` | Codex-adapted skill |
| `.claude/skills/torch_bisect/SKILL.md` | `.codex/skills/torch_bisect/SKILL.md` | Codex-adapted skill |
| `.claude/skills/validate-numerics/SKILL.md` | `.codex/skills/validate-numerics/SKILL.md` | Codex-adapted skill |
| `.claude/commands/create-pr.md` | `.codex/workflows/create-pr.md` | Codex workflow/reference doc |
| `.claude/commands/gen-commit-msg.md` | `.codex/workflows/gen-commit-msg.md` | Codex workflow/reference doc |
| `.claude/commands/review-pr.md` | `.codex/workflows/review-pr.md` | Codex workflow/reference doc |
| `.claude/settings.json` | `.codex/reference/settings.md` | Reference-only settings/permissions notes |
| `.claude/settings.local.json` | `.codex/reference/settings.md` | Reference-only settings/permissions notes |
| `.claude/hooks/check-expert-update.sh` | `.codex/reference/hooks.md` | Reference-only; not executable hook migration |

## Verification

Run the canonical verification commands from
`.omx/plans/test-spec-claude-to-codex-compat.md` after changing this directory.
Those checks validate generated file inventory, custom-agent TOML parseability,
source-to-target coverage, absence of executable hook activation, and source
`.claude` non-modification.

## Notes on `.codex/config.toml`

This migration intentionally does not create `.codex/config.toml`. Standalone
`.codex/agents/*.toml` files are the Codex custom-agent artifacts, and avoiding
project config prevents accidental hooks, sandbox, approval, model pin, or trust
settings.
