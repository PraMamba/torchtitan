# AGENTS.md for `.codex` migration assets

This file is intentionally scoped to the `.codex/` directory and its descendants.
It documents how to maintain the migrated Codex compatibility layer; it is **not** a
repo-root instruction file for all TorchTitan source changes.

## Maintenance principles

- Preserve source traceability back to `.claude` assets.
- Prefer information fidelity when updating migrated rules, skills, agents, or workflows.
- Keep Codex operational artifacts valid:
  - custom agents live in `.codex/agents/*.toml`
  - custom agents must define `name`, `description`, and `developer_instructions`
  - skills live in `.codex/skills/<skill-name>/SKILL.md`
- Do not add executable Codex hook configuration here as part of this migration.
- If repository-wide enforcement is desired later, create or update a repo-root
  `AGENTS.md` in a separate, explicit change.

## Verification before declaring changes complete

Run the coverage and TOML checks documented in `.codex/README.md` and
`.omx/plans/test-spec-claude-to-codex-compat.md` after changing these assets.
